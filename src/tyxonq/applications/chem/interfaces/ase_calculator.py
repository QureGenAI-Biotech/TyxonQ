"""``TyxonQCalculator``：ASE Calculator 适配器——分子动力学生态的唯一枢纽。

所有上层适配（ASE 优化/MD、i-PI driver、OpenMM MLPotential）都经由本类
消费 ``qc_scanner``。内部链路::

    atoms.get_positions() (Å)  →  Bohr 坐标  →  qc_scanner  →  (E Ha, dE/dR Ha/Bohr)
    E → eV；F = -dE/dR → eV/Å（ASE 约定力是能量的负梯度，必须取负）

单位换算常数只在本文件出现一次；对外接口是标准 ASE Calculator::

    from tyxonq.applications.chem.interfaces.ase_calculator import TyxonQCalculator
    atoms.calc = TyxonQCalculator(active_space=(4, 4), method="uccsd")
    atoms.get_potential_energy()   # eV
    atoms.get_forces()             # eV/Å

注意：``interfaces/__init__.py`` 不导出本类（保持包级惰性），需要时按
子模块路径导入；本模块顶层导入 ASE，缺失时构造会给出安装指引。
不声明 ``stress``：本期不支持周期体系（计划 §7 风险 R3），
ASE/i-PI 会明确报属性缺失而不是静默给零。

QM/MM 区域划分模式（E8 阶段 A，见 ``MD_INTEGRATION_PLAN.md`` §5 E8-A）：
传入 ``qm_indices`` + ``atom_charges`` 后，``Atoms`` 视为全体系：
``qm_indices`` 选 QM 子集走 ``qc_scanner``（静电嵌入 MM 电荷），其余原子作
MM 点电荷环境，每步经 ``set_mm_charges`` 更新；输出的全原子力 = QM 梯度 +
MM 反作用力（``mm_gradient``）。**energy 只含 QM 部分**（嵌入后的 E_QM），
MM 力场能由另一力场（如 LAMMPS）提供、由 i-PI ``<forces>`` 求和——
这正是加和型静电嵌入 QM/MM 的标准分工。

周期性固体嵌入（E8 阶段 B）：区域划分模式下另传 ``mm_lattice``（3×3 对角，
单位同 ``unit``）+ ``rcut_ewald``/``rcut_hcore``，嵌入层换成 ``pyscf.qmmm.pbc``
Ewald 求和（QM 区感受 MM 点电荷的全部周期镜像）。晶格固定（不随 ``Atoms.cell``
变）；MM–MM 周期静电由 MM 引擎（如 LAMMPS ``kspace_style pppm``）负责，
与嵌入的 QM–MM 周期静电互补不重叠（QM 原子在 MM 引擎侧电荷置 0）。
"""

from __future__ import annotations

import numpy as np

from .scanner import qc_scanner

# ---- 单位换算（附录 B；只允许在本文件出现） ----
BOHR_TO_ANGSTROM = 0.52917721092  # 1 Bohr = 0.52917721092 Å
HARTREE_TO_EV = 27.211386245988   # 1 Hartree = 27.211386245988 eV

try:
    from ase.calculators.calculator import Calculator as _AseCalculator, all_changes as _all_changes
except ImportError:  # ASE 未安装：提供占位类，构造时给出安装指引
    _AseCalculator = None
    _all_changes = None


class TyxonQCalculator(_AseCalculator if _AseCalculator is not None else object):
    """ASE Calculator，势能面由 ``qc_scanner`` 提供。

    构造参数与 :func:`qc_scanner` 同构（除 ``atom``——几何来自 ASE ``Atoms``，
    元素符号在首次计算时自动读取）。关键参数：

    - ``basis`` / ``charge`` / ``spin`` / ``unit``：平均场设置；
    - ``active_space=(n_elec, n_orb)``：活性空间；
    - ``method``：``"uccsd"`` / ``"rouccsd"`` / ``"hea"``（``"sqd"`` 暂缓）;
    - ``solver_kwargs``：透传给求解器（如 HEA 的 ``runtime="numeric"``）;
    - ``mm_charges=(coords, charges)``：静电嵌入（坐标单位由 ``unit`` 声明）。
    - ``qm_indices``：QM 区原子下标（区域划分模式；``None`` = 全体系皆 QM）；
    - ``atom_charges``：全体系固定电荷表（长度 = 原子数，QM 条目被忽略），
      与 ``qm_indices`` 配套：补集原子按此表作 MM 点电荷环境，每步更新。
      区域划分模式下不得同时传 ``mm_charges``。
    - ``mm_lattice`` / ``rcut_ewald`` / ``rcut_hcore``：周期性固体嵌入（E8 阶段 B，
      仅区域划分模式），语义与 :func:`qc_scanner` 同名参数一致；守卫与已知近似
      见 ``scanner.py`` 与 ``examples/qmmm/md_lammps_qmmm_pbc/VALIDATION.md``。

    ``label`` 等 ASE Calculator 通用参数透传给基类。
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        *,
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
        unit: str = "Angstrom",
        active_space: tuple[int, int],
        method: str = "uccsd",
        sampler=None,
        solver_kwargs: dict | None = None,
        mm_charges=None,
        qm_indices=None,
        atom_charges=None,
        mm_lattice=None,
        rcut_ewald: float | None = None,
        rcut_hcore: float | None = None,
        allow_discontinuous: bool = False,
        verbose: int = 0,
        **ase_kwargs,
    ):
        if _AseCalculator is None:
            raise ImportError(
                "TyxonQCalculator requires ASE. Install it with: "
                "pip install 'tyxonq[md]'   (or: pip install 'ase>=3.23')"
            )
        if qm_indices is not None:
            if mm_charges is not None:
                raise ValueError(
                    "Region-partition mode (qm_indices) supplies MM charges per step "
                    "from atom_charges; do not pass static mm_charges at the same time."
                )
            if atom_charges is None:
                raise ValueError("qm_indices requires atom_charges (per-atom fixed charges).")
        elif mm_lattice is not None:
            raise ValueError(
                "mm_lattice (pbc Ewald embedding) requires region-partition mode "
                "(qm_indices + atom_charges): MM charges must be supplied per step."
            )
        super().__init__(**ase_kwargs)
        self.qm_indices = None if qm_indices is None else np.asarray(qm_indices, dtype=int)
        self.atom_charges = None if atom_charges is None else np.asarray(atom_charges, dtype=float)
        self.tq_kwargs = dict(
            basis=basis,
            charge=charge,
            spin=spin,
            unit=unit,
            active_space=tuple(active_space),
            method=method,
            sampler=sampler,
            solver_kwargs=solver_kwargs,
            mm_charges=mm_charges,
            mm_lattice=mm_lattice,
            rcut_ewald=rcut_ewald,
            rcut_hcore=rcut_hcore,
            allow_discontinuous=allow_discontinuous,
            verbose=verbose,
        )
        self._scan = None

    # ---- ASE 契约 ----

    def calculate(self, atoms=None, properties=("energy",), system_changes=None):
        super().calculate(atoms, properties, system_changes or _all_changes)

        if self.qm_indices is not None:
            self._calculate_qmmm(atoms)
            return

        # 首次计算时按当前 Atoms 的元素符号构建 scanner 配方；
        # 此后几何更新只走 coords 分支（HF 热启动、SCF 不重复冷启动）。
        if self._scan is None:
            atom_spec = [
                (sym, tuple(map(float, pos)))
                for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions())
            ]
            self._scan = qc_scanner(atom_spec, **self.tq_kwargs)

        coords_bohr = atoms.get_positions() / BOHR_TO_ANGSTROM
        e_hartree, de = self._scan(coords_bohr)

        # ASE 约定：energy 单位 eV，forces = -dE/dR 单位 eV/Å。
        self.results["energy"] = float(e_hartree) * HARTREE_TO_EV
        self.results["free_energy"] = self.results["energy"]  # 0 K，二者相同
        self.results["forces"] = -np.asarray(de) * HARTREE_TO_EV / BOHR_TO_ANGSTROM

    def _calculate_qmmm(self, atoms):
        """区域划分模式：QM 子集走 qc_scanner，MM 补集作每步更新的点电荷环境。

        全原子梯度 = QM 子集梯度（含嵌入贡献）⊕ MM 反作用力；
        energy 只是嵌入后的 E_QM（MM 力场能由另一力场在 i-PI 侧求和）。
        """
        qm_idx = self.qm_indices
        natm = len(atoms)
        if qm_idx.size == 0 or qm_idx.max() >= natm or qm_idx.min() < 0:
            raise ValueError(f"qm_indices={qm_idx.tolist()} invalid for {natm} atoms.")
        if self.atom_charges.shape[0] != natm:
            raise ValueError(
                f"atom_charges has {self.atom_charges.shape[0]} entries but Atoms has {natm}."
            )
        mask = np.ones(natm, dtype=bool)
        mask[qm_idx] = False
        mm_idx = np.flatnonzero(mask)

        positions = atoms.get_positions()  # Å
        if self._scan is None:
            symbols = atoms.get_chemical_symbols()
            qm_spec = [(symbols[i], tuple(map(float, positions[i]))) for i in qm_idx]
            kwargs = dict(self.tq_kwargs)
            if kwargs.get("mm_lattice") is not None:
                # pbc 模式：scanner 守卫要求 mm_charges 构造时在场（分子版则保持
                # “初始不带、首帧 set_mm_charges 填配方”的原行为）。两者都在
                # 首帧之前到位，后续每步照旧走 set_mm_charges。
                kwargs["mm_charges"] = (positions[mm_idx], self.atom_charges[mm_idx])
            self._scan = qc_scanner(qm_spec, **kwargs)

        # MM 坐标单位与 scanner 的 self.unit（缺省 Angstrom）一致；
        # 电荷只取 MM 条目（QM 条目的嵌入意义由 QM 哈密顿量自身承担）。
        self._scan.set_mm_charges(positions[mm_idx], self.atom_charges[mm_idx])

        coords_bohr = positions[qm_idx] / BOHR_TO_ANGSTROM
        e_hartree, de_qm = self._scan(coords_bohr)
        de_mm = self._scan.mm_gradient()

        de = np.zeros((natm, 3))
        de[qm_idx] = de_qm
        de[mm_idx] = de_mm

        self.results["energy"] = float(e_hartree) * HARTREE_TO_EV
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = -de * HARTREE_TO_EV / BOHR_TO_ANGSTROM
