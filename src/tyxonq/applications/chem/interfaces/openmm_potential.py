"""``openmm_potential``：TyxonQ 的 OpenMM 直连适配（``openmmml`` 的薄封装）。

链路（计划 §3 P3）::

    TyxonQCalculator(ASE)  →  openmmml.MLPotential('ase')  →  openmm.PythonForce
                                                              （每步回调取 E/F）

单位换算不在本文件发生：OpenMM 侧的 kJ/mol/nm 回转由上游
``openmmml/models/asepotential.py`` 的回调负责，ASE 层的 eV/Å ↔ Ha/Bohr
由 :class:`TyxonQCalculator` 负责（附录 B 常数只出现在 ``ase_calculator.py``）。

模块提供两条 QM/MM 物理层级，按需选择：

- **机械嵌入**（:func:`create_mixed_system`，经上游 ``createMixedSystem``）：
  QM 区只算自身内部能量，QM–MM 相互作用（含静电）全部按经典力场处理，
  QM 电子结构**感受不到 MM 静电极化**。
- **静电嵌入**（:func:`create_qmmm_ee_system`，本模块自组装）：MM 点电荷经
  ``pyscf.qmmm`` 进入 QM 哈密顿量、每步随坐标更新；QM 梯度与 MM 反作用力
  （``mm_gradient``）由同一个 ``PythonForce`` 回调返回，MM 力场能仍由经典力提供。
  与 i-PI 三进程路径（``examples/qmmm/md_lammps_qmmm_embedded/`` 与
  ``md_lammps_qmmm_pbc/``）物理等价，但单进程、无外部通信；簇嵌入版（分子版）。
  周期性固体嵌入仍走 i-PI + LAMMPS pppm 路径（E8-B）或 P4 的 MDI engine。

``openmm`` / ``openmmml`` 惰性导入：本模块顶层不触碰它们，缺失时在
调用处报带安装指引的 ``ImportError``（``pip install "tyxonq[md]"``）。
与 ``ase_calculator.py`` 相同，``interfaces/__init__.py`` 不导出本模块。

已知上游细节（照用不误，但读轨迹时须知）：混合体系（``create_mixed_system``）
模式下，上游回调把 MM 侧力数组建为 ``np.float32``
（``asepotential.py`` 的 ``_computeASE``），即混合体系力精度约 1e-7 相对；
纯 QM 路径无此截断。
"""

from __future__ import annotations

from functools import partial

_INSTALL_HINT = (
    "This function requires OpenMM and OpenMM-ML. Install them with: "
    "pip install 'tyxonq[md]'   (or: pip install 'openmm>=8.1' 'openmmml>=1.7')"
)


def _require_stack():
    """惰性导入 openmm/openmmml/TyxonQCalculator，缺失时给安装指引。"""
    try:
        import openmm  # noqa: F401
        import openmm.app  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    try:
        from openmmml import MLPotential
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    from .ase_calculator import TyxonQCalculator

    return MLPotential, TyxonQCalculator


def _calculator_and_info(info, scanner_kwargs):
    """组装 TyxonQCalculator 与透传 info：``info['charge']`` 映射到 ``charge=``。

    OpenMM 侧约定总电荷放在 ``info={'charge': ...}``（上游 ``asepotential``
    会写进 ``Atoms.info``）；这里同时把它映射到 ``qc_scanner(charge=...)``，
    显式传的 ``charge`` 优先（``setdefault`` 语义）。
    """
    kwargs = dict(scanner_kwargs)
    if info and "charge" in info:
        kwargs.setdefault("charge", int(info["charge"]))
    return kwargs


def create_tyxonq_system(topology, *, info=None, **scanner_kwargs):
    """纯 QM 的 OpenMM ``System``：整个体系由 ``qc_scanner`` 势能面驱动。

    Parameters
    ----------
    topology:
        ``openmm.app.Topology``，所有原子必须声明元素（上游校验）。
    info:
        可选 dict，写入 ASE ``Atoms.info``；含 ``'charge'`` 时自动映射到
        ``qc_scanner(charge=...)``（显式 ``charge=`` 优先）。
    scanner_kwargs:
        与 :class:`TyxonQCalculator` 同构的参数（``active_space`` 必填，
        ``method`` 缺省 ``"uccsd"``）。

    Returns
    -------
    ``openmm.System``：内部是一个 ``PythonForce``，每步经 ``TyxonQCalculator``
    求 (E, F)。
    """
    MLPotential, TyxonQCalculator = _require_stack()
    kwargs = _calculator_and_info(info, scanner_kwargs)
    calc = TyxonQCalculator(**kwargs)
    potential = MLPotential("ase")
    if info is None:
        return potential.createSystem(topology, calculator=calc)
    return potential.createSystem(topology, calculator=calc, info=info)


def create_mixed_system(
    topology,
    mm_system,
    qm_atoms,
    *,
    interpolate=False,
    removeConstraints=True,
    forceGroup=0,
    info=None,
    **scanner_kwargs,
):
    """QM/MM 混合 ``System``：``qm_atoms`` 子集走 TyxonQ，其余照 ``mm_system``。

    上游 ``createMixedSystem`` 语义：删去 QM 子集内部的键/角/二面角与
    nonbonded 自作用（改由 QM 势能面负责），QM–MM 跨区相互作用仍按
    ``mm_system`` 的经典力场算——即**机械嵌入**（见模块 docstring）。

    Parameters
    ----------
    topology:
        全体系 ``openmm.app.Topology``。
    mm_system:
        全体系的经典力场 ``openmm.System``（如 ``ForceField.createSystem`` 产物）。
    qm_atoms:
        QM 子集原子下标。注意 ``scanner_kwargs`` 的 ``active_space`` 等必须
        描述这个**子集**（上游只把子集坐标喂给 calculator）。
    interpolate:
        ``True`` 时产出带全局参数 ``lambda_interpolate`` 的 ``CustomCVForce``：
        0 = 纯经典、1 = 纯 TyxonQ，可经 ``Context.setParameter`` 做 MM↔QM
        自由能微扰。
    info / scanner_kwargs:
        同 :func:`create_tyxonq_system`。
    """
    MLPotential, TyxonQCalculator = _require_stack()
    kwargs = _calculator_and_info(info, scanner_kwargs)
    calc = TyxonQCalculator(**kwargs)
    potential = MLPotential("ase")
    args = dict(calculator=calc, removeConstraints=removeConstraints,
                forceGroup=forceGroup, interpolate=interpolate)
    if info is not None:
        args["info"] = info
    return potential.createMixedSystem(topology, mm_system, qm_atoms, **args)


def create_qmmm_ee_system(
    topology,
    mm_system,
    qm_atoms,
    atom_charges,
    *,
    removeConstraints=True,
    forceGroup=0,
    **scanner_kwargs,
):
    """QM/MM **静电嵌入** ``System``（簇嵌入版）：MM 点电荷进 QM 哈密顿量。

    与 :func:`create_mixed_system`（机械嵌入）的区别：本函数用区域划分模式的
    :class:`TyxonQCalculator`（``qm_indices`` + ``atom_charges``）替换 QM 区，
    MM 电荷每步经 ``set_mm_charges`` 注入嵌入层；回调返回全原子力 =
    QM 梯度（含嵌入贡献）⊕ MM 反作用力（``mm_gradient``）。

    防双计数协议（与 E8 的 LAMMPS 侧一一对应）：
    1. QM 原子在 ``NonbondedForce`` 中的电荷置 0 → QM–MM 库仑只由嵌入算一份；
    2. QM 子集内部的键/角/二面角与约束删除（复用上游 ``_removeBonds`` 协议），
       QM 原子对之间的 nonbonded/例外全清零（QM 内部全归量子）；
    3. QM–MM 的 vdW 与 MM–MM 全部经典项（含静电）原样保留。

    Parameters
    ----------
    topology:
        全体系 ``openmm.app.Topology``（须声明元素）。
    mm_system:
        全体系的经典力场 ``openmm.System``。
    qm_atoms:
        QM 子集原子下标；``scanner_kwargs`` 的 ``active_space`` 描述该子集。
    atom_charges:
        全体系固定电荷表（长度 = 原子数），与区域划分模式同义：
        补集原子按此表作 MM 点电荷环境；QM 条目被忽略（嵌入由哈密顿量承担）。
        QM 原子在 ``mm_system`` 中的经典电荷仍被本函数置 0 防双计数。
    removeConstraints / forceGroup:
        同上游 ``createMixedSystem`` 同名参数。
    scanner_kwargs:
        透传给 :class:`TyxonQCalculator`（``active_space`` 必填）。本路径是簇嵌入，
        不接受 ``mm_lattice``/``rcut_ewald``/``rcut_hcore``（pbc 走 E8-B）。

    Returns
    -------
    ``openmm.System``：经典力场（改写后）+ 一个 ``PythonForce``（嵌入 QM 区）。
    """
    import openmm

    MLPotential, TyxonQCalculator = _require_stack()

    pbc_keys = ("mm_lattice", "rcut_ewald", "rcut_hcore")
    if any(scanner_kwargs.get(k) is not None for k in pbc_keys):
        raise ValueError(
            "create_qmmm_ee_system is the cluster (molecular) embedding path; "
            "periodic Ewald embedding (mm_lattice/rcut_*) belongs to the "
            "i-PI + LAMMPS route (examples/qmmm/md_lammps_qmmm_pbc/)."
        )

    qm_set = set(int(i) for i in qm_atoms)
    natoms = mm_system.getNumParticles()
    if len(atom_charges) != natoms:
        raise ValueError(
            f"atom_charges has {len(atom_charges)} entries but mm_system has {natoms} particles."
        )

    # 1) 删 QM 子集内部键/角/二面角与约束（上游 _removeBonds 协议；self 不参与逻辑）。
    new_system = MLPotential("ase")._removeBonds(mm_system, list(qm_set), True, removeConstraints)

    for force in new_system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            # 2) QM 原子电荷置 0（QM–MM 库仑只由嵌入算）；LJ 参数保留。
            for i in qm_set:
                _charge, sigma, epsilon = force.getParticleParameters(i)
                force.setParticleParameters(i, 0.0, sigma, epsilon)
            # 3) QM 原子对全部清零（内部归量子）；含 QM 的例外库仑项清零。
            for i in sorted(qm_set):
                for j in qm_set:
                    if j < i:
                        force.addException(j, i, 0.0, 1.0, 0.0, True)
            for k in range(force.getNumExceptions()):
                p1, p2, charge_prod, sigma, epsilon = force.getExceptionParameters(k)
                if (p1 in qm_set or p2 in qm_set) and charge_prod._value != 0.0:
                    force.setExceptionParameters(k, p1, p2, 0.0, sigma, epsilon)

    # 4) 区域划分模式的 TyxonQCalculator：全体系 Atoms 由回调逐帧 set_positions
    #    （上游 asepotential 同款语义），电荷经 set_mm_charges 每步注入。
    import ase

    calc = TyxonQCalculator(qm_indices=sorted(qm_set), atom_charges=list(atom_charges),
                            **scanner_kwargs)
    numbers = [atom.element.atomic_number for atom in topology.atoms()]
    qm_atoms_obj = ase.Atoms(numbers=numbers, calculator=calc)
    qm_force = openmm.PythonForce(partial(_compute_qmmm_ee, atoms=qm_atoms_obj))
    qm_force.setForceGroup(forceGroup)
    new_system.addForce(qm_force)
    return new_system


def _compute_qmmm_ee(state, atoms):
    """``PythonForce`` 回调：嵌入单点 + 全原子力（QM 梯度 ⊕ MM 反作用力）。

    与上游 ``asepotential._computeASE`` 同款回转：能量 eV→kJ/mol、
    力 eV/Å→kJ/mol/nm（除 ``ase.units.kJ/ase.units.mol``）。本路径力数组保持
    全精度 float64（不走上游混合模式的 float32 截断）。
    """
    import ase.units
    from openmm import unit

    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    atoms.set_positions(positions)
    energy = atoms.get_potential_energy(apply_constraint=False)   # 嵌入后的 E_QM，eV
    forces = atoms.get_forces(apply_constraint=False)              # 全原子，eV/Å
    return energy / (ase.units.kJ / ase.units.mol), forces * 10 / (ase.units.kJ / ase.units.mol)
