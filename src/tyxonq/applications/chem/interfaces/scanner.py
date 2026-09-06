"""``qc_scanner``：TyxonQ 进入分子动力学生态的唯一门面。

产出一个与 ``pyscf.grad`` scanner 同契约的可调用对象：

    scanner(geometry_bohr) -> (e_tot, de)    # Hartree / Hartree·Bohr⁻¹

核梯度全部复用 ``pyscf/grad/``（依据 ``MD_INTEGRATION_RESEARCH.md`` §3、§4.1），
本文件不重写任何梯度代码。

子空间策略（仅 ``method="sqd"``）：

- ``"frozen"``（默认）：首帧确定行列式子空间后锁存，能量面光滑，
  是唯一允许用于解析梯度 / MD 的模式（§4.4）。
- ``"refresh"`` / ``"adaptive"``：能量面不光滑或分段光滑，默认拒绝；
  需要显式 ``allow_discontinuous=True`` 才放行（§4.4 D：随机子空间下
  力噪声约 4.3e-03 Hartree/Bohr，解析梯度在数学上没有定义）。

静电嵌入的 MD 支持（E8 阶段 A）：

- ``mm_charges=(coords, charges)`` 构造时给定初始 MM 环境；
- :meth:`QCScanner.set_mm_charges` 每个 MD 步更新 MM 坐标/电荷，
  走上游 ``qmmm_for_scf`` 重入路径（对已装饰对象只是 ``mm_mol`` 重赋值，
  ``pyscf/qmmm/itrf.py`` L99-101），不重建平均场；
- :meth:`QCScanner.mm_gradient` 返回当前态对 MM 粒子的梯度（反作用力），
  复用上游 ``QMMMGrad.grad_hcore_mm``（电子部分）+ ``grad_nuc_mm``（核部分），
  密度矩阵取 CASCI 的 AO 1-RDM，不手写任何梯度代码。

周期性固体嵌入（E8 阶段 B，``pyscf.qmmm.pbc`` Ewald 求和）：

- ``mm_charges`` + ``mm_lattice``（3×3 对角，单位同 ``unit``）启用；
  ``rcut_ewald``/``rcut_hcore`` 必给（上游缺省启发式不满足自身校验，
  RB4），构建期与每期都执行带源码位置的显式守卫（RB2/RB3/RB10）；
- :meth:`QCScanner.set_mm_charges` 的 pbc 分支走上游重入路径（重置 5 个
  缓存，``pyscf/qmmm/pbc/itrf.py`` L89-96）并重跑平均场（RB7 协议）；
- :meth:`QCScanner.mm_gradient` 的 pbc 分支另加 ``grad_ewald`` 的 MM 分量。
  已知近似：上游解析式缺 post-HF 轨道响应项（基准偏置 ~4e-5 Ha/Bohr，
  相对净嵌入力 ~3%，光滑；严格 NVE 能量守恒诊断失效），见
  ``examples/qmmm/md_lammps_qmmm_pbc/VALIDATION.md`` §4。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from pyscf import gto, mcscf, qmmm, scf
from pyscf.qmmm.pbc.itrf import add_mm_charges as _pbc_add_mm_charges

from ..algorithms.sqd import pyscf_solver as _sqd_solver

# 对外只暴露具体入口：sqd（SQD 求解器）与 uccsd.py/hea.py 的具体类。
# UCC 是基类，不作为可选方法暴露；开壳层用 ROUCCSD。
_METHOD_REGISTRY = ("sqd", "uccsd", "rouccsd", "hea")


def _make_fcisolver(method: str, subspace: str, sampler, solver_kwargs: dict):
    """按方法名构造 PySCF fcisolver 鸭子类型对象。"""
    solver_kwargs = dict(solver_kwargs)
    if method == "sqd":
        return _sqd_solver.as_pyscf_solver(subspace=subspace, sampler=sampler, **solver_kwargs)

    # VQE 族：惰性导入，避免仅为用 SQD 的用户拉起 openfermion 电路栈。
    # 入口文件是 algorithms/vqe/uccsd.py（UCCSD 闭壳 / ROUCCSD 开壳）与 vqe/hea.py。
    if method in ("uccsd", "rouccsd"):
        if method == "uccsd":
            from ..algorithms.vqe.uccsd import UCCSD as _cls
        else:
            from ..algorithms.vqe.uccsd import ROUCCSD as _cls
        # 真机/采样运行选项（shots/provider/device）不是 UCCSD 构造参数，
        # 打包成 device_opts 透传给 kernel（device 路径，含真机提交）。
        solver_kwargs.setdefault("runtime", "numeric")
        device_opts = {k: solver_kwargs.pop(k) for k in ("shots", "provider", "device")
                       if k in solver_kwargs}
        if device_opts:
            merged = dict(solver_kwargs.get("device_opts") or {})
            merged.update(device_opts)
            solver_kwargs["device_opts"] = merged
        return _cls.as_pyscf_solver(**solver_kwargs)

    from ..algorithms.vqe.hea import HEA

    # 真机/采样运行选项（shots/provider/device）不是 HEA 构造参数，
    # 打包成 device_opts 透传给 HEA.kernel（device 路径，含真机提交）。
    device_opts = {k: solver_kwargs.pop(k) for k in ("shots", "provider", "device")
                   if k in solver_kwargs}
    if device_opts:
        merged = dict(solver_kwargs.get("device_opts") or {})
        merged.update(device_opts)
        solver_kwargs["device_opts"] = merged
    return HEA.as_pyscf_solver(**solver_kwargs)


class QCScanner:
    """惰性重建几何的量子化学能量/梯度扫描器。

    持有构建配方（方法、基组、活性空间、求解器对象），每次调用在
    新几何上重建 ``gto.Mole`` 与平均场；冻结子空间保存在求解器对象里，
    跨几何复用，这正是 ``subspace="frozen"`` 的语义。
    """

    def __init__(
        self,
        atom: str | list,
        *,
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
        unit: str = "Angstrom",
        active_space: tuple[int, int],
        method: str = "sqd",
        sampler: Callable | None = None,
        subspace: str = "frozen",
        solver_kwargs: dict | None = None,
        mm_charges: tuple[np.ndarray, np.ndarray] | None = None,
        mm_lattice: np.ndarray | None = None,
        rcut_ewald: float | None = None,
        rcut_hcore: float | None = None,
        allow_discontinuous: bool = False,
        verbose: int = 0,
    ):
        if method not in _METHOD_REGISTRY:
            raise ValueError(f"method must be one of {_METHOD_REGISTRY}, got {method!r}.")
        if method == "sqd" and subspace != "frozen" and not allow_discontinuous:
            raise ValueError(
                f"subspace={subspace!r} makes the energy surface (piecewise) non-smooth: "
                "analytic gradients are not defined for a stochastic subspace "
                "(force noise ~4.3e-03 Hartree/Bohr, see MD_INTEGRATION_RESEARCH.md §4.4 D). "
                "Use subspace='frozen' for MD, or pass allow_discontinuous=True "
                "if you knowingly accept a non-smooth energy surface."
            )
        n_elec, n_orb = active_space
        if n_elec < 1 or n_orb < 1 or n_elec > 2 * n_orb:
            raise ValueError(f"Invalid active_space={active_space!r}.")

        self.mm_lattice = None
        self.rcut_ewald = rcut_ewald
        self.rcut_hcore = rcut_hcore
        if mm_lattice is not None:
            if mm_charges is None:
                raise ValueError("mm_lattice requires mm_charges=(coords, charges).")
            if rcut_ewald is None or rcut_hcore is None:
                raise ValueError(
                    "rcut_ewald and rcut_hcore (in `unit`) are required with mm_lattice: "
                    "the upstream defaults are heuristics that fail upstream's own "
                    "validation (pyscf/qmmm/pbc/mm_mole.py L56-61, RB4)."
                )
            a = np.asarray(mm_lattice, dtype=float)
            if a.shape != (3, 3):
                raise ValueError(f"mm_lattice must be (3,3), got shape {a.shape}.")
            # RB2：上游只支持对角晶格（pbc/mm_mole.py L54 assert）。
            if np.linalg.norm(a - np.diag(np.diag(a))) > 1e-12:
                raise ValueError(
                    "mm_lattice must be diagonal: upstream pyscf/qmmm/pbc supports "
                    "orthorhombic boxes only (mm_mole.py L54)."
                )
            box_min = float(np.min(np.abs(np.diag(a))))
            # RB2：rcut_ewald < 最小盒边（pbc/mm_mole.py L63 assert）；实空间只生成
            # 最近邻 27 胞镜像（get_lattice_Ls），超出即丢镜像。
            if rcut_ewald >= box_min:
                raise ValueError(
                    f"rcut_ewald={rcut_ewald} must be < min box edge {box_min} "
                    "(pyscf/qmmm/pbc/mm_mole.py L63)."
                )
            # RB3/RB10：rcut_hcore 必须小于半盒边（QM 镜像在盒边处，不是半对角线；
            # pyscf/qmmm/pbc/itrf.py get_hcore L176）。
            if rcut_hcore >= 0.5 * box_min:
                raise ValueError(
                    f"rcut_hcore={rcut_hcore} must be < half the min box edge "
                    f"{0.5 * box_min}: the nearest QM image sits at the box edge, "
                    "not the half-diagonal (pyscf/qmmm/pbc/itrf.py get_hcore L176)."
                )
            self.mm_lattice = a

        self.atom = atom
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.unit = unit
        self.active_space = (int(n_elec), int(n_orb))
        self.method = method
        self.subspace = subspace
        self.sampler = sampler
        self.solver_kwargs = dict(solver_kwargs or {})
        self.mm_charges = mm_charges
        self.verbose = verbose

        self.fcisolver = _make_fcisolver(method, subspace, sampler, self.solver_kwargs)
        self._scanner: Callable | None = None

    # ---- 内部构建 ----

    def _build(self, coords_bohr: np.ndarray | None = None):
        """按配方构建一套 (mol, mf, mc, scanner)。``coords_bohr`` 覆盖初始几何。

        注意：``coords_bohr`` 给出时必须以 Bohr 坐标 + 元素符号直接建 mol，
        不能先按 ``self.unit`` 建再 ``set_geom_``（mol.unit 在 build 后固化，
        会造成坐标单位被错误解读）。
        """
        if coords_bohr is not None:
            coords = np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
            symbols = self._element_symbols()
            if len(symbols) != coords.shape[0]:
                raise ValueError(f"Geometry has {coords.shape[0]} atoms but spec has {len(symbols)}.")
            atom_spec = [(sym, c.tolist()) for sym, c in zip(symbols, coords)]
            mol = gto.M(
                atom=atom_spec,
                basis=self.basis,
                charge=self.charge,
                spin=self.spin,
                unit="Bohr",
                verbose=self.verbose,
            )
            # 保留 mol.unit='Bohr'：pyscf SCF_GradScanner 收到裸坐标数组时，
            # 内部 set_geom_ 按 mol.unit 解读，二者一致才不会拉伸几何。
        else:
            mol = gto.M(
                atom=self.atom,
                basis=self.basis,
                charge=self.charge,
                spin=self.spin,
                unit=self.unit,
                verbose=self.verbose,
            )

        mf = scf.RHF(mol) if self.spin == 0 else scf.ROHF(mol)
        if self.mm_charges is not None:
            mm_coords, mm_q = self.mm_charges
            # 必须显式传 unit：add_mm_charges 缺省按 mol.unit 解读 MM 坐标，
            # 而此处 mol 可能以 Bohr 构建，与 mm_charges 的声明单位（self.unit）不一致。
            if self.mm_lattice is None:
                mf = qmmm.add_mm_charges(
                    mf, np.asarray(mm_coords), np.asarray(mm_q), unit=self.unit
                )
            else:
                # RB3/RB10：rcut_hcore 还须罩住整个 QM 区（get_hcore L183），
                # 与几何相关，只能在构建期校验。mol.atom_coords() 永远返回 Bohr，
                # 先换算到 self.unit 再与 rcut_hcore 比较（单位混用坑见模块 docstring）。
                _BOHR = 0.52917721092
                _scale = _BOHR if self.unit.lower().startswith("ang") else 1.0
                qc = mol.atom_coords().mean(axis=0)
                r_qm = float(np.max(np.linalg.norm(mol.atom_coords() - qc, axis=1))) * _scale
                if self.rcut_hcore <= r_qm:
                    raise ValueError(
                        f"rcut_hcore={self.rcut_hcore} must exceed the QM region "
                        f"radius {r_qm:.4f} (in {self.unit}): all QM atoms must sit "
                        "inside rcut_hcore of the QM center "
                        "(pyscf/qmmm/pbc/itrf.py get_hcore L183)."
                    )
                mf = _pbc_add_mm_charges(
                    mf,
                    np.asarray(mm_coords),
                    self.mm_lattice,
                    np.asarray(mm_q),
                    rcut_ewald=self.rcut_ewald,
                    rcut_hcore=self.rcut_hcore,
                    unit=self.unit,
                )
                # RB9：QMMMSCF.as_scanner = NotImplemented（pbc/itrf.py L112），
                # 而 CASCI scanner 构造时强调 _scf.as_scanner()；用基类版替换。
                mf.as_scanner = lambda mf=mf: scf.hf.SCF.as_scanner(mf)
        mf.run()

        n_elec, n_orb = self.active_space
        mc = mcscf.CASCI(mf, n_orb, n_elec)
        mc.fcisolver = self.fcisolver
        scanner = mc.nuc_grad_method().as_scanner()
        return mol, scanner

    # ---- 内部 ----

    def _element_symbols(self) -> list[str]:
        """初始 atom 规格的元素符号列表（缓存，只构建一次）。"""
        if getattr(self, "_symbols_cache", None) is None:
            tmp = gto.M(atom=self.atom, basis="sto-3g", unit=self.unit, verbose=0)
            self._symbols_cache = [tmp.atom_symbol(i) for i in range(tmp.natm)]
        return self._symbols_cache

    # ---- 对外 ----

    def __call__(self, geometry) -> tuple[float, np.ndarray]:
        """计算给定几何的 ``(e_tot, de)``，原子单位。

        ``geometry`` 支持 ``(natm, 3)`` Bohr 坐标数组、PySCF 原子列表、
        或带 ``atom_coords()`` 的对象。
        """
        coords = self._resolve_geometry(geometry)
        if self._scanner is None:
            self._mol, self._scanner = self._build(coords)
            return self._scanner(coords)
        return self._scanner(coords)

    def _resolve_geometry(self, geometry) -> np.ndarray:
        if hasattr(geometry, "atom_coords"):
            return np.asarray(geometry.atom_coords(), dtype=float)
        if isinstance(geometry, (list, tuple)) and geometry and isinstance(geometry[0], (list, tuple, np.ndarray)) and isinstance(geometry[0][0], str):
            # PySCF 原子规格：临时构建一次取坐标
            tmp = gto.M(atom=geometry, basis=self.basis, charge=self.charge, spin=self.spin, verbose=0)
            return tmp.atom_coords()
        return np.asarray(geometry, dtype=float).reshape(-1, 3)

    @property
    def pyscf_scanner(self):
        """底层 PySCF scanner；尚未构建（从未调用过）时为 ``None``。"""
        return self._scanner

    def as_pyscf_scanner(self, geometry=None):
        """返回底层的 PySCF 核梯度 scanner（``pyscf.lib.GradScanner`` 实例）。

        首次调用会在 ``geometry``（缺省用初始几何）上完成构建与首帧冻结。
        ``pyscf.md`` 积分器（NVE/NVT/NPT）只认 ``lib.GradScanner``，
        走 MD 时用本方法取底层 scanner 交给积分器::

            scan = qc_scanner(...)
            scan(coords_bohr)                       # 首帧：构建并冻结子空间
            md = pyscf.md.NVE(scan.as_pyscf_scanner())
        """
        if self._scanner is None:
            coords = self._resolve_geometry(geometry if geometry is not None else self.atom)
            self._mol, self._scanner = self._build(coords)
        return self._scanner

    def refreeze(self) -> None:
        """丢弃 SQD 冻结子空间，下一次调用重新确定（仅 ``method="sqd"``）。"""
        refreeze = getattr(self.fcisolver, "refreeze", None)
        if refreeze is None:
            raise TypeError(f"method={self.method!r} has no frozen subspace to refreeze.")
        refreeze()

    # ---- 静电嵌入：MD 每步更新与 MM 反作用力（E8 阶段 A） ----

    def set_mm_charges(self, mm_coords, mm_q=None) -> None:
        """每步更新 MM 嵌入环境（坐标必给，电荷缺省沿用上次）。

        坐标单位跟随 ``self.unit``。已构建时走 ``qmmm.add_mm_charges``
        重入路径：上游 ``qmmm_for_scf`` 对已装饰的 SCF 对象只做 ``mm_mol``
        重赋值并重置 ``s1r/s1rr/mm_ewald_pot/qm_ewald_hess/e_nuc`` 五个缓存，
        不重建平均场；未构建时只更新配方，首次 ``_build`` 时生效。
        ``get_hcore``/``energy_nuc`` 每次调用都动态读 ``mm_mol``，故更新立即对后续调用生效。
        pbc 模式额外重跑平均场（RB7 实测：重入后不重算 SCF 会沿用旧轨道）。

        边界：构建时未给 ``mm_charges``（裸建）的 scanner 首次调用本方法时，
        上游 ``add_mm_charges`` 对未装饰的平均场**返回新对象**而不是原地装饰
        （``qmmm_for_scf`` 走 ``QMMMSCF(method, ...)`` 包装路径，
        ``pyscf/qmmm/itrf.py`` L36-60），丢弃返回值会使嵌入静默失效且
        ``mm_gradient`` 报 ``grad_hcore_mm`` 缺失。此场景整体按配方重建，
        让首次嵌入走构建期装饰路径；一次性代价，此后每步仍走原地重入。
        """
        if mm_q is None:
            if self.mm_charges is None:
                raise ValueError("mm_q is required when no mm_charges was given at construction.")
            mm_q = self.mm_charges[1]
        mm_coords = np.asarray(mm_coords, dtype=float).reshape(-1, 3)
        mm_q = np.asarray(mm_q, dtype=float)
        if mm_coords.shape[0] != mm_q.shape[0]:
            raise ValueError(
                f"mm_coords has {mm_coords.shape[0]} sites but mm_q has {mm_q.shape[0]}."
            )
        self.mm_charges = (mm_coords, mm_q)
        if self._scanner is not None:
            # scanner.base 是 CASCI scanner，._scf 是 QMMM 装饰的平均场；
            # add_mm_charges 内部 create_mm_mol 会把 self.unit 的坐标转 Bohr。
            mf = self._scanner.base._scf
            if not isinstance(mf, qmmm.QMMM):
                # 裸建后首次带电荷：上游返回新对象而非原地装饰，无法仅靠回写
                # _scf 保证 CASCI 层引用一致，整体按配方在当前几何重建。
                coords = self._scanner.mol.atom_coords()
                self._scanner = None
                self._mol, self._scanner = self._build(coords)
                self._scanner(coords)
                return
            if self.mm_lattice is None:
                qmmm.add_mm_charges(mf, mm_coords, mm_q, unit=self.unit)
            else:
                # pbc 重入：重置 5 个缓存（pbc/itrf.py L89-96）后必须重跑 SCF（RB7）。
                _pbc_add_mm_charges(
                    mf, mm_coords, self.mm_lattice, mm_q,
                    rcut_ewald=self.rcut_ewald, rcut_hcore=self.rcut_hcore,
                    unit=self.unit,
                )
                mf.run()

    def mm_gradient(self) -> np.ndarray:
        """当前态对 MM 粒子坐标的梯度 ``dE/dR_mm``（Hartree/Bohr）。

        力是其负值。复用上游现成实现，不手写梯度：
        电子部分 ``QMMMGrad.grad_hcore_mm``（``itrf.py`` L345，吃任意 dm，
        点电荷模型下 ``get_zetas()`` 返回 1e16 的极窄高斯即点电荷极限），
        核部分 ``grad_nuc_mm``（L414）。密度矩阵取 CASCI AO 1-RDM。
        必须在至少一次 ``__call__`` 之后调用。

        pbc 模式（``mm_lattice`` 已给）另加 ``grad_ewald(dm, with_mm=True)``
        的 MM 分量（``pbc/itrf.py`` L517）。已知近似：上游解析式未含
        post-HF 轨道响应项（post-HF 架在 HF 轨道上、对轨道非变分；换 FCI 同样），
        基准偏置 ~4e-5 Ha/Bohr ≈ 2.2 meV/Å（相对净嵌入力 ~3%），随几何光滑：
        恒温/恒压 MD 可用，**严格 NVE 能量守恒诊断不成立**；误差随嵌入强度增长、
        无普适上界。详见 ``examples/qmmm/md_lammps_qmmm_pbc/VALIDATION.md`` §4。
        """
        if self._scanner is None or self.mm_charges is None:
            raise RuntimeError(
                "mm_gradient requires a built scanner with mm_charges; call it once first."
            )
        mc = self._scanner.base          # CASCI scanner（grad 对象的 base）
        mf = mc._scf                     # QMMM 装饰的平均场（itrf.py QMMMSCF）
        dm = mc.make_rdm1()              # CASCI 全 1-RDM（核+活性）
        mf_grad = mf.nuc_grad_method()   # → QMMMGrad 混入（itrf.py L209-211）
        de = np.asarray(mf_grad.grad_hcore_mm(dm)) + np.asarray(mf_grad.grad_nuc_mm())
        if self.mm_lattice is not None:
            # de_ewald_mm 初始为 None（pbc/itrf.py L490），grad_ewald 计算并返回。
            _, de_mm_ewald = mf_grad.grad_ewald(dm, with_mm=True)
            de = de + np.asarray(de_mm_ewald)
        return de


def qc_scanner(*args, **kwargs) -> QCScanner:
    """创建 :class:`QCScanner`。参数与 :class:`QCScanner` 相同。

    最小用法::

        scan = qc_scanner("O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
                          basis="sto-3g", active_space=(4, 4))
        e, de = scan(coords_bohr)
    """
    return QCScanner(*args, **kwargs)
