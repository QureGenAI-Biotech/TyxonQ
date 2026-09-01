"""``mdi_engine``：TyxonQ 的 MDI engine——静电嵌入专线（计划 §3.6，P4）。

命令表抄 psi4 ``mdi_engine.py`` 的结构：只注册 ``@DEFAULT`` 一个节点，
手工派发循环（不依赖 ``MDI_Set_Execute_Command_Func``，便于单测）::

    <NATOMS  <COORDS  >COORDS  <ENERGY  <FORCES  <ELEMENTS  <MASSES
    <TOTCHARGE  >TOTCHARGE  <ELEC_MULT  >ELEC_MULT  <DIMENSIONS  EXIT
    静电嵌入三条：>NLATTICE  >CLATTICE  >LATTICE

**单位**：MDI 协议用原子单位（Bohr / Hartree），与 PySCF 原生一致，
全链路不做任何换算（附录 B）。

静电嵌入链路：``>NLATTICE``（MM 粒子数）→ ``>CLATTICE``（电荷，原子单位 e）→
``>LATTICE``（MM 坐标，Bohr）缓存成对，每步 ``>COORDS`` 后整体经
``qc_scanner.set_mm_charges`` 注入 ``pyscf.qmmm.add_mm_charges`` 重入路径
（E8 阶段 A 同款，RB7 协议验证过）；``<FORCES`` 返回全原子力 =
QM 梯度（含嵌入）⊕ MM 反作用力（``mm_gradient``）。已知近似同
``scanner.py`` 的 ``mm_gradient`` 文档（RB5：缺 post-HF 轨道响应，
偏置 ~4.3e-5 Ha/Bohr，严格 NVE 守恒诊断不成立）。

与 i-PI 三进程路径（E8-A）物理等价；MDI 专线存在的意义是直连支持
MDI 的引擎（如带 MDI 包的 LAMMPS、OpenMM-MDI）。簇嵌入版：周期嵌入
走 E8-B（i-PI + LAMMPS pppm）。

``mdi``（PyPI 包名 ``pymdi``）惰性导入，缺失时报带安装指引的
``ImportError``。与其余适配层相同，``interfaces/__init__.py`` 不导出本模块。
"""

from __future__ import annotations

import numpy as np

from .scanner import qc_scanner

_INSTALL_HINT = (
    "The MDI engine requires MDI_Library's Python bindings. Install them with: "
    "pip install 'pymdi>=1.4'   (PyPI package 'pymdi', imports as 'mdi')"
)

# 原子质量回退表（amu）：ase 可用时优先用 ase.data.mass；本表只覆盖教程元素
_FALLBACK_MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}


def _require_mdi():
    try:
        import mdi
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return mdi


class TyxonQMdiEngine:
    """MDI engine：把 ``qc_scanner``（区域划分 + 静电嵌入）暴露为 ``@DEFAULT`` 节点。

    Parameters
    ----------
    symbols:
        全体系元素符号（长度 = 原子数），供 ``<ELEMENTS`` 回报与首帧建配方。
    qm_indices:
        QM 子集原子下标；补集作 MM 点电荷环境（由 ``>LATTICE``/``>CLATTICE``
        每步推送）。
    active_space / basis / charge / spin / method / sampler / solver_kwargs:
        与 :func:`qc_scanner` 同构，描述 **QM 子集**。
    verbose:
        透传给 scanner。

    单位约定：构造与回调全程原子单位（Bohr / Hartree / e）。
    """

    def __init__(
        self,
        symbols,
        qm_indices,
        *,
        active_space: tuple[int, int],
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
        method: str = "uccsd",
        sampler=None,
        solver_kwargs: dict | None = None,
        verbose: int = 0,
    ):
        self.symbols = list(symbols)
        self.qm_indices = np.asarray(qm_indices, dtype=int)
        natoms = len(self.symbols)
        if self.qm_indices.size == 0 or self.qm_indices.max() >= natoms:
            raise ValueError(f"qm_indices={qm_indices} invalid for {natoms} atoms.")
        mask = np.ones(natoms, dtype=bool)
        mask[self.qm_indices] = False
        self.mm_indices = np.flatnonzero(mask)
        self.scanner_kwargs = dict(
            basis=basis, charge=charge, spin=spin, unit="Bohr",
            active_space=tuple(active_space), method=method,
            sampler=sampler, solver_kwargs=solver_kwargs, verbose=verbose,
        )
        self._scan = None
        self._coords = None            # (natoms, 3) Bohr，最近一次 >COORDS
        self._mm_charges = None        # (coords Bohr, charges e)，缓存成对
        self._energy = None
        self._de_qm = None

    # ---- 连接与主循环 ----

    def run(self, hostname: str = "localhost", port: int = 8021, name: str = "QM"):
        """以 ENGINE 角色初始化 TCP 连接、注册节点与命令，进入派发循环。"""
        mdi = _require_mdi()
        mdi.MDI_Init(f"-role ENGINE -name {name} -method TCP -port {port} "
                     f"-hostname {hostname}", None)
        self.register(mdi)
        comm = mdi.MDI_Accept_Communicator()
        while True:
            cmd = mdi.MDI_Recv_Command(comm)
            if not self.execute(cmd, comm):
                break

    def register(self, mdi) -> None:
        """注册 ``@DEFAULT`` 节点与全部命令（照计划 §3.6 命令表）。"""
        mdi.MDI_Register_Node("@DEFAULT")
        for cmd in ("<NATOMS", "<COORDS", "<ENERGY", "<FORCES", "<ELEMENTS",
                    "<MASSES", "<TOTCHARGE", "<ELEC_MULT", "<DIMENSIONS"):
            mdi.MDI_Register_Command("@DEFAULT", cmd)
        for cmd in (">COORDS", ">NLATTICE", ">CLATTICE", ">LATTICE",
                    ">TOTCHARGE", ">ELEC_MULT", "EXIT"):
            mdi.MDI_Register_Command("@DEFAULT", cmd)

    def execute(self, cmd: str, comm) -> bool:
        """派发一条命令；返回 ``False`` 表示收到 ``EXIT``。"""
        mdi = _require_mdi()
        natoms = len(self.symbols)

        if cmd == "<NATOMS":
            mdi.MDI_Send(natoms, 1, mdi.MDI_INT, comm)
        elif cmd == ">COORDS":
            coords = np.asarray(mdi.MDI_Recv(3 * natoms, mdi.MDI_DOUBLE, comm),
                                dtype=float).reshape(natoms, 3)
            self._update_geometry(coords)
        elif cmd == "<ENERGY":
            mdi.MDI_Send(float(self._energy), 1, mdi.MDI_DOUBLE, comm)
        elif cmd == "<FORCES":
            mdi.MDI_Send(self._forces().reshape(-1).tolist(), 3 * natoms,
                         mdi.MDI_DOUBLE, comm)
        elif cmd == "<ELEMENTS":
            from pyscf.gto import ELEMENTS

            numbers = [ELEMENTS.index(s.capitalize()) for s in self.symbols]
            mdi.MDI_Send(numbers, natoms, mdi.MDI_INT, comm)
        elif cmd == "<MASSES":
            mdi.MDI_Send(self._masses(), natoms, mdi.MDI_DOUBLE, comm)
        elif cmd == "<TOTCHARGE":
            mdi.MDI_Send(float(self.scanner_kwargs["charge"]), 1, mdi.MDI_DOUBLE, comm)
        elif cmd == "<ELEC_MULT":
            mdi.MDI_Send(self.scanner_kwargs["spin"] + 1, 1, mdi.MDI_INT, comm)
        elif cmd == "<DIMENSIONS":
            mdi.MDI_Send([3], 1, mdi.MDI_INT, comm)
        elif cmd == ">NLATTICE":
            self._nlattice = int(mdi.MDI_Recv(1, mdi.MDI_INT, comm))
        elif cmd == ">CLATTICE":
            n = getattr(self, "_nlattice", None)
            if n is None:
                raise RuntimeError(">CLATTICE received before >NLATTICE.")
            charges = np.asarray(mdi.MDI_Recv(n, mdi.MDI_DOUBLE, comm), dtype=float)
            self._mm_charges = (self._mm_charges[0] if self._mm_charges else None, charges)
            self._flush_lattice()
        elif cmd == ">LATTICE":
            n = getattr(self, "_nlattice", None)
            if n is None:
                raise RuntimeError(">LATTICE received before >NLATTICE.")
            coords = np.asarray(mdi.MDI_Recv(3 * n, mdi.MDI_DOUBLE, comm),
                                dtype=float).reshape(n, 3)
            self._mm_charges = (coords, self._mm_charges[1] if self._mm_charges else None)
            self._flush_lattice()
        elif cmd in (">TOTCHARGE", ">ELEC_MULT"):
            mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)  # 吸收，不覆盖构造参数
        elif cmd == "EXIT":
            return False
        else:
            raise ValueError(f"Unknown MDI command: {cmd!r}")
        return True

    # ---- 内部 ----

    def _update_geometry(self, coords: np.ndarray) -> None:
        """``>COORDS`` 处理：必要时注入 MM 电荷，然后跑嵌入单点。"""
        self._coords = coords
        if self._mm_charges is not None and all(x is not None for x in self._mm_charges):
            self._inject_mm_charges()
        if self._scan is None:
            qm_spec = [(self.symbols[i], tuple(map(float, coords[i])))
                       for i in self.qm_indices]
            self._scan = qc_scanner(qm_spec, **self.scanner_kwargs)
            if self._mm_charges is not None and all(x is not None for x in self._mm_charges):
                self._scan.set_mm_charges(*self._mm_charges)
        e, de = self._scan(coords[self.qm_indices])
        self._energy = float(e)
        self._de_qm = np.asarray(de)

    def _inject_mm_charges(self) -> None:
        if self._scan is not None:
            self._scan.set_mm_charges(*self._mm_charges)

    def _flush_lattice(self) -> None:
        """电荷与坐标都到齐后立即注入（已建配方时），实现每步更新。"""
        if self._mm_charges is not None and all(x is not None for x in self._mm_charges):
            self._inject_mm_charges()

    def _forces(self) -> np.ndarray:
        """全原子力（Hartree/Bohr）：QM 梯度（含嵌入）⊕ MM 反作用力。"""
        if self._de_qm is None:
            raise RuntimeError("<FORCES requested before any >COORDS/<ENERGY cycle.")
        natoms = len(self.symbols)
        forces = np.zeros((natoms, 3))
        forces[self.qm_indices] = -self._de_qm
        if self._mm_charges is not None and all(x is not None for x in self._mm_charges):
            forces[self.mm_indices] = -np.asarray(self._scan.mm_gradient())
        return forces

    def _masses(self) -> list[float]:
        try:
            from ase.data import atomic_numbers, masses as ase_masses

            return [float(ase_masses[atomic_numbers[s.capitalize()]]) for s in self.symbols]
        except ImportError:
            try:
                return [float(_FALLBACK_MASSES[s]) for s in self.symbols]
            except KeyError as exc:
                raise ValueError(
                    f"No mass for element {exc.args[0]!r}: install ase "
                    "(pip install 'tyxonq[md]') for the full mass table."
                ) from exc
