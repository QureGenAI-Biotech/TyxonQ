"""E9：MDI 专线——TyxonQ 作为 MDI engine 的静电嵌入 QM/MM（两进程：driver + engine）。

拓扑
----
::

    本进程（driver，手写 MDI 客户端）  ←TCP→  TyxonQMdiEngine（子进程，@DEFAULT 节点）

与 E8-A（i-PI + LAMMPS 三进程）物理等价，走的是另一条通信协议：MDI 的
``>NLATTICE`` / ``>CLATTICE`` / ``>LATTICE`` 三条命令把 MM 点电荷推进
QM 哈密顿量（每步更新），``>COORDS`` / ``<ENERGY`` / ``<FORCES`` 往返
几何与能量力。**全程原子单位**（Bohr / Hartree / e），与 PySCF 原生一致，
不做任何换算。

依赖：``pip install 'pymdi>=1.4'``（PyPI 包名 ``pymdi``，import 名 ``mdi``；
注意 ``openmm-ml``→``openmmml`` 同款命名坑）。本教程自带手写 driver，
不需要带 MDI 包的 LAMMPS；若引擎端是支持 MDI 的 LAMMPS/OpenMM-MDI，
TyxonQ engine 侧代码原样可用。

已知近似：MM 反作用力缺 post-HF 轨道响应项（RB5，同 ``scanner.mm_gradient``
文档）；短轨迹只做演示与能量位移验收，不做严格守恒诊断。

体系：两水二聚体（QM 水 + MM 水，tip3p 电荷），UCCSD/STO-3G/CAS(4,4)。

运行::

    PYTHONPATH=src python examples/qmmm/md_mdi_qmmm_embedded.py
"""

from __future__ import annotations

import multiprocessing
import sys
import time

import numpy as np

BOHR_TO_ANGSTROM = 0.52917721092
PORT = 38181

# ---- 体系（Å）：chain 0 = QM 水，chain 1 = MM 水 ----
QM_POS_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])
MM_SHIFT_ANG = np.array([2.9, 0.8, 0.3])
MM_CHARGES = [-0.834, 0.417, 0.417]          # tip3p MM 水
SYMBOLS = ["O", "H", "H", "O", "H", "H"]
QM_INDICES = [0, 1, 2]

DIMER_BOHR = np.vstack([QM_POS_ANG, QM_POS_ANG + MM_SHIFT_ANG]) / BOHR_TO_ANGSTROM


def _engine_proc():
    """子进程：TyxonQMdiEngine 监听并服务（异常时打印后退出）。"""
    import traceback

    try:
        from tyxonq.applications.chem.interfaces.mdi_engine import TyxonQMdiEngine

        engine = TyxonQMdiEngine(SYMBOLS, QM_INDICES, active_space=(4, 4), method="uccsd")
        engine.run(hostname="localhost", port=PORT)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


def main():
    import mdi

    print("[0] 启动 TyxonQMdiEngine 子进程（TCP 端口", PORT, "）")
    proc = multiprocessing.Process(target=_engine_proc)
    proc.start()
    time.sleep(1.0)

    mdi.MDI_Init(f"-role DRIVER -name MM -method TCP -port {PORT} -hostname localhost", None)
    comm = mdi.MDI_Accept_Communicator()

    def cmd(name):
        mdi.MDI_Send_Command(name, comm)

    # ---- 第 1 步：无嵌入基准能量 ----
    cmd(">COORDS")
    mdi.MDI_Send(DIMER_BOHR.reshape(-1).tolist(), 18, mdi.MDI_DOUBLE, comm)
    cmd("<ENERGY")
    e_bare = mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)
    print(f"[1] 裸算 E_QM（无嵌入）     = {e_bare:.8f} Ha")

    # ---- 第 2 步：推入 MM 点电荷晶格（>NLATTICE/>CLATTICE/>LATTICE） ----
    mm_pos = DIMER_BOHR[3:]
    cmd(">NLATTICE")
    mdi.MDI_Send(3, 1, mdi.MDI_INT, comm)
    cmd(">CLATTICE")
    mdi.MDI_Send(MM_CHARGES, 3, mdi.MDI_DOUBLE, comm)
    cmd(">LATTICE")
    mdi.MDI_Send(mm_pos.reshape(-1).tolist(), 9, mdi.MDI_DOUBLE, comm)

    cmd(">COORDS")
    mdi.MDI_Send(DIMER_BOHR.reshape(-1).tolist(), 18, mdi.MDI_DOUBLE, comm)
    cmd("<ENERGY")
    e_emb = mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)
    cmd("<FORCES")
    forces = np.asarray(mdi.MDI_Recv(18, mdi.MDI_DOUBLE, comm)).reshape(6, 3)

    de = e_emb - e_bare
    print(f"[2] 静电嵌入 E_QM           = {e_emb:.8f} Ha")
    print(f"    能量位移 ΔE             = {de:+.6f} Ha  "
          f"（符号随取向：本例 O 对 O 排斥；量级与 §4.1 实测同类）")
    assert abs(de) > 1e-4, "MM 电荷未进入哈密顿量"
    print("    全原子力（Ha/Bohr；MM 行 = 反作用力）：")
    for i, f in enumerate(forces):
        tag = "QM" if i in QM_INDICES else "MM"
        print(f"      [{tag}] atom {i}: {f[0]:10.6f} {f[1]:10.6f} {f[2]:10.6f}")
    assert np.linalg.norm(forces[3:]) > 1e-4, "MM 反作用力为零"

    # ---- 第 3 步：每步更新演示——MM 水沿 x 平移 0.2 Å，能量跟随 ----
    mm_moved = mm_pos + np.array([0.2, 0.0, 0.0]) / BOHR_TO_ANGSTROM
    cmd(">NLATTICE")
    mdi.MDI_Send(3, 1, mdi.MDI_INT, comm)
    cmd(">CLATTICE")
    mdi.MDI_Send(MM_CHARGES, 3, mdi.MDI_DOUBLE, comm)
    cmd(">LATTICE")
    mdi.MDI_Send(mm_moved.reshape(-1).tolist(), 9, mdi.MDI_DOUBLE, comm)
    cmd(">COORDS")
    mdi.MDI_Send(DIMER_BOHR.reshape(-1).tolist(), 18, mdi.MDI_DOUBLE, comm)
    cmd("<ENERGY")
    e_moved = mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)
    print(f"[3] MM 水平移 0.2 Å 后 E_QM = {e_moved:.8f} Ha  "
          f"(Δ = {e_moved - e_emb:+.6f} Ha，set_mm_charges 重入生效)")
    assert abs(e_moved - e_emb) > 1e-6, "每步更新未生效"

    cmd("EXIT")
    proc.join(timeout=30)
    assert not proc.is_alive(), "engine 未干净退出"
    print("\n[OK] MDI 专线静电嵌入验收通过（命令回环 + 能量位移 + 每步更新）。")


if __name__ == "__main__":
    main()
