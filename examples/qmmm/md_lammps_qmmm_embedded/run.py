"""E8-A — TyxonQ 经 i-PI 桥接接入 LAMMPS：区域划分 + 静电嵌入的水二聚体。

体系（6 原子，两个水分子）：
    原子 0-2 = QM 区：CAS(4,4) UCCSD，TyxonQ 负责全部能量/梯度，
               并通过 ``pyscf.qmmm.add_mm_charges`` 被 MM 点电荷静电嵌入（极化）；
    原子 3-5 = MM 区：TIP3P 电荷 + O 原子 LJ，LAMMPS 负责。

能量分工（严格无重叠）：
    E = E_QM(QM 水；哈密顿量嵌入 MM 点电荷)   ← TyxonQ 返回
      + E_MM(MM-MM 全部 + QM-MM vdW)           ← LAMMPS 返回
防双计数三策略（见 in/in.qmmm 头部注释）：
    1. QM 原子电荷置 0 → QM-MM 库仑只由嵌入提供一份；
    2. pair_coeff * * 全置 0，仅打开 O-O 通道 → 无 QM-QM LJ，保留 QM-MM vdW；
    3. TyxonQ 嵌入（qmmm.add_mm_charges）本身不含 MM-MM 能与 vdW（上游 docstring）。

拓扑与 E7 相同（三进程）：
    LAMMPS (fix ipi) ─ socket ─► i-PI server ◄─ socket ─ TyxonQ driver
本教程的差异：TyxonQ driver 收到全部 6 原子坐标，按 ``qm_indices=0 1 2``
切出 QM 子集，返回全原子力 = QM 梯度（含嵌入贡献）⊕ MM 反作用力
（``QCScanner.mm_gradient``，dE/dR_mm，复用上游 QMMMGrad）。

本脚本做的事：
    1. 算参考值：初始几何上 ``qc_scanner(..., mm_charges=...)`` 得 E_QM(嵌入)，
       解析算 QM-MM vdW（只有 O_qm-O_mm 一对，TIP3P 参数）与 MM-MM 库仑
       （单个 MM 水的分子内库仑，coul/cut 截断 10 Å，远大于分子尺寸，无截断效应）；
    2. 模板占位符替换后写进临时工作目录；
    3. 拉起三进程：i-PI server → TyxonQ driver → LAMMPS；
    4. 回读 qmmm.md，校验第 0 步势能 ≈ E_QM + E_LJ + E_coul
       （容差覆盖 i-PI 单位常数回转误差与 UCCSD 数值抖动）。
       LAMMPS 在 ``run 8`` 结束后会直接退出，i-PI server 随之进入收尾等待（
       对已断连的 client 清理可能挂住），故数据写齐后未退出则宽限 30 s 后
       主动结束三进程，验收以 qmmm.md 的数据为准（与 E7 同一验收口径）。

环境要求：PySCF、ASE、i-PI、LAMMPS（fix ipi）。任一缺失时退出（exit 0）。

运行方式：
    python examples/qmmm/md_lammps_qmmm_embedded/run.py
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np


# ---- 依赖守卫 ----
def _has(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


missing = [m for m in ("pyscf", "ase", "ipi") if not _has(m)]
bins = {name: shutil.which(name) for name in ("i-pi", "i-pi-driver-py", "lmp")}
missing_bins = [k for k, v in bins.items() if v is None]
if missing or missing_bins:
    print(f"跳过 E8-A：缺少依赖（模块 {missing}，可执行文件 {missing_bins}）。")
    sys.exit(0)

import tyxonq.applications.chem.interfaces.ipi_driver as ipi_mod
from ipi.utils.units import unit_to_user
from tyxonq.applications.chem.interfaces import qc_scanner

DRIVER_PY = ipi_mod.__file__
HARTREE_TO_KELVIN = unit_to_user("energy", "kelvin", 1.0)
BOHR_TO_ANGSTROM = 0.52917721092
KCALMOL_PER_HARTREE = 627.5094740631  # CODATA

# 与 water_dimer.xyz 一致的全体系几何（Å）：QM 水在原点，MM 水平移 (2.9, 0.8, 0.3)
QM_POS_ANG = np.array(
    [(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)]
)
MM_SHIFT = np.array([2.9, 0.8, 0.3])
MM_POS_ANG = QM_POS_ANG + MM_SHIFT

# TIP3P（与 water_dimer.lmp / in.qmmm 严格一致）
MM_Q = np.array([-0.834, 0.417, 0.417])
LJ_EPS_KCAL, LJ_SIGMA_ANG, LJ_CUTOFF_ANG = 0.1521, 3.1507, 10.0
KCAL_ANG_PER_E2 = 332.06371  # LAMMPS real 库仑常数（kcal·Å·mol⁻¹·e⁻²）


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _lj_energy_hartree():
    """QM-MM vdW 参考值：TIP3P 只有 O 有 LJ，故仅 O_qm-O_mm 一对。"""
    r = np.linalg.norm(MM_POS_ANG[0] - QM_POS_ANG[0])
    sr6 = (LJ_SIGMA_ANG / r) ** 6
    e_kcal = 4.0 * LJ_EPS_KCAL * (sr6 * sr6 - sr6)
    return e_kcal / KCALMOL_PER_HARTREE


def _mm_coulomb_energy_hartree():
    """MM-MM 库仑参考值：单个 MM 水的分子内三对（coul/cut 10 Å 无截断效应）。
    与 LAMMPS real 单位一致：E = Σ q_i q_j / r · 332.06371 kcal/mol。"""
    e_kcal = 0.0
    for i in range(len(MM_Q)):
        for j in range(i + 1, len(MM_Q)):
            r = np.linalg.norm(MM_POS_ANG[i] - MM_POS_ANG[j])
            e_kcal += KCAL_ANG_PER_E2 * MM_Q[i] * MM_Q[j] / r
    return e_kcal / KCALMOL_PER_HARTREE


# ---------------------------------------------------------------------------
print("=" * 72)
print("E8-A — 区域划分 + 静电嵌入：LAMMPS + i-PI server + TyxonQ driver")
print("=" * 72)

# [1] 参考值：第 0 步总势能必须等于 E_QM(嵌入) + E_LJ(QM-MM vdW)
print("\n[1] 计算参考能量（初始几何）...")
scan = qc_scanner(
    "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    basis="sto-3g", active_space=(4, 4), method="uccsd",
    mm_charges=(MM_POS_ANG, MM_Q),
)
e_qm, _ = scan(QM_POS_ANG / BOHR_TO_ANGSTROM)
e_lj = _lj_energy_hartree()
e_coul = _mm_coulomb_energy_hartree()
e_ref = e_qm + e_lj + e_coul
print(f"    E_QM (UCCSD/CAS(4,4)，嵌入于 TIP3P 点电荷) = {e_qm: .8f} Hartree")
print(f"    E_LJ (QM-MM vdW，仅 O_qm-O_mm)             = {e_lj: .8f} Hartree")
print(f"    E_coul (MM-MM 库仑，分子内三对)          = {e_coul: .8f} Hartree")
print(f"    E_ref = E_QM + E_LJ + E_coul               = {e_ref: .8f} Hartree")

# [2] 准备工作目录：模板占位符替换
in_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "in")
workdir = tempfile.mkdtemp(prefix="tyxonq_e8a_")
port_tq, port_lmp = _free_port(), _free_port()
xyz = os.path.join(workdir, "water_dimer.xyz")
shutil.copy(os.path.join(in_dir, "water_dimer.xyz"), xyz)

xml = open(os.path.join(in_dir, "input.xml")).read()
xml = xml.replace("__PORT_TQ__", str(port_tq))
xml = xml.replace("__PORT_LMP__", str(port_lmp))
xml = xml.replace("__XYZ__", xyz)
with open(os.path.join(workdir, "input.xml"), "w") as fh:
    fh.write(xml)

lmp_in = open(os.path.join(in_dir, "in.qmmm")).read()
lmp_in = lmp_in.replace("__PORT_LMP__", str(port_lmp))
with open(os.path.join(workdir, "in.qmmm"), "w") as fh:
    fh.write(lmp_in)
shutil.copy(os.path.join(in_dir, "water_dimer.lmp"), workdir)
print(f"\n[2] 工作目录：{workdir}（TyxonQ 端口 {port_tq}，LAMMPS 端口 {port_lmp}）")

# [3] 启动顺序：server 先，两个 client 后（顺序写反连不上）
print("\n[3] 启动三进程：i-PI server → TyxonQ driver → LAMMPS ...")
logs = {}
def _open_log(name):
    fh = open(os.path.join(workdir, name), "w")
    logs[name] = fh
    return fh

server = subprocess.Popen(
    [bins["i-pi"], "input.xml"], cwd=workdir,
    stdout=_open_log("server.log"), stderr=subprocess.STDOUT,
)
time.sleep(2.0)
tq_driver = subprocess.Popen(
    [
        bins["i-pi-driver-py"], "-a", "127.0.0.1", "-p", str(port_tq),
        "-m", "custom", "-P", DRIVER_PY,
        # 区域划分：driver 拿全 6 原子坐标，只把 0-2 当 QM；
        # atom_charges 给全 6 原子，QM 原子的 0.0 仅作占位（QM 子集不用）。
        "-o", (
            f"{xyz},basis=sto-3g,active_space=4 4,method=uccsd,"
            "qm_indices=0 1 2,atom_charges=0.0 0.0 0.0 -0.834 0.417 0.417"
        ),
    ],
    cwd=workdir, stdout=_open_log("driver.log"), stderr=subprocess.STDOUT,
)
N_ROWS = 8  # total_steps，与 input.xml 保持一致；变步数两处同改。


def _wait_outputs(props_path, n_rows, timeout=900, poll=5.0):
    """等 qmmm.md 写齐 n_rows 行数据（跳过 # 头），超时抛错。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with open(props_path) as fh:
                if sum(1 for ln in fh if not ln.startswith("#")) >= n_rows:
                    return
        except FileNotFoundError:
            pass
        time.sleep(poll)
    raise RuntimeError(f"timed out waiting for {n_rows} rows in qmmm.md")


lammps = subprocess.Popen(
    [bins["lmp"], "-in", "in.qmmm", "-log", "log.lammps", "-screen", "none"],
    cwd=workdir, stdout=_open_log("lammps.stdout"), stderr=subprocess.STDOUT,
)

# 验收以“数据写齐”为准：LAMMPS 跑完 8 步即退出，之后 i-PI server
# 的收尾等待（对已断连 client 的清理）可能挂住，不阻塞校验。
_wait_outputs(os.path.join(workdir, "qmmm.md"), N_ROWS)

try:
    try:
        server.wait(timeout=30)
        print("    三进程已全部正常收尾。")
    except subprocess.TimeoutExpired:
        print("    数据已写齐；i-PI server 停在收尾等待，主动结束三进程。")
        for p in (server, tq_driver, lammps):
            p.terminate()
        for p in (server, tq_driver, lammps):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
finally:
    for fh in logs.values():
        fh.close()

# [4] 回读 i-PI 输出并校验
props = os.path.join(workdir, "qmmm.md")
with open(props) as fh:
    lines = [ln for ln in fh if not ln.startswith("#")]
rows = [ln.split() for ln in lines]
print("\n[4] i-PI 记录的总势能（QM 力来自 TyxonQ 嵌入，MM 力来自 LAMMPS）：")
print("    step   potential / Hartree   (pot - E_ref)")
for row in rows:
    step, pot_k = int(float(row[0])), float(row[3])
    pot = pot_k / HARTREE_TO_KELVIN
    print(f"    {step:4d}   {pot:18.8f}      {pot - e_ref:+.3e}")

# 第 0 步在初始几何上求值：容差覆盖单位回转误差 + UCCSD 数值抖动（~1e-8 Ha）
pot0 = float(rows[0][3]) / HARTREE_TO_KELVIN
assert abs(pot0 - e_ref) < 2e-4, f"step-0 potential {pot0} != E_ref {e_ref}"
print(f"\n    step-0 校验通过：|pot - (E_QM + E_LJ + E_coul)| = {abs(pot0 - e_ref):.2e} Hartree")
print(f"\n完成。全部日志与输入保留在 {workdir}")
