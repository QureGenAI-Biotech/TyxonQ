"""E8-B — 周期性固体 QM/MM：TyxonQ（pbc Ewald 嵌入）+ i-PI + LAMMPS（pppm）。

体系与 E8-A 相同（6 原子双水，20 Å 盒），差异只在静电的周期性：
    原子 0-2 = QM 区：CAS(4,4) UCCSD，嵌入层用 ``pyscf.qmmm.pbc.add_mm_charges``
               （Ewald 求和：QM 区感受 MM 点电荷的全部周期镜像，
               晶格 20 Å 立方，rcut_ewald=8 Å，rcut_hcore=9 Å——守卫见
               ``scanner.py``，风险取证见同目录 ``VALIDATION.md``）；
    原子 3-5 = MM 区：TIP3P，MM-MM 周期静电由 LAMMPS ``kspace_style pppm`` 负责。

能量分工（严格无重叠，周期版）：
    E = E_QM(QM 水；pbc Ewald 嵌入 MM 点电荷)   ← TyxonQ 返回
      + E_MM(MM-MM 周期静电 + QM-MM vdW)        ← LAMMPS 返回
防双计数（与 E8-A 同，另加周期互补，见 in/in.qmmm 头部注释）：
    1. QM 原子电荷置 0 → pppm/实空间都没有 QM-MM 库仑，归嵌入算一份；
    2. pair_coeff * * 全置 0，仅打开 O-O 通道 → 无 QM-QM LJ，保留 QM-MM vdW；
    3. pbc 嵌入不含 MM-MM（上游 docstring 明示；RB1 大盒极限与簇嵌入一致也印证），
       与 LAMMPS 的 MM-MM pppm 互补不重叠（RB6）。

拓扑与 E7/E8-A 相同（三进程）：
    LAMMPS (fix ipi) ─ socket ─► i-PI server ◄─ socket ─ TyxonQ driver
driver 的 ``-o`` 串比 E8-A 多 ``mm_lattice``/``rcut_ewald``/``rcut_hcore``。

本脚本做的事：
    1. 算参考值：``qc_scanner(..., mm_charges=..., mm_lattice=..., rcut_*=...)``
       得 E_QM(pbc 嵌入)；解析 Ewald 和算 MM-MM 周期静电（实空间 erfc +
       倒空间 + 自能，内部用两个不同 κ 互检——精确 Ewald 总和与 κ 无关）；
       解析算 QM-MM vdW（仅 O_qm-O_mm 一对）；
    2. 模板占位符替换后写进临时工作目录；
    3. 拉起三进程：i-PI server → TyxonQ driver → LAMMPS；
    4. 回读 qmmm.md，校验第 0 步势能 ≈ E_QM + E_LJ + E_ewald
       （容差 2e-4 Ha，覆盖 pppm 对精确 Ewald 的偏差、i-PI 单位回转与
       UCCSD 数值抖动）。收尾逻辑与 E8-A 一致（数据写齐即验收 + 30 s 宽限）。

环境要求：PySCF、ASE、i-PI、LAMMPS（fix ipi）。任一缺失时退出（exit 0）。

运行方式：
    python examples/qmmm/md_lammps_qmmm_pbc/run.py
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
    print(f"跳过 E8-B：缺少依赖（模块 {missing}，可执行文件 {missing_bins}）。")
    sys.exit(0)

import tyxonq.applications.chem.interfaces.ipi_driver as ipi_mod
from ipi.utils.units import unit_to_user
from tyxonq.applications.chem.interfaces import qc_scanner

DRIVER_PY = ipi_mod.__file__
HARTREE_TO_KELVIN = unit_to_user("energy", "kelvin", 1.0)
BOHR_TO_ANGSTROM = 0.52917721092
ANG2BOHR = 1.0 / BOHR_TO_ANGSTROM
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

# pbc 嵌入参数（与 driver 的 -o 串严格一致；守卫与取值依据见
# 同目录 VALIDATION.md §3：rcut_ewald < 盒边，QM 区半径 < rcut_hcore < 半盒边）
BOX_ANG = 20.0
RCUT_EWALD, RCUT_HCORE = 8.0, 9.0


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


def _mm_ewald_energy_hartree(coords_ang, charges, box_ang, kappa_ang, tol=1e-10):
    """MM-MM 周期静电的精确 Ewald 和（tinfoil 边界，Hartree；原子单位内算）。

    E = ½Σ_{ij,L}′ q_i q_j erfc(κ|r|)/|r|  +  (2π/V)Σ_{G≠0} e^{-G²/4κ²}/G² |S(G)|²
        − (κ/√π)Σ_i q_i²
    精确总和与 κ 无关（实空间/倒空间拆分任意）：调用方用两个 κ 互检。
    倒空间截断自适应：取到 exp(−G²/4κ²) < tol（κ 越小要得越多，
    手拍固定截断会欠收敛——精确总和必须与 κ 无关，这是实现正确性自检）。
    """
    from scipy.special import erfc

    coords = np.asarray(coords_ang, dtype=float) * ANG2BOHR
    box = box_ang * ANG2BOHR
    kappa = kappa_ang * ANG2BOHR
    q = np.asarray(charges, dtype=float)
    vol = box**3

    # 实空间：参考胞内全部对（i<j）+ 与 26 个最近邻镜像的全部对；
    # erfc(κ·20 Å) 在 κ≥0.3 时 <1e-8，镜像贡献可忽略但照算（便宜）。
    e_real = 0.0
    n = len(q)
    Ls = np.array(
        [[nx, ny, nz] for nx in (-1, 0, 1) for ny in (-1, 0, 1) for nz in (-1, 0, 1)]
    ) * box
    for L in Ls:
        for i in range(n):
            j0 = i + 1 if np.all(L == 0.0) else 0
            for j in range(j0, n):
                r = np.linalg.norm(coords[i] - coords[j] - L)
                if r > 1e-10:
                    e_real += q[i] * q[j] * erfc(kappa * r) / r

    # 倒空间：S(G) = Σ_i q_i e^{-iG·r_i}；g_shell 自适应（见 docstring）。
    dg = 2 * np.pi / box
    g_shell = int(np.ceil(np.sqrt(-4 * kappa**2 * np.log(tol)) / dg))
    ns = np.arange(-g_shell, g_shell + 1)
    e_k = 0.0
    for nx in ns:
        for ny in ns:
            for nz in ns:
                if nx == ny == nz == 0:
                    continue
                G2 = dg * dg * (nx * nx + ny * ny + nz * nz)
                G = dg * np.array([nx, ny, nz])
                phase = coords @ G
                s_c = float(q @ np.cos(phase))
                s_s = float(q @ np.sin(phase))
                e_k += np.exp(-G2 / (4 * kappa**2)) / G2 * (s_c * s_c + s_s * s_s)
    e_k *= 2 * np.pi / vol

    e_self = -(kappa / np.sqrt(np.pi)) * float(q @ q)
    return e_real + e_k + e_self


# ---------------------------------------------------------------------------
print("=" * 72)
print("E8-B — 周期性固体 QM/MM：LAMMPS(pppm) + i-PI server + TyxonQ(pbc Ewald)")
print("=" * 72)

# [1] 参考值：第 0 步总势能必须等于 E_QM(pbc 嵌入) + E_LJ + E_ewald(MM-MM)
print("\n[1] 计算参考能量（初始几何）...")
scan = qc_scanner(
    "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    basis="sto-3g", active_space=(4, 4), method="uccsd",
    mm_charges=(MM_POS_ANG, MM_Q),
    mm_lattice=np.eye(3) * BOX_ANG,
    rcut_ewald=RCUT_EWALD, rcut_hcore=RCUT_HCORE,
)
e_qm, _ = scan(QM_POS_ANG / BOHR_TO_ANGSTROM)
e_lj = _lj_energy_hartree()
# Ewald 参考值：两个 κ 互检（精确总和与 κ 无关），倒空间截断自适应
e_ew_a = _mm_ewald_energy_hartree(MM_POS_ANG, MM_Q, BOX_ANG, kappa_ang=0.4)
e_ew_b = _mm_ewald_energy_hartree(MM_POS_ANG, MM_Q, BOX_ANG, kappa_ang=0.7)
assert abs(e_ew_a - e_ew_b) < 1e-9, f"Ewald κ-自检失败：{e_ew_a} vs {e_ew_b}"
e_coul = e_ew_a
e_ref = e_qm + e_lj + e_coul
print(f"    E_QM (UCCSD/CAS(4,4)，pbc Ewald 嵌入)       = {e_qm: .8f} Hartree")
print(f"    E_LJ (QM-MM vdW，仅 O_qm-O_mm)             = {e_lj: .8f} Hartree")
print(f"    E_ewald (MM-MM 周期静电，解析 Ewald)       = {e_coul: .8f} Hartree")
print(f"        κ 互检：{e_ew_a:.12f} vs {e_ew_b:.12f}（差 {abs(e_ew_a - e_ew_b):.1e}）")
print(f"    E_ref = E_QM + E_LJ + E_ewald              = {e_ref: .8f} Hartree")

# [2] 准备工作目录：模板占位符替换
in_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "in")
workdir = tempfile.mkdtemp(prefix="tyxonq_e8b_")
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
        # 区域划分 + pbc 嵌入：driver 拿全 6 原子坐标，只把 0-2 当 QM；
        # mm_lattice 是行主序 9 个数（-o 串里逗号是分隔符，只能用空格）。
        "-o", (
            f"{xyz},basis=sto-3g,active_space=4 4,method=uccsd,"
            "qm_indices=0 1 2,atom_charges=0.0 0.0 0.0 -0.834 0.417 0.417,"
            f"mm_lattice={BOX_ANG} 0 0 0 {BOX_ANG} 0 0 0 {BOX_ANG},"
            f"rcut_ewald={RCUT_EWALD},rcut_hcore={RCUT_HCORE}"
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
print("\n[4] i-PI 记录的总势能（QM 力来自 TyxonQ pbc 嵌入，MM 力来自 LAMMPS pppm）：")
print("    step   potential / Hartree   (pot - E_ref)")
for row in rows:
    step, pot_k = int(float(row[0])), float(row[3])
    pot = pot_k / HARTREE_TO_KELVIN
    print(f"    {step:4d}   {pot:18.8f}      {pot - e_ref:+.3e}")

# 第 0 步在初始几何上求值：容差覆盖 pppm 对精确 Ewald 的偏差、
# i-PI 单位常数回转误差与 UCCSD 数值抖动（~1e-8 Ha）。
pot0 = float(rows[0][3]) / HARTREE_TO_KELVIN
assert abs(pot0 - e_ref) < 2e-4, f"step-0 potential {pot0} != E_ref {e_ref}"
print(f"\n    step-0 校验通过：|pot - (E_QM + E_LJ + E_ewald)| = {abs(pot0 - e_ref):.2e} Hartree")
print(f"\n完成。全部日志与输入保留在 {workdir}")
