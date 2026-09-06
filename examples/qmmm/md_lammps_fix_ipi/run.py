"""E7 — TyxonQ 经 i-PI 桥接接入 LAMMPS：三进程编排脚本。

拓扑（设计目标：让 TyxonQ 方便接入 LAMMPS 做固体物理 QM/MM）：

    LAMMPS (fix ipi, client/力场端)
            │  socket (i-PI 协议)
            ▼
    i-PI server（MD 引擎：NVT / PIMD / 系综都在这里）
            ▲
            │  socket (i-PI 协议)
    TyxonQ driver (i-pi-driver-py -m custom, client/QM 力端)

``<forces>`` 中两个力分量加权求和（已取证 ``InputForceComponent.weight``）——
这正是 i-PI 官方文档所说的「mixing first-principles calculations and
empirical force fields」，也是固体 QM/MM 里 delta 学习 / 嵌入修正类
配方的组合基础。

本脚本做的事：
    1. 算参考值：初始几何上直接调 ``qc_scanner`` 得 E_QM，用 numpy 按
       in/in.qmmm 的 LJ 参数解析算 E_MM；
    2. 把 in/ 下模板的端口/路径占位符替换后写进临时工作目录；
    3. 按启动顺序拉起三个进程：i-PI server → TyxonQ driver → LAMMPS；
    4. 等模拟跑完，回读 qmmm.md，校验第 0 步势能 ≈ E_QM + E_MM
       （容差覆盖 i-PI 单位常数的回转误差，见 test_ipi_driver.py 注释）。

环境要求：PySCF、ASE、i-PI（pip install -U ipi）、LAMMPS（需编译进
fix ipi，即 MISC 包；conda-forge 的 lammps 默认包含）。任一缺失时
直接退出（exit 0），方便 CI。

运行方式：
    python examples/qmmm/md_lammps_fix_ipi/run.py

手动三窗口启动（不用本脚本时）见同目录 README。
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
    print(f"跳过 E7：缺少依赖（模块 {missing}，可执行文件 {missing_bins}）。")
    sys.exit(0)

import tyxonq.applications.chem.interfaces.ipi_driver as ipi_mod
from ipi.utils.units import unit_to_user
from tyxonq.applications.chem.interfaces import qc_scanner

DRIVER_PY = ipi_mod.__file__
HARTREE_TO_KELVIN = unit_to_user("energy", "kelvin", 1.0)  # 从 i-PI 单位表取，不写死
BOHR_TO_ANGSTROM = 0.52917721092
KCALMOL_PER_HARTREE = 627.5094740631  # CODATA

# 与 E3/E4 同一参考水分子几何（Å）
H2O_POS_ANG = np.array(
    [(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)]
)
XYZ = """3
water for E7
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
"""

# LJ 参数必须与 in/in.qmmm 保持一致（epsilon / kcal·mol⁻¹，sigma / Å，cutoff / Å）
LJ_EPS_KCAL, LJ_SIGMA_ANG, LJ_CUTOFF_ANG = 0.001, 1.5, 10.0


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _lj_energy_hartree(pos_ang):
    """按 in.qmmm 的 lj/cut 参数，对裸几何解析算 MM 能量（参考值）。"""
    e_kcal = 0.0
    for i in range(len(pos_ang)):
        for j in range(i + 1, len(pos_ang)):
            r = np.linalg.norm(pos_ang[i] - pos_ang[j])
            if r < LJ_CUTOFF_ANG:
                sr6 = (LJ_SIGMA_ANG / r) ** 6
                e_kcal += 4.0 * LJ_EPS_KCAL * (sr6 * sr6 - sr6)
    return e_kcal / KCALMOL_PER_HARTREE


# ---------------------------------------------------------------------------
print("=" * 72)
print("E7 — LAMMPS(fix ipi) + i-PI server + TyxonQ driver 三进程 QM/MM 管线")
print("=" * 72)

# [1] 参考值：第 0 步的总势能必须等于 E_QM + E_MM
print("\n[1] 计算参考能量（初始几何）...")
scan = qc_scanner(
    "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
    basis="sto-3g", active_space=(4, 4), method="uccsd",
)
e_qm, _ = scan(H2O_POS_ANG / BOHR_TO_ANGSTROM)
e_mm = _lj_energy_hartree(H2O_POS_ANG)
e_ref = e_qm + e_mm
print(f"    E_QM (TyxonQ UCCSD/CAS(4,4)) = {e_qm: .8f} Hartree")
print(f"    E_MM (LAMMPS lj/cut)         = {e_mm: .8f} Hartree")
print(f"    E_ref = E_QM + E_MM          = {e_ref: .8f} Hartree")

# [2] 准备工作目录：模板占位符替换
in_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "in")
workdir = tempfile.mkdtemp(prefix="tyxonq_e7_")
port_tq, port_lmp = _free_port(), _free_port()
xyz = os.path.join(workdir, "water.xyz")
with open(xyz, "w") as fh:
    fh.write(XYZ)

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
shutil.copy(os.path.join(in_dir, "water.lmp"), workdir)
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
time.sleep(2.0)  # 等两个 ffsocket 都绑好端口
tq_driver = subprocess.Popen(
    [
        bins["i-pi-driver-py"], "-a", "127.0.0.1", "-p", str(port_tq),
        "-m", "custom", "-P", DRIVER_PY,
        "-o", f"{xyz},basis=sto-3g,active_space=4 4,method=uccsd",
    ],
    cwd=workdir, stdout=_open_log("driver.log"), stderr=subprocess.STDOUT,
)
lammps = subprocess.Popen(
    [bins["lmp"], "-in", "in.qmmm", "-log", "log.lammps", "-screen", "none"],
    cwd=workdir, stdout=_open_log("lammps.stdout"), stderr=subprocess.STDOUT,
)

try:
    rc_server = server.wait(timeout=900)
finally:
    # server 结束后两个 client 都会因断连自行退出；给足时间再兜底
    for p in (tq_driver, lammps):
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
    for fh in logs.values():
        fh.close()

if rc_server != 0:
    for name in ("server.log", "driver.log", "log.lammps"):
        path = os.path.join(workdir, name)
        if os.path.exists(path):
            print(f"---- {name} ----")
            print(open(path).read()[-3000:])
    raise RuntimeError(f"i-PI server exited with code {rc_server}")
print("    三进程已全部正常收尾。")

# [4] 回读 i-PI 输出并校验
props = os.path.join(workdir, "qmmm.md")
with open(props) as fh:
    lines = [ln for ln in fh if not ln.startswith("#")]
rows = [ln.split() for ln in lines]
print("\n[4] i-PI 记录的总势能（QM 力来自 TyxonQ，MM 力来自 LAMMPS）：")
print("    step   potential / Hartree   (pot - E_ref)")
for row in rows:
    step, pot_k = int(float(row[0])), float(row[3])
    pot = pot_k / HARTREE_TO_KELVIN
    print(f"    {step:4d}   {pot:18.8f}      {pot - e_ref:+.3e}")

# 第 0 步在初始几何上求值，必须与参考值一致（容差覆盖 i-PI 单位回转误差）
pot0 = float(rows[0][3]) / HARTREE_TO_KELVIN
assert abs(pot0 - e_ref) < 2e-4, f"step-0 potential {pot0} != E_ref {e_ref}"
print(f"\n    step-0 校验通过：|pot - (E_QM + E_MM)| = {abs(pot0 - e_ref):.2e} Hartree")
print(f"\n完成。全部日志与输入保留在 {workdir}")
