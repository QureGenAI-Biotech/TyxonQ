"""E4 — i-PI server + TyxonQ driver 双进程 MD（含 PIMD 变体）。

E1–E3 里势能面都跑在「本进程」内；本教程展示真正的 client-server 架构：

    i-PI (server)                        TyxonQ driver (client 子进程)
    ─ 读 input.xml                        ─ i-pi-driver-py -m custom -P ipi_driver.py
    ─ 跑 MD 积分 / 系综 / 珠子            ─ 每步通过 socket 收到几何（Bohr）
    ─ 通过 socket 发 POSDATA/GETFORCE     ─ 调 TyxonQCalculator 算 E/F，回包

这正是生产上把昂贵量子化学势能面接入 i-PI 高级采样能力的标准姿势
（参考 i-PI 官方 demos/para-h2-tutorial 的 input.xml 结构，已按
ipi 3.3.0 的 ``Simulation.load_from_xml`` 解析链逐字段验证）。

环境要求：
    - PySCF、ASE、i-PI（``pip install -U ipi``；勿用 ``pip install i-PI``，
      那是废弃占位包）
    - ``i-pi`` 与 ``i-pi-driver-py`` 两个命令行工具在 PATH 上

预期运行时间：约 1~3 分钟（H2O / STO-3G / CAS(4,4) / UCCSD；
第 1 部分经典 NVT 10 步 + 第 2 部分 PIMD nbeads=2 跑 5 步）。

依赖缺失时的行为：直接退出（exit 0），不报错——方便在没有 i-PI 的
CI 环境里被批量执行。

运行方式：
    python examples/qmmm/md_ipi_driver_server.py

教程结构：
    第 1 步  生成运行目录：water.xyz 模板 + input.xml（ffsocket/inet）
    第 2 步  起双进程：i-PI server + TyxonQ driver，跑一小段经典 NVT
    第 3 步  读 i-PI 输出的 properties/轨迹，验证势能面确实来自 TyxonQ
    第 4 步  PIMD 变体：同一条势能面，只改 input.xml 的 nbeads 与恒温器，
             即可跑路径积分分子动力学——这是 i-PI 的独占卖点

姊妹教程：
    md_ase_optimize_and_md.py (E3) — 本进程内的 ASE 全家桶用法
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

# ---- 依赖守卫：缺 pyscf/ase/ipi 或两个可执行工具时优雅退出 ----
def _has(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


missing = [m for m in ("pyscf", "ase", "ipi") if not _has(m)]
bins = {name: shutil.which(name) for name in ("i-pi", "i-pi-driver-py")}
if missing or any(v is None for v in bins.values()):
    print(
        "跳过本教程：缺少依赖"
        + (f"（模块 {missing}）" if missing else "")
        + (f"（可执行文件 {[k for k, v in bins.items() if v is None]}）"
           if any(v is None for v in bins.values()) else "")
        + "。安装：pip install pyscf ase && pip install -U ipi"
    )
    sys.exit(0)

# driver 文件路径（-m custom -P 需要）
import tyxonq.applications.chem.interfaces.ipi_driver as ipi_mod
from ipi.utils.units import unit_to_user

DRIVER_PY = ipi_mod.__file__
# 不写死换算常数：直接从 i-PI 自己的单位表取（1 Hartree = ? K）
HARTREE_TO_KELVIN = unit_to_user("energy", "kelvin", 1.0)

# 水分子模板（Å）——driver 的 -o 参数串第一个 token
XYZ = """3
water template for TyxonQDriver
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
"""

# i-PI input.xml。要点（均已对照 ipi 3.3.0 inputs/ schema 核实）：
#   - ffsocket 的 mode 是属性，address/port/timeout 是子元素；
#   - address 写 127.0.0.1 而非 localhost：macOS 上 localhost 可能解析到
#     ::1，而 driver 用 AF_INET（IPv4），二者不匹配会连不上；
#   - 分子体系必须显式给盒子（否则零晶胞在发 POSDATA 时求逆崩溃），
#     cell 是扁平 9 元数组（io_xml.read_array 不支持嵌套括号），
#     上三角约束自动满足（对角 20 Å 盒子，无 pbc 用途）；
#   - {kelvin}/{femtosecond} 是 i-PI 的带单位字面量语法。
INPUT_XML = """<simulation verbosity='medium'>
  <output prefix='water'>
    <properties filename='md' stride='1'>
      [ step, time{femtosecond}, temperature{kelvin}, potential{kelvin}, kinetic_cv{kelvin} ]
    </properties>
    <trajectory filename='pos' stride='1' format='xyz' cell_units='angstrom'>
      positions{angstrom}
    </trajectory>
  </output>
  <total_steps> __TOTAL_STEPS__ </total_steps>
  <prng> <seed> 42 </seed> </prng>
  <ffsocket mode='inet' name='tyxonq'>
    <address>127.0.0.1</address>
    <port> __PORT__ </port>
    <timeout> 600 </timeout>
  </ffsocket>
  <system>
    <initialize nbeads='__NBEADS__'>
      <file mode='xyz' units='angstrom'> __XYZ__ </file>
      <velocities mode='thermal' units='kelvin'> 300 </velocities>
    </initialize>
    <forces>
      <force forcefield='tyxonq'/>
    </forces>
    <ensemble>
      <temperature units='kelvin'> 300 </temperature>
    </ensemble>
    <motion mode='dynamics'>
      <dynamics mode='nvt'>
        <thermostat mode='__THERMOSTAT__'>
          <tau units='femtosecond'> 100 </tau>
        </thermostat>
        <timestep units='femtosecond'> 0.5 </timestep>
      </dynamics>
    </motion>
    <cell mode='manual' units='angstrom'>
      [ 20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0 ]
    </cell>
  </system>
</simulation>
"""


def _free_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def run_ipi(label, workdir, nbeads, total_steps, thermostat):
    """起 i-PI server + TyxonQ driver 双进程，阻塞到模拟结束。"""
    port = _free_port()
    xyz = os.path.join(workdir, "water.xyz")
    with open(xyz, "w") as fh:
        fh.write(XYZ)

    xml = INPUT_XML.replace("__TOTAL_STEPS__", str(total_steps))
    xml = xml.replace("__PORT__", str(port))
    xml = xml.replace("__NBEADS__", str(nbeads))
    xml = xml.replace("__XYZ__", xyz)
    xml = xml.replace("__THERMOSTAT__", thermostat)
    fn_xml = os.path.join(workdir, f"input_{label}.xml")
    with open(fn_xml, "w") as fh:
        fh.write(xml)

    # -o 参数串：第一个 token 是模板文件，其余 key=value 透传给
    # TyxonQCalculator。注意逗号是上游分隔符，所以 active_space 写 "4 4"。
    driver_cmd = [
        bins["i-pi-driver-py"],
        "-a", "127.0.0.1", "-p", str(port),
        "-m", "custom", "-P", DRIVER_PY,
        "-o", f"{xyz},basis=sto-3g,active_space=4 4,method=uccsd",
    ]

    log_server = open(os.path.join(workdir, f"server_{label}.log"), "w")
    log_driver = open(os.path.join(workdir, f"driver_{label}.log"), "w")
    server = subprocess.Popen(
        [bins["i-pi"], fn_xml], cwd=workdir,
        stdout=log_server, stderr=subprocess.STDOUT,
    )
    time.sleep(2.0)  # 等 server 绑好 socket（实测 <1 s）
    driver = subprocess.Popen(
        driver_cmd, cwd=workdir, stdout=log_driver, stderr=subprocess.STDOUT,
    )

    try:
        rc = server.wait(timeout=600)
    finally:
        # server 结束（或超时）后断开 driver；driver 检测到断连自行退出
        if driver.poll() is None:
            driver.wait(timeout=60) if rc == 0 else driver.kill()
        log_server.close()
        log_driver.close()

    if rc != 0:
        for name in (f"server_{label}.log", f"driver_{label}.log"):
            print(f"---- {name} ----")
            print(open(os.path.join(workdir, name)).read())
        raise RuntimeError(f"i-PI server exited with code {rc} (see logs above)")
    return port


# ---------------------------------------------------------------------------
# 第 1~3 步：经典 NVT（nbeads=1，Langevin 恒温）
# ---------------------------------------------------------------------------
print("=" * 70)
print("E4 — i-PI server + TyxonQ driver 双进程 MD")
print("=" * 70)

workdir = tempfile.mkdtemp(prefix="tyxonq_ipi_e4_")
print(f"\n运行目录：{workdir}\n")

print("[1] 经典 NVT：nbeads=1，10 步 × 0.5 fs（Langevin，300 K）...")
t0 = time.time()
run_ipi("md", workdir, nbeads=1, total_steps=10, thermostat="langevin")
print(f"    完成，用时 {time.time() - t0:.1f} s")

# i-PI 把 <properties> 写到 <prefix>.<filename>，即 water.md
props = os.path.join(workdir, "water.md")
with open(props) as fh:
    lines = [ln for ln in fh if not ln.startswith("#")]
cols = list(zip(*[ln.split() for ln in lines]))
steps = [int(float(c)) for c in cols[0]]  # i-PI 全列用科学计数法输出
pot_kelvin = [float(c) for c in cols[3]]
print("\n[2] i-PI 记录的势能面取值（每步都来自 TyxonQ UCCSD 单点+梯度）：")
print("    step   potential / Hartree")
for s, pk in zip(steps, pot_kelvin):
    print(f"    {s:4d}   {pk / HARTREE_TO_KELVIN:18.8f}")

# i-PI 轨迹文件名带珠子下标：water.pos_0.xyz（nbeads=1 只有第 0 个珠子）
traj = os.path.join(workdir, "water.pos_0.xyz")
n_frames = sum(1 for ln in open(traj) if ln.strip()) // 4  # xyz 帧 = 4 行
print(f"\n[3] 轨迹 water.pos_0.xyz 共 {n_frames} 帧（i-PI 额外写入初态帧，"
      f"= total_steps + 1）")

# ---------------------------------------------------------------------------
# 第 4 步：PIMD 变体——同一条势能面，只改 XML
# ---------------------------------------------------------------------------
print("\n[4] PIMD 变体：nbeads=2，pile_g 恒温，5 步（每步 2 个珠子各算一次力）...")
print("    —— 只改了 input.xml 的 nbeads 与 thermostat，势能面代码零改动，")
print("       这正是 i-PI 相对 ASE/pyscf.md 的独占能力。")
t0 = time.time()
run_ipi("pimd", workdir, nbeads=2, total_steps=5, thermostat="pile_g")
print(f"    完成，用时 {time.time() - t0:.1f} s")

# PIMD 跑的输出同样落在 water.md（同 prefix 会被续写），读最后 5 行即可
with open(props) as fh:
    lines = [ln for ln in fh if not ln.startswith("#")]
last = lines[-5:]
print("\n    PIMD 段的质心势能（kinetic_cv 含珠子涨落，属正常）：")
for ln in last:
    s, _, temp_k, pot_k, _ = ln.split()
    s = int(float(s))
    print(f"    step {s:2d}   T = {float(temp_k):7.1f} K   V = {float(pot_k) / HARTREE_TO_KELVIN:18.8f} Hartree")

print("\n完成。运行目录保留在 " + workdir + "，可自行查看 input_*.xml 与日志。")
