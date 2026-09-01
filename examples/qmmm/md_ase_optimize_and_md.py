"""E3 — 用 ASE 生态消费 TyxonQ 势能面：结构优化、Langevin MD、振动频率。

本教程展示「枢纽成立」后的典型用法：一旦把 ``TyxonQCalculator`` 挂到
ASE 的 ``Atoms`` 上，ASE 全家桶（优化器、MD 积分器、振动分析）都可以
直接用，TyxonQ 只负责在每个几何上给出能量与力。

环境要求：
    - PySCF、ASE（ASE >= 3.23；本仓库开发环境为 3.29）
    - 可选：matplotlib（只在第 4 步画能量曲线，缺失自动跳过）

预期运行时间：约 1~2 分钟（H2O / STO-3G / CAS(4,4) / UCCSD，
每次单点约 0.02~0.05 s，全教程约 100+ 次单点）。

依赖缺失时的行为：直接退出（exit 0），不报错——方便在没有 ASE 的
CI 环境里被批量执行。

运行方式：
    python examples/qmmm/md_ase_optimize_and_md.py

教程结构：
    第 1 步  挂载计算器：TyxonQCalculator 与 qc_scanner 的关系
    第 2 步  BFGS 结构优化：把扰动过的水分子放回平衡几何
    第 3 步  Langevin 恒温 MD：300 K 下跑一小段，观察能量
    第 4 步  振动频率：在优化后的几何上做有限差分振动分析

姊妹教程：
    md_qc_scanner_basics.py   (E1) — 不涉及 ASE 的裸 scanner 用法
    md_pyscf_native_aimd.py   (E2) — 用 pyscf.md 跑 NVE（零 ASE 依赖）
"""

import sys

# ---- 依赖守卫：缺 pyscf 或 ase 时优雅退出 ----
try:
    from pyscf import gto  # noqa: F401  # 仅用于探测依赖
    from ase import Atoms, units
    from ase.optimize import BFGS
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.vibrations import Vibrations
except ImportError as exc:  # pragma: no cover - 仅缺依赖时走到
    print(f"跳过本教程：缺少依赖（{exc}）。请安装 PySCF 与 ASE。")
    sys.exit(0)

from tyxonq.applications.chem.interfaces.ase_calculator import TyxonQCalculator


# ---------------------------------------------------------------------------
# 体系：一个水分子（故意给一个略微拉长的初始几何，方便看优化效果）
# 坐标单位是 Å——ASE 的默认单位，Calculator 内部会自动换算成 Bohr。
# ---------------------------------------------------------------------------
atoms = Atoms(
    "OHH",
    positions=[
        (0.0, 0.0, 0.15),    # O：z 方向略微偏移
        (0.0, 0.80, -0.50),  # H：O-H 键长拉长到约 1.03 Å
        (0.0, -0.80, -0.50),
    ],
)

# ---------------------------------------------------------------------------
# 第 1 步：挂载计算器
#
# TyxonQCalculator 是 qc_scanner 的 ASE 外壳：
#   - 构造参数与 qc_scanner 同构（basis / active_space / method / ...）；
#   - 几何不用传——每次计算从 Atoms 实时读取；
#   - 输出自动换算成 ASE 单位：能量 eV，力 eV/Å（且已按 ASE 约定取负）。
# 首次调用时按 Atoms 的元素符号构建内部扫描器，此后只做坐标更新，
# SCF 热启动，不重复冷启动。
# ---------------------------------------------------------------------------
atoms.calc = TyxonQCalculator(
    basis="sto-3g",
    active_space=(4, 4),   # CAS(4,4)：水的价层活性空间
    method="uccsd",        # 也可以换 "hea"（配 solver_kwargs={"runtime": "numeric"}）
)

e0 = atoms.get_potential_energy()  # 单位 eV
print(f"初始几何能量: {e0:.6f} eV")
print(f"初始最大力  : {abs(atoms.get_forces()).max():.4f} eV/Å")

# ---------------------------------------------------------------------------
# 第 2 步：BFGS 结构优化
#
# ASE 的 BFGS 只需要 atoms 上有计算器即可工作：它反复调用
# get_forces()，沿力方向做拟牛顿更新。fmax 是收敛阈值（eV/Å）。
# ---------------------------------------------------------------------------
print("\n== 第 2 步：BFGS 结构优化 ==")
opt = BFGS(atoms, logfile="-")  # logfile="-" 表示打印到屏幕
opt.run(fmax=0.01, steps=25)

print(f"优化后能量: {atoms.get_potential_energy():.6f} eV")
assert atoms.get_potential_energy() < e0, "优化应当降低能量"
oh = atoms.get_distance(0, 1)
print(f"优化后 O-H 键长: {oh:.4f} Å")
assert 0.9 < oh < 1.1, "水分子平衡 O-H 键长应在 1.0 Å 附近"

# ---------------------------------------------------------------------------
# 第 3 步：Langevin 恒温 MD
#
# Langevin 恒温器 = 牛顿方程 + 随机力 + 摩擦项，把体系耦合到温度 T 的热浴。
# 步长经验值：最快振动周期（O-H 伸缩 ~10 fs）的 1/10 量级，取 0.5 fs。
# 注意：TyxonQCalculator 本身与积分器无关——换任何 ASE 积分器（Verlet、
# Nose-Hoover、NPT...）都是同样的挂法。
# ---------------------------------------------------------------------------
print("\n== 第 3 步：Langevin 恒温 MD（300 K, 20 步） ==")
MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=None)
Stationary(atoms)  # 去掉整体平动，避免体系漂移

dyn = Langevin(
    atoms,
    timestep=0.5 * units.fs,   # 0.5 fs
    temperature_K=300.0,
    friction=0.01 / units.fs,  # 摩擦系数：1/100 fs，弱耦合
)

history = {"step": [], "epot_ev": [], "ekin_ev": [], "temp_k": []}

def _record():
    """每步记录势能/动能/瞬时温度。"""
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * units.kB * len(atoms))  # 3N 自由度的均分温度
    history["step"].append(len(history["step"]))
    history["epot_ev"].append(epot)
    history["ekin_ev"].append(ekin)
    history["temp_k"].append(temp)

dyn.attach(_record, interval=1)  # 每 1 步记录一次
dyn.run(steps=20)

print(f"MD 结束：势能 {history['epot_ev'][-1]:.6f} eV，"
      f"瞬时温度 {history['temp_k'][-1]:.1f} K")
# 恒温浴下势能有涨落是正常的（与 NVE 守恒不同，见 E2）
assert max(history["epot_ev"]) - min(history["epot_ev"]) < 1.0, \
    "20 步内势能涨落不应超过 1 eV（ Sanity check，非物理判据）"

# ---------------------------------------------------------------------------
# 第 4 步：振动频率
#
# ase.vibrations.Vibrations 用中心差分对 Hessian 采样（每个原子 ±delta
# 两个方向，共 6×N 次单点），再对角化得到频率。对 3 原子水分子共 9 个
# 模式：3 个平动 + 3 个转动（自由分子）+ 3 个真实振动。
# ---------------------------------------------------------------------------
print("\n== 第 4 步：振动频率（中心差分, delta=0.01 Å） ==")
vib = Vibrations(atoms, delta=0.01, name="h2o_vib")
vib.run()
vib.summary()          # 打印全部 9 个模式（含平动/转动的 ~0 cm^-1）

# 只关心真实振动：取后 3 个模式的波数（cm^-1）。
# get_frequencies() 返回复数：实部是频率，虚部表示不稳定方向（软模）。
freqs_cm = vib.get_frequencies()[-3:].real
print("\n真实振动模式（cm^-1）:", [round(float(f), 1) for f in freqs_cm])
# 参考值：真实水分子约 1600（弯曲）/ 3700 / 3800（伸缩）；STO-3G 极小基组下
# 弯曲模会偏低（本教程实测约 2160），这里只验证「链路打通 + 量级正确」。
for f in freqs_cm:
    assert f > 500.0, f"振动频率 {f:.1f} cm^-1 过低，链路可能出错"
vib.clean()  # 删除中心差分产生的临时 .log 文件

# ---------------------------------------------------------------------------
# 可选：画 MD 能量曲线（没有 matplotlib 就跳过）
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境也能保存
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(history["step"], history["epot_ev"], label="势能 $E_\\mathrm{pot}$")
    ax.plot(history["step"], history["ekin_ev"], label="动能 $E_\\mathrm{kin}$")
    ax.plot(history["step"],
            [p + k for p, k in zip(history["epot_ev"], history["ekin_ev"])],
            label="总能量", ls="--")
    ax.set_xlabel("MD 步数")
    ax.set_ylabel("能量 / eV")
    ax.set_title("TyxonQ(UCCSD) 驱动的 Langevin MD (300 K)")
    ax.legend()
    fig.tight_layout()
    out = "qmmm_md_langevin_energy.png"
    fig.savefig(out, dpi=150)
    print(f"\n能量曲线已保存至 {out}")
except ImportError:
    print("\n（未安装 matplotlib，跳过绘图）")

print("\n全部完成：TyxonQCalculator 已作为标准 ASE 计算器接入 "
      "优化 / MD / 振动分析三类工作流。")
