"""E5 — OpenMM 直连纯 QM：MLPotential('ase') 建 System，OpenMM 积分器跑 MD。

本教程展示 TyxonQ 势能面经 ``openmm_potential`` 薄封装进入 OpenMM 生态：
    topology  →  create_tyxonq_system(...)  →  openmm.System（PythonForce 每步
                                               回调 TyxonQCalculator 取 E/F）
    VerletIntegrator + Context  →  标准 OpenMM 工作流（报告器、平台选择等全可用）

物理说明：这是**纯 QM** 路径——整个体系都在量子势能面上，没有 MM 区。
QM/MM（机械嵌入）见姊妹教程 ``md_openmm_qmmm_solvated.py``（E6）；
**静电嵌入**（MM 电荷进 QM 哈密顿量）不走本路径，见
``md_lammps_qmmm_embedded/``（E8-A）与 ``md_lammps_qmmm_pbc/``（E8-B）。

环境要求：
    - PySCF、ASE、OpenMM (>=8.1)、OpenMM-ML (openmmml >= 1.7)
    - 安装：pip install 'tyxonq[md]'

预期运行时间：约 1~2 分钟（H2O / STO-3G / CAS(4,4) / UCCSD，
20 步 NVE，每步一次量子单点约 0.05~1 s）。

依赖缺失时的行为：直接退出（exit 0），不报错——方便在没装 OpenMM 的
CI 环境里被批量执行。

运行方式：
    python examples/qmmm/md_openmm_pure_qm.py

教程结构：
    第 1 步  建拓扑与 System：create_tyxonq_system 的输入输出
    第 2 步  NVE 短 MD：velocity-Verlet 积分 20 步，观察总能量守恒
    第 3 步  守恒诊断：漂移量级与 UCCSD 数值抖动的关系
"""

import sys

# ---- 依赖守卫：缺任一依赖时优雅退出 ----
try:
    import pyscf  # noqa: F401
    import ase  # noqa: F401
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from openmmml import MLPotential  # noqa: F401
except ImportError as exc:  # pragma: no cover - 仅缺依赖时走到
    print(f"跳过本教程：缺少依赖（{exc}）。请安装：pip install 'tyxonq[md]'")
    sys.exit(0)

import numpy as np

from tyxonq.applications.chem.interfaces import qc_scanner
from tyxonq.applications.chem.interfaces.openmm_potential import create_tyxonq_system

HARTREE_TO_KJMOL = 2625.4996394798  # CODATA
BOHR_TO_ANGSTROM = 0.52917721092

# 与全套教程一致的水几何（Å）：略微拉长，让 MD 有东西可振动
H2O_POS_ANG = [(0.0, 0.0, 0.15), (0.0, 0.80, -0.50), (0.0, -0.80, -0.50)]


# ---------------------------------------------------------------------------
# 第 1 步：拓扑 → System
#
# OpenMM 的入口是 Topology（元素 + 键连关系），不是坐标串。水分子逐原子
# 声明元素（上游 asepotential 要求元素必须定义，否则报错）。
# create_tyxonq_system 内部：TyxonQCalculator(**kwargs) → MLPotential('ase')
# → createSystem(topology, calculator=...)，返回的 System 里是一个
# PythonForce，每步把坐标喂给 calculator 取回 (E, F)。
# ---------------------------------------------------------------------------
top = app.Topology()
elem_o, elem_h = app.Element.getBySymbol("O"), app.Element.getBySymbol("H")
chain = top.addChain()
res = top.addResidue("HOH", chain)
o = top.addAtom("O", elem_o, res)
h1 = top.addAtom("H1", elem_h, res)
h2 = top.addAtom("H2", elem_h, res)
top.addBond(o, h1)
top.addBond(o, h2)

system = create_tyxonq_system(
    top,
    basis="sto-3g",
    active_space=(4, 4),   # CAS(4,4)：水的价层活性空间
    method="uccsd",        # 目标量子算法（也可 "hea" + solver_kwargs）
)
print("System 建好：")
for i in range(system.getNumForces()):
    f = system.getForce(i)
    print(f"    Force[{i}]: {type(f).__name__}")

# 独立参考：同一几何直接走 qc_scanner（验证 System 构建没走样）
scan = qc_scanner(
    [("O", H2O_POS_ANG[0]), ("H", H2O_POS_ANG[1]), ("H", H2O_POS_ANG[2])],
    basis="sto-3g", active_space=(4, 4), method="uccsd",
)
e_ref_ha, _ = scan(np.array(H2O_POS_ANG) / BOHR_TO_ANGSTROM)
print(f"qc_scanner 参考能量 = {e_ref_ha:.8f} Hartree")

# ---------------------------------------------------------------------------
# 第 2 步：NVE 短 MD
#
# 标准 OpenMM 工作流：Integrator + Context + setPositions/setVelocities。
# 量子力每步一次单点（~0.05~1 s），所以步长/步数都不必大——
# 本教程只验证「链路通 + 守恒合理」。
# OpenMM 的 ``VerletIntegrator`` 就是 velocity-Verlet（严格辛，漂移有界不累积）。
# 实测（2026-09-01）：守恒带 ~2e-5 Ha；若不手动剥离质心平动，CMMotionRemover
# 会在第一步吃掉初始速度里的质心动能（~1e-3 Ha 随机），造成一次性台阶。
# ---------------------------------------------------------------------------
DT_FS, N_STEPS = 0.2, 20

integ = openmm.VerletIntegrator(DT_FS * unit.femtoseconds)
ctx = openmm.Context(system, integ)
ctx.setPositions(np.array(H2O_POS_ANG) * 0.1 * unit.nanometers)   # Å → nm
ctx.setVelocitiesToTemperature(300 * unit.kelvin)

# 剥离质心平动：setVelocitiesToTemperature 连 3 个平动自由度一起抽样（平均携 ~3/2 kT
# 动能），会被 System 里的 CMMotionRemover 在第一步一次性剥掉，造成 ~1e-3 Ha 的
# 守恒台阶（实测 2026-09-01）。自己先剥掉，漂移从 step-0 起就干净。
v = ctx.getState(getVelocities=True).getVelocities(asNumpy=True) \
    .value_in_unit(unit.nanometers / unit.picoseconds)
m = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
              for i in range(system.getNumParticles())])
v -= (m @ v) / m.sum()   # v_cm = Σ mᵢvᵢ / Σ mᵢ，逐原子减去（广播）
ctx.setVelocities(v * unit.nanometers / unit.picoseconds)

print(f"\nNVE：{N_STEPS} 步 × {DT_FS} fs，300 K 初始速度")
print("    step   potential/Ha    kinetic/Ha     total/Ha     drift/Ha")
totals = []
for step in range(N_STEPS + 1):
    state = ctx.getState(getEnergy=True)
    pot = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJMOL
    kin = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJMOL
    tot = pot + kin
    totals.append(tot)
    if step % 4 == 0:
        drift = tot - totals[0]
        print(f"    {step:4d}   {pot:14.8f}   {kin:11.8f}   {tot:12.8f}   {drift:+.2e}")
    integ.step(1) if step < N_STEPS else None

# ---------------------------------------------------------------------------
# 第 3 步：守恒诊断
#
# 漂移来源：① velocity-Verlet 积分截断（非平衡初始几何下辛积分器的修正哈密顿量误差，
# 有界不累积，实测 ~2e-5 Ha）；② UCCSD 数值求解器的跨调用抖动（~1e-8 Ha，无偏）。
# 两者都远小于化学精度（~1.6e-3 Ha）。若换更大的活性空间/基组，积分步长要相应减小。
# ---------------------------------------------------------------------------
drift_max = max(abs(t - totals[0]) for t in totals)
print(f"\n总能量最大漂移 = {drift_max:.3e} Hartree（容差 5e-4）")
assert drift_max < 5e-4, f"NVE drift too large: {drift_max}"

# 第 0 步势能应与独立参考一致（UCCSD 抖动 ~1e-8，容差 1e-6）
integ2 = openmm.VerletIntegrator(DT_FS * unit.femtoseconds)
ctx2 = openmm.Context(system, integ2)
ctx2.setPositions(np.array(H2O_POS_ANG) * 0.1 * unit.nanometers)
state0_pot = ctx2.getState(getEnergy=True).getPotentialEnergy() \
    .value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJMOL
print(f"step-0 势能 {state0_pot:.8f} vs 参考 {e_ref_ha:.8f} Hartree"
      f"（差 {abs(state0_pot - e_ref_ha):.2e}）")
assert abs(state0_pot - e_ref_ha) < 1e-6

print("\n完成。OpenMM 纯 QM 链路（create_tyxonq_system → velocity-Verlet NVE）验证通过。")
