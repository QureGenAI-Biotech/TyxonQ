"""E6 — OpenMM QM/MM 溶剂化体系：createMixedSystem + λ 插值（自由能微扰）。

体系：双水分子（chain 0 = QM 水 / chain 1 = MM 水，几何与 E8 系列教程一致），
``create_mixed_system(interpolate=True)`` 产出带 ``lambda_interpolate`` 的
混合 System：
    λ = 0  →  全体系经典力场（tip3p）
    λ = 1  →  QM 水内部能量换由 TyxonQ（UCCSD/CAS(4,4)）计算
    中间值 →  两者线性插值（Context.setParameter 可扫，供自由能微扰）

**机械嵌入边界（必读）**：本路径下 QM-MM 相互作用（含静电）按经典力场算，
QM 电子结构**感受不到 MM 电荷的极化**——与 E8 系列（i-PI 三进程、静电嵌入，
MM 电荷进 QM 哈密顿量）是两种物理层级。本教程验证的是「机械嵌入管线 +
能量分解正确」，不是静电嵌入的替代品。

能量分解恒等式（验收依据）：
    E(λ=1) − E(λ=0) = E_QM(TyxonQ, QM 水) − E_bonded(tip3p, QM 水内部键角)
    （QM 水内部 nonbonded 在 tip3p 里全被例外清零，故不出现）

环境要求：
    - PySCF、ASE、OpenMM (>=8.1)、OpenMM-ML (openmmml >= 1.7)
    - 安装：pip install 'tyxonq[md]'

预期运行时间：约 2~3 分钟（~15 次量子单点，每次 0.05~1 s）。

依赖缺失时的行为：直接退出（exit 0），不报错。

运行方式：
    python examples/qmmm/md_openmm_qmmm_solvated.py

教程结构：
    第 1 步  拓扑 + 纯经典参考能量（tip3p 全体系）
    第 2 步  混合 System（interpolate=True）：λ=0/1 能量分解恒等式验收
    第 3 步  λ 扫描：0 → 1 五点，观察 MM↔QM 平滑过渡
    第 4 步  λ=1 短 MD：5 步 velocity-Verlet，确认混合体系动力学可跑
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
from tyxonq.applications.chem.interfaces.openmm_potential import create_mixed_system

HARTREE_TO_KJMOL = 2625.4996394798  # CODATA
BOHR_TO_ANGSTROM = 0.52917721092

# 与 E8 系列教程同一几何约定（Å）：QM 水在原点，MM 水平移 (2.9, 0.8, 0.3)
QM_POS_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])
MM_SHIFT = np.array([2.9, 0.8, 0.3])
POSITIONS = np.vstack([QM_POS_ANG, QM_POS_ANG + MM_SHIFT])
QM_ATOMS = [0, 1, 2]


def _two_water_topology():
    """逐水一个 chain，residue 'HOH'，原子名 O/H1/H2（tip3p.xml 模板约定）。"""
    top = app.Topology()
    elem_o, elem_h = app.Element.getBySymbol("O"), app.Element.getBySymbol("H")
    for _ in range(2):
        chain = top.addChain()
        res = top.addResidue("HOH", chain)
        o = top.addAtom("O", elem_o, res)
        h1 = top.addAtom("H1", elem_h, res)
        h2 = top.addAtom("H2", elem_h, res)
        top.addBond(o, h1)
        top.addBond(o, h2)
    return top


def _energy_hartree(ctx):
    st = ctx.getState(getEnergy=True)
    return st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJMOL


def _new_context(system):
    integ = openmm.VerletIntegrator(0.2 * unit.femtoseconds)
    ctx = openmm.Context(system, integ)
    ctx.setPositions(POSITIONS * 0.1 * unit.nanometers)  # Å → nm
    return ctx, integ


# ---------------------------------------------------------------------------
# 第 1 步：纯经典参考（tip3p 全体系）
#
# rigidWater=False：保留键/角项（缺省 True 会换成约束，能量分解就得另算约束项）。
# NoCutoff：气相双水，不引周期/截断，与能量分解恒等式的推导口径一致。
# ---------------------------------------------------------------------------
top = _two_water_topology()
mm_system = app.ForceField("tip3p.xml").createSystem(
    top, nonbondedMethod=app.NoCutoff, rigidWater=False)

ctx_mm, _ = _new_context(mm_system)
e_mm_full = _energy_hartree(ctx_mm)
print(f"[1] 纯经典参考   E_MM(全体系, tip3p)   = {e_mm_full: .8f} Hartree")

# QM 水的经典内部键角能（从 mm_system 的真实参数取值，在 QM 几何上求值）
e_bonded_qm_kj = 0.0
for f in mm_system.getForces():
    if isinstance(f, openmm.HarmonicBondForce):
        for i in range(f.getNumBonds()):
            p1, p2, length, k = f.getBondParameters(i)
            if {p1, p2} <= set(QM_ATOMS):
                r = np.linalg.norm(POSITIONS[p1] - POSITIONS[p2]) * 0.1  # Å → nm
                e_bonded_qm_kj += 0.5 * k.value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer**2) * \
                    (r - length.value_in_unit(unit.nanometer))**2
    elif isinstance(f, openmm.HarmonicAngleForce):
        for i in range(f.getNumAngles()):
            p1, p2, p3, theta0, k = f.getAngleParameters(i)
            if {p1, p2, p3} <= set(QM_ATOMS):
                v1 = POSITIONS[p1] - POSITIONS[p2]
                v2 = POSITIONS[p3] - POSITIONS[p2]
                cos_t = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                theta = np.arccos(np.clip(cos_t, -1.0, 1.0))
                e_bonded_qm_kj += 0.5 * k.value_in_unit(
                    unit.kilojoule_per_mole / unit.radian**2) * \
                    (theta - theta0.value_in_unit(unit.radian))**2
e_bonded_qm = e_bonded_qm_kj / HARTREE_TO_KJMOL
print(f"    QM 水内部键角能(tip3p)              = {e_bonded_qm: .8f} Hartree")

# 独立量子参考：QM 水裸算（机械嵌入下 QM 哈密顿量里没有 MM 电荷）
scan = qc_scanner(
    [("O", tuple(QM_POS_ANG[0])), ("H", tuple(QM_POS_ANG[1])),
     ("H", tuple(QM_POS_ANG[2]))],
    basis="sto-3g", active_space=(4, 4), method="uccsd",
)
e_qm, _ = scan(QM_POS_ANG / BOHR_TO_ANGSTROM)
print(f"    E_QM(UCCSD/CAS(4,4), 裸 QM 水)      = {e_qm: .8f} Hartree")

# ---------------------------------------------------------------------------
# 第 2 步：混合 System + 能量分解恒等式
#
# interpolate=True：产出 CustomCVForce，全局参数 lambda_interpolate
# （0=全经典，1=QM 内部换 TyxonQ）。上游会自动：删 QM 内部键角与
# nonbonded 自作用（tip3p 里本来就全例外清零），保留 QM-MM 跨区经典项。
# ---------------------------------------------------------------------------
system = create_mixed_system(
    top, mm_system, qm_atoms=QM_ATOMS, interpolate=True,
    basis="sto-3g", active_space=(4, 4), method="uccsd",
)
ctx, integ = _new_context(system)

ctx.setParameter("lambda_interpolate", 0.0)
e_lam0 = _energy_hartree(ctx)
ctx.setParameter("lambda_interpolate", 1.0)
e_lam1 = _energy_hartree(ctx)

print(f"\n[2] 混合体系（interpolate=True）")
print(f"    E(λ=0)  全经典                      = {e_lam0: .8f} Hartree")
print(f"    E(λ=1)  QM 内部换 TyxonQ            = {e_lam1: .8f} Hartree")

# 验收 1：λ=0 必须严格回到纯经典（同一套力场项，只是装进了 CV）
assert abs(e_lam0 - e_mm_full) < 1e-8, f"λ=0 {e_lam0} != MM {e_mm_full}"
print(f"    λ=0 vs 纯经典：差 {abs(e_lam0 - e_mm_full):.2e}（< 1e-8 ✓）")

# 验收 2：分解恒等式（容差覆盖 UCCSD 抖动 ~1e-8 Ha，取 2e-4 留余量）
delta = e_lam1 - e_lam0
expected = e_qm - e_bonded_qm
assert abs(delta - expected) < 2e-4, f"分解恒等式不成立: {delta} vs {expected}"
print(f"    E(λ=1)−E(λ=0) = {delta: .8f}")
print(f"    E_QM−E_bonded = {expected: .8f}（差 {abs(delta - expected):.2e} < 2e-4 ✓）")

# ---------------------------------------------------------------------------
# 第 3 步：λ 扫描（自由能微扰的入口演示）
#
# 每个 λ 一次量子单点（PythonForce 不感知 λ，CV 只在外面加权）——生产上
# 的 FEP/TI 采样就在这个 setParameter 循环外套采样器。
# ---------------------------------------------------------------------------
print("\n[3] λ 扫描：")
print("    λ       E(λ)/Hartree")
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    ctx.setParameter("lambda_interpolate", lam)
    print(f"    {lam:4.2f}   {_energy_hartree(ctx): .8f}")

# ---------------------------------------------------------------------------
# 第 4 步：λ=1 短 MD（混合体系动力学可跑性）
#
# 注意：混合模式下上游回调把 MM 侧力建为 float32（asepotential._computeASE），
# 力精度约 1e-7 相对——对短演示无碍，生产长轨迹须知悉。
# ---------------------------------------------------------------------------
ctx.setParameter("lambda_interpolate", 1.0)
ctx.setVelocitiesToTemperature(300 * unit.kelvin)
v = ctx.getState(getVelocities=True).getVelocities(asNumpy=True) \
    .value_in_unit(unit.nanometers / unit.picoseconds)
m = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
              for i in range(system.getNumParticles())])
ctx.setVelocities((v - (m @ v) / m.sum()) * unit.nanometers / unit.picoseconds)

print("\n[4] λ=1 短 MD（5 步 × 0.2 fs）：")
print("    step   potential/Ha    kinetic/Ha     |F|max / (kJ/mol/nm)")
for step in range(6):
    st = ctx.getState(getEnergy=True, getForces=True)
    pot = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJMOL
    kin = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJMOL
    fmax = np.abs(st.getForces().value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)).max()
    print(f"    {step:4d}   {pot:14.8f}   {kin:11.8f}   {fmax:12.2f}")
    assert np.isfinite(fmax) and fmax > 0.0
    if step < 5:
        integ.step(1)

print("\n完成。OpenMM 机械嵌入 QM/MM（create_mixed_system + λ 插值）验证通过。")
print("要静电嵌入（MM 电荷进 QM 哈密顿量）请改用 i-PI 三进程路径：")
print("    examples/qmmm/md_lammps_qmmm_embedded/  (E8-A, 簇)")
print("    examples/qmmm/md_lammps_qmmm_pbc/       (E8-B, 周期)")
