"""E-EE：OpenMM 原生**静电嵌入** QM/MM（簇嵌入版，单进程，无 MDI / i-PI）。

物理层级（与 E6 机械嵌入对照）
--------------------------------
- E6（``md_openmm_qmmm_solvated.py``）：机械嵌入——QM 区只算自身内部能量，
  MM 点电荷只以经典力作用于 QM 原子核，QM 电子结构感受不到 MM 极化。
- 本教程：静电嵌入——MM 点电荷经 ``pyscf.qmmm`` 进入 QM 哈密顿量，**每步随
  坐标更新**（``set_mm_charges`` 重入）；QM 梯度与 MM 反作用力
  （``mm_gradient``）由同一个 ``PythonForce`` 回调返回，与经典力场（MM-MM
  静电、QM-MM vdW、MM 键合项）在 OpenMM 内自动求和。

防双计数协议（``create_qmmm_ee_system`` 内部，与 E8 的 LAMMPS 侧一一对应）：
QM 原子在经典 ``NonbondedForce`` 中电荷置 0（QM-MM 库仑只由嵌入算一份）；
QM 子集内部键/角与约束删除、QM 原子对 nonbonded 全清零（内部全归量子）；
QM-MM vdW 与 MM 侧全部经典项原样保留。

已知近似（如实告知）：MM 反作用力的解析式缺 post-HF 轨道响应项（RB5 归因，
见 ``examples/qmmm/md_lammps_qmmm_pbc/VALIDATION.md`` §4），偏置 ~4.3e-5
Ha/Bohr ≈ 2.2 meV/Å，随几何光滑；恒温系综与优化可用，**严格 NVE 守恒诊断
失效**——本教程的漂移容差据此放宽。

体系：两水二聚体（chain 0 = QM 水，chain 1 = MM 水，tip3p），全程 UCCSD。

运行（需 ``pip install 'tyxonq[md]'``，即 openmm/openmmml/ase）::

    PYTHONPATH=src python examples/qmmm/md_openmm_qmmm_electrostatic.py
"""

from __future__ import annotations

import numpy as np

import openmm
import openmm.app as app
import openmm.unit as unit

from tyxonq.applications.chem.interfaces import qc_scanner
from tyxonq.applications.chem.interfaces.openmm_potential import create_qmmm_ee_system

# ---- 常数（附录 B 口径；教程侧独立书写） ----
BOHR_TO_ANGSTROM = 0.52917721092
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KJMOL = HARTREE_TO_EV * 96.4853321233100184

# ---- 体系：两水二聚体（Å）；QM = chain 0，MM = chain 1 ----
QM_POS_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])
MM_SHIFT = np.array([2.9, 0.8, 0.3])
QM_ATOMS = [0, 1, 2]
TIP3P_CHARGES = [-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]  # 全体系，QM 条目被嵌入层忽略


def _two_water_topology():
    from openmm.app import Element, Topology

    top = Topology()
    elem_o, elem_h = Element.getBySymbol("O"), Element.getBySymbol("H")
    for _ in range(2):
        chain = top.addChain()
        res = top.addResidue("HOH", chain)
        o = top.addAtom("O", elem_o, res)
        h1 = top.addAtom("H1", elem_h, res)
        h2 = top.addAtom("H2", elem_h, res)
        top.addBond(o, h1)
        top.addBond(o, h2)
    return top


def _strip_python_force(system):
    """移除 PythonForce → 同一手术后的纯经典参考体系（能量分解用）。"""
    for i in range(system.getNumForces()):
        if isinstance(system.getForce(i), openmm.PythonForce):
            system.removeForce(i)
            return system
    raise AssertionError("no PythonForce in system")


def _strip_com_momentum(ctx, system):
    """手动剥离质心平动（CMMotionRemover 会在第一步一次性剥掉，造成守恒台阶）。"""
    v = ctx.getState(getVelocities=True).getVelocities(asNumpy=True) \
        .value_in_unit(unit.nanometers / unit.picoseconds)
    m = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                  for i in range(system.getNumParticles())])
    v -= (m @ v) / m.sum()
    ctx.setVelocities(v * unit.nanometers / unit.picoseconds)


def main():
    positions = np.vstack([QM_POS_ANG, QM_POS_ANG + MM_SHIFT])
    top = _two_water_topology()
    mm_system = app.ForceField("tip3p.xml").createSystem(
        top, nonbondedMethod=app.NoCutoff, rigidWater=False)

    # ---- 第 1 步：参考能量——裸算 / 嵌入 / 极化效应 ----
    qm_spec = [("O", tuple(QM_POS_ANG[0])), ("H", tuple(QM_POS_ANG[1])),
               ("H", tuple(QM_POS_ANG[2]))]
    mm_idx = [3, 4, 5]
    scan_emb = qc_scanner(qm_spec, basis="sto-3g", active_space=(4, 4), method="uccsd",
                          mm_charges=(positions[mm_idx], np.array(TIP3P_CHARGES)[mm_idx]))
    e_qm_emb, _ = scan_emb(QM_POS_ANG / BOHR_TO_ANGSTROM)
    scan_bare = qc_scanner(qm_spec, basis="sto-3g", active_space=(4, 4), method="uccsd")
    e_qm_bare, _ = scan_bare(QM_POS_ANG / BOHR_TO_ANGSTROM)
    de_pol = e_qm_emb - e_qm_bare
    print("[1] 嵌入效应（MM 电荷进 QM 哈密顿量）")
    print(f"    E_QM 裸算       = {e_qm_bare:.8f} Ha")
    print(f"    E_QM 静电嵌入   = {e_qm_emb:.8f} Ha")
    print(f"    ΔE 极化         = {de_pol:+.6f} Ha  "
          f"（符号随取向：本例 O 对 O，库仑排斥主导）")

    # ---- 第 2 步：能量分解验收——E_total = E_QM(嵌入) + E_MM(经典) ----
    system = create_qmmm_ee_system(top, mm_system, qm_atoms=QM_ATOMS,
                                   atom_charges=TIP3P_CHARGES,
                                   basis="sto-3g", active_space=(4, 4), method="uccsd")
    integ = openmm.VerletIntegrator(0.2 * unit.femtoseconds)
    ctx = openmm.Context(system, integ)
    ctx.setPositions(positions * 0.1 * unit.nanometers)
    e_total = ctx.getState(getEnergy=True).getPotentialEnergy() \
        .value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJMOL

    classical = _strip_python_force(create_qmmm_ee_system(
        *_two_water_setup(), qm_atoms=QM_ATOMS, atom_charges=TIP3P_CHARGES,
        basis="sto-3g", active_space=(4, 4), method="uccsd"))
    ctx_c = openmm.Context(classical, openmm.VerletIntegrator(0.2 * unit.femtoseconds))
    ctx_c.setPositions(positions * 0.1 * unit.nanometers)
    e_mm = ctx_c.getState(getEnergy=True).getPotentialEnergy() \
        .value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJMOL

    decomp_err = abs(e_total - (e_qm_emb + e_mm))
    print("[2] 能量分解（步 0）")
    print(f"    E_total(OpenMM) = {e_total:.8f} Ha")
    print(f"    E_QM(嵌入)+E_MM = {e_qm_emb + e_mm:.8f} Ha   "
          f"(E_MM 经典 = {e_mm:.8f} Ha)")
    print(f"    |分解残差|      = {decomp_err:.2e} Ha  (容差 2e-4)")
    assert decomp_err < 2e-4

    # ---- 第 3 步：短 NVE 轨迹——每步都是全新的嵌入量子单点 ----
    ctx.setVelocitiesToTemperature(300 * unit.kelvin)
    _strip_com_momentum(ctx, system)
    print("[3] NVE 10 步 × 0.2 fs（RB5 近似：反作用力缺轨道响应项，严格守恒诊断失效）")
    print("    step   kinetic/Ha     potential/Ha   total/Ha")
    totals = []
    for step in range(10):
        integ.step(1)
        st = ctx.getState(getEnergy=True)
        kin = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJMOL
        pot = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJMOL
        totals.append(kin + pot)
        print(f"    {step + 1:>4}   {kin:12.6f}   {pot:12.6f}   {kin + pot:12.6f}")
    drift = max(totals) - min(totals)
    print(f"    总能量极差 = {drift:.2e} Ha  (容差 5e-3：RB5 偏置上限 ~4.3e-5 Ha/Bohr)")
    assert drift < 5e-3

    print("\n[OK] OpenMM 原生静电嵌入 QM/MM 验收通过（能量分解 + 短轨迹）。")


def _two_water_setup():
    """供分解参考用的独立经典 System（避免与主路径共享被手术对象）。"""
    top = _two_water_topology()
    mm_system = app.ForceField("tip3p.xml").createSystem(
        top, nonbondedMethod=app.NoCutoff, rigidWater=False)
    return top, mm_system


if __name__ == "__main__":
    main()
