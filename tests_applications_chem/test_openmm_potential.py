"""openmm_potential 测试（用例 9）。

依据 ``MD_INTEGRATION_PLAN.md`` §3 P3 / §4 用例表：
单水 topology 建纯 QM System，``Context`` 取 (E, F) 与 ``qc_scanner`` 按
kJ/mol/nm 换算一致；``create_mixed_system``（interpolate=True）能建成且
含 ``lambda_interpolate``，λ=0/1 两端能量有限可算。

全程 ``method="uccsd"``（目标量子算法，测试规范）；UCCSD 数值求解器有
~1e-8 Ha 的跨调用抖动，容差按此放宽而非换经典求解器。

机械嵌入的物理边界（QM 区感受不到 MM 静电极化）在模块 docstring 文档化，
不在本用例断言范围内；静电嵌入的另一条交付路径是 i-PI 三进程（用例 11/12），
OpenMM 原生静电嵌入（``create_qmmm_ee_system``，簇嵌入）由本文件后三个测试覆盖。
"""

from __future__ import annotations

import numpy as np
import pytest


def _has_pyscf():
    try:
        import pyscf  # noqa: F401
        return True
    except Exception:
        return False


def _has_ase():
    try:
        import ase  # noqa: F401
        return True
    except Exception:
        return False


def _has_openmm():
    try:
        import openmm  # noqa: F401
        from openmmml import MLPotential  # noqa: F401
        return True
    except Exception:
        return False


needs_pyscf = pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping.")
needs_ase = pytest.mark.skipif(not _has_ase(), reason="ASE not installed; skipping.")
needs_openmm = pytest.mark.skipif(not _has_openmm(), reason="OpenMM/OpenMM-ML not installed; skipping.")

# 换算常数（测试侧独立书写，与 ase_calculator / 附录 B 一致）
BOHR_TO_ANGSTROM = 0.52917721092
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KJMOL = HARTREE_TO_EV * 96.4853321233100184   # eV → kJ/mol
EV_A_TO_KJMOL_NM = 96.4853321233100184 * 10.0            # eV/Å → kJ/mol/nm
BOHR_TO_NM = BOHR_TO_ANGSTROM * 0.1

# 与用例 7 相同的水几何（Å）
H2O_POS_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])
# MM 水相对平移（与 E8 系列教程同一几何约定，用于混合体系）
MM_SHIFT = np.array([2.9, 0.8, 0.3])


def _water_topology(positions_ang):
    """逐水一个 chain、residue 名 'HOH'、原子名 O/H1/H2（tip3p.xml 模板约定）。"""
    from openmm.app import Element, Topology

    top = Topology()
    elem_o, elem_h = Element.getBySymbol("O"), Element.getBySymbol("H")
    n_water = len(positions_ang) // 3
    for w in range(n_water):
        chain = top.addChain()
        res = top.addResidue("HOH", chain)
        o = top.addAtom("O", elem_o, res)
        h1 = top.addAtom("H1", elem_h, res)
        h2 = top.addAtom("H2", elem_h, res)
        top.addBond(o, h1)
        top.addBond(o, h2)
    return top


def _context(system, positions_ang):
    import openmm
    import openmm.unit as unit

    integ = openmm.VerletIntegrator(0.001 * unit.femtoseconds)
    ctx = openmm.Context(system, integ)
    ctx.setPositions(positions_ang * 0.1 * unit.nanometers)  # Å → nm
    return ctx


def _scanner_reference(positions_ang):
    """同一几何直接走 qc_scanner：返回 (E Ha, dE/dR Ha/Bohr)。"""
    from tyxonq.applications.chem.interfaces import qc_scanner

    spec = [(sym, tuple(map(float, p))) for sym, p in
            zip(["O", "H", "H"], positions_ang)]
    scan = qc_scanner(spec, basis="sto-3g", active_space=(4, 4), method="uccsd")
    return scan(positions_ang / BOHR_TO_ANGSTROM)


@needs_pyscf
@needs_ase
@needs_openmm
def test_pure_qm_system_matches_scanner():
    """纯 QM System：Context 的 (E, F) 与 scanner 按 kJ/mol/nm 换算一致。"""
    import openmm
    from tyxonq.applications.chem.interfaces.openmm_potential import create_tyxonq_system

    top = _water_topology(H2O_POS_ANG)
    system = create_tyxonq_system(top, basis="sto-3g", active_space=(4, 4), method="uccsd")
    ctx = _context(system, H2O_POS_ANG)
    state = ctx.getState(getEnergy=True, getForces=True)

    e_kj = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
    f_kj_nm = np.asarray(state.getForces().value_in_unit(
        openmm.unit.kilojoule_per_mole / openmm.unit.nanometer))

    e_ha, de_ha_bohr = _scanner_reference(H2O_POS_ANG)
    # UCCSD 跨调用抖动 ~1e-8 Ha；换算链 PythonForce→ASE→scanner 不引入额外近似
    assert e_kj == pytest.approx(e_ha * HARTREE_TO_KJMOL, rel=1e-7)
    expected_f = -np.asarray(de_ha_bohr) * HARTREE_TO_KJMOL / BOHR_TO_NM  # F = -dE/dR
    np.testing.assert_allclose(f_kj_nm, expected_f, rtol=1e-6, atol=1e-3)


@needs_pyscf
@needs_ase
@needs_openmm
def test_info_charge_mapping():
    """``info={'charge': 0}`` 与显式 ``charge=0`` 等价（映射经 setdefault）。"""
    from tyxonq.applications.chem.interfaces.openmm_potential import create_tyxonq_system

    top = _water_topology(H2O_POS_ANG)
    sys_a = create_tyxonq_system(top, info={"charge": 0}, basis="sto-3g",
                                 active_space=(4, 4), method="uccsd")
    sys_b = create_tyxonq_system(top, charge=0, basis="sto-3g",
                                 active_space=(4, 4), method="uccsd")

    import openmm
    e_a = _context(sys_a, H2O_POS_ANG).getState(getEnergy=True) \
        .getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
    e_b = _context(sys_b, H2O_POS_ANG).getState(getEnergy=True) \
        .getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
    assert e_a == pytest.approx(e_b, rel=1e-7)


@needs_pyscf
@needs_ase
@needs_openmm
def test_mixed_system_builds_with_lambda_interpolate():
    """混合体系（interpolate=True）：含 lambda_interpolate；λ=0/1 能量有限可算。"""
    import openmm
    import openmm.app as app
    from tyxonq.applications.chem.interfaces.openmm_potential import create_mixed_system

    positions = np.vstack([H2O_POS_ANG, H2O_POS_ANG + MM_SHIFT])
    top = _water_topology(positions)
    mm_system = app.ForceField("tip3p.xml").createSystem(
        top, nonbondedMethod=app.NoCutoff)

    system = create_mixed_system(
        top, mm_system, qm_atoms=[0, 1, 2], interpolate=True,
        basis="sto-3g", active_space=(4, 4), method="uccsd")

    cvs = [f for f in system.getForces() if isinstance(f, openmm.CustomCVForce)]
    assert len(cvs) == 1
    cv = cvs[0]
    params = [cv.getGlobalParameterName(i) for i in range(cv.getNumGlobalParameters())]
    assert "lambda_interpolate" in params

    ctx = _context(system, positions)
    energies = {}
    for lam in (0.0, 1.0):
        ctx.setParameter("lambda_interpolate", lam)
        e = ctx.getState(getEnergy=True).getPotentialEnergy() \
            .value_in_unit(openmm.unit.kilojoule_per_mole)
        assert np.isfinite(e)
        energies[lam] = e
    # λ=0 是纯经典（全 tip3p），λ=1 把 QM 水内部换成 TyxonQ：两端能量应不同
    assert energies[0.0] != pytest.approx(energies[1.0], abs=1e-9)


@needs_pyscf
@needs_ase
@needs_openmm
def test_mixed_system_removes_qm_internal_mm_terms():
    """混合体系（interpolate=False）：QM 子集内部经典项被移除、QM-MM 保留。

    判据：原 ``mm_system`` 里 QM 水的键/角项（HarmonicBond/Angle）在混合
    体系里全部消失（改由 TyxonQ 势能面负责），而 MM 水自身的与跨区项保留。
    """
    import openmm
    import openmm.app as app
    from tyxonq.applications.chem.interfaces.openmm_potential import create_mixed_system

    positions = np.vstack([H2O_POS_ANG, H2O_POS_ANG + MM_SHIFT])
    top = _water_topology(positions)
    # rigidWater=False：缺省 True 会把水的键/角换成约束，就看不到键合项移除了
    mm_system = app.ForceField("tip3p.xml").createSystem(
        top, nonbondedMethod=app.NoCutoff, rigidWater=False)

    def _bonded_terms(system):
        qm = {0, 1, 2}
        n_bond_in_qm = n_angle_in_qm = 0
        n_bond_total = 0
        for f in system.getForces():
            if isinstance(f, openmm.HarmonicBondForce):
                n_bond_total = f.getNumBonds()
                for i in range(f.getNumBonds()):
                    p1, p2, *_ = f.getBondParameters(i)
                    if p1 in qm and p2 in qm:
                        n_bond_in_qm += 1
            elif isinstance(f, openmm.HarmonicAngleForce):
                for i in range(f.getNumAngles()):
                    p1, p2, p3, *_ = f.getAngleParameters(i)
                    if {p1, p2, p3} <= qm:
                        n_angle_in_qm += 1
        return n_bond_in_qm, n_angle_in_qm, n_bond_total

    b0, a0, total0 = _bonded_terms(mm_system)
    assert b0 == 2 and a0 == 1  # 原体系：QM 水有 2 键 1 角（sanity）

    system = create_mixed_system(
        top, mm_system, qm_atoms=[0, 1, 2], interpolate=False,
        basis="sto-3g", active_space=(4, 4), method="uccsd")
    b1, a1, total1 = _bonded_terms(system)
    assert (b1, a1) == (0, 0)        # QM 内部键/角全部移除
    assert total1 == total0 - 2      # 只剩 MM 水的 2 个键


# ---- 静电嵌入（create_qmmm_ee_system，簇嵌入）----
# 两水二聚体：chain 0 = QM 水，chain 1 = MM 水（tip3p 电荷）。
TIP3P_CHARGES = [-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]
KC_KJ_NM = 138.935456  # OpenMM 库仑常数，kJ·nm/(mol·e²)


def _ee_setup():
    """两水二聚体的 topology / 经典 System / 坐标（rigidWater=False 保留键合项）。"""
    import openmm.app as app

    positions = np.vstack([H2O_POS_ANG, H2O_POS_ANG + MM_SHIFT])
    top = _water_topology(positions)
    mm_system = app.ForceField("tip3p.xml").createSystem(
        top, nonbondedMethod=app.NoCutoff, rigidWater=False)
    return top, mm_system, positions


def _nonbonded_of(system):
    import openmm

    for f in system.getForces():
        if isinstance(f, openmm.NonbondedForce):
            return f
    raise AssertionError("no NonbondedForce in system")


def _strip_python_force(system):
    """移除 PythonForce，得到同一手术后的纯经典参考体系（含 MM 水自身键合项）。"""
    import openmm

    for i in range(system.getNumForces()):
        if isinstance(system.getForce(i), openmm.PythonForce):
            system.removeForce(i)
            return system
    raise AssertionError("no PythonForce in system")


@needs_pyscf
@needs_ase
@needs_openmm
def test_ee_system_surgery_and_guards():
    """防双计数手术：QM 电荷置 0、含 QM 例外清零、QM 内部键合项移除；pbc 参数拒收。"""
    import openmm
    from tyxonq.applications.chem.interfaces.openmm_potential import create_qmmm_ee_system

    top, mm_system, positions = _ee_setup()

    with pytest.raises(ValueError, match="cluster"):
        create_qmmm_ee_system(top, mm_system, [0, 1, 2], TIP3P_CHARGES,
                              mm_lattice=np.eye(3) * 2.0,
                              basis="sto-3g", active_space=(4, 4))

    system = create_qmmm_ee_system(top, mm_system, qm_atoms=[0, 1, 2],
                                   atom_charges=TIP3P_CHARGES,
                                   basis="sto-3g", active_space=(4, 4), method="uccsd")
    nb = _nonbonded_of(system)
    for i in (0, 1, 2):
        charge, *_ = nb.getParticleParameters(i)
        assert charge._value == 0.0                # QM 电荷置 0（防双计数）
    for i in (3, 4, 5):
        charge, *_ = nb.getParticleParameters(i)
        assert charge._value == pytest.approx(TIP3P_CHARGES[i])  # MM 电荷保留（供 MM-MM）
    for k in range(nb.getNumExceptions()):
        p1, p2, charge_prod, *_ = nb.getExceptionParameters(k)
        if p1 in (0, 1, 2) or p2 in (0, 1, 2):
            assert charge_prod._value == 0.0       # 含 QM 的例外库仑全清零
    # QM 内部键/角移除（同用例 4 判据）
    n_bond_in_qm = 0
    for f in system.getForces():
        if isinstance(f, openmm.HarmonicBondForce):
            for i in range(f.getNumBonds()):
                p1, p2, *_ = f.getBondParameters(i)
                if p1 in (0, 1, 2) and p2 in (0, 1, 2):
                    n_bond_in_qm += 1
    assert n_bond_in_qm == 0


@needs_pyscf
@needs_ase
@needs_openmm
def test_ee_energy_decomposition():
    """总势能 = E_QM(嵌入) + E_MM(同一手术后的纯经典体系)。

    嵌入参考直接走 ``qc_scanner(mm_charges=...)``；经典参考用
    ``_strip_python_force`` 移除 PythonForce 后的同一 System（含 MM 水自身
    键合项与 QM-MM LJ）。另断言嵌入能量与裸算不同，证明 MM 电荷确实进了
    QM 哈密顿量；符号依赖几何取向（本例 O 对 O，库仑排斥主导，嵌入升能）。
    """
    import openmm
    from tyxonq.applications.chem.interfaces import qc_scanner
    from tyxonq.applications.chem.interfaces.openmm_potential import create_qmmm_ee_system

    top, mm_system, positions = _ee_setup()

    mm_idx = [3, 4, 5]
    scan = qc_scanner([("O", tuple(H2O_POS_ANG[0])), ("H", tuple(H2O_POS_ANG[1])),
                       ("H", tuple(H2O_POS_ANG[2]))],
                      basis="sto-3g", active_space=(4, 4), method="uccsd",
                      mm_charges=(positions[mm_idx], np.array(TIP3P_CHARGES)[mm_idx]))
    e_qm_emb, _ = scan(H2O_POS_ANG / BOHR_TO_ANGSTROM)
    scan_bare = qc_scanner([("O", tuple(H2O_POS_ANG[0])), ("H", tuple(H2O_POS_ANG[1])),
                            ("H", tuple(H2O_POS_ANG[2]))],
                           basis="sto-3g", active_space=(4, 4), method="uccsd")
    e_qm_bare, _ = scan_bare(H2O_POS_ANG / BOHR_TO_ANGSTROM)
    # 嵌入改变能量（符号随取向）：本例 O 对 O 排斥，e_qm_emb 高于裸算 ~3e-3 Ha
    assert e_qm_emb != pytest.approx(e_qm_bare, abs=1e-6)
    assert e_qm_emb > e_qm_bare  # 当前几何取向：排斥主导（实测归档，非普适结论）

    system = create_qmmm_ee_system(top, mm_system, qm_atoms=[0, 1, 2],
                                   atom_charges=TIP3P_CHARGES,
                                   basis="sto-3g", active_space=(4, 4), method="uccsd")
    ctx = _context(system, positions)
    e_total = ctx.getState(getEnergy=True).getPotentialEnergy() \
        .value_in_unit(openmm.unit.kilojoule_per_mole)

    classical = _strip_python_force(create_qmmm_ee_system(
        top, _ee_setup()[1], qm_atoms=[0, 1, 2], atom_charges=TIP3P_CHARGES,
        basis="sto-3g", active_space=(4, 4), method="uccsd"))
    e_mm = _context(classical, positions).getState(getEnergy=True) \
        .getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
    e_ref = e_qm_emb * HARTREE_TO_KJMOL + e_mm
    assert e_total == pytest.approx(e_ref, rel=1e-6)


@needs_pyscf
@needs_ase
@needs_openmm
def test_ee_forces_match_region_partition():
    """全原子力 = （区域划分 Calculator 的 QM 梯度 ⊕ MM 反作用力）+ 纯经典力。

    区域划分参考走 ASE 端 ``TyxonQCalculator(qm_indices=..., atom_charges=...)``；
    经典参考用 ``_strip_python_force`` 后的同一 System（含 MM 水自身键合项与
    QM-MM LJ，QM 电荷已置 0），叠加后与 Context 力逐分量对齐。
    """
    import ase
    import openmm
    from tyxonq.applications.chem.interfaces.ase_calculator import TyxonQCalculator
    from tyxonq.applications.chem.interfaces.openmm_potential import create_qmmm_ee_system

    top, mm_system, positions = _ee_setup()

    calc = TyxonQCalculator(qm_indices=[0, 1, 2], atom_charges=TIP3P_CHARGES,
                            basis="sto-3g", active_space=(4, 4), method="uccsd")
    ref_atoms = ase.Atoms("OHHOHH", positions=positions, calculator=calc)
    f_qmmm_ev_a = ref_atoms.get_forces()  # 嵌入 QM 梯度 ⊕ MM 反作用力，eV/Å

    system = create_qmmm_ee_system(top, mm_system, qm_atoms=[0, 1, 2],
                                   atom_charges=TIP3P_CHARGES,
                                   basis="sto-3g", active_space=(4, 4), method="uccsd")
    ctx = _context(system, positions)
    f_ctx = np.asarray(ctx.getState(getForces=True).getForces().value_in_unit(
        openmm.unit.kilojoule_per_mole / openmm.unit.nanometer))

    classical = _strip_python_force(create_qmmm_ee_system(
        top, _ee_setup()[1], qm_atoms=[0, 1, 2], atom_charges=TIP3P_CHARGES,
        basis="sto-3g", active_space=(4, 4), method="uccsd"))
    f_classical = np.asarray(_context(classical, positions).getState(getForces=True)
                             .getForces().value_in_unit(
                                 openmm.unit.kilojoule_per_mole / openmm.unit.nanometer))

    expected = f_qmmm_ev_a * EV_A_TO_KJMOL_NM + f_classical  # eV/Å → kJ/mol/nm
    np.testing.assert_allclose(f_ctx, expected, rtol=1e-5, atol=1e-2)
