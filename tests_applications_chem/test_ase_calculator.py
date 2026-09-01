"""TyxonQCalculator 测试（用例 7）。

依据 ``MD_INTEGRATION_PLAN.md`` §3 P1 / §4 用例表：
单位换算（附录 B，rtol=1e-10）、力符号（F = -dE/dR）、1 步 BFGS 能量下降。
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


needs_pyscf = pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping.")
needs_ase = pytest.mark.skipif(not _has_ase(), reason="ASE not installed; skipping.")

# 与附录 B / ase_calculator.py 中同一组常数（测试侧独立书写以防互相抄袭）。
BOHR_TO_ANGSTROM = 0.52917721092
HARTREE_TO_EV = 27.211386245988

H2O_POS_ANG = [(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)]


def _make_atoms():
    from ase import Atoms
    return Atoms("OHH", positions=H2O_POS_ANG)


def _make_calc():
    from tyxonq.applications.chem.interfaces.ase_calculator import TyxonQCalculator
    return TyxonQCalculator(basis="sto-3g", active_space=(4, 4), method="uccsd")


@needs_pyscf
@needs_ase
def test_units_and_force_sign_match_scanner():
    """单位换算与力符号：ASE 结果 == scanner 原始值按附录 B 换算（rtol=1e-10），力取负。"""
    from tyxonq.applications.chem.interfaces import qc_scanner

    atoms = _make_atoms()
    calc = _make_calc()
    atoms.calc = calc

    e_ev = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # 同一几何直接走 scanner（Å → Bohr）；构造几何带显式坐标避免被解析成 Z-matrix，
    # 首次调用会用下面的 Bohr 坐标覆盖。
    scan = qc_scanner(
        "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        basis="sto-3g", active_space=(4, 4), method="uccsd",
    )
    e_ha, de_ha_bohr = scan(np.array(H2O_POS_ANG) / BOHR_TO_ANGSTROM)

    assert np.isclose(e_ev, e_ha * HARTREE_TO_EV, rtol=1e-10, atol=0.0)
    expected_forces = -np.asarray(de_ha_bohr) * HARTREE_TO_EV / BOHR_TO_ANGSTROM
    assert np.allclose(forces, expected_forces, rtol=1e-10, atol=0.0)
    # free_energy 与 energy 同值（0 K）
    assert atoms.get_potential_energy(force_consistent=True) == pytest.approx(e_ev, abs=0.0)


@needs_pyscf
@needs_ase
def test_implemented_properties_no_stress():
    calc = _make_calc()
    assert set(calc.implemented_properties) == {"energy", "free_energy", "forces"}


@needs_pyscf
@needs_ase
def test_forces_point_downhill():
    """力指向能量下降方向：沿力方向微移，能量应降低。"""
    atoms = _make_atoms()
    atoms.calc = _make_calc()

    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    assert np.abs(f0).max() > 1e-6  # 非平衡几何，力不为为零

    step = 1e-3 * f0 / np.abs(f0).max()  # 沿力方向小幅移动（Å）
    atoms.positions = atoms.positions + step
    e1 = atoms.get_potential_energy()
    assert e1 < e0


@needs_pyscf
@needs_ase
def test_one_step_bfgs_lowers_energy():
    """1 步 BFGS 优化能量下降。"""
    from ase.optimize import BFGS

    atoms = _make_atoms()
    # 轻微扰动，避免恰在驻点上
    rng = np.random.default_rng(7)
    atoms.positions = atoms.positions + rng.normal(scale=0.02, size=atoms.positions.shape)
    atoms.calc = _make_calc()

    e_before = atoms.get_potential_energy()
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=1e-3, steps=1)
    e_after = atoms.get_potential_energy()
    assert e_after < e_before
