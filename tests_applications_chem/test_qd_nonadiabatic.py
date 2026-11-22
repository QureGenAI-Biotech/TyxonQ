"""Tier2 非绝热引擎验收：Tully PES + Ehrenfest + FSSH + ASE 薄壳。

验收门（小体系精确解/解析对照，实验对标延后）：
Tully 三模型能量/NAC、Ehrenfest 能量守恒、FSSH 跳跃统计、ASE 薄壳单位换算。
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from tyxonq.applications.chem.dynamics.models.pes import tully_model
from tyxonq.applications.chem.dynamics.ehrenfest import run_ehrenfest
from tyxonq.applications.chem.dynamics.surface_hopping import run_fssh


def _has_ase() -> bool:
    return importlib.util.find_spec("ase") is not None


def test_tully_model_a_avoided_crossing():
    """Model A：x=0 为避免交叉点（NAC 峰、能隙 = 2C）。"""
    m = tully_model("A")
    evals, _ = m.adiabatic(0.0)
    gap = float(evals[1] - evals[0])
    assert gap == pytest.approx(2 * 0.005)
    assert m.nac(0.0) > m.nac(2.0) > 0


def test_tully_model_b_dual_crossing():
    m = tully_model("B")
    evals, _ = m.adiabatic(0.0)
    assert evals[0] < evals[1]
    evals_far, _ = m.adiabatic(10.0)
    assert abs(evals_far[0] - (-0.1)) < 0.01 or abs(evals_far[1] - 0.0) < 0.01


def test_tully_model_c_extended_coupling():
    m = tully_model("C")
    evals, _ = m.adiabatic(0.0)
    assert abs(evals[1] - evals[0]) > 0.01


def test_tully_gradient_matches_finite_difference():
    """解析梯度 vs 数值差分（Hellmann-Feynman）。"""
    for kind in ("A", "B", "C"):
        m = tully_model(kind)
        for x in (-1.5, 0.3, 2.0):
            grad = m.adiabatic_gradient(x)
            h = 1e-5
            e_p, _ = m.adiabatic(x + h)
            e_m, _ = m.adiabatic(x - h)
            fd = (e_p - e_m) / (2 * h)
            assert np.allclose(grad, fd, atol=1e-6), f"{kind} x={x}"


def test_ehrenfest_energy_conservation():
    """Ehrenfest 能量漂移 < 1e-5（velocity-Verlet + 矩阵指数传播）。"""
    pes = tully_model("A")
    res = run_ehrenfest(pes, x0=-5.0, v0=1.0, dt=0.01, n_steps=1000)
    drift = abs(res.energies[-1] - res.energies[0])
    assert drift < 1e-5, f"能量漂移 {drift}"
    assert np.allclose(res.adiabatic_populations.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(res.diabatic_populations.sum(axis=1), 1.0, atol=1e-10)


def test_ehrenfest_adiabatic_limit():
    """慢速通过避免交叉：绝热极限下主要留在原绝热面。"""
    pes = tully_model("A")
    res = run_ehrenfest(pes, x0=-3.0, v0=0.05, dt=0.02, n_steps=2000)
    assert res.adiabatic_populations[-1, 0] > 0.5


def test_fssh_seed_reproducible():
    pes = tully_model("A")
    r1 = run_fssh(pes, -5.0, 1.0, 0, 0.01, 500, seed=42)
    r2 = run_fssh(pes, -5.0, 1.0, 0, 0.01, 500, seed=42)
    assert np.allclose(r1.x, r2.x)
    assert len(r1.hops) == len(r2.hops)


def test_fssh_hop_statistics():
    """快速通过时有非绝热跳跃；跳跃记录完整。"""
    pes = tully_model("A")
    hops_total = 0
    for seed in range(20):
        r = run_fssh(pes, -5.0, 1.0, 0, 0.01, 1500, seed=seed)
        hops_total += len(r.hops)
        for h in r.hops:
            assert "frustrated" in h and "from" in h and "to" in h
    assert hops_total > 0


def test_fssh_frustrated_hop():
    """低动能时跳跃被阻（frustrated hop 反弹）。"""
    pes = tully_model("A")
    r = run_fssh(pes, -5.0, 0.001, 0, 0.01, 100, seed=0)
    successful = [h for h in r.hops if not h["frustrated"]]
    assert len(successful) == 0


@pytest.mark.skipif(not _has_ase(), reason="ASE 未安装")
def test_ase_calculator_energy_forces_and_units():
    """ASE 返回 eV 与 eV/Å，且力严格对应所选绝热面的负梯度。"""
    from ase import Atoms

    from tyxonq.applications.chem.interfaces.dynamics_md import (
        BOHR_TO_ANGSTROM,
        HARTREE_TO_EV,
        NonadiabaticASECalculator,
    )

    pes = tully_model("A")
    x_angstrom = 1.2
    x_bohr = x_angstrom / BOHR_TO_ANGSTROM
    for surface in (0, 1):
        atoms = Atoms("H", positions=[[x_angstrom, 0.0, 0.0]])
        atoms.calc = NonadiabaticASECalculator(pes, surface=surface)
        evals, _ = pes.adiabatic(x_bohr)
        gradient = pes.adiabatic_gradient(x_bohr)
        assert atoms.get_potential_energy() == pytest.approx(
            evals[surface] * HARTREE_TO_EV
        )
        expected_force = -gradient[surface] * HARTREE_TO_EV / BOHR_TO_ANGSTROM
        assert atoms.get_forces()[0, 0] == pytest.approx(expected_force)
        assert np.array_equal(atoms.get_forces()[0, 1:], np.zeros(2))


@pytest.mark.skipif(not _has_ase(), reason="ASE 未安装")
def test_ase_velocity_verlet_runs_and_conserves_energy():
    """标准 ASE MD 驱动器可消费薄壳，短时单面轨迹总能量守恒。"""
    from ase import Atoms
    from ase.md.verlet import VelocityVerlet

    from tyxonq.applications.chem.interfaces.dynamics_md import NonadiabaticASECalculator

    atoms = Atoms("H", positions=[[-2.0, 0.0, 0.0]])
    atoms.calc = NonadiabaticASECalculator(tully_model("A"), surface=0)
    dynamics = VelocityVerlet(atoms, timestep=0.01, logfile=None)
    initial_energy = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    dynamics.run(20)
    final_energy = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    assert np.isfinite(atoms.positions).all()
    assert not np.allclose(atoms.positions[0, 0], -2.0)
    assert abs(final_energy - initial_energy) < 1e-5
