"""Tier 1b OLED 轨道验收：激子动力学（严格解析/ED 对照，实验对照延后）。

验收门（用户 2026-11-12 拍板：小体系精确解搞定，实验对标延后）：

- 二聚体 Rabi 解析：P_0(t)=cos^2(Jt)（零失谐）与 k=2sqrt(J^2+(delta/2)^2)；
- Frenkel 链 ED 本征谱：开边界链 eps_k = 2J cos(k pi/(n+1)) 解析对照；
- 数值 vs 线路布居一致性：n=4 短窗两档对齐（采样噪声内）；
- MSD/L_D 标量：初态 MSD(0)=0、ballistic 短时行为、L_D=sqrt(MSD(tau))。

单位：原子单位 hbar=1。设备档 provider="simulator"。
"""

from __future__ import annotations

import numpy as np
import pytest

from tyxonq.applications.chem.dynamics.exciton_dynamics import (
    ballistic_velocity,
    diffusion_length,
    diffusivity_from_msd,
    mean_squared_displacement,
    mobility_from_diffusivity,
    population_trajectory,
    rabi_frequency_two_site,
)
from tyxonq.applications.chem.dynamics.models.exciton import (
    frenkel_chain,
    two_site_dimer,
)
from tyxonq.applications.chem.dynamics.observables import build_pauli_matrix


def test_dimer_rabi_analytic_zero_detuning():
    """零失谐二聚体：P_0(t)=cos^2(Jt)，机器精度对照解析。"""
    J = 0.5
    model = two_site_dimer(hopping=J, detuning=0.0)
    times = np.linspace(0.0, 3.0, 61)
    out = population_trajectory(model, times, site0=0)
    pops = out["populations"]
    expected = np.cos(J * times) ** 2
    assert np.allclose(pops[:, 0], expected, atol=1e-10)
    assert np.allclose(pops.sum(axis=1), 1.0, atol=1e-12)
    assert rabi_frequency_two_site(J, 0.0) == pytest.approx(2 * J)


def test_dimer_rabi_with_detuning():
    """失谐二聚体：振荡频率 2sqrt(J^2+(delta/2)^2)，振幅 <1。"""
    J, delta = 0.5, 0.3
    model = two_site_dimer(hopping=J, detuning=delta)
    times = np.linspace(0.0, 6.0, 121)
    out = population_trajectory(model, times, site0=0)
    pops = out["populations"]
    k = rabi_frequency_two_site(J, delta)
    amp = J ** 2 / (J ** 2 + (delta / 2.0) ** 2)
    expected = 1.0 - amp * np.sin(k * times / 2.0) ** 2
    assert np.allclose(pops[:, 0], expected, atol=1e-10)
    assert pops[:, 1].max() < 0.999


def test_frenkel_chain_eigenvalues_analytic():
    """开边界纯净链：ED 本征值 = 2J cos(k pi/(n+1))（单激子子空间）。"""
    n, J = 5, 0.7
    model = frenkel_chain(n, hopping=J, epsilon=0.0)
    H = build_pauli_matrix(model.pauli_items(), n)
    single_exciton = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            idx_a, idx_b = 1 << a, 1 << b
            single_exciton[a, b] = H[idx_a, idx_b].real
    evals = np.sort(np.linalg.eigvalsh(single_exciton))
    expected = np.sort([2 * J * np.cos(k * np.pi / (n + 1)) for k in range(1, n + 1)])
    assert np.allclose(evals, expected, atol=1e-12)


def test_frenkel_chain_identity_offset():
    """identity 偏移：整体能移 epsilon 不改变布居动力学。"""
    m1 = frenkel_chain(4, hopping=0.5, epsilon=0.0)
    m2 = frenkel_chain(4, hopping=0.5, epsilon=3.0)
    times = np.linspace(0, 2.0, 21)
    p1 = population_trajectory(m1, times)["populations"]
    p2 = population_trajectory(m2, times)["populations"]
    assert np.allclose(p1, p2, atol=1e-10)


def test_population_numeric_vs_circuits():
    """n=4 纯净链：线路档（Trotter）布居与数值精确档一致。

    容差 = 采样噪声 + 一阶 Trotter 误差（独立细步长 dt_trotter=0.02
    抑制 Trotter 误差后，残差以采样噪声为主）。
    """
    model = frenkel_chain(4, hopping=0.5, epsilon=0.0)
    times = np.linspace(0.0, 2.0, 21)
    num = population_trajectory(model, times)
    circ = population_trajectory(
        model, times, provider="simulator", shots=32768, dt_trotter=0.02
    )
    assert np.allclose(circ["populations"], num["populations"], atol=2e-2)


def test_msd_ballistic_and_diffusion_length():
    """MSD(0)=0；纯净链短时 ballistic；L_D=sqrt(MSD(tau)) 手算对照。"""
    model = frenkel_chain(5, hopping=0.5)
    times = np.linspace(0.0, 2.0, 41)
    out = population_trajectory(model, times)
    msd = mean_squared_displacement(out["populations"], out["site0"])
    assert abs(msd[0]) < 1e-12
    assert np.all(msd[1:] >= 0)
    v = ballistic_velocity(out["populations"], times, out["site0"])
    assert 0 < v < 10.0
    tau = 1.0
    ld = diffusion_length(out["populations"], times, tau, out["site0"])
    assert ld["L_D"] == pytest.approx(np.sqrt(float(np.interp(tau, times, msd))))
    assert not ld["extrapolated"]
    ld_ex = diffusion_length(out["populations"], times, 99.0, out["site0"])
    assert ld_ex["extrapolated"]


def test_mobility_einstein_conversion():
    """mu = D/(k_B T)（原子单位）。"""
    D, T = 2.0, 300.0
    mu = mobility_from_diffusivity(D, T)
    kB = 3.166811563e-6
    assert mu == pytest.approx(D / (kB * T))
    with pytest.raises(ValueError):
        mobility_from_diffusivity(D, 0.0)


def test_diffusivity_requires_enough_points():
    model = frenkel_chain(3, hopping=0.5)
    times = np.linspace(0, 1.0, 5)
    out = population_trajectory(model, times)
    with pytest.raises(ValueError):
        diffusivity_from_msd(out["populations"], times[:2])


def test_disorder_seed_reproducible():
    """无序种子可复现。"""
    m1 = frenkel_chain(4, hopping=0.5, disorder_sigma=0.2, seed=42)
    m2 = frenkel_chain(4, hopping=0.5, disorder_sigma=0.2, seed=42)
    assert np.allclose(m1.site_energies, m2.site_energies)
    m3 = frenkel_chain(4, hopping=0.5, disorder_sigma=0.2, seed=7)
    assert not np.allclose(m1.site_energies, m3.site_energies)
