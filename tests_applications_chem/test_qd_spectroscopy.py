"""Tier 2 谱学两路线验收：直接关联函数与踢动线性响应。

两条路线均以单比特二能级解析解为小体系金标准；实验数据与真机定标延后。
"""

from __future__ import annotations

import numpy as np

from tyxonq.applications.chem.dynamics.exact import correlation_exact, kicked_evolution
from tyxonq.applications.chem.dynamics.observables import pauli_expectation
from tyxonq.applications.chem.dynamics.spectroscopy import correlation_to_spectrum


def test_correlation_route_matches_two_level_analytic():
    """H=ωZ/2、|0⟩：C_XX(t)=exp(iωt)，对称关联谱峰位于 ω。"""
    omega = 0.7
    ham = [(omega / 2, [("Z", 0)])]
    x_obs = [(1.0, [("X", 0)])]
    psi0 = np.array([1.0, 0.0], dtype=complex)
    times = np.linspace(0.0, 60.0, 1201)
    corr = correlation_exact(x_obs, x_obs, ham, psi0, times, 1)
    np.testing.assert_allclose(corr, np.exp(1j * omega * times), atol=1e-12)

    frequencies, spectrum = correlation_to_spectrum(
        corr.real, times, window="hann", n_frequency=4001
    )
    peak = frequencies[np.argmax(spectrum)]
    assert abs(peak - omega) < 0.04


def test_kicked_route_matches_two_level_retarded_response():
    """弱 X 踢动的响应/(2h)=sin(ωt)，且谱峰与直接关联路线一致。"""
    omega = 0.7
    ham = [(omega / 2, [("Z", 0)])]
    x_obs = [(1.0, [("X", 0)])]
    psi0 = np.array([1.0, 0.0], dtype=complex)
    times = np.linspace(0.0, 20.0, 401)
    h_kick = 1e-3
    states_plus, states_minus = kicked_evolution(
        ham, x_obs, h_kick, psi0, times, 1
    )
    response = np.array([
        (pauli_expectation(plus, x_obs, 1)
         - pauli_expectation(minus, x_obs, 1)) / 2
        for plus, minus in zip(states_plus, states_minus)
    ])
    np.testing.assert_allclose(
        response / (2 * h_kick), np.sin(omega * times), atol=2e-5
    )
    frequencies, spectrum = correlation_to_spectrum(
        response, times, window="hann", n_frequency=2001
    )
    peak = frequencies[np.argmax(spectrum)]
    assert abs(peak - omega) < 0.2
