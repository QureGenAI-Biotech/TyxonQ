"""Q3：直接关联函数与踢动线性响应两条谱学路线。

仅核心依赖；预期运行时间 < 2 秒。
两条路线均对照单比特二能级解析解，不调用真机。
"""

import numpy as np

from tyxonq.applications.chem.dynamics.exact import correlation_exact, kicked_evolution
from tyxonq.applications.chem.dynamics.observables import pauli_expectation
from tyxonq.applications.chem.dynamics.spectroscopy import correlation_to_spectrum


def main():
    omega = 0.7
    ham = [(omega / 2, [("Z", 0)])]
    x_obs = [(1.0, [("X", 0)])]
    psi0 = np.array([1.0, 0.0], dtype=complex)
    times = np.linspace(0.0, 60.0, 1201)

    corr = correlation_exact(x_obs, x_obs, ham, psi0, times, 1)
    freq_c, spec_c = correlation_to_spectrum(
        corr.real, times, window="hann", n_frequency=2001
    )
    print(f"direct correlation peak: {freq_c[np.argmax(spec_c)]:.4f}")

    h_kick = 0.005
    plus, minus = kicked_evolution(ham, x_obs, h_kick, psi0, times, 1)
    response = np.array([
        (pauli_expectation(p, x_obs, 1) - pauli_expectation(m, x_obs, 1)) / 2
        for p, m in zip(plus, minus)
    ])
    freq_r, spec_r = correlation_to_spectrum(
        response, times, window="hann", n_frequency=2001
    )
    response_error = np.max(np.abs(response / (2 * h_kick) - np.sin(omega * times)))
    print(f"kicked response peak:    {freq_r[np.argmax(spec_r)]:.4f}")
    print(f"analytic frequency:      {omega:.4f}")
    print(f"linear-response max error vs sin(ωt): {response_error:.3e}")


if __name__ == "__main__":
    main()
