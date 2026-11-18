"""Q8：经典芯片上的 QD MVP 可信度验收（完全离线）。


演示目的不是“程序能运行”，而是分别检查解析金标准、Trotter/踢动/时间窗
收敛和有限 shots 统计区间。随机采样误差不要求逐点单调。
"""

from __future__ import annotations

import numpy as np

from tyxonq.applications.chem.dynamics.benchmarks import (
    comparison_report,
    confidence_interval,
    convergence_report,
    pauli_standard_error,
)
from tyxonq.applications.chem.dynamics.circuits import (
    measure_pauli_expectation,
    trotter_expectation_trajectory,
)
from tyxonq.applications.chem.dynamics.exact import (
    correlation_exact,
    expectation_trajectory,
    kicked_evolution,
)
from tyxonq.applications.chem.dynamics.observables import (
    bitstring_to_state,
    pauli_expectation,
)
from tyxonq.applications.chem.dynamics.spectroscopy import correlation_to_spectrum


def trotter_convergence():
    """非对易二比特模型的一阶 Trotter 误差随 dt 缩小。"""
    ham = [(0.5, [("Z", 0), ("Z", 1)]), (0.3, [("X", 0)]), (0.2, [("X", 1)])]
    obs = [(1.0, [("Z", 0), ("Z", 1)])]
    psi0 = bitstring_to_state("01", 2)
    times = np.array([0.0, 0.4, 0.8, 1.2])
    reference = expectation_trajectory(ham, obs, psi0, times, 2)
    errors = []
    dts = [0.2, 0.1, 0.05]
    for dt in dts:
        actual = trotter_expectation_trajectory(
            ham, obs, times, 2, dt=dt, init_bitstring="01", shots=0
        )
        errors.append(float(np.max(np.abs(actual - reference))))
    return convergence_report(dts, errors)


def kick_convergence():
    """中心差分响应/(2h) → 单比特解析 sin(ωt)，预期二阶收敛。"""
    omega = 0.7
    ham = [(omega / 2, [("Z", 0)])]
    x_obs = [(1.0, [("X", 0)])]
    psi0 = np.array([1.0, 0.0], dtype=complex)
    times = np.linspace(0.0, 10.0, 201)
    errors = []
    strengths = [0.04, 0.02, 0.01]
    for strength in strengths:
        plus, minus = kicked_evolution(ham, x_obs, strength, psi0, times, 1)
        response = np.array([
            (pauli_expectation(p, x_obs, 1) - pauli_expectation(m, x_obs, 1)) / 2
            for p, m in zip(plus, minus)
        ])
        analytic = 2 * strength * np.sin(omega * times)
        errors.append(float(np.max(np.abs(response - analytic))))
    return convergence_report(strengths, errors)


def window_convergence():
    """延长物理时间窗降低 Hann 谱的带外泄漏；增加绘图点数不等价。"""
    omega = 0.7
    ham = [(omega / 2, [("Z", 0)])]
    x_obs = [(1.0, [("X", 0)])]
    psi0 = np.array([1.0, 0.0], dtype=complex)
    errors = []
    durations = [10.0, 20.0, 40.0]
    for duration in durations:
        times = np.linspace(0.0, duration, int(duration / 0.05) + 1)
        corr = correlation_exact(x_obs, x_obs, ham, psi0, times, 1).real
        frequencies, spectrum = correlation_to_spectrum(
            corr, times, window="hann", n_frequency=4001
        )
        sideband = (frequencies >= 1.0) & (frequencies <= 1.5)
        leakage = np.sum(spectrum[sideband]) / np.max(spectrum)
        errors.append(float(leakage))
    return convergence_report(durations, errors)


def shot_intervals():
    """实际线路有限采样的 95% 区间；单个区间不覆盖真值不构成失败。"""
    from tyxonq.core.ir.circuit import Circuit

    omega = 1.4
    time = 1.0
    exact = float(np.cos(omega * time))
    ham = [(omega / 2, [("Z", 0)])]
    x_obs = [(1.0, [("X", 0)])]
    from tyxonq.applications.chem.dynamics.circuits import trotter_ops

    circuit = Circuit(1)
    circuit.ops.append(("h", 0))
    for _ in range(20):
        circuit.ops.extend(trotter_ops(ham, time / 20))
    rows = []
    for shots in (512, 2048, 8192):
        estimate = measure_pauli_expectation(circuit, x_obs, 1, shots=shots)
        sigma = pauli_standard_error(estimate, shots)
        lower, upper = confidence_interval(estimate, sigma)
        rows.append({
            "shots": shots,
            "estimate": estimate,
            "standard_error": sigma,
            "ci95": [lower, upper],
            "contains_exact": bool(lower <= exact <= upper),
        })
    return exact, rows


def main():
    analytic_times = np.linspace(0.0, 5.0, 101)
    omega = 0.7
    corr = correlation_exact(
        [(1.0, [("X", 0)])], [(1.0, [("X", 0)])],
        [(omega / 2, [("Z", 0)])], np.array([1.0, 0.0]),
        analytic_times, 1,
    )
    analytic_error = float(np.max(np.abs(corr - np.exp(1j * omega * analytic_times))))
    trotter = trotter_convergence()
    kicks = kick_convergence()
    windows = window_convergence()
    exact, shots = shot_intervals()

    print(f"analytic correlation max error: {analytic_error:.3e}")
    for name, report in (("Trotter", trotter), ("kick", kicks), ("window", windows)):
        print(f"\n{name} convergence")
        for parameter, error in zip(report["parameters"], report["errors"]):
            print(f"  parameter={parameter:8.4f}  error={error:.6e}")
        print(f"  monotonic={report['monotonically_decreasing']}")
    print(f"\nshot statistics (exact={exact:.8f})")
    for row in shots:
        print(
            f"  shots={row['shots']:5d} estimate={row['estimate']: .6f} "
            f"sigma={row['standard_error']:.6f} CI95={row['ci95']}"
        )
    return comparison_report(
        name="q8_solid_mvp",
        curves={"analytic_correlation_real": corr.real},
        metadata={
            "analytic_times": analytic_times,
            "trotter_convergence": trotter,
            "analytic_max_error": analytic_error,
            "kick_convergence": kicks,
            "window_convergence": windows,
            "shot_exact": exact,
            "shot_statistics": shots,
            "note": "有限 shots 的随机误差不要求逐点单调；报告标准误和置信区间。",
        },
    )


if __name__ == "__main__":
    main()
