"""D1：dynamics 直观算法模块（observables/exact/circuits/spectroscopy/pulse_dynamics）测试。

验证门（QD_RESEARCH 对应）：

- QD-RB2：数值/线路交叉验证（Trotter Trotter 化 vs expm 精确解）；
- QD-RB3：bit-order 单一约定（x@q0 → ⟨Z0⟩=-1；数值 LSB 与 counts 聚合一致）；
- 谱学：关联函数 → 谱峰位 vs 已知频率（二能级能隙）；
- 惰性导出：``dynamics.__getattr__`` PEP 562 生效；
- 既有 ``DynamicsNumericRuntime`` 导出零回归。

设备档一律 provider="simulator"（真机层另有 opt-in 测试）。
单位：原子单位 ħ=1。
"""

from __future__ import annotations

import numpy as np
import pytest

from tyxonq.applications.chem.dynamics import DynamicsNumericRuntime  # 零回归门
from tyxonq.applications.chem.dynamics.observables import (
    bitstring_to_state,
    build_pauli_matrix,
    pauli_expectation,
    site_populations,
    state_to_bitstring,
)
from tyxonq.applications.chem.dynamics.exact import (
    correlation_exact,
    evolve_exact,
    expectation_trajectory,
)
from tyxonq.applications.chem.dynamics.circuits import (
    build_trotter_circuit,
    measure_pauli_expectation,
    trotter_expectation_trajectory,
    trotter_ops,
)
from tyxonq.applications.chem.dynamics.spectroscopy import (
    apply_window,
    correlation_to_spectrum,
    error_metrics,
)


# ------------------------------------------------------------------ #
# observables（QD-RB3：LSB 约定 + 矩阵正确性）
# ------------------------------------------------------------------ #
def test_build_pauli_matrix_lsb_order():
    """X0（qubit 0）= kron(I, X)（q0 为最右因子/LSB）。"""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    mat = build_pauli_matrix([(1.0, [("X", 0)])], 2)
    assert np.allclose(mat, np.kron(I, X))
    mat1 = build_pauli_matrix([(1.0, [("X", 1)])], 2)
    assert np.allclose(mat1, np.kron(X, I))


def test_bitstring_roundtrip_lsb():
    """位串右端 = qubit 0（"10" → q0=0,q1=1）。"""
    psi = bitstring_to_state("10", 2)
    assert abs(psi[0b10]) == 1.0
    assert state_to_bitstring(psi, 2) == "10"


def test_site_populations_and_expectation():
    psi = (bitstring_to_state("10", 2) + bitstring_to_state("01", 2)) / np.sqrt(2)
    pops = site_populations(psi, 2)
    assert np.allclose(pops, 0.5)  # 每个 site 各 0.5
    val = pauli_expectation(psi, [(1.0, [("Z", 0)])], 2)
    assert abs(val) < 1e-12  # 均衡叠加 → ⟨Z0⟩=0


# ------------------------------------------------------------------ #
# exact（谱投影演化 vs expm；能量守恒）
# ------------------------------------------------------------------ #
def test_evolve_exact_matches_expm():
    from scipy.linalg import expm

    ham = [(0.5, [("Z", 0), ("Z", 1)]), (0.3, [("X", 0)]), (0.2, [("Y", 1)])]
    psi0 = bitstring_to_state("01", 2)
    times = [0.0, 0.3, 0.7, 1.1]
    states = evolve_exact(ham, psi0, times, 2)
    H = build_pauli_matrix(ham, 2)
    for i, t in enumerate(times):
        ref = expm(-1j * t * H) @ psi0
        assert np.allclose(states[i], ref, atol=1e-12)


def test_energy_conservation_and_correlation_at_origin():
    ham = [(0.4, [("X", 0), ("X", 1)]), (0.6, [("Z", 0)])]
    psi0 = bitstring_to_state("10", 2)
    times = np.linspace(0.0, 2.0, 41)
    traj = expectation_trajectory(ham, ham, psi0, times, 2)
    assert np.allclose(traj, traj[0], atol=1e-10), "H 期望应守恒（能量）"
    A = [(1.0, [("X", 0)])]
    C = correlation_exact(A, A, ham, psi0, times, 2)
    assert abs(C[0].imag) < 1e-12  # C(0)=⟨A²⟩ 实数


# ------------------------------------------------------------------ #
# circuits（QD-RB2/RB3：Trotter vs 数值交叉）
# ------------------------------------------------------------------ #
def test_trotter_vs_exact_single_qubit():
    """H=0.7 Z0，|+> 初态：⟨X(t)⟩=cos(1.4 t)；线路档对齐数值（采样噪声内）。"""
    from tyxonq.core.ir.circuit import Circuit

    ham = [(0.7, [("Z", 0)])]
    x_obs = [(1.0, [("X", 0)])]
    psi0 = np.zeros(4, dtype=complex)
    psi0[0] = psi0[1] = 1 / np.sqrt(2)
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    num = expectation_trajectory(ham, x_obs, psi0, times, 2)
    assert np.allclose(num, np.cos(1.4 * np.asarray(times)), atol=1e-9)
    dt = 0.05
    circ_sim = []
    for t in times:
        circ = Circuit(2)
        circ.ops.append(("h", 0))
        for _ in range(int(round(t / dt))):
            circ.ops.extend(trotter_ops(ham, dt))
        circ_sim.append(measure_pauli_expectation(circ, x_obs, 2, shots=65536))
    # 容差 = 采样噪声（σ≈1/√65536≈0.004；含 3σ 余量）
    assert np.allclose(circ_sim, num, atol=2e-2)


def test_trotter_vs_exact_two_qubit_coupled():
    """H=0.5 Z0Z1+0.3 X0+0.2 X1：整段 Trotter 线路 vs expm 轨迹（含初态制备）。"""
    ham = [(0.5, [("Z", 0), ("Z", 1)]), (0.3, [("X", 0)]), (0.2, [("X", 1)])]
    obs = [(1.0, [("Z", 0), ("Z", 1)])]
    psi0 = bitstring_to_state("01", 2)  # q0=1
    times = [0.0, 0.4, 0.8, 1.2]
    num = expectation_trajectory(ham, obs, psi0, times, 2)
    traj = trotter_expectation_trajectory(
        ham, obs, times, 2, dt=0.01, init_bitstring="01", shots=16384
    )
    # 容差 = 采样噪声（σ≈1/√16384≈0.008）+ 一阶 Trotter 离散误差
    assert np.allclose(traj, num, atol=2e-2)


def test_first_order_trotter_error_scales_with_dt():
    """一阶 Trotter 细化应降低误差；shots=0 排除采样涨落干扰。"""
    ham = [(0.5, [("Z", 0), ("Z", 1)]), (0.3, [("X", 0)]), (0.2, [("X", 1)])]
    obs = [(1.0, [("Z", 0), ("Z", 1)])]
    psi0 = bitstring_to_state("01", 2)
    times = np.array([0.0, 0.4, 0.8, 1.2])
    reference = expectation_trajectory(ham, obs, psi0, times, 2)
    errors = []
    for dt in (0.2, 0.1, 0.05):
        actual = trotter_expectation_trajectory(
            ham, obs, times, 2, dt=dt, init_bitstring="01", shots=0
        )
        errors.append(float(np.max(np.abs(actual - reference))))
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.05)


def test_bitorder_gate_z0_flips_on_x_at_q0():
    """QD-RB3 门：x@q0 → ⟨Z0⟩=-1、⟨Z1⟩=+1（counts 聚合与数值约定一致）。"""
    from tyxonq.core.ir.circuit import Circuit

    circ = Circuit(2)
    circ.ops.append(("x", 0))
    z0 = measure_pauli_expectation(circ, [(1.0, [("Z", 0)])], 2, shots=4096)
    z1 = measure_pauli_expectation(circ, [(1.0, [("Z", 1)])], 2, shots=4096)
    assert z0 == pytest.approx(-1.0, abs=1e-9)
    assert z1 == pytest.approx(1.0, abs=1e-9)


def test_build_trotter_circuit_init_bitstring_validation():
    ham = [(0.5, [("Z", 0)])]
    with pytest.raises(ValueError):
        build_trotter_circuit(ham, 2, 0.1, 3, init_bitstring="010")  # 长度不符
    with pytest.raises(ValueError):
        build_trotter_circuit(ham, 2, 0.1, 3, init_bitstring="0a")  # 非二进制
    circ = build_trotter_circuit(ham, 2, 0.1, 0, init_bitstring="10")
    assert any(op[0] == "x" for op in circ.ops)


# ------------------------------------------------------------------ #
# spectroscopy（谱峰位 vs 已知频率）
# ------------------------------------------------------------------ #
def test_correlation_to_spectrum_peak_position():
    """二能级能隙 0.7：⟨X0(t)⟩~cos(0.7t) 的关联谱峰位 ≈ 0.7。"""
    ham = [(0.35, [("Z", 0)])]
    A = [(1.0, [("X", 0)])]
    psi0 = np.zeros(4, dtype=complex)
    psi0[0] = psi0[1] = 1 / np.sqrt(2)
    times = np.linspace(0.0, 60.0, 1201)
    C = correlation_exact(A, A, ham, psi0, times, 2).real
    freqs, spec = correlation_to_spectrum(C, times, window="hann", n_frequency=4001)
    peak = float(freqs[int(np.argmax(spec))])
    # 容差 = 频率网格分辨率 + Hann 主瓣宽（2/T≈0.033）
    assert abs(peak - 0.7) < 0.05, f"谱峰 {peak} 应在能隙 0.7 附近"


def test_apply_window_variants():
    times = np.linspace(0.0, 1.0, 101)
    corr = np.cos(4.0 * times)
    for w in ("none", "hann", "gaussian"):
        out = apply_window(corr, times, window=w)
        assert out.shape == corr.shape
    with pytest.raises(ValueError):
        apply_window(corr, times, window="bogus")


def test_error_metrics_keys():
    m = error_metrics(np.zeros(3), np.ones(3))
    assert set(m) == {"max_abs", "rmse", "relative_l2"}
    assert m["max_abs"] == 1.0


# ------------------------------------------------------------------ #
# pulse_dynamics（数值档链 + 编排层守卫）
# ------------------------------------------------------------------ #
def test_pulse_dynamics_numeric_and_guards():
    """数值档走 prog.state；非 PulseProgram/非法 provider 被拒绝。"""
    from tyxonq.applications.chem.dynamics import evolve_with_pulses
    from tyxonq.core.ir.pulse import PulseProgram

    prog = PulseProgram(1)
    prog.gaussian(0, amp=0.3, duration=100, sigma=25.0)
    out = evolve_with_pulses(prog)  # 数值档默认
    assert out["schema"] == "tyxonq.qd.pulse.v1"
    assert out["mode"] == "numeric"
    assert np.asarray(out["state"]).size >= 2
    with pytest.raises(TypeError):
        evolve_with_pulses("not-a-program")
    with pytest.raises(ValueError):
        evolve_with_pulses(prog, provider="bogus-vendor")


def test_dynamics_lazy_exports():
    """PEP 562 惰性导出生效 + 未知属性 AttributeError。"""
    import tyxonq.applications.chem.dynamics as dyn

    for name in ("evolve_exact", "correlation_to_spectrum", "evolve_with_pulses",
                 "site_populations", "build_trotter_circuit"):
        assert callable(getattr(dyn, name))
    with pytest.raises(AttributeError):
        getattr(dyn, "no_such_symbol")
