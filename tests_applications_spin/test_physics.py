"""独立矩阵、精确演化和实验可观测量约定的验证。"""

import numpy as np
import pytest
from scipy.linalg import expm

from tyxonq.core.ir import Circuit
from tyxonq.applications.spin import SpinChainModel, build_evolution
from tyxonq.applications.spin.circuits import cartan_ops, fold_local_rotations
from tyxonq.applications.spin.models import PAULIS, site_operator
from tyxonq.applications.spin.simulation import pauli_expectations, statevector
from tyxonq.applications.spin.spectra import compute_dsf, response_from_kicks, spectrum_for_plot
from tyxonq.applications.spin.workflow import exact_response
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine


def matrix_of(circuit):
    """按计算基逐列执行线路，得到独立于 Hamiltonian 构造的线路矩阵。"""
    engine = StatevectorEngine(backend_name="numpy")
    basis = np.eye(2**circuit.num_qubits, dtype=complex)
    return np.column_stack([engine.state(Circuit(circuit.num_qubits, ops=circuit.ops, inputs=column))
                            for column in basis.T])


def aligned_error(actual, expected):
    phase = np.vdot(expected, actual)
    phase = phase / abs(phase)
    return np.max(np.abs(actual - phase * expected))


@pytest.mark.parametrize("abc", [(0, 0, 0), (.21, -.37, .19), (-.9, .31, 1.1), (.4, .4, 0)])
def test_cartan_matrix(abc):
    h = sum(angle * np.kron(PAULIS[axis], PAULIS[axis]) for angle, axis in zip(abc, "XYZ"))
    assert aligned_error(matrix_of(Circuit(2, ops=cartan_ops(0, 1, *abc))), expm(1j * h)) < 1e-10


@pytest.mark.parametrize("jxy,jz", [(1, 0), (1, 1), (.5, 1), (-.7, .3)])
def test_model_and_pair_time_normalization(jxy, jz):
    model = SpinChainModel(2, jxy, jz)
    expected = (jxy * (np.kron(PAULIS["X"], PAULIS["X"]) + np.kron(PAULIS["Y"], PAULIS["Y"]))
                + jz * np.kron(PAULIS["Z"], PAULIS["Z"])) / 4
    np.testing.assert_allclose(model.sparse_matrix().toarray(), expected, atol=1e-14)
    for order in (1, 2):
        circuit = build_evolution(model, .13, 3, order)
        assert aligned_error(matrix_of(circuit), expm(-1j * .39 * expected)) < 1e-10
        assert all(op[0] != "measure_z" for op in circuit.ops)


def test_trotter_second_order_convergence():
    model = SpinChainModel(4, .8, 1.2)
    exact = expm(-.8j * model.sparse_matrix().toarray())
    errors = []
    for steps in (4, 8, 16):
        actual = matrix_of(build_evolution(model, .8 / steps, steps))
        overlap = np.vdot(exact, actual)
        errors.append(np.linalg.norm(actual - exact * overlap / abs(overlap)))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


def test_q0_is_leftmost_and_zero_step_is_identity():
    np.testing.assert_allclose(statevector(Circuit(2, ops=[("x", 0)])), [0, 0, 1, 0])
    assert build_evolution(SpinChainModel(4), .2, 0).ops == []


def test_rotation_folding_preserves_matrix_and_dependencies():
    # 包含跨不相交门的合并、精确抵消，以及禁止跨过同一线 CNOT 的情况。
    ops = [("s", 0), ("x", 1), ("rz", 0, .2), ("rx", 0, .3), ("rx", 0, -.3),
           ("sdg", 0), ("cx", 0, 1), ("rz", 0, .7), ("rz", 1, .4), ("rz", 0, -.2)]
    folded = fold_local_rotations(2, ops)
    assert len(folded) < len(ops)
    assert aligned_error(matrix_of(Circuit(2, ops=folded)), matrix_of(Circuit(2, ops=ops))) < 1e-12
    assert len([op for op in folded if op[0] == "cx"]) == 1


@pytest.mark.parametrize("alpha,beta", [("Z", "Z"), ("X", "X"), ("Y", "Y"), ("X", "Y")])
def test_kick_difference_matches_direct_commutator(alpha, beta):
    # 使用不具备论文对称性的随机纯态，同时覆盖同向和交叉方向。
    model = SpinChainModel(4, .6, 1)
    rng = np.random.default_rng(7)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    times = np.array([0, .13, .26])
    measured = exact_response(model, psi, times, 1, beta, alpha)
    b = site_operator(4, 1, beta).toarray() / 2
    for k, time in enumerate(times):
        u = expm(-1j * time * model.sparse_matrix().toarray())
        for site in range(4):
            a = u.conj().T @ (site_operator(4, site, alpha).toarray() / 2) @ u
            expected = -1j * np.vdot(psi, (a @ b - b @ a) @ psi)
            np.testing.assert_allclose(measured[k, site], expected, atol=1e-12)


def test_dsf_quadrature_axes_and_plot_copy():
    times = np.arange(4) * .2
    green = np.array([[123, -999], [.1, -.2], [.3, .4], [-.2, .1]])
    q, w, spectrum = compute_dsf(green, times, 0, n_momentum=8, n_frequency=7)
    expected = np.zeros((7, 8))
    for iw, omega in enumerate(w):
        for iq, momentum in enumerate(q):
            value = sum(.2 * np.exp(1j * omega * times[k] - 1j * momentum * j) * green[k, j]
                        for k in range(1, 4) for j in range(2))
            expected[iw, iq] = -value.imag / np.pi
    np.testing.assert_allclose(spectrum, expected, atol=1e-14)
    assert q[-1] < 2 * np.pi and w[-1] < np.pi / .2
    original = spectrum.copy()
    mirrored = spectrum_for_plot(spectrum, mirror=True)
    np.testing.assert_allclose(mirrored[:, 1], (original[:, 1] + original[:, 7]) / 2)
    np.testing.assert_array_equal(spectrum, original)
    np.testing.assert_array_equal(spectrum_for_plot(np.zeros((2, 3)), normalize=True), 0)
    assert np.min(spectrum_for_plot(spectrum, clip_negative=True)) >= 0
    np.testing.assert_allclose(response_from_kicks(4 * green, np.zeros_like(green)), green)


@pytest.mark.parametrize("alpha,beta", [("Z", "Z"), ("X", "Y"), ("Y", "X")])
def test_native_kicks_match_commutator_and_exact_spectrum(alpha, beta):
    # 两位链的偶键演化与精确指数相等，因此同时检验门符号、响应和谱。
    model = SpinChainModel(2, .7, 1)
    rng = np.random.default_rng(42)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    times = np.arange(5) * .13
    observations = []
    for sign in (1, -1):
        values = []
        for step in range(len(times)):
            ops = [("r" + beta.lower(), 0, sign * np.pi / 2)] + build_evolution(model, .13, step).ops
            values.append(pauli_expectations(statevector(Circuit(2, ops=ops, inputs=psi)), 2, alpha))
        observations.append(np.array(values))
    measured = response_from_kicks(*observations)
    reference = exact_response(model, psi, times, 0, beta, alpha)
    np.testing.assert_allclose(measured, reference, atol=1e-12)
    np.testing.assert_allclose(compute_dsf(measured, times, 0)[2],
                               compute_dsf(reference, times, 0)[2], atol=1e-12)


def test_invalid_input_rejected():
    for n in (0, 3, 4.0, True):
        with pytest.raises(ValueError):
            SpinChainModel(n)
    with pytest.raises(ValueError):
        build_evolution(SpinChainModel(4), .2, -1)
    with pytest.raises(ValueError):
        compute_dsf(np.zeros((3, 4)), [0, .2, .5], 1)
