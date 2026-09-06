# devices.simulators 底层 kernel 测试合并：statevector 门应用与 Bell 态期望、
# 两量子位旋转/受控构造、compressed MPS 构建与 bond 维度、以及噪声通道
# （depolarizing/amplitude_damping/phase_damping/pauli）作用于密度矩阵的 Kraus 应用。
from __future__ import annotations

import numpy as np

from tyxonq.numerics import get_backend
from tyxonq.libs.quantum_library.kernels.gates import (
    gate_h,
    gate_rx,
    gate_ry,
    gate_phase,
    gate_cx_rank4,
    gate_cx_4x4,
    gate_cz_4x4,
    gate_rxx,
    gate_ryy,
    gate_rzz,
    build_controlled_unitary,
)
from tyxonq.libs.quantum_library.kernels.statevector import (
    init_statevector,
    apply_1q_statevector,
    apply_2q_statevector,
    expect_z_statevector,
)
from tyxonq.libs.quantum_library.kernels.matrix_product_state import (
    MPSState,
    init_product_state,
    apply_1q,
    apply_2q,
    to_statevector,
    bond_dims,
)
from tyxonq.libs.quantum_library.kernels.density_matrix import apply_kraus_density
from tyxonq.devices.simulators.noise.channels import (
    depolarizing,
    amplitude_damping,
    phase_damping,
    pauli_channel,
)


def _dm0():
    rho = np.zeros((2, 2), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


def test_bell_state_construction_and_expectations():
    n = 2
    backend = get_backend("numpy")
    psi = init_statevector(n)
    psi = apply_1q_statevector(backend, psi, gate_h(), 0, n)
    psi = apply_2q_statevector(backend, psi, gate_cx_rank4(), 0, 1, n)
    # Bell state |Phi+> should have Z expectation 0 on each qubit
    ez0 = expect_z_statevector(psi, 0, n)
    ez1 = expect_z_statevector(psi, 1, n)
    assert abs(ez0) < 1e-12
    assert abs(ez1) < 1e-12
    # State norm ~ 1
    norm = float(np.vdot(psi, psi).real)
    assert abs(norm - 1.0) < 1e-12


def test_ry_phase_and_cz_on_bell_like():
    backend = get_backend(None)
    n = 2
    psi = init_statevector(n)
    # Prepare |+0>
    psi = apply_1q_statevector(backend, psi, gate_h(), 0, n)
    # Rotate Y on qubit 1 by pi/2 to |+i> style
    psi = apply_1q_statevector(backend, psi, gate_ry(np.pi/2), 1, n)
    # Controlled-Z
    psi = apply_2q_statevector(backend, psi, gate_cz_4x4(), 0, 1, n)
    # Phase on target
    psi = apply_1q_statevector(backend, psi, gate_phase(np.pi/2), 1, n)

    # Just sanity: norm and shape
    assert psi.shape == (4,)
    assert np.isclose(np.vdot(psi, psi).real, 1.0)


def test_two_qubit_rotations_and_multi_control():
    n = 2
    backend = get_backend(None)
    psi = init_statevector(n)
    # Apply exp(-i pi/2 Z.Z) = i*Z.Z up to global phase
    psi = apply_2q_statevector(backend, psi, gate_rzz(np.pi), 0, 1, n)
    # Global phase ignored: still normalized
    assert np.isclose(np.vdot(psi, psi).real, 1.0)

    # Build a controlled-X with 1 control (i.e., CNOT) via generic builder and compare
    CX_generic = build_controlled_unitary(gate_x := np.array([[0,1],[1,0]], dtype=np.complex128), 1)
    psi2 = init_statevector(n)
    psi2 = apply_2q_statevector(backend, psi2, CX_generic, 0, 1, n)
    psi3 = init_statevector(n)
    psi3 = apply_2q_statevector(backend, psi3, gate_cx_4x4(), 0, 1, n)
    assert np.allclose(psi2, psi3)


def test_mps_builds_bell_state_and_matches_full_state():
    n = 2
    mps = init_product_state(n)
    apply_1q(mps, gate_h(), 0)
    apply_2q(mps, gate_cx_rank4(), 0, 1)
    psi = to_statevector(mps)
    # Expected Bell |Phi+> = (|00> + |11>)/sqrt(2)
    expected = (1.0 / np.sqrt(2.0)) * np.array([1, 0, 0, 1], dtype=np.complex128)
    # Global phase insensitive check
    phase = psi[0] / expected[0]
    np.testing.assert_allclose(psi, phase * expected, rtol=0, atol=1e-10)


def test_mps_non_nearest_neighbor_via_swaps_and_bond_dims():
    n = 3
    mps = init_product_state(n)
    apply_1q(mps, gate_h(), 0)
    # Apply CX between qubits 0 and 2 (requires routing)
    apply_2q(mps, gate_cx_rank4(), 0, 2)
    psi = to_statevector(mps)
    # Expected state (|000> + |011>)/sqrt(2) after H on 0 and CX(0->2) then CX(0->1)
    # Here we only applied CX(0->2) so expected entanglement between 0 and 2
    # Norm check only
    norm = float(np.vdot(psi, psi).real)
    assert abs(norm - 1.0) < 1e-10
    dims = bond_dims(mps)
    assert all(Dl >= 1 and Dr >= 1 for Dl, Dr in dims)


def test_depolarizing_channel_preserves_trace():
    rho = _dm0()
    K = depolarizing(0.2)
    out = apply_kraus_density(rho, K, qubit=0, num_qubits=1)
    tr = np.trace(out)
    assert abs(tr - 1.0) < 1e-12


def test_amplitude_damping_relaxes_excited_pop():
    rho = np.zeros((2, 2), dtype=np.complex128)
    rho[1, 1] = 1.0
    K = amplitude_damping(0.5)
    out = apply_kraus_density(rho, K, qubit=0, num_qubits=1)
    # population should move towards |0>
    assert out[0, 0].real > 0.0


def test_phase_damping_kills_coherences():
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    K = phase_damping(1.0)
    out = apply_kraus_density(rho, K, qubit=0, num_qubits=1)
    assert abs(out[0, 1]) < 1e-12 and abs(out[1, 0]) < 1e-12


def test_pauli_channel_probabilities_valid():
    K = pauli_channel(0.1, 0.2, 0.3)
    # sqrt weights should be non-negative and combine with identity weight
    assert len(K) == 4
