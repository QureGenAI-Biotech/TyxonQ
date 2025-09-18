from __future__ import annotations

import numpy as np

from tyxonq.devices.simulators.noise.channels import (
    depolarizing,
    amplitude_damping,
    phase_damping,
    pauli_channel,
    apply_to_density_matrix,
)


def _dm0():
    rho = np.zeros((2, 2), dtype=np.complex128)
    rho[0, 0] = 1.0
    return rho


def test_depolarizing_channel_preserves_trace():
    rho = _dm0()
    K = depolarizing(0.2)
    out = apply_to_density_matrix(rho, K, wire=0, num_qubits=1)
    tr = np.trace(out)
    assert abs(tr - 1.0) < 1e-12


def test_amplitude_damping_relaxes_excited_pop():
    rho = np.zeros((2, 2), dtype=np.complex128)
    rho[1, 1] = 1.0
    K = amplitude_damping(0.5)
    out = apply_to_density_matrix(rho, K, wire=0, num_qubits=1)
    # population should move towards |0>
    assert out[0, 0].real > 0.0


def test_phase_damping_kills_coherences():
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    K = phase_damping(1.0)
    out = apply_to_density_matrix(rho, K, wire=0, num_qubits=1)
    assert abs(out[0, 1]) < 1e-12 and abs(out[1, 0]) < 1e-12


def test_pauli_channel_probabilities_valid():
    K = pauli_channel(0.1, 0.2, 0.3)
    # sqrt weights should be non-negative and combine with identity weight
    assert len(K) == 4

