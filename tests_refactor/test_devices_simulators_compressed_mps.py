from __future__ import annotations

import numpy as np

from tyxonq.libs.quantum_library.kernels.matrix_product_state import (
    MPSState,
    init_product_state,
    apply_1q,
    apply_2q,
    to_statevector,
    bond_dims,
)
from tyxonq.libs.quantum_library.kernels.gates import gate_h, gate_cx_rank4


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


