from __future__ import annotations

import numpy as np

from tyxonq.numerics import get_backend
from tyxonq.devices.simulators.gates import (
    init_statevector,
    gate_h,
    gate_cx_rank4,
    apply_1q_statevector,
    apply_2q_statevector,
    expect_z_statevector,
)


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


