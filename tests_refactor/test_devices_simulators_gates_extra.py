import numpy as np

from tyxonq.libs.quantum_library.kernels.gates import (
    gate_h,
    gate_rx,
    gate_ry,
    gate_phase,
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
)
from tyxonq.numerics.api import get_backend


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


