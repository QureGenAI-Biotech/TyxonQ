from __future__ import annotations

import numpy as np

from tyxonq.libs.quantum_library.kernels.pauli import (
    ps2xyz,
    xyz2ps,
    pauli_string_to_matrix,
    pauli_string_sum_dense,
    heisenberg_hamiltonian,
)


def test_ps2xyz_and_back_xyz2ps():
    ps = [1, 2, 2, 0, 3]
    xyz = ps2xyz(ps)
    assert set(xyz.keys()) == {"x", "y", "z"}
    ps_back = xyz2ps(xyz, n=len(ps))
    assert ps_back == ps


def test_pauli_string_to_matrix_single_qubit():
    X = pauli_string_to_matrix([1])
    expected = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    np.testing.assert_allclose(X, expected)


def test_pauli_string_sum_dense_zz_on_two_qubits():
    H = pauli_string_sum_dense([[3, 3]])
    expected = np.diag([1, -1, -1, 1]).astype(np.complex128)
    np.testing.assert_allclose(H, expected)


def test_heisenberg_hamiltonian_line2_with_zz_only():
    H = heisenberg_hamiltonian(2, [(0, 1)], hzz=1.0, hxx=0.0, hyy=0.0)
    expected = np.diag([1, -1, -1, 1]).astype(np.complex128)
    np.testing.assert_allclose(H, expected)


