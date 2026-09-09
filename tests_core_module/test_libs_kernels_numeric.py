import math
import numpy as np

from tyxonq.numerics.api import get_backend
from tyxonq.libs.quantum_library.kernels.statevector import (
    init_statevector,
    apply_1q_statevector,
    apply_2q_statevector,
    expect_z_statevector,
)
from tyxonq.libs.quantum_library.kernels.gates import gate_h, gate_rz, gate_cx_4x4
from tyxonq.libs.quantum_library.kernels.density_matrix import (
    init_density,
    apply_1q_density,
    apply_2q_density,
    exp_z_density,
)
from tyxonq.libs.quantum_library.kernels.unitary import get_unitary
from tyxonq.libs.quantum_library.kernels.pauli import (
    ps2xyz,
    xyz2ps,
    pauli_string_to_matrix,
    pauli_string_sum_dense,
    heisenberg_hamiltonian,
)


def _all_backends():
    for name in ("numpy", "pytorch"):
        try:
            yield name, get_backend(name)
        except Exception:
            continue


def test_statevector_kernels_basic():
    for name, K in _all_backends():
        psi = init_statevector(2, backend=K)
        psi = apply_1q_statevector(K, psi, gate_h(), 0, 2)
        psi = apply_2q_statevector(K, psi, gate_cx_4x4(), 0, 1, 2)
        e0 = expect_z_statevector(psi, 0, 2, backend=K)
        e1 = expect_z_statevector(psi, 1, 2, backend=K)
        assert abs(e0) < 1e-9
        assert abs(e1 - 0.0) < 1e-9


def test_density_matrix_kernels_basic():
    for name, K in _all_backends():
        rho = init_density(1, backend=K)
        rho = apply_1q_density(K, rho, gate_rz(0.0), 0, 1)
        e = exp_z_density(K, rho, 0, 1)
        assert abs(e - 1.0) < 1e-9


# 以下用例迁自已删除的 test_core_operations.py（core.operations/core.measurements 属
# 死脚手架）；仅保留真正有物理价值、且依赖存活模块（kernels.unitary / kernels.pauli）
# 的断言：单比特/双比特门矩阵定义与 Pauli 字符串核函数。
def test_h_unitary_matches_definition():
    u = get_unitary("h")
    expected = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    np.testing.assert_allclose(u, expected, rtol=0, atol=1e-12)


def test_rz_unitary_matches_definition():
    theta = np.pi
    u = get_unitary("rz", theta)
    # RZ(theta) = exp(-i theta/2 Z) = diag(e^{-i theta/2}, e^{i theta/2})
    expected = np.diag([np.exp(-1j * theta / 2.0), np.exp(1j * theta / 2.0)])
    np.testing.assert_allclose(u, expected, rtol=0, atol=1e-12)


def test_cx_unitary_shape_and_action():
    u = get_unitary("cx")
    assert u.shape == (4, 4)
    # |10> -> |11>, |11> -> |10>, others unchanged
    basis = np.eye(4, dtype=np.complex128)
    out_10 = u @ basis[:, 2]
    out_11 = u @ basis[:, 3]
    np.testing.assert_allclose(out_10, basis[:, 3])
    np.testing.assert_allclose(out_11, basis[:, 2])


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


