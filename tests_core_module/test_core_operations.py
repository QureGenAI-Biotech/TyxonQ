# 核心 operations 测试合并：GateSpec / Operation / registry / get_unitary 的单元矩阵与
# 梯度元数据；同时并入 Pauli kernels（libs.quantum_library.kernels.pauli）与
# measurements 结构（core.measurements）的最小校验。
from __future__ import annotations

import numpy as np

from tyxonq.core.operations import GateSpec, Operation, registry, get_unitary
from tyxonq.core.measurements import Expectation, Probability, Sample
from tyxonq.libs.quantum_library.kernels.pauli import (
    ps2xyz,
    xyz2ps,
    pauli_string_to_matrix,
    pauli_string_sum_dense,
    heisenberg_hamiltonian,
)


def test_gate_spec_and_registry_minimal():
    h = GateSpec(name="h", num_qubits=1, generator=None, differentiable=True)
    registry.register(h)
    assert registry.get("h") is h

    op = Operation(name="h", wires=(0,))
    assert op.name == "h"
    assert op.wires == (0,)


def test_default_gate_registry_contains_h_rz_cx():
    for name in ["h", "rz", "cx"]:
        spec = registry.get(name)
        assert spec is not None
        assert spec.name == name
        assert spec.num_qubits in (1, 2)
    rz = registry.get("rz")
    assert rz is not None and rz.is_shiftable and rz.num_params == 1


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


def test_registry_contains_gradient_metadata_for_rz():
    spec = registry.get("rz")
    assert spec is not None
    assert spec.is_shiftable is True
    assert spec.shift_coeffs == (0.5,)


def test_gate_spec_gradient_metadata_defaults():
    spec = GateSpec(name="custom", num_qubits=1)
    assert spec.num_params == 0
    assert spec.is_shiftable is False
    assert spec.shift_coeffs is None
    assert spec.gradient_method is None


def test_gate_spec_parameter_shift_metadata():
    rz = GateSpec(
        name="rz",
        num_qubits=1,
        num_params=1,
        is_shiftable=True,
        shift_coeffs=(0.5,),  # typical coefficient for parameter-shift of RZ
        gradient_method="parameter-shift",
        generator="Z",
    )
    registry.register(rz)
    got = registry.get("rz")
    assert got is not None
    assert got.is_shiftable is True
    assert got.num_params == 1
    assert got.shift_coeffs == (0.5,)
    assert got.gradient_method == "parameter-shift"


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


def test_measurement_structs_minimal():
    e = Expectation(obs="Z", wires=(0,))
    p = Probability(wires=(0, 1))
    s = Sample(wires=(0,), shots=1000)
    assert e.obs == "Z" and e.wires == (0,)
    assert p.wires == (0, 1)
    assert s.shots == 1000
