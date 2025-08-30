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


