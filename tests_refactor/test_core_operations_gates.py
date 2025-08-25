from __future__ import annotations

import numpy as np

from tyxonq.core.operations import registry
from tyxonq.core.operations import get_unitary


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

