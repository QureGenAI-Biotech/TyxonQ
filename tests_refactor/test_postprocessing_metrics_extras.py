from __future__ import annotations

import numpy as np

from tyxonq.postprocessing.metrics import (
    taylorlnm,
    truncated_free_energy,
    reduced_wavefunction,
)


def test_taylorlnm_small_matrix_and_truncated_free_energy():
    # rho diagonal pure |0>
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    h = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    # truncated free energy should be close to energy term for small k
    F = truncated_free_energy(rho, h, beta=1.0, k=2)
    # Energy is -1.0, truncated entropy approx >= 0 so F <= -1.0 approximately
    assert F <= -0.999999 + 1e-6


def test_reduced_wavefunction_simple_projection():
    # |psi> = |00> + |11> normalized
    psi = (1.0 / np.sqrt(2.0)) * np.array([1, 0, 0, 1], dtype=np.complex128)
    # project qubit 1 (LSB) to 0, remaining ket should be |0>
    out = reduced_wavefunction(psi, cut=[1], measure=[0])
    # remaining qubit is MSB; amplitude should be [1, 0] up to norm
    out = out / np.linalg.norm(out)
    np.testing.assert_allclose(out, np.array([1.0, 0.0], dtype=np.complex128))


