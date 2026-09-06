# postprocessing 指标测试合并：counts 归一化/KL 散度/期望值与 CSV 往返（postprocessing.
# metrics + io），以及量子信息 kernel（quantum_info 的截断自由能与约化波函数投影）。
from __future__ import annotations

import numpy as np

from tyxonq.postprocessing.metrics import normalized_count, kl_divergence, expectation
from tyxonq.postprocessing.io import counts_to_csv, csv_to_counts
from tyxonq.libs.quantum_library.kernels.quantum_info import (
    taylorlnm,
    truncated_free_energy,
    reduced_wavefunction,
)


def test_metrics_and_io_roundtrip():
    counts = {"00": 60, "01": 20, "10": 15, "11": 5}
    norm = normalized_count(counts)
    assert abs(sum(norm.values()) - 1.0) < 1e-12
    kl = kl_divergence(counts, counts)
    assert abs(kl) < 1e-12
    expz0 = expectation(counts, z=[0])
    assert -1.0 <= expz0 <= 1.0

    csv = counts_to_csv(counts)
    rec = csv_to_counts(csv)
    assert rec == counts


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
