from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tyxonq.numerics.backends.base import BackendUnavailableError
from tyxonq.numerics.backends.metal_backend import MetalBackend


pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU is unavailable")


def test_metal_array_algebra_and_materialization():
    backend = MetalBackend()
    a = backend.array([[1.0, 2.0], [3.0, 4.0]], dtype=backend.float32)
    result = backend.matmul(a, a)
    assert result.dtype == mx.float32
    np.testing.assert_allclose(backend.to_numpy(result), [[7.0, 10.0], [15.0, 22.0]])
    assert backend.scalar(backend.sum(result)) == 54.0
    assert backend.shape(result) == (2, 2)


def test_metal_rejects_fp64_before_execution():
    backend = MetalBackend()
    with pytest.raises(BackendUnavailableError, match="does not support FP64"):
        backend.array([1.0], dtype=np.float64)
    with pytest.raises(BackendUnavailableError, match="does not support FP64"):
        backend.zeros((2,), dtype=backend.float64)


def test_mlx_cpu_supports_fp64_golden_operations():
    backend = MetalBackend("cpu")
    a = backend.array([[2.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    values, vectors = backend.eigh(a)
    np.testing.assert_allclose(backend.to_numpy(values), [2.0, 4.0])
    np.testing.assert_allclose(backend.to_numpy(vectors), np.eye(2))


def test_metal_random_reduction_and_autodiff_bridge():
    backend = MetalBackend()
    samples = backend.normal(backend.rng(7), (32,), dtype=backend.float32)
    assert samples.shape == (32,)
    counts = backend.bincount(backend.array([0, 1, 1], dtype=mx.int32), minlength=3)
    np.testing.assert_array_equal(backend.to_numpy(counts), [1, 2, 0])

    def loss(x):
        return mx.sum(x * x)

    value, gradient = backend.value_and_grad(loss)(backend.array([2.0, 3.0]))
    assert backend.scalar(value) == pytest.approx(13.0)
    np.testing.assert_allclose(backend.to_numpy(gradient), [4.0, 6.0])


def test_unsupported_nonzero_does_not_fall_back_to_host():
    backend = MetalBackend()
    with pytest.raises(NotImplementedError, match="no data-dependent nonzero"):
        backend.nonzero(backend.array([0, 1]))
