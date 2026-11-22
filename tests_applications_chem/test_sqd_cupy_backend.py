from __future__ import annotations

import importlib.util

import pytest

from tyxonq.numerics import get_backend


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="cupy not installed")
def test_cupy_backend_protocol():
    backend = get_backend("cupy")
    assert backend.name == "cupy"
    array = backend.asarray([1.0, 2.0])
    assert backend.to_numpy(backend.matmul(array, array)).tolist() == [1.0, 4.0]


def test_cupynumeric_is_removed():
    with pytest.raises(ValueError):
        get_backend("cupynumeric")
