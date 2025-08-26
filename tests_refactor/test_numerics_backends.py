import importlib

import numpy as np
import pytest

from tyxonq.config import normalize_backend_name
from tyxonq.numerics import get_backend


def test_factory_returns_numpy_backend_and_basic_ops():
    be = get_backend(normalize_backend_name("numpy(cpu)"))
    assert be.name == "numpy"
    a = be.array([[1.0, 2.0], [3.0, 4.0]])
    b = be.array([[1.0], [0.5]])
    c = be.matmul(a, b)
    assert be.to_numpy(c).shape == (2, 1)
    d = be.einsum("ij,jk->ik", be.array([[1, 0], [0, 1]]), be.array([[2], [3]]))
    assert be.to_numpy(d).tolist() == [[2], [3]]
    rng = be.rng(0)
    x = be.normal(rng, (2,))
    assert len(be.to_numpy(x)) == 2
    y = be.detach(a)
    assert isinstance(be.to_numpy(y), np.ndarray)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_factory_returns_pytorch_backend():
    be = get_backend("pytorch")
    assert be.name == "pytorch"
    # vectorize_or_fallback is tested elsewhere; here we just smoke-test array
    t = be.asarray([1, 2, 3]) if hasattr(be, "asarray") else be.array([1, 2, 3])
    assert be.to_numpy(t).shape == (3,)


@pytest.mark.skipif(importlib.util.find_spec("cupynumeric") is None, reason="cupynumeric not installed")
def test_factory_returns_cunumeric_backend():
    be = get_backend("cupynumeric")
    assert be.name == "cupynumeric"

