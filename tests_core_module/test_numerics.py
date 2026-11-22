# numerics 测试：ArrayBackend 工厂（numpy/pytorch/cupy）与全局后端切换。
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from tyxonq.config import normalize_backend_name
from tyxonq.numerics import get_backend, set_backend, use_backend


class NumpyBackend:
    name = "numpy"

    def array(self, data: Any, dtype: Any | None = None) -> Any:
        return np.array(data, dtype=dtype)

    def asarray(self, data: Any) -> Any:
        return np.asarray(data)

    def to_numpy(self, data: Any) -> np.ndarray:  # type: ignore[override]
        return np.asarray(data)

    def matmul(self, a: Any, b: Any) -> Any:
        return a @ b

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return np.einsum(subscripts, *operands)

    def rng(self, seed: int | None = None) -> Any:
        return np.random.default_rng(seed)

    def normal(self, rng: Any, shape, dtype: Any | None = None) -> Any:
        out = rng.normal(size=shape)
        return out.astype(dtype) if dtype is not None else out

    def requires_grad(self, x: Any, flag: bool = True) -> Any:
        return x

    def detach(self, x: Any) -> Any:
        return np.asarray(x)


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
    t = be.asarray([1, 2, 3]) if hasattr(be, "asarray") else be.array([1, 2, 3])
    assert be.to_numpy(t).shape == (3,)


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="cupy not installed")
def test_factory_returns_cupy_backend():
    be = get_backend("cupy")
    assert be.name == "cupy"


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown numerics backend"):
        get_backend("cupynumeric")


def test_set_backend_by_name_affects_get_backend_none():
    set_backend("numpy")
    b = get_backend(None)
    assert getattr(b, "name", "") == "numpy"


def test_use_backend_context_manager_restores_previous():
    set_backend("numpy")
    b0 = get_backend(None)
    assert getattr(b0, "name", "") == "numpy"
    with use_backend("numpy"):
        b1 = get_backend(None)
        assert getattr(b1, "name", "") == "numpy"
    b2 = get_backend(None)
    assert getattr(b2, "name", "") == "numpy"
