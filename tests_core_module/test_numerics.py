# numerics 测试合并：ArrayBackend/vectorize_or_fallback 的向量化策略、后端工厂
# （numpy/pytorch/cupynumeric）、vectorization_checks 安全性检测，以及全局后端的
# set_backend/use_backend 切换与恢复。
from __future__ import annotations

import importlib
import warnings
from typing import Any

import numpy as np
import pytest

from tyxonq.config import normalize_backend_name
from tyxonq.numerics import (
    ArrayBackend,
    get_backend,
    set_backend,
    use_backend,
    vectorize_or_fallback,
)
from tyxonq.numerics.vectorization_checks import safe_for_vectorization, warn_as_error


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


def _unsafe_fn(x):
    # Simulate an in-place like pattern by emitting a warning
    warnings.warn("AliasWarning: potential alias detected", category=UserWarning)
    return x


def test_vectorize_or_fallback_off_policy():
    backend: ArrayBackend = NumpyBackend()  # type: ignore[assignment]

    def add_one(x):
        return x + 1

    wrapped = vectorize_or_fallback(add_one, backend, policy="off")
    assert wrapped(2) == 3


def test_vectorize_or_fallback_generic_vectorization():
    backend: ArrayBackend = NumpyBackend()  # type: ignore[assignment]

    def square(x):
        return x * x

    wrapped = vectorize_or_fallback(square, backend, policy="auto")
    out = wrapped([1, 2, 3])
    assert out == [1, 4, 9]


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


def test_safe_for_vectorization_simple():
    assert safe_for_vectorization(lambda x: x, args=(1,), kwargs={}) is True
    assert safe_for_vectorization(_unsafe_fn, args=(1,), kwargs={}) is False


def test_warn_as_error_context():
    with warn_as_error(["AliasWarning"]) as caught:
        try:
            warnings.warn("AliasWarning: test", category=UserWarning)
        except Warning:
            pass
    assert caught["raised"] is True


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
