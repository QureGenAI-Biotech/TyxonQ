from typing import Any

import numpy as np

from tyxonq.numerics import ArrayBackend, vectorize_or_fallback


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


