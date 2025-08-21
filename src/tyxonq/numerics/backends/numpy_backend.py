from __future__ import annotations

from typing import Any, Tuple

import numpy as np


class NumpyBackend:
    name = "numpy"

    def array(self, data: Any, dtype: Any | None = None) -> Any:
        return np.array(data, dtype=dtype)

    def asarray(self, data: Any) -> Any:
        return np.asarray(data)

    def to_numpy(self, data: Any) -> np.ndarray:  # type: ignore[override]
        return np.asarray(data)

    def matmul(self, a: Any, b: Any) -> Any:
        return np.matmul(a, b)

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return np.einsum(subscripts, *operands)

    def rng(self, seed: int | None = None) -> Any:
        return np.random.default_rng(seed)

    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        out = rng.normal(size=shape)
        return out.astype(dtype) if dtype is not None else out

    def requires_grad(self, x: Any, flag: bool = True) -> Any:
        return x

    def detach(self, x: Any) -> Any:
        return np.asarray(x)


