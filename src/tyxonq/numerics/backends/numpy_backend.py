from __future__ import annotations

from typing import Any, Tuple

import numpy as np


class NumpyBackend:
    name = "numpy"

    # dtype constants
    import numpy as _np  # local alias to avoid polluting module
    complex64 = _np.complex64
    complex128 = _np.complex128
    float32 = _np.float32
    float64 = _np.float64
    int32 = _np.int32
    int64 = _np.int64
    bool = _np.bool_
    int = _np.int64

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

    # Array ops
    def reshape(self, a: Any, shape: Any) -> Any:
        return np.reshape(a, shape)

    def moveaxis(self, a: Any, source: int, destination: int) -> Any:
        return np.moveaxis(a, source, destination)

    def sum(self, a: Any, axis: int | None = None) -> Any:
        return np.sum(a, axis=axis)

    def mean(self, a: Any, axis: int | None = None) -> Any:
        return np.mean(a, axis=axis)

    def abs(self, a: Any) -> Any:
        return np.abs(a)

    def real(self, a: Any) -> Any:
        return np.real(a)

    def conj(self, a: Any) -> Any:
        return np.conj(a)

    def diag(self, a: Any) -> Any:
        return np.diag(a)

    def zeros(self, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        return np.ones(shape, dtype=dtype)

    def zeros_like(self, a: Any) -> Any:
        return np.zeros_like(a)

    def ones_like(self, a: Any) -> Any:
        return np.ones_like(a)

    def eye(self, n: int, dtype: Any | None = None) -> Any:
        return np.eye(n, dtype=dtype)

    def kron(self, a: Any, b: Any) -> Any:
        return np.kron(a, b)

    def square(self, a: Any) -> Any:
        return np.square(a)

    # Elementary math
    def exp(self, a: Any) -> Any:
        return np.exp(a)

    def sin(self, a: Any) -> Any:
        return np.sin(a)

    def cos(self, a: Any) -> Any:
        return np.cos(a)

    def sqrt(self, a: Any) -> Any:
        return np.sqrt(a)

    def log(self, a: Any) -> Any:
        return np.log(a)

    def log2(self, a: Any) -> Any:
        return np.log2(a)

    # Linear algebra
    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]:
        return np.linalg.svd(a, full_matrices=full_matrices)

    def rng(self, seed: int | None = None) -> Any:
        return np.random.default_rng(seed)

    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        out = rng.normal(size=shape)
        return out.astype(dtype) if dtype is not None else out

    # Discrete ops / sampling helpers
    def choice(self, rng: Any, a: int, *, size: int, p: Any | None = None) -> Any:
        return rng.choice(a, size=size, p=p)

    def bincount(self, x: Any, minlength: int = 0) -> Any:
        return np.bincount(x, minlength=minlength)

    def nonzero(self, x: Any) -> Any:
        return np.nonzero(x)

    def requires_grad(self, x: Any, flag: bool = True) -> Any:
        return x

    def detach(self, x: Any) -> Any:
        return np.asarray(x)

    # --- K-like helpers (no-op/finite-diff implementations) ---
    def jit(self, fn):  # numpy has no JIT; return original
        return fn

    def value_and_grad(self, fn, argnums: int | tuple[int, ...] = 0):
        # Simple finite-difference fallback; not efficient but keeps API uniform
        eps = 1e-6

        def wrapped(*args: Any, **kwargs: Any):
            import numpy as _np

            def _to_tuple(idx) -> tuple[int, ...]:
                return (idx,) if isinstance(idx, int) else tuple(idx)

            arg_idx = _to_tuple(argnums)
            args_list = list(args)
            val = fn(*args_list, **kwargs)
            grads: list[Any] = []
            for ai in arg_idx:
                x = _np.asarray(args_list[ai], dtype=float)
                g = _np.zeros_like(x)
                it = _np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    x_plus = x.copy(); x_plus[idx] += eps
                    x_minus = x.copy(); x_minus[idx] -= eps
                    args_list[ai] = x_plus; f_plus = fn(*args_list, **kwargs)
                    args_list[ai] = x_minus; f_minus = fn(*args_list, **kwargs)
                    g[idx] = (f_plus - f_minus) / (2 * eps)
                    it.iternext()
                args_list[ai] = x
                grads.append(g)
            return val, grads[0] if len(grads) == 1 else tuple(grads)

        return wrapped


