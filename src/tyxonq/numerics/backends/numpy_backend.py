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


