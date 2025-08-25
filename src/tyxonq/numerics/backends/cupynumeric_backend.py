from __future__ import annotations

from typing import Any, Tuple

try:
    import cupynumeric as cn
except Exception:  # pragma: no cover - optional dependency
    cn = None  # type: ignore


class CuPyNumericBackend:
    """Array backend backed by cupynumeric (GPU/accelerated)."""

    name = "cupynumeric"

    def array(self, data: Any, dtype: Any | None = None) -> Any:
        if cn is None:
            raise RuntimeError("cupynumeric not available")
        return cn.array(data, dtype=dtype)

    def asarray(self, data: Any) -> Any:
        if cn is None:
            raise RuntimeError("cupynumeric not available")
        return cn.asarray(data)

    def to_numpy(self, data: Any):  # type: ignore[override]
        if cn is None:
            raise RuntimeError("cupynumeric not available")
        import numpy as np

        return np.asarray(data)

    def matmul(self, a: Any, b: Any) -> Any:
        if cn is None:
            raise RuntimeError("cupynumeric not available")
        return a @ b

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        if cn is None:
            raise RuntimeError("cupynumeric not available")
        return cn.einsum(subscripts, *operands)

    def rng(self, seed: int | None = None) -> Any:
        if cn is None:
            raise RuntimeError("cupynumeric not available")
        import numpy as np

        return np.random.default_rng(seed)

    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        if cn is None:
            raise RuntimeError("cupynumeric not available")
        import numpy as np

        out = rng.normal(size=shape)
        return out.astype(dtype) if dtype is not None else out

    def requires_grad(self, x: Any, flag: bool = True) -> Any:
        return x

    def detach(self, x: Any) -> Any:
        return x

    # K-like helpers (no-op on jit; finite-diff gradient via numpy fallback)
    def jit(self, fn):
        return fn

    def value_and_grad(self, fn, argnums: int | tuple[int, ...] = 0):
        # Use numpy fallback by converting arrays
        import numpy as _np

        eps = 1e-6

        def to_np(a):
            try:
                import cupy as _cp  # type: ignore
                return _cp.asnumpy(a) if hasattr(a, "__array__") else _np.asarray(a)
            except Exception:
                return _np.asarray(a)

        def wrapped(*args: Any, **kwargs: Any):
            def _to_tuple(idx) -> tuple[int, ...]:
                return (idx,) if isinstance(idx, int) else tuple(idx)

            arg_idx = _to_tuple(argnums)
            args_list = list(args)
            val = fn(*args_list, **kwargs)
            grads: list[Any] = []
            for ai in arg_idx:
                x = to_np(args_list[ai]).astype(float)
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


