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


