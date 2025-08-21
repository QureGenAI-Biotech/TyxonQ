from __future__ import annotations

from typing import Any, Tuple

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


class PyTorchBackend:
    """Array backend backed by PyTorch tensors."""

    name = "pytorch"

    def array(self, data: Any, dtype: Any | None = None) -> Any:
        if torch is None:
            raise RuntimeError("torch not available")
        return torch.tensor(data, dtype=dtype)

    def asarray(self, data: Any) -> Any:
        if torch is None:
            raise RuntimeError("torch not available")
        return torch.as_tensor(data)

    def to_numpy(self, data: Any):  # type: ignore[override]
        if torch is None:
            raise RuntimeError("torch not available")
        return data.detach().cpu().numpy() if hasattr(data, "detach") else data

    def matmul(self, a: Any, b: Any) -> Any:
        if torch is None:
            raise RuntimeError("torch not available")
        return a @ b

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        if torch is None:
            raise RuntimeError("torch not available")
        return torch.einsum(subscripts, *operands)

    def rng(self, seed: int | None = None) -> Any:
        if torch is None:
            raise RuntimeError("torch not available")
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        return g

    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        if torch is None:
            raise RuntimeError("torch not available")
        return torch.normal(mean=0.0, std=1.0, size=shape, generator=rng, dtype=dtype)

    def requires_grad(self, x: Any, flag: bool = True) -> Any:
        if torch is None:
            raise RuntimeError("torch not available")
        if hasattr(x, "requires_grad"):
            x.requires_grad_(flag)
        return x

    def detach(self, x: Any) -> Any:
        if torch is None:
            raise RuntimeError("torch not available")
        return x.detach() if hasattr(x, "detach") else x

    # Optional: expose vmap if available
    def vmap(self, fn):  # pragma: no cover - thin wrapper
        if torch is None:
            raise RuntimeError("torch not available")
        try:
            from torch.func import vmap as torch_vmap  # type: ignore

            return torch_vmap(fn)
        except Exception:  # pragma: no cover
            def _fallback(*args: Any, **kwargs: Any):
                return fn(*args, **kwargs)

            return _fallback


