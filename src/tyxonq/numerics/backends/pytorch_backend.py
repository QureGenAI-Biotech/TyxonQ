from __future__ import annotations

from typing import Any, Tuple

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


class PyTorchBackend:
    """Array backend backed by PyTorch tensors."""

    name = "pytorch"
    available = _TORCH_AVAILABLE

    # dtype constants (only defined if torch is importable)
    if _TORCH_AVAILABLE:  # pragma: no cover - simple attribute wiring
        complex64 = torch.complex64
        complex128 = torch.complex128
        float32 = torch.float32
        float64 = torch.float64
        int32 = torch.int32
        int64 = torch.int64
        bool = torch.bool
        int = torch.int64

    def _to_torch_dtype(self, dtype: Any | None):  # pragma: no cover - small mapper
        if torch is None:
            raise RuntimeError("torch not available")
        if dtype is None:
            return None
        if isinstance(dtype, torch.dtype):
            return dtype
        try:
            import numpy as _np

            mapping = {
                _np.float32: torch.float32,
                _np.float64: torch.float64,
                _np.complex64: torch.complex64,
                _np.complex128: torch.complex128,
                float: torch.float32,
                complex: torch.complex64,
            }
            return mapping.get(dtype, None)
        except Exception:
            return None

    def array(self, data: Any, dtype: Any | None = None) -> Any:
        td = self._to_torch_dtype(dtype)
        return torch.tensor(data, dtype=td)

    def asarray(self, data: Any) -> Any:
        return torch.as_tensor(data)

    def to_numpy(self, data: Any):  # type: ignore[override]
        return data.detach().cpu().numpy() if hasattr(data, "detach") else data

    def matmul(self, a: Any, b: Any) -> Any:
        return a @ b

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return torch.einsum(subscripts, *operands)

    # Array ops
    def reshape(self, a: Any, shape: Any) -> Any:
        return torch.reshape(a, shape)

    def moveaxis(self, a: Any, source: int, destination: int) -> Any:
        return torch.movedim(a, source, destination)

    def sum(self, a: Any, axis: int | None = None) -> Any:
        return torch.sum(a, dim=axis) if axis is not None else torch.sum(a)

    def mean(self, a: Any, axis: int | None = None) -> Any:
        return torch.mean(a, dim=axis) if axis is not None else torch.mean(a)

    def abs(self, a: Any) -> Any:
        return torch.abs(a)

    def real(self, a: Any) -> Any:
        return torch.real(a)

    def conj(self, a: Any) -> Any:
        return torch.conj(a)

    def diag(self, a: Any) -> Any:
        return torch.diag(a)

    def zeros(self, shape: tuple[int, ...], dtype: Any | None = None) -> Any:
        td = self._to_torch_dtype(dtype)
        return torch.zeros(shape, dtype=td)

    def ones(self, shape: tuple[int, ...], dtype: Any | None = None) -> Any:
        td = self._to_torch_dtype(dtype)
        return torch.ones(shape, dtype=td)

    def zeros_like(self, a: Any) -> Any:
        return torch.zeros_like(a)

    def ones_like(self, a: Any) -> Any:
        return torch.ones_like(a)

    def eye(self, n: int, dtype: Any | None = None) -> Any:
        td = self._to_torch_dtype(dtype)
        return torch.eye(n, dtype=td)

    def kron(self, a: Any, b: Any) -> Any:
        return torch.kron(a, b)

    # Elementary math
    def exp(self, a: Any) -> Any:
        return torch.exp(a)

    def sin(self, a: Any) -> Any:
        return torch.sin(a)

    def cos(self, a: Any) -> Any:
        return torch.cos(a)

    def sqrt(self, a: Any) -> Any:
        return torch.sqrt(a)

    def square(self, a: Any) -> Any:
        return torch.square(a)

    def log(self, a: Any) -> Any:
        return torch.log(a)

    def log2(self, a: Any) -> Any:
        return torch.log2(a)

    # Linear algebra
    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]:
        # torch.linalg.svd returns U, S, Vh similar to numpy when full_matrices=False
        U, S, Vh = torch.linalg.svd(a, full_matrices=full_matrices)
        return U, S, Vh

    def rng(self, seed: int | None = None) -> Any:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        return g

    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        return torch.normal(mean=0.0, std=1.0, size=shape, generator=rng, dtype=dtype)

    # Discrete ops / sampling helpers
    def choice(self, rng: Any, a: int, *, size: int, p: Any | None = None) -> Any:
        # PyTorch doesn't expose choice with prob directly; implement via multinomial
        if p is None:
            probs = torch.full((a,), 1.0 / a, dtype=torch.float64)
        else:
            probs = torch.as_tensor(p, dtype=torch.float64)
        idx = torch.multinomial(probs, num_samples=size, replacement=True, generator=rng)
        return idx.cpu().numpy() if hasattr(idx, 'cpu') else idx

    def bincount(self, x: Any, minlength: int = 0) -> Any:
        t = torch.as_tensor(x, dtype=torch.int64)
        return torch.bincount(t, minlength=minlength).cpu().numpy()

    def nonzero(self, x: Any) -> Any:
        t = torch.as_tensor(x)
        nz = torch.nonzero(t, as_tuple=False).squeeze(-1)
        return (nz.cpu().numpy(),)

    def requires_grad(self, x: Any, flag: bool = True) -> Any:
        if hasattr(x, "requires_grad"):
            x.requires_grad_(flag)
        return x

    def detach(self, x: Any) -> Any:
        return x.detach() if hasattr(x, "detach") else x

    # Optional: expose vmap if available
    def vmap(self, fn):  # pragma: no cover - thin wrapper
        try:
            from torch.func import vmap as torch_vmap  # type: ignore

            return torch_vmap(fn)
        except Exception:  # pragma: no cover
            def _fallback(*args: Any, **kwargs: Any):
                return fn(*args, **kwargs)

            return _fallback

    # --- K-like helpers ---
    def jit(self, fn):  # PyTorch eager; return original or torch.compile if available
        try:
            compiled = torch.compile(fn)  # type: ignore[attr-defined]
            return compiled
        except Exception:
            return fn

    def value_and_grad(self, fn, argnums: int | tuple[int, ...] = 0):

        def wrapped(*args: Any, **kwargs: Any):
            # ensure selected args require grad
            arg_idx = (argnums,) if isinstance(argnums, int) else tuple(argnums)
            args_list = list(args)
            for i in arg_idx:
                xi = args_list[i]
                if hasattr(xi, 'requires_grad'):
                    xi.requires_grad_(True)
                    args_list[i] = xi
            y = fn(*args_list, **kwargs)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, dtype=torch.float32)
            grads = torch.autograd.grad(y, [args_list[i] for i in arg_idx], allow_unused=True, retain_graph=False, create_graph=False)
            return y, grads[0] if len(grads) == 1 else tuple(grads)

        return wrapped


