"""Apple Metal array backend implemented with MLX.

MLX operations stay on the selected device. Unsupported operations raise instead
of silently moving data to NumPy; lazy arrays are materialized at host boundaries.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

from .base import BackendUnavailableError

mx: Any = None
try:
    import mlx.core as mx
    _mlx_available = True
except ImportError:  # pragma: no cover - optional dependency
    _mlx_available = False


class MetalBackend:
    name = "metal"
    available = _mlx_available
    dtypestr = "complex64"
    rdtypestr = "float32"

    if _mlx_available:
        complex64 = mx.complex64
        float32 = mx.float32
        int8 = mx.int8
        int32 = mx.int32
        int64 = mx.int64
        bool = mx.bool_
        int = mx.int64
    complex128 = np.complex128
    float64 = np.float64

    def __init__(self, device: str | None = None) -> None:
        if mx is None:
            raise BackendUnavailableError(
                "Metal backend requires MLX. Install it with `pip install 'tyxonq[metal]'` "
                "or `python -m pip install mlx` on Apple Silicon."
            )
        requested = device or "auto"
        if requested in ("auto", "metal", "metal:0", "gpu"):
            if not mx.metal.is_available():
                raise BackendUnavailableError("Metal backend requires an available Metal GPU")
            self.device = mx.gpu
        elif requested == "cpu":
            self.device = mx.cpu
        else:
            raise BackendUnavailableError(f"Unsupported MLX device: {requested!r}")
        self.device_name = "cpu" if self.device == mx.cpu else "metal:0"

    def _run(self, operation: Callable[[], Any]) -> Any:
        with mx.stream(self.device):
            return operation()

    def _dtype(self, dtype: Any | None) -> Any | None:
        if dtype is None:
            return None
        if dtype in (np.float64, np.dtype("float64"), "float64", "complex128", np.complex128):
            if self.device == mx.gpu:
                raise BackendUnavailableError(
                    "MLX Metal GPU does not support FP64/complex128; use the MLX CPU device "
                    "for FP64 array operations or the mixed-precision Metal SBD solver."
                )
            return mx.float64 if dtype not in ("complex128", np.complex128) else None
        if dtype in (np.float32, np.dtype("float32"), float, "float32"):
            return mx.float32
        if dtype in (np.complex64, np.dtype("complex64"), complex, "complex64"):
            return mx.complex64
        if dtype in (np.int8, np.dtype("int8"), "int8"):
            return mx.int8
        if dtype in (np.int32, np.dtype("int32"), "int32"):
            return mx.int32
        if dtype in (np.int64, np.dtype("int64"), int, "int64"):
            return mx.int64
        if dtype in (np.bool_, np.dtype("bool"), bool, "bool"):
            return mx.bool_
        return dtype

    def set_dtype(self, dtype_str: str) -> Tuple[Any, Any]:
        if dtype_str == "complex64":
            self.dtypestr, self.rdtypestr = "complex64", "float32"
            return self.complex64, self.float32
        if dtype_str == "complex128":
            self._dtype("complex128")
            raise BackendUnavailableError("MLX does not provide complex128")
        raise ValueError("dtype must be 'complex64' or 'complex128'")

    def array(self, data: Any, dtype: Any | None = None) -> Any:
        return self._run(lambda: mx.array(data, dtype=self._dtype(dtype)))

    def asarray(self, data: Any, dtype: Any | None = None) -> Any:
        if isinstance(data, mx.array) and dtype is None:
            return data
        return self.array(data, dtype)

    def evaluate(self, *arrays: Any) -> None:
        mx.eval(*arrays)

    def synchronize(self) -> None:
        mx.synchronize(self.device)

    def to_numpy(self, data: Any) -> np.ndarray:
        if not isinstance(data, mx.array):
            return np.asarray(data)
        mx.eval(data)
        return np.asarray(data)

    def scalar(self, data: Any) -> Any:
        mx.eval(data)
        return data.item()

    def matmul(self, a: Any, b: Any) -> Any:
        return self._run(lambda: a @ b)

    def dot(self, a: Any, b: Any) -> Any:
        return self._run(lambda: mx.matmul(a, b))

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return self._run(lambda: mx.einsum(subscripts, *operands))

    def reshape(self, a: Any, shape: Any) -> Any:
        return self._run(lambda: mx.reshape(a, shape))

    def moveaxis(self, a: Any, source: int, destination: int) -> Any:
        return self._run(lambda: mx.moveaxis(a, source, destination))

    def sum(self, a: Any, axis: Any = None) -> Any:
        return self._run(lambda: mx.sum(a, axis=axis))

    def mean(self, a: Any, axis: Any = None) -> Any:
        return self._run(lambda: mx.mean(a, axis=axis))

    def abs(self, a: Any) -> Any:
        return self._run(lambda: mx.abs(a))

    def real(self, a: Any) -> Any:
        return self._run(lambda: mx.real(a))

    def imag(self, a: Any) -> Any:
        return self._run(lambda: mx.imag(a))

    def conj(self, a: Any) -> Any:
        return self._run(lambda: mx.conj(a))

    def diag(self, a: Any) -> Any:
        return self._run(lambda: mx.diag(a))

    def zeros(self, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        return self._run(lambda: mx.zeros(shape, dtype=self._dtype(dtype) or mx.float32))

    def ones(self, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        return self._run(lambda: mx.ones(shape, dtype=self._dtype(dtype) or mx.float32))

    def zeros_like(self, a: Any) -> Any:
        return self._run(lambda: mx.zeros_like(a))

    def ones_like(self, a: Any) -> Any:
        return self._run(lambda: mx.ones_like(a))

    def eye(self, n: int, dtype: Any | None = None) -> Any:
        return self._run(lambda: mx.eye(n, dtype=self._dtype(dtype) or mx.float32))

    def kron(self, a: Any, b: Any) -> Any:
        return self._run(lambda: mx.kron(a, b))

    def square(self, a: Any) -> Any:
        return self._run(lambda: mx.square(a))

    def copy(self, a: Any) -> Any:
        return self.array(a)

    def allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return self.scalar(self._run(lambda: mx.allclose(a, b, rtol=rtol, atol=atol)))

    def isclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> Any:
        return self._run(lambda: mx.isclose(a, b, rtol=rtol, atol=atol))

    def exp(self, a: Any) -> Any:
        return self._run(lambda: mx.exp(a))

    def sin(self, a: Any) -> Any:
        return self._run(lambda: mx.sin(a))

    def cos(self, a: Any) -> Any:
        return self._run(lambda: mx.cos(a))

    def sqrt(self, a: Any) -> Any:
        return self._run(lambda: mx.sqrt(a))

    def log(self, a: Any) -> Any:
        return self._run(lambda: mx.log(a))

    def log2(self, a: Any) -> Any:
        return self._run(lambda: mx.log2(a))

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return self._run(lambda: mx.stack(arrays, axis=axis))

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        return self._run(lambda: mx.concatenate(arrays, axis=axis))

    def arange(self, start: Any, stop: Any | None = None, step: Any = 1) -> Any:
        return self._run(lambda: mx.arange(start) if stop is None else mx.arange(start, stop, step))

    def linspace(self, start: Any, stop: Any, num: int = 50) -> Any:
        return self._run(lambda: mx.linspace(start, stop, num))

    def transpose(self, a: Any, axes: Any | None = None) -> Any:
        return self._run(lambda: mx.transpose(a, axes=axes))

    def norm(self, a: Any, ord: Any | None = None, axis: Any | None = None) -> Any:
        return self._run(lambda: mx.linalg.norm(a, ord=ord, axis=axis))

    def cast(self, a: Any, dtype: Any) -> Any:
        return self._run(lambda: a.astype(self._dtype(dtype)))

    def sign(self, a: Any) -> Any:
        return self._run(lambda: mx.sign(a))

    def outer(self, a: Any, b: Any) -> Any:
        return self._run(lambda: mx.outer(a, b))

    def shape(self, a: Any) -> Tuple[int, ...]:
        return tuple(a.shape)

    def adjoint(self, a: Any) -> Any:
        return self._run(lambda: mx.conj(mx.swapaxes(a, -1, -2)))

    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]:
        if full_matrices:
            raise NotImplementedError("MLX SVD currently exposes reduced matrices only")
        return self._run(lambda: mx.linalg.svd(a))

    def eigh(self, a: Any) -> Tuple[Any, Any]:
        return self._run(lambda: mx.linalg.eigh(a))

    def eig(self, a: Any) -> Tuple[Any, Any]:
        return self._run(lambda: mx.linalg.eig(a))

    def solve(self, a: Any, b: Any, assume_a: str = "gen") -> Any:
        if assume_a != "gen":
            raise NotImplementedError("MLX solve supports general matrices only")
        return self._run(lambda: mx.linalg.solve(a, b))

    def inv(self, a: Any) -> Any:
        return self._run(lambda: mx.linalg.inv(a))

    def expm(self, a: Any) -> Any:
        raise NotImplementedError("MLX has no native matrix exponential; no host fallback is performed")

    def tensordot(self, a: Any, b: Any, axes: Any = 2) -> Any:
        return self._run(lambda: mx.tensordot(a, b, axes=axes))

    def rng(self, seed: int | None = None) -> Any:
        return mx.random.key(0 if seed is None else int(seed))

    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        return self._run(lambda: mx.random.normal(shape, dtype=self._dtype(dtype) or mx.float32, key=rng))

    def choice(self, rng: Any, a: int, *, size: int, p: Any | None = None) -> Any:
        probabilities = mx.full((a,), 1.0 / a) if p is None else self.asarray(p, mx.float32)
        return self._run(lambda: mx.random.categorical(mx.log(probabilities), num_samples=size, key=rng))

    def bincount(self, x: Any, minlength: int = 0) -> Any:
        length = max(int(minlength), self.scalar(mx.max(x)) + 1)
        labels = self.arange(length)
        return self.sum(self.asarray(x)[:, None] == labels[None, :], axis=0)

    def nonzero(self, x: Any) -> Any:
        raise NotImplementedError("MLX has no data-dependent nonzero operator on Metal GPU")

    def requires_grad(self, x: Any, flag: bool = True) -> Any:
        return x if flag else mx.stop_gradient(x)

    def detach(self, x: Any) -> Any:
        return mx.stop_gradient(x)

    def value_and_grad(self, fn: Callable[..., Any], argnums: int | tuple[int, ...] = 0) -> Callable[..., Any]:
        return mx.value_and_grad(fn, argnums=argnums)


__all__ = ["MetalBackend"]
