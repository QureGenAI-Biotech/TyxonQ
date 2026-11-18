"""NVIDIA GPU array backend backed by CuPy.

CuPy replaces the end-of-life cuPyNumeric backend. It provides NumPy-compatible
array operations on CUDA devices and is an optional, explicitly selected
backend; it is never chosen by the SQD auto policy.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .numpy_backend import NumpyBackend

try:  # optional dependency
    import cupy as cp

    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional package
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False


class CuPyBackend(NumpyBackend):
    """Array operations on CuPy arrays (NVIDIA CUDA only)."""

    name = "cupy"
    available = _CUPY_AVAILABLE

    if _CUPY_AVAILABLE:
        complex64 = cp.complex64
        complex128 = cp.complex128
        float32 = cp.float32
        float64 = cp.float64
        int8 = cp.int8
        int32 = cp.int32
        int64 = cp.int64
        bool = cp.bool_
        int = cp.int64

    def __init__(self) -> None:
        if cp is None:
            raise RuntimeError("cupy backend requested but cupy is not installed")

    def set_dtype(self, dtype_str: str) -> Tuple[Any, Any]:
        if dtype_str == "complex64":
            self.dtypestr, self.rdtypestr = "complex64", "float32"
            return cp.complex64, cp.float32
        if dtype_str == "complex128":
            self.dtypestr, self.rdtypestr = "complex128", "float64"
            return cp.complex128, cp.float64
        raise ValueError(f"Unsupported dtype: {dtype_str}. Use 'complex64' or 'complex128'.")

    def array(self, data: Any, dtype: Any | None = None) -> Any:
        return cp.array(data, dtype=dtype)

    def asarray(self, data: Any, dtype: Any | None = None) -> Any:
        return cp.asarray(data, dtype=dtype) if dtype is not None else cp.asarray(data)

    def to_numpy(self, data: Any) -> np.ndarray:
        return cp.asnumpy(data) if isinstance(data, cp.ndarray) else np.asarray(data)

    def dot(self, a: Any, b: Any) -> Any:
        return cp.matmul(a, b)

    def matmul(self, a: Any, b: Any) -> Any:
        return cp.matmul(a, b)

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return cp.einsum(subscripts, *operands)

    def reshape(self, a: Any, shape: Any) -> Any:
        return cp.reshape(a, shape)

    def moveaxis(self, a: Any, source: int, destination: int) -> Any:
        return cp.moveaxis(a, source, destination)

    def sum(self, a: Any, axis: int | None = None) -> Any:
        return cp.sum(a, axis=axis)

    def mean(self, a: Any, axis: int | None = None) -> Any:
        return cp.mean(a, axis=axis)

    def abs(self, a: Any) -> Any:
        return cp.abs(a)

    def real(self, a: Any) -> Any:
        return cp.real(a)

    def imag(self, a: Any) -> Any:
        return cp.imag(a)

    def conj(self, a: Any) -> Any:
        return cp.conj(a)

    def diag(self, a: Any) -> Any:
        return cp.diag(a)

    def zeros(self, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        return cp.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        return cp.ones(shape, dtype=dtype)

    def zeros_like(self, a: Any) -> Any:
        return cp.zeros_like(a)

    def ones_like(self, a: Any) -> Any:
        return cp.ones_like(a)

    def eye(self, n: int, dtype: Any | None = None) -> Any:
        return cp.eye(n, dtype=dtype)

    def kron(self, a: Any, b: Any) -> Any:
        return cp.kron(a, b)

    def square(self, a: Any) -> Any:
        return cp.square(a)

    def copy(self, a: Any) -> Any:
        return cp.copy(a)

    def allclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return bool(cp.allclose(a, b, rtol=rtol, atol=atol))

    def isclose(self, a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-8) -> Any:
        return cp.isclose(a, b, rtol=rtol, atol=atol)

    def exp(self, a: Any) -> Any:
        return cp.exp(a)

    def sin(self, a: Any) -> Any:
        return cp.sin(a)

    def cos(self, a: Any) -> Any:
        return cp.cos(a)

    def sqrt(self, a: Any) -> Any:
        return cp.sqrt(a)

    def log(self, a: Any) -> Any:
        return cp.log(a)

    def log2(self, a: Any) -> Any:
        return cp.log2(a)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return cp.stack(arrays, axis=axis)

    def concatenate(self, arrays: Any, axis: int = 0) -> Any:
        return cp.concatenate(arrays, axis=axis)

    def arange(self, start: Any, stop: Any | None = None, step: Any = 1) -> Any:
        return cp.arange(start) if stop is None else cp.arange(start, stop, step)

    def linspace(self, start: Any, stop: Any, num: int = 50) -> Any:
        return cp.linspace(start, stop, num)

    def transpose(self, a: Any, axes: Any | None = None) -> Any:
        return cp.transpose(a, axes=axes)

    def norm(self, a: Any, ord: Any | None = None, axis: Any | None = None) -> Any:
        return cp.linalg.norm(a, ord=ord, axis=axis)

    def cast(self, a: Any, dtype: Any) -> Any:
        return cp.asarray(a, dtype=dtype)

    def sign(self, a: Any) -> Any:
        return cp.sign(a)

    def outer(self, a: Any, b: Any) -> Any:
        return cp.outer(a, b)

    def shape(self, a: Any) -> Tuple[int, ...]:
        return tuple(a.shape)

    def adjoint(self, a: Any) -> Any:
        return cp.conj(cp.swapaxes(a, -1, -2))

    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]:
        return cp.linalg.svd(a, full_matrices=full_matrices)

    def eigh(self, a: Any) -> Tuple[Any, Any]:
        return cp.linalg.eigh(a)

    def eig(self, a: Any) -> Tuple[Any, Any]:
        return cp.linalg.eig(a)

    def solve(self, a: Any, b: Any, assume_a: str = "gen") -> Any:
        return cp.linalg.solve(a, b)

    def inv(self, a: Any) -> Any:
        return cp.linalg.inv(a)

    def expm(self, a: Any) -> Any:
        import scipy.linalg

        return cp.asarray(scipy.linalg.expm(self.to_numpy(a)))

    def tensordot(self, a: Any, b: Any, axes: Any = 2) -> Any:
        return cp.tensordot(a, b, axes=axes)

    def rng(self, seed: int | None = None) -> Any:
        factory = getattr(cp.random, "default_rng", None)
        return factory(seed) if factory is not None else cp.random.RandomState(seed)

    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any:
        out = rng.normal(size=shape)
        return cp.asarray(out, dtype=dtype) if dtype is not None else cp.asarray(out)

    def choice(self, rng: Any, a: int, *, size: int, p: Any | None = None) -> Any:
        return cp.asarray(rng.choice(a, size=size, p=p))

    def bincount(self, x: Any, minlength: int = 0) -> Any:
        return cp.bincount(x, minlength=minlength)

    def nonzero(self, x: Any) -> Any:
        return cp.nonzero(x)

    def requires_grad(self, x: Any, flag: bool = True) -> Any:
        return x

    def detach(self, x: Any) -> Any:
        return cp.asarray(x)

    def jit(self, fn: Any) -> Any:
        return fn

    def jacfwd(self, fn: Any, argnums: int = 0) -> Any:
        raise NotImplementedError("CuPyBackend does not provide autodiff; use pytorch for gradients")

    def hessian(self, fn: Any, argnums: int = 0) -> Any:
        raise NotImplementedError("CuPyBackend does not provide autodiff; use pytorch for gradients")

    def value_and_grad(self, fn: Any, argnums: int | tuple[int, ...] = 0) -> Any:
        raise NotImplementedError("CuPyBackend does not provide autodiff; use pytorch for gradients")


__all__ = ["CuPyBackend"]
