from __future__ import annotations

from typing import Any, Protocol, Tuple
import numpy as np
from .backends.numpy_backend import NumpyBackend
from .context import get_configured_backend_instance, get_configured_backend_name
try:  # optional imports
    from .backends.pytorch_backend import PyTorchBackend  # type: ignore
except Exception:  # pragma: no cover
    PyTorchBackend = None  # type: ignore
try:
    from .backends.cupy_backend import CuPyBackend  # type: ignore
except Exception:  # pragma: no cover
    CuPyBackend = None  # type: ignore
try:
    from .backends.metal_backend import MetalBackend  # type: ignore
except Exception:  # pragma: no cover
    MetalBackend = None  # type: ignore


class ArrayBackend(Protocol):
    """Unified array/tensor backend protocol.

    Implementations should provide a cohesive set of array creation, basic
    arithmetic, and optional vectorization helpers. This protocol is purposely
    minimal at the skeleton stage and will be extended as features are
    migrated.

    Required attributes and methods:
        name: Backend name.
        array(data, dtype): Create an array.
        asarray(data): Convert to backend-native array.
        to_numpy(data): Convert to NumPy ndarray.
        matmul(a, b): Matrix multiplication.
        einsum(subscripts, *operands): Einstein summation.
        rng(seed): Random generator handle.
        normal(rng, shape, dtype): Normal distribution array.
        requires_grad(x, flag): Mark tensor for autodiff if supported.
        detach(x): Detach tensor from autodiff graph if supported.

    Optional methods:
        vmap(fn): Return a vectorized version of `fn` along the leading axis.
    """

    name: str

    # Dtype constants (backend-specific dtype tokens)
    complex64: Any
    complex128: Any
    float32: Any
    float64: Any
    int32: Any
    int64: Any
    int8: Any
    bool: Any
    # Common alias
    int: Any

    # Creation and conversion
    def array(self, data: Any, dtype: Any | None = None) -> Any: ...
    def asarray(self, data: Any) -> Any: ...
    def to_numpy(self, data: Any) -> "np.ndarray": ...  # type: ignore[name-defined]

    # Algebra / array ops
    def matmul(self, a: Any, b: Any) -> Any: ...
    def einsum(self, subscripts: str, *operands: Any) -> Any: ...
    def reshape(self, a: Any, shape: Tuple[int, ...] | Any) -> Any: ...
    def moveaxis(self, a: Any, source: int, destination: int) -> Any: ...
    def sum(self, a: Any, axis: int | None = None) -> Any: ...
    def mean(self, a: Any, axis: int | None = None) -> Any: ...
    def abs(self, a: Any) -> Any: ...
    def real(self, a: Any) -> Any: ...
    def imag(self, a: Any) -> Any: ...
    def conj(self, a: Any) -> Any: ...
    def diag(self, a: Any) -> Any: ...
    def zeros(self, shape: Tuple[int, ...], dtype: Any | None = None) -> Any: ...
    def ones(self, shape: Tuple[int, ...], dtype: Any | None = None) -> Any: ...
    def zeros_like(self, a: Any) -> Any: ...
    def ones_like(self, a: Any) -> Any: ...
    def eye(self, n: int, dtype: Any | None = None) -> Any: ...
    def kron(self, a: Any, b: Any) -> Any: ...
    def square(self, a: Any) -> Any: ...
    
    # Additional array operations
    def stack(self, arrays: Any, axis: int = 0) -> Any: ...
    def concatenate(self, arrays: Any, axis: int = 0) -> Any: ...
    def arange(self, start: Any, stop: Any | None = None, step: Any = 1) -> Any: ...
    def linspace(self, start: Any, stop: Any, num: int = 50) -> Any: ...
    def transpose(self, a: Any, axes: Any | None = None) -> Any: ...
    def norm(self, a: Any, ord: Any | None = None, axis: Any | None = None) -> Any: ...
    def cast(self, a: Any, dtype: Any) -> Any: ...
    def sign(self, a: Any) -> Any: ...
    def outer(self, a: Any, b: Any) -> Any: ...

    # Elementary math
    def exp(self, a: Any) -> Any: ...
    def sin(self, a: Any) -> Any: ...
    def cos(self, a: Any) -> Any: ...
    def sqrt(self, a: Any) -> Any: ...
    def log(self, a: Any) -> Any: ...
    def log2(self, a: Any) -> Any: ...

    # Discrete ops / sampling helpers
    def choice(self, rng: Any, a: int, *, size: int, p: Any | None = None) -> Any: ...
    def bincount(self, x: Any, minlength: int = 0) -> Any: ...
    def nonzero(self, x: Any) -> Any: ...

    # Linear algebra
    def svd(self, a: Any, full_matrices: bool = False) -> Tuple[Any, Any, Any]: ...
    def eigh(self, a: Any) -> Tuple[Any, Any]: ...
    def eig(self, a: Any) -> Tuple[Any, Any]: ...
    def solve(self, a: Any, b: Any, assume_a: str = 'gen') -> Any: ...
    def inv(self, a: Any) -> Any: ...
    def expm(self, a: Any) -> Any: ...
    def tensordot(self, a: Any, b: Any, axes: Any = 2) -> Any: ...

    # Random
    def rng(self, seed: int | None = None) -> Any: ...
    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any: ...

    # Autodiff bridge (optional)
    def requires_grad(self, x: Any, flag: bool = True) -> Any: ...
    def detach(self, x: Any) -> Any: ...


def get_backend(name: str | None = None, device: str | None = None) -> ArrayBackend:
    """Factory function that returns an ArrayBackend instance by canonical name.

    This function provides the primary interface for accessing TyxonQ's numerical
    backends. It supports automatic backend selection, configuration-based defaults,
    and explicit backend specification.

    Args:
        name (str | None): Canonical name of the backend to retrieve.
            Supported values:
            - "numpy": CPU-based NumPy backend (always available)
            - "pytorch": PyTorch backend with CUDA/MPS support (requires torch)
            - "cupy": NVIDIA GPU array backend (requires cupy)
            - None: Use configured default or fallback to NumPy

    Returns:
        ArrayBackend: An instance of the requested backend that provides
            a unified array interface for numerical computations.

    Raises:
        RuntimeError: If the requested backend is not available or its
            dependencies are not installed.

    Examples:
        >>> # Get the default backend (usually NumPy)
        >>> backend = get_backend(None)
        >>> arr = backend.array([1, 2, 3])
        
        >>> # Explicitly request NumPy backend
        >>> numpy_backend = get_backend("numpy")
        >>> x = numpy_backend.zeros((2, 3))
        
        >>> # Request PyTorch backend (if available)
        >>> try:
        ...     torch_backend = get_backend("pytorch")
        ...     tensor = torch_backend.array([1.0, 2.0])
        ... except RuntimeError as e:
        ...     print(f"PyTorch not available: {e}")
        
    Notes:
        - The function first checks for a globally configured backend instance
        - If no explicit name is provided, it falls back to the configured name
        - NumPy backend is used when no backend was configured
        - Unknown backend names raise ValueError instead of silently falling back
        - Backend availability is checked before instantiation
        
    See Also:
        NumericBackend: Class-level proxy for accessing the current backend.
        set_backend: Function for configuring the global backend.
    """

    # If no explicit name is provided, try global configuration first
    if name is None:
        inst = get_configured_backend_instance()
        if inst is not None:
            return inst  # type: ignore[return-value]
        name = get_configured_backend_name()

    if name == "auto":
        from .accelerator import cuda_available, metal_available

        if cuda_available():
            name, device = "cupy", device or "cuda"
        elif metal_available():
            name, device = "metal", device or "metal:0"
        else:
            name, device = "pytorch", "cpu"

    if name is None or name == "numpy":
        return NumpyBackend()  # type: ignore[return-value]
    if name == "pytorch":
        if PyTorchBackend is None:
            raise RuntimeError("pytorch backend requested but torch module not importable")
        if not getattr(PyTorchBackend, "available", False):
            raise RuntimeError("pytorch backend requested but torch is not installed")
        return PyTorchBackend(device=device)  # type: ignore[return-value]
    if name == "cupy":
        if device not in (None, "auto", "cuda") and not str(device).startswith("cuda:"):
            raise RuntimeError("cupy backend supports only CUDA devices")
        if CuPyBackend is None:
            raise RuntimeError("cupy backend requested but module not importable")
        if not getattr(CuPyBackend, "available", False):
            raise RuntimeError("cupy backend requested but cupy is not installed")
        return CuPyBackend()  # type: ignore[return-value]
    if name == "metal":
        if MetalBackend is None or not getattr(MetalBackend, "available", False):
            raise RuntimeError("metal backend requested but MLX is not installed")
        return MetalBackend(device=device)  # type: ignore[return-value]
    raise ValueError(f"Unknown numerics backend: {name!r}")


