"""Numerics backends and vectorization utilities."""

from __future__ import annotations
from typing import Any, Tuple

from .api import ArrayBackend, VectorizationPolicy, vectorize_or_fallback, get_backend
from .context import set_backend as _set_backend_internal, use_backend

__all__ = [
    "ArrayBackend",
    "VectorizationPolicy",
    "vectorize_or_fallback",
    "get_backend",
    "set_backend",
    "set_dtype",
    "use_backend",
]


def set_backend(name_or_instance: Any) -> Any:
    """Set global numerics backend and return the backend instance.
    
    Args:
        name_or_instance: Backend name ('numpy', 'pytorch', 'cupynumeric') or backend instance
        
    Returns:
        The backend instance
    """
    # If it's a string, create instance and store it
    if isinstance(name_or_instance, str):
        bk = get_backend(name_or_instance)
        _set_backend_internal(bk)  # Store the instance, not the name
        return bk
    else:
        _set_backend_internal(name_or_instance)
        return name_or_instance


def set_dtype(dtype_str: str) -> Tuple[Any, Any]:
    """Set default dtype for the current backend.
    
    This is a convenience function that calls set_dtype() on the current backend instance.
    It sets both complex dtype (for quantum states) and real dtype (for parameters).
    
    Args:
        dtype_str: One of "complex64" or "complex128"
        
    Returns:
        Tuple of (complex_dtype, real_dtype) from the current backend
        
    Example:
        >>> import tyxonq as tq
        >>> tq.set_backend("pytorch")
        >>> ctype, rtype = tq.set_dtype("complex64")
        >>> # Now backend uses complex64 for states, float32 for parameters
    """
    backend = get_backend(None)  # Get current backend
    return backend.set_dtype(dtype_str)



class _classproperty(property):
    def __get__(self, obj, owner):
        return self.fget(owner)


class NumericBackend:
    """Class-level proxy providing unified access to the current numerical backend.

    This class serves as a convenient interface for accessing array operations
    without requiring explicit backend instantiation. All methods and properties
    are implemented as class-level operations that delegate to the currently
    configured backend.

    The NumericBackend provides a consistent API across different numerical
    libraries (NumPy, PyTorch, CuPy) while allowing runtime backend switching
    for performance optimization and hardware targeting.

    Key Features:
        - **No instantiation required**: All operations are class methods
        - **Automatic backend delegation**: Forwards calls to the active backend
        - **Unified dtype access**: Consistent data type properties across backends
        - **Hardware acceleration**: Supports CPU, GPU, and distributed computing
        - **Automatic differentiation**: Gradient computation when supported

    Supported Backends:
        - **numpy**: CPU-based computations with full NumPy compatibility
        - **pytorch**: GPU acceleration and automatic differentiation via PyTorch
        - **cupynumeric**: Distributed GPU computing via CuPy/Legate

    Examples:
        >>> # Create arrays using the current backend
        >>> x = NumericBackend.array([1, 2, 3])
        >>> y = NumericBackend.zeros((2, 3), dtype=NumericBackend.complex128)
        
        >>> # Perform operations
        >>> result = NumericBackend.matmul(x, y.T)
        >>> eigenvals = NumericBackend.svd(result)[1]
        
        >>> # Access data types
        >>> complex_dtype = NumericBackend.complex64
        >>> float_dtype = NumericBackend.float32
        
        >>> # Mathematical functions
        >>> exp_result = NumericBackend.exp(x)
        >>> trig_result = NumericBackend.sin(NumericBackend.pi * x)
        
        >>> # Automatic differentiation (PyTorch backend)
        >>> x_grad = NumericBackend.requires_grad(x, True)
        >>> if hasattr(NumericBackend, 'value_and_grad'):
        ...     val, grad = NumericBackend.value_and_grad(lambda t: NumericBackend.sum(t**2))(x_grad)

    Notes:
        - Backend selection is controlled via `get_backend()` and configuration
        - Not all backends support all operations (e.g., automatic differentiation)
        - Performance characteristics vary significantly between backends
        - Some operations may fall back gracefully when unsupported
        
    See Also:
        get_backend: Factory function for explicit backend selection.
        set_backend: Configure the global default backend.
        ArrayBackend: Base interface implemented by all backends.
    """

    # Dtype constants
    @_classproperty
    def complex64(cls):  # type: ignore[override]
        return get_backend(None).complex64

    @_classproperty
    def complex128(cls):  # type: ignore[override]
        return get_backend(None).complex128

    @_classproperty
    def float32(cls):  # type: ignore[override]
        return get_backend(None).float32

    @_classproperty
    def float64(cls):  # type: ignore[override]
        return get_backend(None).float64

    @_classproperty
    def int32(cls):  # type: ignore[override]
        return get_backend(None).int32

    @_classproperty
    def int64(cls):  # type: ignore[override]
        return get_backend(None).int64

    @_classproperty
    def bool(cls):  # noqa: A003
        return get_backend(None).bool

    @_classproperty
    def int(cls):  # noqa: A003
        return get_backend(None).int

    # Creation and conversion
    @classmethod
    def array(cls, data, dtype=None):
        return get_backend(None).array(data, dtype=dtype)

    @classmethod
    def array_to_tensor(cls, data, dtype=None):
        """Convert array-like data to backend tensor.
        
        This is an alias for array() with clearer semantic meaning in quantum
        computing contexts where 'tensor' refers to the backend representation.
        
        Args:
            data: Array-like data (list, numpy array, etc.)
            dtype: Target data type (optional)
        
        Returns:
            Backend tensor representation
        
        Examples:
            >>> psi = NumericBackend.array_to_tensor([1.0, 0.0])  # |0⟩ state
            >>> H = NumericBackend.array_to_tensor([[1, 1], [1, -1]], dtype=NumericBackend.complex128)
        """
        return get_backend(None).array(data, dtype=dtype)

    @classmethod
    def asarray(cls, data):
        return get_backend(None).asarray(data)

    @classmethod
    def to_numpy(cls, data):
        return get_backend(None).to_numpy(data)

    # Algebra / ops
    @classmethod
    def matmul(cls, a, b):
        return get_backend(None).matmul(a, b)

    @classmethod
    def einsum(cls, subscripts: str, *operands):
        return get_backend(None).einsum(subscripts, *operands)

    @classmethod
    def reshape(cls, a, shape):
        return get_backend(None).reshape(a, shape)

    @classmethod
    def moveaxis(cls, a, source, destination):
        return get_backend(None).moveaxis(a, source, destination)

    @classmethod
    def sum(cls, a, axis=None):
        return get_backend(None).sum(a, axis=axis)

    @classmethod
    def mean(cls, a, axis=None):
        return get_backend(None).mean(a, axis=axis)

    @classmethod
    def abs(cls, a):
        return get_backend(None).abs(a)

    @classmethod
    def real(cls, a):
        return get_backend(None).real(a)

    @classmethod
    def imag(cls, a):
        return get_backend(None).imag(a)

    @classmethod
    def conj(cls, a):
        return get_backend(None).conj(a)

    @classmethod
    def adjoint(cls, a):
        """Compute the Hermitian conjugate (adjoint) of a matrix.
        
        For a matrix A, returns A† = (A*)ᵀ (conjugate transpose).
        
        Args:
            a: Input array/tensor (typically a matrix)
        
        Returns:
            Hermitian conjugate of a
        
        Examples:
            >>> # Pauli Y matrix
            >>> Y = NumericBackend.array([[0, -1j], [1j, 0]])
            >>> Y_dag = NumericBackend.adjoint(Y)  # Y† = Y (Y is Hermitian)
            
            >>> # Create state |ψ⟩ = (|0⟩ + |1⟩)/√2
            >>> psi = NumericBackend.array([1/np.sqrt(2), 1/np.sqrt(2)])
            >>> psi_col = psi.reshape((-1, 1))  # Column vector
            >>> bra = NumericBackend.adjoint(psi_col)  # Row vector ⟨ψ|
        
        Note:
            This is equivalent to conj(a).T or conj(transpose(a, axes=[1, 0]))
        """
        b = get_backend(None)
        # Use backend's adjoint if available, otherwise compute manually
        adj = getattr(b, "adjoint", None)
        if callable(adj):
            return adj(a)
        # Fallback: conjugate transpose
        return b.conj(b.transpose(a))

    @classmethod
    def diag(cls, a):
        return get_backend(None).diag(a)

    @classmethod
    def zeros(cls, shape, dtype=None):
        return get_backend(None).zeros(shape, dtype=dtype)

    @classmethod
    def zeros_like(cls, a):
        return get_backend(None).zeros_like(a)

    @classmethod
    def ones_like(cls, a):
        return get_backend(None).ones_like(a)

    @classmethod
    def eye(cls, n, dtype=None):
        return get_backend(None).eye(n, dtype=dtype)

    @classmethod
    def kron(cls, a, b):
        return get_backend(None).kron(a, b)
    
    # Additional array operations
    @classmethod
    def stack(cls, arrays, axis=0):
        return get_backend(None).stack(arrays, axis=axis)

    @classmethod
    def concatenate(cls, arrays, axis=0):
        return get_backend(None).concatenate(arrays, axis=axis)

    @classmethod
    def arange(cls, start, stop=None, step=1):
        return get_backend(None).arange(start, stop, step)

    @classmethod
    def linspace(cls, start, stop, num=50):
        return get_backend(None).linspace(start, stop, num)

    @classmethod
    def transpose(cls, a, axes=None):
        return get_backend(None).transpose(a, axes=axes)

    @classmethod
    def norm(cls, a, ord=None, axis=None):
        return get_backend(None).norm(a, ord=ord, axis=axis)

    @classmethod
    def cast(cls, a, dtype):
        return get_backend(None).cast(a, dtype)

    @classmethod
    def sign(cls, a):
        return get_backend(None).sign(a)

    @classmethod
    def outer(cls, a, b):
        return get_backend(None).outer(a, b)
    
    # Linear algebra (extended)
    @classmethod
    def eigh(cls, a):
        return get_backend(None).eigh(a)

    @classmethod
    def eig(cls, a):
        return get_backend(None).eig(a)

    @classmethod
    def solve(cls, a, b, assume_a='gen'):
        return get_backend(None).solve(a, b, assume_a=assume_a)

    @classmethod
    def inv(cls, a):
        return get_backend(None).inv(a)

    @classmethod
    def expm(cls, a):
        return get_backend(None).expm(a)

    @classmethod
    def tensordot(cls, a, b, axes=2):
        return get_backend(None).tensordot(a, b, axes=axes)
    
    @classmethod
    def svd(cls, a, full_matrices: bool = False):
        return get_backend(None).svd(a, full_matrices=full_matrices)

    # Elementary math
    @classmethod
    def exp(cls, a):
        return get_backend(None).exp(a)

    @classmethod
    def sin(cls, a):
        return get_backend(None).sin(a)

    @classmethod
    def cos(cls, a):
        return get_backend(None).cos(a)

    @classmethod
    def sqrt(cls, a):
        return get_backend(None).sqrt(a)

    # Random
    @classmethod
    def rng(cls, seed=None):
        return get_backend(None).rng(seed)

    @classmethod
    def normal(cls, rng, shape, dtype=None):
        return get_backend(None).normal(rng, shape, dtype=dtype)

    # Autodiff
    @classmethod
    def requires_grad(cls, x, flag=True):
        return get_backend(None).requires_grad(x, flag)

    @classmethod
    def detach(cls, x):
        return get_backend(None).detach(x)

    # Optional helpers
    @classmethod  # pragma: no cover
    def vmap(cls, fn):
        return get_backend(None).vmap(fn)

    @classmethod  # pragma: no cover
    def jit(cls, fn):
        b = get_backend(None)
        j = getattr(b, "jit", None)
        return j(fn) if callable(j) else fn

    @classmethod  # pragma: no cover
    def value_and_grad(cls, fn, argnums=0):
        b = get_backend(None)
        vag = getattr(b, "value_and_grad", None)
        if callable(vag):
            return vag(fn, argnums=argnums)
        raise AttributeError("Active backend does not provide value_and_grad")

    @classmethod  # pragma: no cover
    def __repr__(cls) -> str:
        b = get_backend(None)
        return f"<NumericBackend {b.name}>"

__all__.append("NumericBackend")


