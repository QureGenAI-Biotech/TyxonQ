"""Numerics backends and vectorization utilities."""

from .api import ArrayBackend, VectorizationPolicy, vectorize_or_fallback, get_backend
from .context import set_backend, use_backend

__all__ = [
    "ArrayBackend",
    "VectorizationPolicy",
    "vectorize_or_fallback",
    "get_backend",
    "set_backend",
    "use_backend",
]



class _classproperty(property):
    def __get__(self, obj, owner):
        return self.fget(owner)


class NumericBackend:
    """Class-level proxy to the current backend (no instantiation required)."""

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
    def conj(cls, a):
        return get_backend(None).conj(a)

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


