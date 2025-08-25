from __future__ import annotations

from typing import Any, Callable, Literal, Protocol, Tuple
from .backends.numpy_backend import NumpyBackend
from .context import get_configured_backend_instance, get_configured_backend_name
try:  # optional imports
    from .backends.pytorch_backend import PyTorchBackend  # type: ignore
except Exception:  # pragma: no cover
    PyTorchBackend = None  # type: ignore
try:
    from .backends.cupynumeric_backend import CuPyNumericBackend  # type: ignore
except Exception:  # pragma: no cover
    CuPyNumericBackend = None  # type: ignore


VectorizationPolicy = Literal["auto", "force", "off"]


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

    # Creation and conversion
    def array(self, data: Any, dtype: Any | None = None) -> Any: ...
    def asarray(self, data: Any) -> Any: ...
    def to_numpy(self, data: Any) -> "np.ndarray": ...  # type: ignore[name-defined]

    # Algebra
    def matmul(self, a: Any, b: Any) -> Any: ...
    def einsum(self, subscripts: str, *operands: Any) -> Any: ...

    # Random
    def rng(self, seed: int | None = None) -> Any: ...
    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any: ...

    # Autodiff bridge (optional)
    def requires_grad(self, x: Any, flag: bool = True) -> Any: ...
    def detach(self, x: Any) -> Any: ...


def vectorize_or_fallback(
    fn: Callable[..., Any],
    backend: ArrayBackend,
    policy: VectorizationPolicy = "auto",
    *,
    enable_checks: bool = True,
) -> Callable[..., Any]:
    """Wrap a function with vectorization behavior and safe fallback.

    Behavior:
    - policy == "off": return the original function.
    - If backend provides `vmap` and policy != "off": try vectorized execution.
      On any exception, log (if available) and fallback to eager execution.
    - If no `vmap` is provided: emulate vectorization by applying the function
      element-wise along the leading axis when the first positional argument is
      a sequence; otherwise run eagerly.

    Parameters:
        fn: Function to wrap.
        backend: Array backend instance implementing the protocol.
        policy: Vectorization policy ("auto" | "force" | "off").
        enable_checks: Placeholder for future safety checks before vectorizing.

    Returns:
        A callable that applies `fn` with the chosen vectorization strategy.
    """

    if policy == "off":
        return fn

    # Prefer backend-provided vmap if available
    vmap_fn = getattr(backend, "vmap", None)
    if callable(vmap_fn):
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return vmap_fn(fn)(*args, **kwargs)  # type: ignore[misc]
            except Exception:
                # Fallback to eager execution on any vectorization error
                return fn(*args, **kwargs)
        return wrapped

    # Generic lightweight vectorization fallback along leading axis
    def _generic_vectorized_call(*args: Any, **kwargs: Any) -> Any:
        if not args:
            return fn(*args, **kwargs)
        first = args[0]
        try:
            return [fn(a, *args[1:], **kwargs) for a in first]  # type: ignore[call-arg]
        except TypeError:
            # Not iterable; run eagerly
            return fn(*args, **kwargs)

    return _generic_vectorized_call


def get_backend(name: str | None) -> ArrayBackend:
    """Factory returning an ArrayBackend by canonical name.

    Supported:
        - 'numpy'
        - 'pytorch' (requires torch)
        - 'cupynumeric' (requires cunumeric)
    """

    # If no explicit name is provided, try global configuration first
    if name is None:
        inst = get_configured_backend_instance()
        if inst is not None:
            return inst  # type: ignore[return-value]
        name = get_configured_backend_name()

    if name is None or name == "numpy":
        return NumpyBackend()  # type: ignore[return-value]
    if name == "pytorch":
        if PyTorchBackend is None:
            raise RuntimeError("pytorch backend requested but torch not available")
        return PyTorchBackend()  # type: ignore[return-value]
    if name == "cupynumeric":
        if CuPyNumericBackend is None:
            raise RuntimeError("cupynumeric backend requested but cupynumeric not available")
        return CuPyNumericBackend()  # type: ignore[return-value]
    # default fallback
    return NumpyBackend()  # type: ignore[return-value]


