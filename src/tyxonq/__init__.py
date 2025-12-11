"""TyxonQ package initializer (refactor minimal).

Lightweight exports for cloud API convenience while preserving side-effect
constraints. Heavy modules are not imported here.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import sys as _sys
import importlib as _importlib

__version__ = "0.9.9"
__author__ = "TyxonQ Development Team"

# --- Cloud API re-exports and module aliases ---
# We alias `tyxonq.api` and `tyxonq.apis` to `tyxonq.cloud.apis` so both old and
# new entrypoints work:
#   1) tyxonq.api.submit_task()
#   2) tyxonq.apis.submit_task()
#   3) tyxonq.run(provider=..., device=...)
#   4) tyxonq.submit_task(provider=..., device=...)
#   5) tyxonq.set_token()
try:
    from .cloud import api as _cloud_apis  # type: ignore
except Exception:  # pragma: no cover - keep package import robust
    _cloud_apis = None  # type: ignore

if _cloud_apis is not None:
    # Submodule import compatibility: `import tyxonq.api`
    _sys.modules[__name__ + ".api"] = _cloud_apis

    # Attribute access compatibility: `tyxonq.api` / `tyxonq.apis`
    api = _cloud_apis  # type: ignore

    # Top-level convenience wrappers
    def set_token(
        token: Optional[str] = None,
        provider: Optional[Union[str, Any]] = "tyxonq",
        device: Optional[Union[str, Any]] = "homebrew_s2",
    ) -> Dict[str, Any]:
        """Configure TyxonQ cloud API token (non-persistent).

        Parameters:
            token: API key string. If None, only defaults are set.
            provider: Cloud provider id. Default "tyxonq".
            device: Device id (without prefix). Default "homebrew_s2".
        Security:
            Token is never persisted to disk here.
        """

        return _cloud_apis.set_token(  # type: ignore[attr-defined]
            token=token,
            provider=provider,
            device=device,
        )

    def submit_task(
        *,
        provider: Optional[Union[str, Any]] = None,
        device: Optional[Union[str, Any]] = None,
        token: Optional[str] = None,
        **task_kws: Any,
    ):
        """Submit a task to a provider/device.

        Delegates to ``tyxonq.cloud.apis.submit_task`` and returns a Task or a list of Tasks.
        """

        return _cloud_apis.run(  # type: ignore[attr-defined]
            provider=provider, device=device, token=token, **task_kws
        )

    def run(
        circuit: Union[Any, Sequence[Any]],
        *,
        provider: Optional[Union[str, Any]] = None,
        device: Optional[Union[str, Any]] = None,
        shots: int = 1024,
        wait: bool = True,
        mitigated: bool = False,
        **task_kws: Any,
    ) -> Union[Dict[str, int], List[Dict[str, int]], Any]:
        """Submit circuit(s) and optionally wait for results.

        - If ``wait`` is True (default), returns counts dict or a list of counts.
        - If ``wait`` is False, returns the Task or list of Tasks.
        """

        tasks = _cloud_apis.run(  # type: ignore[attr-defined]
            provider=provider,
            device=device,
            shots=shots,
            circuit=circuit,
            **task_kws,
        )

        # Normalize to list of Task objects
        if isinstance(tasks, list):
            task_list = tasks
        else:
            task_list = [tasks]

        if not wait:
            return tasks

        # Fetch details via unified cloud API helper (supports polling)
        results = [_cloud_apis.get_task_details(t, wait=True) for t in task_list]  # type: ignore[attr-defined]
        return results if isinstance(tasks, list) else results[0]

    __all__ = [
        "api",
        "apis",
        "set_token",
        "submit_task",
        "run",
    ]
else:
    __all__ = []


# --- Compatibility alias: expose `tyxonq.chem` to point to applications.chem ---
# from .applications import chem as _chem_app  # type: ignore
# # Package-level alias
# _sys.modules[__name__ + ".chem"] = _chem_app

# --- Top-level core IR exports ---
try:
    from .core import Circuit, Hamiltonian  # type: ignore
    
    # expose in module namespace and __all__
    __all__.extend(["Circuit", "Hamiltonian"])
except Exception:
    pass

# --- Top-level global pipeline defaults (device/postprocessing/compile) ---
try:
    from .core.ir.circuit import (
        set_global_device_defaults as _set_dev_defaults,
        get_global_device_defaults as _get_dev_defaults,
        set_global_postprocessing_defaults as _set_post_defaults,
        get_global_postprocessing_defaults as _get_post_defaults,
        set_global_compile_defaults as _set_compile_defaults,
        get_global_compile_defaults as _get_compile_defaults,
    )

    def device(**options: Any) -> Dict[str, Any]:
        return _set_dev_defaults(dict(options))

    def get_device_defaults() -> Dict[str, Any]:
        return _get_dev_defaults()

    def postprocessing(**options: Any) -> Dict[str, Any]:
        return _set_post_defaults(dict(options))


    def get_postprocessing_defaults() -> Dict[str, Any]:
        return _get_post_defaults()

    def compile(**options: Any) -> Dict[str, Any]:
        return _set_compile_defaults(dict(options))

    def get_compile() -> Dict[str, Any]:
        return _get_compile_defaults()

    __all__.extend([
        "device",
        "get_device_defaults",
        "postprocessing",
        "get_postprocessing_defaults",
        "compile",
        "get_compile",
    ])
except Exception:
    pass

# --- Top-level numerics backend selection convenience ---
try:
    from .numerics.context import set_backend as _set_backend  # type: ignore
    from .numerics.api import get_backend as _get_backend  # type: ignore
    from .numerics import NumericBackend as _NumericBackend  # type: ignore
    from .numerics import set_dtype as _set_dtype  # type: ignore

    def set_backend(name_or_instance: Any):
        """Set global/default numerics backend (e.g., 'numpy' | 'pytorch').

        Also exposes ``tyxonq.backend`` and ``tyxonq.rdtypestr`` for legacy helpers.
        """

        def _assign_backend_alias(bk: Any) -> Any:
            try:
                globals()["backend"] = bk
                try:
                    globals()["rdtypestr"] = getattr(bk, "rdtypestr", "float64")
                except Exception:
                    globals()["rdtypestr"] = "float64"
                try:
                    globals()["dtypestr"] = getattr(bk, "dtypestr", "complex128")
                except Exception:
                    globals()["dtypestr"] = "complex128"
                return bk
            except Exception:
                return None

        # String name path with minimal fallback for cupynumeric → numpy
        if isinstance(name_or_instance, str):
            name = str(name_or_instance).lower()
            if name == "cupynumeric":
                try:
                    bk = _get_backend(name)
                    _set_backend(bk)  # Store instance, not name
                    return _assign_backend_alias(bk)
                except Exception:
                    try:
                        import warnings as _warnings
                        _warnings.warn("cupynumeric not installed, falling back to numpy backend", UserWarning)
                    except Exception:
                        pass
                    bk = _get_backend("numpy")
                    _set_backend(bk)
                    return _assign_backend_alias(bk)
            else:
                bk = _get_backend(name)
                _set_backend(bk)  # Store instance, not name
                return _assign_backend_alias(bk)

        # Instance path
        _set_backend(name_or_instance)
        return _assign_backend_alias(name_or_instance)

    def get_backend(name: Any | None = None):
        """Get an ArrayBackend instance.

        - name is optional; when None, returns the currently configured backend.
        - name can be 'numpy' | 'pytorch' | 'cupynumeric' to explicitly fetch.
        """

        return _get_backend(name)
    
    def set_dtype(dtype_str: str):
        """Set default dtype for the current backend.
        
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
        result = _set_dtype(dtype_str)
        # Update global attributes
        try:
            bk = _get_backend(None)
            globals()["dtypestr"] = getattr(bk, "dtypestr", "complex128")
            globals()["rdtypestr"] = getattr(bk, "rdtypestr", "float64")
        except Exception:
            pass
        return result

    # expose in module namespace and __all__
    __all__.extend(["set_backend", "get_backend", "set_dtype"])
except Exception:
    pass

# --- Thin forwarding for missing attributes: backend / rdtypestr / array_to_tensor / dtypestr ---
def __getattr__(name: str):
    """Lazy attribute bridge.

    If `backend`, `rdtypestr`, `array_to_tensor`, or `dtypestr` is not initialized,
    fetch current backend via get_backend(None) and expose these symbols.
    """
    if name in ("backend", "rdtypestr", "dtypestr", "array_to_tensor"):
        try:
            # get current or default backend lazily
            bk = get_backend(None)  # type: ignore[name-defined]
        except Exception:
            bk = None
        
        if name == "backend":
            globals()["backend"] = bk
            # also try to expose rdtypestr along the way
            try:
                globals()["rdtypestr"] = getattr(bk, "rdtypestr", "float64")
            except Exception:
                globals()["rdtypestr"] = "float64"
            # expose dtypestr
            try:
                globals()["dtypestr"] = getattr(bk, "dtypestr", "complex128")
            except Exception:
                globals()["dtypestr"] = "complex128"
            return bk
        
        if name == "rdtypestr":
            try:
                rd = getattr(bk, "rdtypestr", "float64")
            except Exception:
                rd = "float64"
            globals()["rdtypestr"] = rd
            return rd
        
        if name == "dtypestr":
            try:
                dt = getattr(bk, "dtypestr", "complex128")
            except Exception:
                dt = "complex128"
            globals()["dtypestr"] = dt
            return dt
        
        if name == "array_to_tensor":
            # Expose array_to_tensor from NumericBackend
            try:
                from .numerics import NumericBackend as _NB
                array_to_tensor_fn = _NB.array_to_tensor
                globals()["array_to_tensor"] = array_to_tensor_fn
                return array_to_tensor_fn
            except Exception:
                raise AttributeError(f"module '{__name__}' could not load 'array_to_tensor'")
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# --- Top-level noise controls (simulator) ---
try:
    from .devices import base as _devbase  # type: ignore

    def enable_noise(enabled: bool = True, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Globally enable/disable simulator noise and optionally set default config.

        Example:
            tyxonq.enable_noise(True, {"type": "depolarizing", "p": 0.01})
        """

        return _devbase.enable_noise(enabled=enabled, config=config)

    def is_noise_enabled() -> bool:
        return _devbase.is_noise_enabled()

    def get_noise_config() -> Dict[str, Any]:
        return _devbase.get_noise_config()

    __all__.extend(["enable_noise", "is_noise_enabled", "get_noise_config"])
except Exception:
    pass
