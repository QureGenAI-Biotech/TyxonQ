"""TyxonQ package initializer (refactor minimal).

Lightweight exports for cloud API convenience while preserving side-effect
constraints. Heavy modules are not imported here.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import sys as _sys
import importlib as _importlib

__version__ = "0.5.0"
__author__ = "TyxonQ Authors"

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
        provider: Optional[Union[str, Any]] = None,
        device: Optional[Union[str, Any]] = None,
        cached: bool = True,
        clear: bool = False,
    ) -> Dict[str, Any]:
        """Set API token at package level.

        Delegates to ``tyxonq.cloud.apis.set_token``.
        """

        return _cloud_apis.set_token(  # type: ignore[attr-defined]
            token=token,
            provider=provider,
            device=device,
            cached=cached,
            clear=clear,
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

        return _cloud_apis.submit_task(  # type: ignore[attr-defined]
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

        tasks = _cloud_apis.submit_task(  # type: ignore[attr-defined]
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

        results = [t.results(blocked=True, mitigated=mitigated) for t in task_list]
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
try:  # best-effort; keep import robust if chem is not present
    from .applications import chem as _chem_app  # type: ignore
    # Package-level alias
    _sys.modules[__name__ + ".chem"] = _chem_app
    # Common submodules to maintain dotted imports like `tyxonq.chem.static.uccsd`
    for _sub in ("constants", "molecule", "dynamic", "dynamic.model", "static", "utils"):
        try:
            _m = _importlib.import_module(f"tyxonq.applications.chem.{_sub}")
            _sys.modules[f"tyxonq.chem.{_sub}"] = _m
        except Exception:
            pass
except Exception:
    pass
