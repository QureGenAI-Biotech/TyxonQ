from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Type


_CACHE: Dict[str, Any] = {}


def discover(group: str) -> Dict[str, Type[Any]]:
    """Discover extension classes for a given entry point group.

    Placeholder implementation: returns an empty mapping until a concrete
    plugin system is required. This keeps the surface stable for tests and
    documentation.
    """

    return {}


def _load_by_path(path: str) -> Any:
    """Load an object by fully-qualified module path `module:attr` or
    `module.attr`.
    """

    if ":" in path:
        mod_name, attr = path.split(":", 1)
    elif "." in path:
        mod_name, attr = path.rsplit(".", 1)
    else:
        raise ValueError(f"Invalid object path: {path}")
    module = import_module(mod_name)
    return getattr(module, attr)


def get_device(name: str) -> Any:
    """Return a device instance by name.

    For the initial refactor, this function only supports fully-qualified
    class paths and caches the instance. A future version may integrate with
    entry points and configuration files.
    """

    if name in _CACHE:
        return _CACHE[name]
    obj = _load_by_path(name)
    instance = obj() if callable(obj) else obj
    _CACHE[name] = instance
    return instance


def get_compiler(name: str) -> Any:
    """Return a compiler instance by name using the same convention as
    `get_device`.
    """

    if name in _CACHE:
        return _CACHE[name]
    obj = _load_by_path(name)
    instance = obj() if callable(obj) else obj
    _CACHE[name] = instance
    return instance


# Sentinel used by tests to validate dynamic loading and caching behavior.
class _X:  # noqa: N801 (test-only symbol)
    def __call__(self) -> str:
        return "ok"


