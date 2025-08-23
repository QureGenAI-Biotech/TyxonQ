from __future__ import annotations

"""Global configuration for numerics backends.

Provides a simple global/default backend selection with a context manager.
This avoids plumbing the backend through every API when a project-wide default
is desired, while still allowing per-call/per-instance overrides.
"""

from contextlib import contextmanager
from typing import Any, Generator, Optional

# Store either a configured backend instance or a name to be resolved by api.get_backend
_CURRENT_BACKEND_INSTANCE: Optional[Any] = None
_CURRENT_BACKEND_NAME: Optional[str] = None


def set_backend(name_or_instance: Any) -> None:
    """Set the global/default backend by name or instance.

    Passing a string sets the backend name to be lazily resolved.
    Passing an instance pins the exact backend object to use.
    """
    global _CURRENT_BACKEND_INSTANCE, _CURRENT_BACKEND_NAME
    if isinstance(name_or_instance, str):
        _CURRENT_BACKEND_NAME = name_or_instance
        _CURRENT_BACKEND_INSTANCE = None
    else:
        _CURRENT_BACKEND_INSTANCE = name_or_instance
        _CURRENT_BACKEND_NAME = None


def get_configured_backend_instance() -> Optional[Any]:
    return _CURRENT_BACKEND_INSTANCE


def get_configured_backend_name() -> Optional[str]:
    return _CURRENT_BACKEND_NAME


@contextmanager
def use_backend(name_or_instance: Any) -> Generator[None, None, None]:
    """Temporarily set the global backend within a context."""
    global _CURRENT_BACKEND_INSTANCE, _CURRENT_BACKEND_NAME
    prev_inst, prev_name = _CURRENT_BACKEND_INSTANCE, _CURRENT_BACKEND_NAME
    try:
        set_backend(name_or_instance)
        yield
    finally:
        _CURRENT_BACKEND_INSTANCE, _CURRENT_BACKEND_NAME = prev_inst, prev_name


