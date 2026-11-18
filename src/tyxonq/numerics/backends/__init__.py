"""Array backends and shared device data structures."""

from __future__ import annotations

from .base import BackendCapabilities, BackendUnavailableError, MemoryInfo
from .numpy_backend import NumpyBackend

__all__ = [
    "BackendCapabilities",
    "BackendUnavailableError",
    "MemoryInfo",
    "NumpyBackend",
]
