from __future__ import annotations

"""Global configuration and normalization utilities for TyxonQ.

This module centralizes:
- Package-level constants and defaults
- Backend name normalization
- Default dtype strings for numerics

It is intentionally lightweight and free of heavy dependencies so that it can
be imported early by subsystems (core, numerics, devices) without side effects.
"""

from typing import Any
from abc import ABCMeta

# ---- Core constants ----

# Package canonical name
PACKAGE_NAME: str = "tyxonq"

# Default numeric dtype strings for new architecture (complex64/float32)
DEFAULT_COMPLEX_DTYPE_STR: str = "complex64"
DEFAULT_REAL_DTYPE_STR: str = "float32"

# Canonical backend names supported by numerics
SUPPORTED_BACKENDS: tuple[str, ...] = ("numpy", "pytorch", "cupy")


# ---- Lightweight runtime string subtypes for readability ----

class BackendName(str, metaclass=ABCMeta):
    """Type for backend names (e.g., 'numpy', 'pytorch', 'cupy')."""

    @classmethod
    def __instancecheck__(cls, instance: Any) -> bool:  # pragma: no cover
        return isinstance(instance, str)


# ---- Normalizers and helpers ----

def normalize_backend_name(name: str) -> BackendName:
    """Normalize user/backend alias to canonical backend name.

    Canonical names: 'numpy', 'cupy', 'pytorch'
    Aliases:
        - 'cpu' -> 'numpy'
        - 'gpu' -> 'cupy'
        - 'torch', 'pt' -> 'pytorch'
        - 'numpy(cpu)' -> 'numpy'
        - 'cupy(gpu)' -> 'cupy'
    """

    s = name.strip().lower()
    if s in {"cpu", "numpy", "numpy(cpu)"}:
        return BackendName("numpy")
    if s in {"gpu", "cupy", "cupy(gpu)"}:
        return BackendName("cupy")
    if s in {"torch", "pt", "pytorch"}:
        return BackendName("pytorch")
    return BackendName(s)


def default_dtypes() -> tuple[str, str]:
    """Return default complex/real dtype strings for numerics."""

    return DEFAULT_COMPLEX_DTYPE_STR, DEFAULT_REAL_DTYPE_STR


__all__ = [
    "PACKAGE_NAME",
    "DEFAULT_COMPLEX_DTYPE_STR",
    "DEFAULT_REAL_DTYPE_STR",
    "SUPPORTED_BACKENDS",
    "BackendName",
    "normalize_backend_name",
    "default_dtypes",
]


