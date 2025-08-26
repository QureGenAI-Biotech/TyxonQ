from __future__ import annotations

"""Global configuration and normalization utilities for TyxonQ.

This module centralizes:
- Package-level constants and defaults
- Backend name normalization and vectorization policy checks
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
SUPPORTED_BACKENDS: tuple[str, ...] = ("numpy", "pytorch", "cupynumeric")


# ---- Lightweight runtime string subtypes for readability ----

class BackendName(str, metaclass=ABCMeta):
    """Type for backend names (e.g., 'numpy', 'pytorch', 'cupynumeric')."""

    @classmethod
    def __instancecheck__(cls, instance: Any) -> bool:  # pragma: no cover
        return isinstance(instance, str)


class VectorizationPolicy(str, metaclass=ABCMeta):  # "auto" | "force" | "off"
    """Type for vectorization policy indicators."""

    @classmethod
    def __instancecheck__(cls, instance: Any) -> bool:  # pragma: no cover
        return isinstance(instance, str)


# ---- Normalizers and helpers ----

def normalize_backend_name(name: str) -> BackendName:
    """Normalize user/backend alias to canonical backend name.

    Canonical names: 'numpy', 'cupynumeric', 'pytorch'
    Aliases:
        - 'cpu' -> 'numpy'
        - 'gpu' -> 'cupynumeric'
        - 'torch', 'pt' -> 'pytorch'
        - 'numpy(cpu)' -> 'numpy'
        - 'cupynumeric(gpu)' -> 'cupynumeric'
    """

    s = name.strip().lower()
    if s in {"cpu", "numpy", "numpy(cpu)"}:
        return BackendName("numpy")
    if s in {"gpu", "cupynumeric", "cupynumeric(gpu)"}:
        return BackendName("cupynumeric")
    if s in {"torch", "pt", "pytorch"}:
        return BackendName("pytorch")
    return BackendName(s)


def is_valid_vectorization_policy(value: str) -> bool:
    """Return True if value is a supported vectorization policy."""

    return value in {"auto", "force", "off"}


def default_dtypes() -> tuple[str, str]:
    """Return default complex/real dtype strings for numerics."""

    return DEFAULT_COMPLEX_DTYPE_STR, DEFAULT_REAL_DTYPE_STR


__all__ = [
    "PACKAGE_NAME",
    "DEFAULT_COMPLEX_DTYPE_STR",
    "DEFAULT_REAL_DTYPE_STR",
    "SUPPORTED_BACKENDS",
    "BackendName",
    "VectorizationPolicy",
    "normalize_backend_name",
    "is_valid_vectorization_policy",
    "default_dtypes",
]


