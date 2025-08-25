from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal
from abc import ABCMeta


# ---- Core constants ----
# Package name for global lookups and integration points
PACKAGE_NAME: str = "tyxonq"

# Default numeric dtype strings for new architecture (complex64/float32)
DEFAULT_COMPLEX_DTYPE_STR: str = "complex64"
DEFAULT_REAL_DTYPE_STR: str = "float32"

# Canonical backend names supported by numerics
SUPPORTED_BACKENDS: tuple[str, ...] = ("numpy", "pytorch", "cupynumeric")


# Lightweight runtime string subtypes for readability and isinstance checks
class BackendName(str, metaclass=ABCMeta):
    """Type for backend names (e.g., 'numpy', 'torch', 'cunumeric')."""

    @classmethod
    def __instancecheck__(cls, instance: Any) -> bool:
        # Treat any string as a valid backend name at runtime
        return isinstance(instance, str)


class VectorizationPolicy(str, metaclass=ABCMeta):  # "auto" | "force" | "off"
    """Type for vectorization policy indicators."""

    @classmethod
    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, str)


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
    """Return default complex/real dtype strings for numerics.

    This is a stable source of truth for dtype defaults used across the
    refactored architecture.
    """
    return DEFAULT_COMPLEX_DTYPE_STR, DEFAULT_REAL_DTYPE_STR


@dataclass(frozen=True)
class Problem:
    """Domain problem wrapper for input to app/compilers.

    Fields:
        kind: Problem category (e.g., "hamiltonian", "circuit").
        payload: Arbitrary structured data describing the problem.
    """

    kind: Literal["hamiltonian", "circuit", "pulse", "custom"]
    payload: Dict[str, Any]


