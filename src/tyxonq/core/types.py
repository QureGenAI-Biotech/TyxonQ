from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal
from abc import ABCMeta


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


@dataclass(frozen=True)
class Problem:
    """Domain problem wrapper for input to app/compilers.

    Fields:
        kind: Problem category (e.g., "hamiltonian", "circuit").
        payload: Arbitrary structured data describing the problem.
    """

    kind: Literal["hamiltonian", "circuit", "pulse", "custom"]
    payload: Dict[str, Any]


