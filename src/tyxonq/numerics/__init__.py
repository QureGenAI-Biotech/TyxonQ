"""Numerics backends and vectorization utilities."""

from .api import ArrayBackend, VectorizationPolicy, vectorize_or_fallback, get_backend
from .context import set_backend, use_backend

__all__ = [
    "ArrayBackend",
    "VectorizationPolicy",
    "vectorize_or_fallback",
    "get_backend",
    "set_backend",
    "use_backend",
]


