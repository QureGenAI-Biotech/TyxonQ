"""Core package exposing IR, operations, and measurement primitives.

At this stage of the refactor, only `ir` is provided. Operations and
measurements will be added in subsequent increments.
"""

from .ir import Circuit, Hamiltonian

__all__ = ["Circuit", "Hamiltonian"]


