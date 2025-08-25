"""Deprecated: use tyxonq.libs.quantum_library.dynamics

This module re-exports symbols for backwards compatibility.
"""

from .dynamics import (
    PauliStringSum2COO,
    evolve_state_numeric,
    expval_dense,
)

__all__ = ["PauliStringSum2COO", "evolve_state_numeric", "expval_dense"]


