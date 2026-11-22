"""Backward-compatible function aliases for the pre-refactor GPU API.

Module paths with a ``torch_`` prefix were removed; these callable aliases stay
so existing scripts can switch to the unified entry point gradually.
"""

from __future__ import annotations

from .batch import make_cuda_sci_solver, make_sbd_sci_solver
from .sbd_solver import SBDContractor, solve_sci_sbd
from .solver import solve_sci_cuda, wait_for_davidson_memory

__all__ = [
    "SBDContractor",
    "make_cuda_sci_solver",
    "make_sbd_sci_solver",
    "solve_sci_cuda",
    "solve_sci_sbd",
    "wait_for_davidson_memory",
]
