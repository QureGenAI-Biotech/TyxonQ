"""Selected Basis Diagonalization (SBD), a first-class numerics algorithm.

SBD is not a backend: it is an eigensolver used by SQD and other selected-CI
workflows. CUDA kernels and the explicit build remain opt-in.
"""

from __future__ import annotations

from .metal import MetalSelectedCIContractor, MetalSelectedS2Contractor
from .metal import davidson_lowest as solve_metal_davidson
from .cuda import (
    SBDContractor,
    convert_raw_rdms,
    interleaved_phase,
    is_available,
    load_extension,
    normalize_problem,
    solve_selected_basis,
)

__all__ = [
    "MetalSelectedCIContractor",
    "MetalSelectedS2Contractor",
    "SBDContractor",
    "convert_raw_rdms",
    "interleaved_phase",
    "is_available",
    "load_extension",
    "normalize_problem",
    "solve_selected_basis",
    "solve_metal_davidson",
]
