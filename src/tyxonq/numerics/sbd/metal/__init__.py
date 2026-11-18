"""Single-device MLX/Metal Selected Basis Diagonalization."""

from .solver import (
    MetalDavidsonResult,
    MetalSelectedCIContractor,
    MetalSelectedS2Contractor,
    absorb_h1e,
    davidson_lowest,
    make_hdiag,
    make_spin_square_diagonal,
    metal_gpu_execution,
)

__all__ = [
    "MetalDavidsonResult",
    "MetalSelectedCIContractor",
    "MetalSelectedS2Contractor",
    "absorb_h1e",
    "davidson_lowest",
    "make_hdiag",
    "make_spin_square_diagonal",
    "metal_gpu_execution",
]
