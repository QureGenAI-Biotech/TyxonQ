"""Density matrix simulator package.

Use `DensityMatrixEngine` to simulate mixed states via a dense density matrix.
It natively supports Kraus noise channels and is suitable for noise studies.
"""

from .engine import DensityMatrixEngine

__all__ = ["DensityMatrixEngine"]


