"""Legacy noise channel interface.

This module provides backward compatibility for old noise simulation code.
New code should import from:
    - tyxonq.libs.quantum_library.noise (standard channels)
    - tyxonq.libs.quantum_library.kernels.density_matrix (apply functions)
"""

from __future__ import annotations

from typing import List
import numpy as np

# Import standard channels from core library
from ....libs.quantum_library.noise import (
    depolarizing_channel,
    amplitude_damping_channel,
    phase_damping_channel,
    pauli_channel as pauli_channel_impl,
)


# Legacy function names for backward compatibility
def depolarizing(p: float) -> List[np.ndarray]:
    """Legacy wrapper for depolarizing_channel."""
    return depolarizing_channel(p)


def amplitude_damping(gamma: float) -> List[np.ndarray]:
    """Legacy wrapper for amplitude_damping_channel."""
    return amplitude_damping_channel(gamma)


def phase_damping(lmbda: float) -> List[np.ndarray]:
    """Legacy wrapper for phase_damping_channel."""
    return phase_damping_channel(lmbda)


def pauli_channel(px: float, py: float, pz: float) -> List[np.ndarray]:
    """Legacy wrapper for pauli_channel."""
    return pauli_channel_impl(px, py, pz)


