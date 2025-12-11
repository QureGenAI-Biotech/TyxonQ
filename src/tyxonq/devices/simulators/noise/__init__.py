"""Noise channel interface for simulators.

This module provides backward compatibility by re-exporting noise channels
from the core quantum library. The actual implementations are in:
    libs.quantum_library.noise (standard channels)
    libs.quantum_library.kernels.* (apply functions)

For new code, import directly from libs.quantum_library.noise.
"""

from __future__ import annotations

# Re-export standard noise channels from core library
from ....libs.quantum_library.noise import (
    depolarizing_channel,
    amplitude_damping_channel,
    phase_damping_channel,
    pauli_channel,
    measurement_channel,
)

__all__ = [
    # Standard noise channels
    "depolarizing_channel",
    "amplitude_damping_channel",
    "phase_damping_channel",
    "pauli_channel",
    "measurement_channel",
]
