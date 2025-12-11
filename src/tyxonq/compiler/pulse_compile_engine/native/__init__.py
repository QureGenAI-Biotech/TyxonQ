"""TyxonQ native pulse compiler implementation.

This module implements TyxonQ's native pulse-level compilation pipeline,
providing gate-to-pulse decomposition, calibration management, and pulse scheduling.

Key Components:
    - PulseCompiler: Main compiler class
    - GateToPulsePass: Decompose standard gates to pulse sequences
    - PulseLoweringPass: Inline defcal (pulse calibration) definitions
    - PulseSchedulingPass: Optimize pulse timing and scheduling

References:
    - QuTiP-qip: Physics model reference (Quantum 6, 630, 2022)
"""

from __future__ import annotations

from .pulse_compiler import PulseCompiler
from .gate_to_pulse import GateToPulsePass
from .pulse_lowering import PulseLoweringPass
from .pulse_scheduling import PulseSchedulingPass

__all__ = [
    "PulseCompiler",
    "GateToPulsePass",
    "PulseLoweringPass",
    "PulseSchedulingPass",
]
