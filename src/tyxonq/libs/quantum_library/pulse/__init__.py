"""
Pulse optimization module for TyxonQ.

This module provides advanced pulse optimization capabilities including:
- PyTorch autograd integration for gradient-based optimization
- GRAPE (Gradient Ascent Pulse Engineering) algorithm
- Pulse VQE and Pulse QAOA support
- Integration with three-level systems and ZZ crosstalk models

Submodules:
    autograd: PyTorch autograd support for pulse parameter optimization
    grape: GRAPE optimizer for high-fidelity gate synthesis (future)
"""

from .autograd import (
    DifferentiablePulseSimulation,
    optimize_pulse_parameters
)

__all__ = [
    "DifferentiablePulseSimulation",
    "optimize_pulse_parameters",
]
