from __future__ import annotations

from typing import Any
from tyxonq.core.ir.circuit import Circuit


def get_circuit_summary(circuit: Circuit):
    """Thin wrapper to Circuit.get_circuit_summary()."""
    return circuit.get_circuit_summary()


def count_circuit_flop(circuit: Circuit):
    """Thin wrapper to Circuit.count_flop()."""
    return circuit.count_flop()


__all__ = ["get_circuit_summary", "count_circuit_flop"]


