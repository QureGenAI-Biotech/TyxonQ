"""Pulse-level compilation engine for TyxonQ.

This module provides pulse-level compilation capabilities, converting gate-level
circuits into pulse sequences for hardware execution. This is a core differentiating
feature of TyxonQ compared to traditional quantum frameworks.

Architecture:
    - native/: TyxonQ's native pulse compiler implementation
    - (future) ibm/: IBM Pulse specification support
    - (future) rigetti/: Rigetti Pulse specification support

Dual-mode support (per Memory 8b12df21):
    - Mode A (Chain): Gate Circuit → Pulse Sequence → Execution
    - Mode B (Direct): Hamiltonian → Direct Pulse Evolution (see libs.quantum_library.pulse_simulation)

Example:
    >>> from tyxonq.compiler.pulse_compile_engine.native import PulseCompiler
    >>> compiler = PulseCompiler()
    >>> pulse_circuit = compiler.compile(gate_circuit)
"""

from __future__ import annotations

__all__ = [
    "PulseCompiler",
    "DefcalLibrary",
    "save_pulse_circuit",
    "load_pulse_circuit",
    "serialize_pulse_circuit_to_json",
    "deserialize_pulse_circuit_from_json",
]

# Lazy import to avoid circular dependencies
def __getattr__(name: str):
    if name == "PulseCompiler":
        from .native.pulse_compiler import PulseCompiler
        return PulseCompiler
    elif name == "DefcalLibrary":
        from .defcal_library import DefcalLibrary
        return DefcalLibrary
    elif name in ("save_pulse_circuit", "load_pulse_circuit",
                  "serialize_pulse_circuit_to_json", "deserialize_pulse_circuit_from_json"):
        from .serialization import (
            save_pulse_circuit,
            load_pulse_circuit,
            serialize_pulse_circuit_to_json,
            deserialize_pulse_circuit_from_json
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
