from __future__ import annotations

from typing import Any, List, Tuple

from .circuit import Circuit


class CircuitBuilder:
    """Lightweight circuit recorder for user-facing qfunc-style building.

    Usage:
        with CircuitBuilder(num_qubits=2) as cb:
            cb.h(0)
            cb.cx(0, 1)
            cb.measure_z(1)
        circuit = cb.circuit()
    """

    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = int(num_qubits)
        self._ops: List[Tuple[Any, ...]] = []
        self._closed = False

    # Context manager protocol
    def __enter__(self) -> "CircuitBuilder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._closed = True

    # Recording
    def _append(self, name: str, *args: Any) -> None:
        self._ops.append((name,) + args)

    # Common gate helpers (aligned with defaults)
    def h(self, q: int) -> None:
        self._append("h", q)

    def rz(self, q: int, theta: Any) -> None:
        self._append("rz", q, theta)

    def cx(self, c: int, t: int) -> None:
        self._append("cx", c, t)

    # Measurement helpers
    def measure_z(self, q: int) -> None:
        self._append("measure_z", q)

    # Finalize
    def circuit(self) -> Circuit:
        return Circuit(num_qubits=self._num_qubits, ops=list(self._ops))


