from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Public helpers
from .unitary import get_unitary  # noqa: F401


@dataclass(frozen=True)
class GateSpec:
    """Metadata describing a gate/operator.

    Fields:
        name: Canonical gate name (e.g., "h", "cx").
        num_qubits: Arity of the operation.
        generator: Optional generator information for gradients.
        differentiable: Whether parameterized variants are differentiable.
    """

    name: str
    num_qubits: int
    generator: Optional[Any] = None
    differentiable: bool = True
    # Gradient-related metadata
    num_params: int = 0
    is_shiftable: bool = False
    shift_coeffs: Optional[Tuple[float, ...]] = None
    gradient_method: Optional[str] = None  # e.g., "parameter-shift", "finite-diff", "adjoint"


@dataclass(frozen=True)
class Operation:
    """A placed operation referring to a gate by name on specific wires."""

    name: str
    wires: Tuple[int, ...]
    params: Tuple[Any, ...] = ()


class _Registry:
    """Simple registry of gate specifications by name."""

    def __init__(self) -> None:
        self._registry: Dict[str, GateSpec] = {}

    def register(self, spec: GateSpec) -> None:
        self._registry[spec.name] = spec

    def get(self, name: str) -> Optional[GateSpec]:
        return self._registry.get(name)

    def clear(self) -> None:
        self._registry.clear()


registry = _Registry()

__all__ = ["GateSpec", "Operation", "registry", "get_unitary"]


# Register default gate specs aligned with migration plan and common dialect

def _register_defaults() -> None:
    if registry.get("h") is None:
        registry.register(GateSpec(name="h", num_qubits=1, differentiable=False))
    if registry.get("rz") is None:
        registry.register(
            GateSpec(
                name="rz",
                num_qubits=1,
                generator="Z",
                differentiable=True,
                num_params=1,
                is_shiftable=True,
                shift_coeffs=(0.5,),
                gradient_method="parameter-shift",
            )
        )
    if registry.get("cx") is None:
        registry.register(GateSpec(name="cx", num_qubits=2, differentiable=False))


_register_defaults()


