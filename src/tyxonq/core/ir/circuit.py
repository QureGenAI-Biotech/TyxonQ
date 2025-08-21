from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Dict


@dataclass
class Circuit:
    """Minimal intermediate representation (IR) for a quantum circuit.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        ops: A sequence of operation descriptors. The concrete type is left
            open for backends/compilers to interpret (e.g., gate tuples, IR
            node objects). Keeping this generic allows the IR to evolve while
            tests exercise the structural contract.
    """

    num_qubits: int
    ops: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hamiltonian:
    """IR for a Hamiltonian.

    The `terms` field may contain a backend-specific structure, such as a
    Pauli-sum, sparse representation, or dense matrix. The type is intentionally
    loose at this stage and will be specialized by compiler stages or devices.
    """

    terms: Any


