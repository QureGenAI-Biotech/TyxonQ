from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from ...core.ir import Circuit


def _apply_single_term(c: Circuit, ps: Sequence[int], theta: float) -> Circuit:
    """Apply exp(-i theta P) via native gates for Pauli strings.

    Supported patterns per qubit code: 0=I, 1=X, 2=Y, 3=Z
    - Single-qubit X: H-RZ-H
    - Single-qubit Y: S†-H-RZ-H-S
    - Single-qubit Z: RZ
    - Two-qubit XX, YY, ZZ: basis transformation + CX-RZ-CX
    - Multi-qubit Pauli strings: general decomposition

    Strategy:
    1. Transform all non-Z Paulis to Z basis via basis rotations
    2. Apply ZZ...Z interaction via CNOT ladder
    3. Undo basis transformations
    """

    n = len(ps)
    # Find non-identity qubits
    nz: List[int] = [i for i, v in enumerate(ps) if v != 0]
    if not nz:
        return c
    
    # Single-qubit case
    if len(nz) == 1:
        q = nz[0]
        if ps[q] == 1:  # X
            return c.h(q).rz(q, 2.0 * theta).h(q)
        elif ps[q] == 2:  # Y
            # Y = S† X S, so exp(-iθY) = S† exp(-iθX) S = S† H RZ H S
            return c.sdg(q).h(q).rz(q, 2.0 * theta).h(q).s(q)
        elif ps[q] == 3:  # Z
            return c.rz(q, 2.0 * theta)
        return c
    
    # Multi-qubit case: transform to Z basis, apply ZZ...Z, transform back
    # Step 1: Basis transformations to map all Paulis to Z
    for q in nz:
        if ps[q] == 1:  # X -> Z via H
            c.h(q)
        elif ps[q] == 2:  # Y -> Z via S† H
            c.sdg(q)
            c.h(q)
        # Z stays as Z (no transformation needed)
    
    # Step 2: Apply exp(-i theta Z⊗Z⊗...⊗Z) via CNOT ladder
    # CNOT ladder: propagate parity to last qubit, rotate, then undo
    for i in range(len(nz) - 1):
        c.cx(nz[i], nz[i + 1])
    
    # Rotation on last qubit
    c.rz(nz[-1], 2.0 * theta)
    
    # Undo CNOT ladder
    for i in range(len(nz) - 2, -1, -1):
        c.cx(nz[i], nz[i + 1])
    
    # Step 3: Undo basis transformations
    for q in reversed(nz):  # Reverse order for proper unwinding
        if ps[q] == 1:  # Z -> X via H
            c.h(q)
        elif ps[q] == 2:  # Z -> Y via H S
            c.h(q)
            c.s(q)
    
    return c


def build_trotter_circuit(
    pauli_terms: Sequence[Sequence[int]] | Any,
    *,
    weights: Sequence[float] | None = None,
    time: float,
    steps: int,
    num_qubits: int | None = None,
    order: str = "first",
) -> Circuit:
    """Construct a first-order Trotterized circuit for H = sum_j w_j P_j.

    Parameters
    ----------
    pauli_terms: list[list[int]]
        Each P_j encoded as length-n with entries in {0,1,2,3} for I,X,Y,Z.
    weights: list[float] | None
        Coefficients w_j. Defaults to 1.0 for all.
    time: float
        Evolution time t.
    steps: int
        Number of Trotter steps.
    num_qubits: Optional[int]
        Number of qubits (required if cannot infer from terms).
    order: str
        "first" (only supported currently).
    """

    if not isinstance(pauli_terms, (list, tuple)):
        raise NotImplementedError("Dense Hamiltonian input not yet supported; pass Pauli term list instead")
    if not pauli_terms:
        n = int(num_qubits or 0)
        return Circuit(n)
    n = int(num_qubits or len(pauli_terms[0]))
    w = list(weights) if weights is not None else [1.0] * len(pauli_terms)
    dt = float(time) / float(max(1, int(steps)))

    c = Circuit(n)
    if order != "first":
        raise NotImplementedError("Only first-order Trotter is supported in this template")

    for _ in range(max(1, int(steps))):
        for ps, coeff in zip(pauli_terms, w):
            theta = float(coeff) * dt
            c = _apply_single_term(c, ps, theta)
    # By default add Z measurements on all qubits; Circuit.run will also auto-add if absent
    for q in range(n):
        c = c.measure_z(q)
    return c


__all__ = ["build_trotter_circuit"]


