from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from ...core.ir import Circuit


def _apply_single_term(c: Circuit, ps: Sequence[int], theta: float) -> Circuit:
    """Apply exp(-i theta P) via native gates for limited Pauli patterns.

    Supported patterns per qubit code: 0=I, 1=X, 2=Y, 3=Z
    - Single-qubit Z: RZ
    - Single-qubit X: H-RZ-H
    - Two-qubit ZZ: CX-RZ-CX

    This is a minimal template to get examples running; it can be extended.
    """

    n = len(ps)
    # count non-identity qubits
    nz: List[int] = [i for i, v in enumerate(ps) if v != 0]
    if not nz:
        return c
    if len(nz) == 1:
        q = nz[0]
        if ps[q] == 3:  # Z
            return c.rz(q, 2.0 * theta)  # exp(-i theta Z) = RZ(2 theta)
        if ps[q] == 1:  # X
            return c.h(q).rz(q, 2.0 * theta).h(q)
        # Y: S^† H RZ H S (not implemented)
        raise NotImplementedError("Single-qubit Y rotation not yet implemented in trotter template")
    if len(nz) == 2 and ps[nz[0]] == 3 and ps[nz[1]] == 3:
        a, b = nz
        # exp(-i theta Z.Z) = CX(a->b) RZ(2 theta) on b then CX
        return c.cx(a, b).rz(b, 2.0 * theta).cx(a, b)
    raise NotImplementedError("Pauli pattern not supported by minimal trotter template")


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


