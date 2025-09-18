from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from tyxonq.core.ir import Circuit


def qaoa_ising(
    num_qubits: int,
    nlayers: int,
    pauli_z_terms: Sequence[Sequence[int]],
    weights: Sequence[float],
    params: Sequence[float],
    mixer: str = "X",
    full_coupling: bool = False,
) -> Circuit:
    """Build QAOA(Ising) ansatz as IR ops.

    - cost layer: Z rotations and ZZ (as RZZ) recorded in ops; decompose stage will lower them.
    - mixer: X/XY/ZZ using RX/RXX/RYX variants; decompose stage handles to {H,RZ,CX}.
    """

    ops: list[tuple[Any, ...]] = []
    for q in range(num_qubits):
        ops.append(("h", q))

    for j in range(nlayers):
        # cost terms
        for k, term in enumerate(pauli_z_terms):
            one_indices: list[int] = [i for i, v in enumerate(term) if int(v) == 1]
            if len(one_indices) == 1:
                q = one_indices[0]
                ops.append(("rz", q, 2.0 * float(weights[k]) * float(params[2 * j])))
            elif len(one_indices) == 2:
                q0, q1 = one_indices
                ops.append(("rzz", q0, q1, float(weights[k]) * float(params[2 * j])))
            else:
                raise ValueError("Invalid number of Z terms for QAOA Ising")

        # mixer pairs
        pairs: list[Tuple[int, int]] = []
        if not full_coupling:
            pairs = [(q0, q0 + 1) for q0 in list(range(0, num_qubits - 1, 2)) + list(range(1, num_qubits - 1, 2))]
            if num_qubits > 1:
                pairs.append((num_qubits - 1, 0))
        else:
            for q0 in range(num_qubits - 1):
                for q1 in range(q0 + 1, num_qubits):
                    pairs.append((q0, q1))

        theta_m = float(params[2 * j + 1])
        m = mixer.upper()
        if m == "X":
            for q in range(num_qubits):
                ops.append(("rx", q, theta_m))
        elif m == "XY":
            for q0, q1 in pairs:
                ops.append(("rxx", q0, q1, theta_m))
                ops.append(("ryy", q0, q1, theta_m))
        elif m == "ZZ":
            for q0, q1 in pairs:
                ops.append(("rzz", q0, q1, theta_m))
        else:
            raise ValueError("Invalid mixer type")

    return Circuit(num_qubits=num_qubits, ops=ops)


