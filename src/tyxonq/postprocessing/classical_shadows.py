from __future__ import annotations

"""Classical shadows helpers (postprocessing layer).

Lightweight utilities for generating random Pauli bases and estimating Z
expectation values from classical counts. This belongs to the postprocessing
layer as part of analysis flows after execution.
"""

from typing import Dict, List, Sequence
import random


def random_pauli_basis(num_qubits: int, *, include_i: bool = False, seed: int | None = None) -> List[str]:
    """Generate a random single-qubit Pauli basis for each qubit.

    Returns a list of length ``num_qubits`` with entries from {X, Y, Z} by default,
    or {I, X, Y, Z} if ``include_i=True``.
    """
    rng = random.Random(seed)
    choices = ["I", "X", "Y", "Z"] if include_i else ["X", "Y", "Z"]
    return [rng.choice(choices) for _ in range(num_qubits)]


def estimate_z_from_counts(counts: Dict[str, int], qubit: int) -> float:
    """Estimate Z expectation value on a qubit from bitstring counts.

    Assumes bitstrings are composed of '0'/'1' with qubit indexing by position.
    Rightmost bit is qubit 0.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    z_sum = 0
    for bitstr, c in counts.items():
        bit = bitstr[-(qubit + 1)]
        z_sum += c if bit == "0" else -c
    return float(z_sum) / float(total)


def random_pauli_bases(num_qubits: int, num_shots: int, *, include_i: bool = False, seed: int | None = None) -> List[List[str]]:
    """Generate ``num_shots`` random bases (one per shot)."""
    rng = random.Random(seed)
    choices = ["I", "X", "Y", "Z"] if include_i else ["X", "Y", "Z"]
    return [[rng.choice(choices) for _ in range(num_qubits)] for _ in range(num_shots)]


def bitstrings_to_bits(bitstrings: Sequence[str], num_qubits: int) -> List[List[int]]:
    """Convert list of little-endian bitstrings to per-shot bit arrays.

    Rightmost bit corresponds to qubit 0.
    """
    out: List[List[int]] = []
    for s in bitstrings:
        s = s.strip()
        if len(s) != num_qubits:
            # pad left if needed
            s = s.rjust(num_qubits, "0")
        out.append([int(b) for b in reversed(s)])
    return out


def estimate_expectation_pauli_product(
    num_qubits: int,
    pauli_ops: Dict[int, str],
    bases: Sequence[Sequence[str]],
    outcomes: Sequence[Sequence[int]],
) -> float:
    """Estimate expectation of a Pauli product using classical shadows.

    Parameters
    ----------
    num_qubits:
        Number of qubits.
    pauli_ops:
        Mapping qubit index -> one of {"I","X","Y","Z"}. "I" may be omitted.
    bases:
        For each shot, the measurement basis per qubit.
    outcomes:
        For each shot, the measured 0/1 outcome per qubit (little-endian: index 0 is qubit 0).

    Returns
    -------
    float
        Estimated expectation value.
    """
    if not bases:
        return 0.0
    shots = len(bases)
    total = 0.0
    for s in range(shots):
        contrib = 1.0
        for q, P in pauli_ops.items():
            if P == "I":
                continue
            if bases[s][q] != P:
                contrib = 0.0
                break
            bit = outcomes[s][q]
            sign = +1.0 if bit == 0 else -1.0
            contrib *= sign
        total += contrib
    return total / float(shots)


__all__ = [
    "random_pauli_basis",
    "estimate_z_from_counts",
    "random_pauli_bases",
    "bitstrings_to_bits",
    "estimate_expectation_pauli_product",
]


