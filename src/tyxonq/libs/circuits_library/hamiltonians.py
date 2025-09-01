"""Reusable Hamiltonian builders and converters.

Representation:
- Hamiltonian is a list of (coefficient, [(Pauli, qubit), ...])
- Example: [(0.5, [("Z", 0)]), (0.7, [("Z", 0), ("Z", 1)])]

This module avoids heavy numeric deps; it's just construction helpers.
"""

from __future__ import annotations

from typing import List, Tuple

Hamiltonian = List[Tuple[float, List[Tuple[str, int]]]]


def build_tfim_terms(n: int, hzz: float = 1.0, hx: float = -1.0, pbc: bool = False) -> Hamiltonian:
    """Build TFIM Hamiltonian terms: hzz * sum Z_i Z_{i+1} + hx * sum X_i."""
    terms: Hamiltonian = []
    for i in range(n - 1):
        terms.append((hzz, [("Z", i), ("Z", i + 1)]))
    if pbc and n > 1:
        terms.append((hzz, [("Z", n - 1), ("Z", 0)]))
    for i in range(n):
        terms.append((hx, [("X", i)]))
    return terms


def pauli_terms_from_openfermion(qubit_operator) -> Tuple[List[List[int]], List[float], int]:
    """Convert OpenFermion QubitOperator to (lsb, wb, n_qubits).

    - lsb: list of length n lists, each with 0(I)/1(X)/2(Y)/3(Z)
    - wb: list of real weights (float)
    - n_qubits: number of qubits inferred
    """
    # QubitOperator.terms is a dict: {((q0,'X'), (q1,'Z'), ...): coeff}
    terms = []; weights = []; max_idx = -1
    for term, coeff in getattr(qubit_operator, "terms").items():
        if term:
            max_idx = max(max_idx, max(q for q, _ in term))
        terms.append(term)
        # keep real part; imaginary parts should cancel in Hermitian ops
        weights.append(float(getattr(coeff, "real", coeff.real if hasattr(coeff, "real") else coeff)))
    n_qubits = max_idx + 1 if max_idx >= 0 else 0

    lsb: List[List[int]] = []
    wb: List[float] = []
    for term, coeff in zip(terms, weights):
        codes = [0] * n_qubits
        for q, p in term:
            if p == 'X':
                codes[q] = 1
            elif p == 'Y':
                codes[q] = 2
            elif p == 'Z':
                codes[q] = 3
        lsb.append(codes)
        wb.append(float(coeff))
    return lsb, wb, n_qubits
