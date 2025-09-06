from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as _np


def term_expectation_from_counts(counts: Dict[str, int], idxs: Sequence[int]) -> float:
    """Compute ⟨Z^{\otimes idxs}⟩ from bitstring counts.

    Assumes the circuit has already applied basis rotations for X/Y terms
    and measurements are performed in Z basis on all qubits.
    """
    total = sum(counts.values()) or 1
    acc = 0.0
    for bitstr, cnt in counts.items():
        s = 1.0
        for q in idxs:
            s *= (1.0 if bitstr[q] == "0" else -1.0)
        acc += s * cnt
    return acc / total


def _normalize_term_entry(entry: Any) -> Tuple[Tuple[Tuple[int, str], ...], float | None]:
    """Accept (term, coeff) or term-only entries and return (term, coeff?)."""
    if isinstance(entry, tuple) and entry and isinstance(entry[0], tuple):
        # Could be (term, coeff) or (term,) already
        if len(entry) >= 2 and isinstance(entry[1], (int, float)):
            return tuple(entry[0]), float(entry[1])
        return tuple(entry[0]), None
    # Fallback: treat as term only
    return tuple(entry), None  # type: ignore[return-value]


def expval_pauli_term(counts: Dict[str, int], idxs: Sequence[int]) -> float:
    """Thin wrapper for a single Pauli-Z product expectation from counts."""
    return term_expectation_from_counts(counts, idxs)


def expval_pauli_terms(counts: Dict[str, int], terms: Sequence[Any]) -> List[float]:
    """Return expectations for a list of Pauli terms under Z-basis counts.

    Each element in `terms` can be either:
    - term_only: Tuple[(q, letter), ...]
    - (term, coeff): Tuple[Tuple[(q, letter), ...], coeff]
    Coeff is ignored here; use `expval_pauli_sum` if energy aggregation is needed.
    """
    expvals: List[float] = []
    for entry in terms:
        term, _ = _normalize_term_entry(entry)
        idxs = [int(q) for (q, _p) in term]
        expvals.append(term_expectation_from_counts(counts, idxs))
    return expvals


def _expval_pauli_sum_analytic(expectations: Dict[str, float], items: Sequence[Any], identity_const: float = 0.0, *, probabilities: _np.ndarray | None = None, num_qubits: int | None = None) -> Dict[str, Any]:
    """Aggregate energy when analytic per-qubit Z expectations are provided.

    expectations: mapping like {"Z0": <Z0>, "Z1": <Z1>, ...} coming from statevector engine when shots=0.
    For a product of Z on indices [q0, q1, ...], expectation equals product of individual expectations.
    """
    energy = float(identity_const)
    expvals: List[float] = []
    for entry in items:
        term, coeff = _normalize_term_entry(entry)
        idxs = [int(q) for (q, _p) in term]
        # If full probabilities available, compute exact Z-product via probabilities; else multiply single Z expectations
        if probabilities is not None and num_qubits is not None:
            prod = 0.0
            dim = 1 << int(num_qubits)
            probs = _np.asarray(probabilities, dtype=float).reshape(-1)
            for idx in range(dim):
                bit = 1.0
                for q in idxs:
                    # Big-endian: q=0 is leftmost; index bit at (n-1-q)
                    bit *= (1.0 if ((idx >> (num_qubits - 1 - q)) & 1) == 0 else -1.0)
                prod += bit * float(probs[idx])
        else:
            prod = 1.0
            for q in idxs:
                prod *= float(expectations.get(f"Z{q}", 0.0))
        expvals.append(prod)
        if coeff is not None:
            energy += float(coeff) * prod
    return {"energy": float(energy), "expvals": expvals}


def expval_pauli_sum(counts: Dict[str, int] | None, items: Sequence[Any], identity_const: float = 0.0, *, expectations: Dict[str, float] | None = None, probabilities: _np.ndarray | None = None, num_qubits: int | None = None) -> Dict[str, Any]:
    """Aggregate energy for a Pauli-sum from counts or analytic expectations.

    Parameters:
        counts: bitstring histogram {bitstr: count}. If None and expectations provided, use analytic path.
        items:  list of either term_only or (term, coeff)
        identity_const: constant term to add
        expectations: optional mapping of per-qubit Z expectations (shots=0, statevector engine)

    Returns:
        {"energy": float, "expvals": List[float]}
    """
    if expectations is not None and (counts is None or len(counts) == 0):
        return _expval_pauli_sum_analytic(expectations, items, identity_const=identity_const, probabilities=probabilities, num_qubits=num_qubits)

    counts = counts or {}
    energy = float(identity_const)
    expvals: List[float] = []
    for entry in items:
        term, coeff = _normalize_term_entry(entry)
        idxs = [int(q) for (q, _p) in term]
        ev = term_expectation_from_counts(counts, idxs)
        expvals.append(ev)
        if coeff is not None:
            energy += coeff * float(ev)
    return {"energy": float(energy), "expvals": expvals}


