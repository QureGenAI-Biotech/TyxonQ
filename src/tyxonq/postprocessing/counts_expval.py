from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


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


def expval_pauli_sum(counts: Dict[str, int], items: Sequence[Any], identity_const: float = 0.0) -> Dict[str, Any]:
    """Aggregate energy for a Pauli-sum from counts.

    Parameters:
        counts: bitstring histogram {bitstr: count}
        items:  list of either term_only or (term, coeff)
        identity_const: constant term to add

    Returns:
        {"energy": float, "expvals": List[float]}
    """
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


