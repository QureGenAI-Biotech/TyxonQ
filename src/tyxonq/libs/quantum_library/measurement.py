from __future__ import annotations

from typing import Dict, List, Tuple


def term_expectation_from_counts(counts: Dict[str, int], idxs: List[int]) -> float:
    """[DEPRECATED] Moved to tyxonq.postprocessing.methods.term_expectation_from_counts.

    保留薄重导出以兼容现有调用；新代码请改用 postprocessing.methods。
    """
    from tyxonq.postprocessing.counts_expval import term_expectation_from_counts as _impl
    return _impl(counts, idxs)


def group_qubit_operator(qop, n_qubits: int) -> Tuple[float, Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]]]:
    """[DEPRECATED] Use compiler.rewrite.measurement grouping and metadata instead.

    此函数将逐步废弃，当前仅用于过渡期。
    """
    identity_const = 0.0
    groups: Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]] = {}
    for term, coeff in qop.terms.items():
        if term == ():
            identity_const += float(getattr(coeff, "real", float(coeff)))
            continue
        bases = ["I"] * n_qubits
        for (q, p) in term:
            bases[q] = p.upper()
        groups.setdefault(tuple(bases), []).append((tuple(term), float(getattr(coeff, "real", float(coeff)))))
    return identity_const, groups


def group_hamiltonian_terms(hamiltonian, n_qubits: int) -> Tuple[float, Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]]]:
    """[DEPRECATED] Use compiler.rewrite.measurement grouping and metadata instead.

    Hamiltonian is List[(coeff, [(P,q), ...])].
    """
    identity_const = 0.0
    groups: Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]] = {}
    for coeff, ops in hamiltonian:
        if not ops:
            identity_const += float(coeff)
            continue
        bases = ["I"] * n_qubits
        term_tuple: Tuple[Tuple[int, str], ...] = tuple((q, p) for (p, q) in ops)
        for (q, p) in term_tuple:
            bases[q] = p.upper()
        groups.setdefault(tuple(bases), []).append((term_tuple, float(coeff)))
    return identity_const, groups


