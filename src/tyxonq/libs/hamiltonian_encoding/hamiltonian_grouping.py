from __future__ import annotations

from typing import Any, Dict, List, Tuple

__all__ = [
	"group_qubit_operator_terms",
	"group_hamiltonian_pauli_terms",
]


def group_qubit_operator_terms(qop: Any, n_qubits: int) -> Tuple[float, Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]]]:
	"""Group an OpenFermion-like QubitOperator into product-basis measurement groups.

	Returns:
		(identity_const, groups)
		groups maps basis tuple (length n_qubits, entries in {I,X,Y,Z}) to a list of
		(term_tuple, coeff) where term_tuple is ((q, P), ...) with P in {X,Y,Z}.
	"""
	identity_const = 0.0
	groups: Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]] = {}
	terms = getattr(qop, "terms", {})
	for term, coeff in terms.items():
		if term == ():
			try:
				identity_const += float(getattr(coeff, "real", float(coeff)))
			except Exception:
				identity_const += float(coeff)
			continue
		bases = ["I"] * n_qubits
		for (q, p) in term:
			bases[int(q)] = str(p).upper()
		try:
			coeff_val = float(getattr(coeff, "real", float(coeff)))
		except Exception:
			coeff_val = float(coeff)
		groups.setdefault(tuple(bases), []).append((tuple((int(q), str(p).upper()) for (q, p) in term), coeff_val))
	return identity_const, groups


def group_hamiltonian_pauli_terms(hamiltonian: List[Tuple[float, List[Tuple[str, int]]]], n_qubits: int) -> Tuple[float, Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]]]:
	"""Group a Pauli-sum list [(coeff, [(P, q), ...]), ...] into product-basis groups.

	Returns:
		(identity_const, groups) with the same structure as group_qubit_operator_terms.
	"""
	identity_const = 0.0
	groups: Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]] = {}
	for coeff, ops in hamiltonian:
		if not ops:
			identity_const += float(coeff)
			continue
		bases = ["I"] * n_qubits
		# ops structure: [(P, q), ...]
		term_tuple: Tuple[Tuple[int, str], ...] = tuple((int(q), str(p).upper()) for (p, q) in ops)
		for (q, p) in term_tuple:
			bases[int(q)] = p
		groups.setdefault(tuple(bases), []).append((term_tuple, float(coeff)))
	return identity_const, groups
