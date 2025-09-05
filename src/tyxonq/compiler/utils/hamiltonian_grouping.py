from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ...libs.hamiltonian_encoding.hamiltonian_grouping import (
	group_qubit_operator_terms as _group_qubit_operator_terms,
	group_hamiltonian_pauli_terms as _group_hamiltonian_pauli_terms,
)

__all__ = [
	"group_qubit_operator_terms",
	"group_hamiltonian_pauli_terms",
]


def group_qubit_operator_terms(qop: Any, n_qubits: int) -> Tuple[float, Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]]]:
	return _group_qubit_operator_terms(qop, n_qubits)


def group_hamiltonian_pauli_terms(hamiltonian: List[Tuple[float, List[Tuple[str, int]]]], n_qubits: int) -> Tuple[float, Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]]]:
	return _group_hamiltonian_pauli_terms(hamiltonian, n_qubits)
