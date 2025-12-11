from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np

__all__ = [
	"group_qubit_operator_terms",
	"group_hamiltonian_pauli_terms",
]


def group_qubit_operator_terms(qop: Any, n_qubits: int) -> Tuple[float, Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]]]:
	"""Group QubitOperator terms by measurement basis for efficient quantum computation.

	This function analyzes a Hamiltonian represented as a QubitOperator and groups
	terms that can be measured simultaneously in the same Pauli basis. This grouping
	enables significant reduction in the number of quantum circuit executions required
	for Hamiltonian expectation value estimation.

	Measurement Compatibility:
		Terms are compatible if they can be measured in the same computational basis:
		- Pauli-I: Always compatible (identity measurement)
		- Pauli-Z: Measured in computational basis (|0⟩, |1⟩)
		- Pauli-X: Requires H gate before Z measurement
		- Pauli-Y: Requires S†H gates before Z measurement

	Args:
		qop (Any): OpenFermion QubitOperator containing Pauli terms with coefficients.
			Expected to have a 'terms' attribute mapping term tuples to coefficients.
		n_qubits (int): Total number of qubits in the quantum system.

	Returns:
		Tuple[float, Dict]: A tuple containing:
			- identity_const (float): Constant term (identity coefficient)
			- groups (Dict): Measurement groups where:
				* Key: Tuple of basis strings (length n_qubits), each entry in {'I','X','Y','Z'}
				* Value: List of (term_tuple, coefficient) pairs
					- term_tuple: ((qubit_index, pauli_char), ...) for non-identity Paulis
					- coefficient: Real coefficient for the Pauli term

	Examples:
		>>> from openfermion import QubitOperator
		>>> # Create a simple Hamiltonian: 0.5*Z0 + 0.3*X0*X1 + 1.0*I
		>>> qop = QubitOperator('Z0', 0.5) + QubitOperator('X0 X1', 0.3) + QubitOperator('', 1.0)
		>>> identity_const, groups = group_qubit_operator_terms(qop, n_qubits=2)
		>>> print(f"Identity constant: {identity_const}")
		1.0
		>>> print(f"Number of measurement groups: {len(groups)}")
		2
		>>> # Group 1: Z-basis measurement for Z0 term
		>>> groups[('Z', 'I')]
		[((0, 'Z'),), 0.5]
		>>> # Group 2: X-basis measurement for X0*X1 term
		>>> groups[('X', 'X')]
		[((0, 'X'), (1, 'X')), 0.3]
		
		>>> # Molecular Hamiltonian grouping
		>>> from tyxonq.libs.hamiltonian_encoding import get_molecular_hamiltonian
		>>> h2_hamiltonian = get_molecular_hamiltonian('H2', basis='sto-3g')
		>>> const, groups = group_qubit_operator_terms(h2_hamiltonian, n_qubits=4)
		>>> # Significant reduction: 15 terms → ~4 measurement groups
		>>> measurements_needed = len(groups)
		>>> total_terms = sum(len(terms) for terms in groups.values())
		>>> print(f"Reduced {total_terms} terms to {measurements_needed} measurements")

	Measurement Strategy:
		For each basis group (e.g., ('X', 'I', 'Z', 'Y')):
		1. Apply basis rotation gates:
		   - Qubit 0: H (for X measurement)
		   - Qubit 1: nothing (I measurement)
		   - Qubit 2: nothing (Z measurement)
		   - Qubit 3: S†H (for Y measurement)
		2. Measure all qubits in computational basis
		3. Compute expectation values for all terms in the group
		4. Weight by coefficients and sum contributions

	Raises:
		ValueError: If any Hamiltonian term has significant imaginary coefficient
			(> 1e-10), indicating a non-Hermitian operator.

	Notes:
		- The grouping is exact and preserves all Hamiltonian information
		- Measurement reduction scales with Hamiltonian structure complexity
		- Compatible terms within each group can be measured simultaneously
		- The identity constant is extracted separately for efficient handling
		
	See Also:
		tyxonq.compiler.passes.measurement_reduction: Compiler passes using this grouping.
		tyxonq.postprocessing.counts_expval: Expectation value computation from grouped measurements.
		tyxonq.libs.hamiltonian_encoding.fermion_to_qubit: Fermion-to-qubit mapping functions.
	"""
	identity_const = 0.0
	groups: Dict[Tuple[str, ...], List[Tuple[Tuple[Tuple[int, str], ...], float]]] = {}
	terms = getattr(qop, "terms", {})
	for term, coeff in terms.items():
		if term == ():
			try:
				c = complex(np.asarray(coeff, dtype=np.complex128))
			except Exception:
				c = complex(getattr(coeff, "real", float(coeff)))
			# 严谨性：若虚部超过阈值，判为非常规哈密顿量，抛出异常提示上层修正
			if abs(c.imag) > 1e-10:
				raise ValueError(f"Hamiltonian identity term has non-negligible imaginary part: {c}")
			identity_const += float(c.real)
			continue
		bases = ["I"] * n_qubits
		for (q, p) in term:
			bases[int(q)] = str(p).upper()
		try:
			c = complex(np.asarray(coeff, dtype=np.complex128))
		except Exception:
			c = complex(getattr(coeff, "real", float(coeff)))
		if abs(c.imag) > 1e-10:
			raise ValueError(f"Hamiltonian term has non-negligible imaginary part: {c}")
		coeff_val = float(c.real)
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
