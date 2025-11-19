from __future__ import annotations

from typing import List, Tuple, Sequence
import numpy as np
from functools import lru_cache

from tyxonq.core.ir.circuit import Circuit
from openfermion import QubitOperator
from tyxonq.numerics import NumericBackend as nb
from tyxonq.libs.quantum_library.kernels.pauli import pauli_string_sum_dense
from tyxonq.libs.quantum_library.dynamics import expectation
from tyxonq.libs.circuits_library.qiskit_real_amplitudes import build_circuit_from_template


class HEANumericRuntime:
    def __init__(self, n: int, layers: int, hamiltonian: List[Tuple[float, List[Tuple[str, int]]]], numeric_engine: str | None = None, *, circuit_template: list | None = None, qop: QubitOperator | None = None):
        self.n = int(n)
        self.layers = int(layers)
        self.hamiltonian = list(hamiltonian)
        self.numeric_engine = (numeric_engine or "statevector").lower()
        self.circuit_template = circuit_template
        # Optional: pre-mapped QubitOperator cache for reuse
        self._qop_cached = qop

    def _build(self, params: Sequence[float], get_circuit) -> Circuit:
        # Prefer external template if provided; otherwise use supplied builder
        if self.circuit_template is not None:
            return build_circuit_from_template(self.circuit_template, np.asarray(params, dtype=np.float64), n_qubits=self.n)
        return get_circuit(params)
    
    @staticmethod
    @lru_cache(maxsize=64)
    def _hamiltonian_matrix_cached(key: tuple) -> object:
        """Build and cache Hamiltonian matrix from Hamiltonian key.
        
        Converts Hamiltonian format to Pauli string format and builds dense matrix.
        Uses pauli_string_sum_dense which returns backend-native matrix (NumPy/PyTorch).
        """
        n_qubits, items = key
        pauli_terms = []
        weights = []
        pauli_map = {'X': 1, 'Y': 2, 'Z': 3}
        
        # Convert Hamiltonian format to Pauli string format [0,1,2,3] = [I,X,Y,Z]
        for coeff, ops in items:
            term = [0] * n_qubits
            for pauli_char, qubit_idx in ops:
                term[qubit_idx] = pauli_map.get(str(pauli_char).upper(), 0)
            pauli_terms.append(term)
            weights.append(float(coeff))
        
        # Build dense Hamiltonian matrix
        # pauli_string_sum_dense returns backend-native matrix (NumPy array or PyTorch tensor)
        return pauli_string_sum_dense(pauli_terms, weights)
    
    def _qop_key(self) -> tuple:
        """Generate cache key for Hamiltonian from internal format."""
        items = []
        for coeff, ops in self.hamiltonian:
            items.append((float(coeff), tuple((str(P).upper(), int(q)) for (P, q) in ops)))
        return (int(self.n), tuple(items))
    
    def _get_hamiltonian_matrix(self) -> object:
        """Get cached Hamiltonian matrix.
        
        Returns Hamiltonian matrix that can be used with circuit.state() and expectation().
        Uses caching to avoid repeated matrix construction.
        """
        return self._hamiltonian_matrix_cached(self._qop_key())

    def energy(self, params: Sequence[float], get_circuit):
        """Compute energy using circuit.state() + Hamiltonian matrix approach.
        
        This is the single energy function used by both energy() and value_and_grad().
        Returns tensor/array (not float) to preserve autograd chain for gradient computation.
        """
        c = self._build(params, get_circuit)
        psi = c.state()  # Returns backend tensor (NumPy/PyTorch), preserves autograd
        H = self._get_hamiltonian_matrix()  # Cached Hamiltonian matrix
        
        # Compute <ψ|H|ψ> using backend-agnostic expectation function
        # This preserves autograd chain for PyTorch backend
        energy_tensor = expectation(psi, H, backend=nb)
        # Return as tensor (NOT float) so gradient computation works
        return energy_tensor

    def energy_and_grad(self, params: Sequence[float], get_circuit) -> Tuple[float, np.ndarray]:
        """Compute energy and gradient with PyTorch autograd support.
        
        Uses the same energy() function for both value and gradient computation.
        
        When PyTorch backend is active:
        - Uses PyTorch's autograd mechanism for gradient computation
        - Enables GPU acceleration and efficient batching
        
        When NumPy backend is active:
        - Uses numerical gradient (finite difference)
        """
        # Use the single energy() function for value_and_grad
        # nb.value_and_grad will handle both PyTorch autograd and NumPy finite difference
        vag = nb.value_and_grad(self.energy, argnums=0)
        e, g = vag(np.asarray(params, dtype=np.float64), get_circuit)
        return float(e), np.asarray(g, dtype=np.float64)


