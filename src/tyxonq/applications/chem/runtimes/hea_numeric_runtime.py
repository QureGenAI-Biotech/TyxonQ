from __future__ import annotations

from typing import List, Tuple, Sequence
import numpy as np
from functools import lru_cache

from tyxonq.core.ir.circuit import Circuit
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from tyxonq.numerics import NumericBackend as nb
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

    def _state(self, c: Circuit) -> np.ndarray:
        if self.numeric_engine == "statevector":
            eng = StatevectorEngine()
            psi = np.asarray(eng.state(c), dtype=np.complex128)
            return psi
        elif self.numeric_engine == "mps":
            # TODO: Add MPS numeric path; fallback to exact for now
            eng = StatevectorEngine()
            return np.asarray(eng.state(c), dtype=np.complex128)
        else:
            eng = StatevectorEngine()
            return np.asarray(eng.state(c), dtype=np.complex128)

    @staticmethod
    @lru_cache(maxsize=64)
    def _to_qubit_operator_cached(key: tuple) -> QubitOperator:
        n_qubits, items = key
        qop = QubitOperator()
        for coeff, ops in items:
            if not ops:
                qop += float(coeff)
            else:
                term = tuple((int(q), str(P).upper()) for (P, q) in ops)
                qop += QubitOperator(term, float(coeff))
        return qop

    def _qop_key(self) -> tuple:
        items = []
        for coeff, ops in self.hamiltonian:
            items.append((float(coeff), tuple((str(P).upper(), int(q)) for (P, q) in ops)))
        return (int(self.n), tuple(items))

    def _to_qubit_operator(self) -> QubitOperator:
        if self._qop_cached is not None:
            return self._qop_cached
        return self._to_qubit_operator_cached(self._qop_key())

    @staticmethod
    @lru_cache(maxsize=64)
    def _sparse_from_qop_key(key: tuple) -> object:
        n_qubits, items = key
        # Rebuild qop from key
        qop = QubitOperator()
        for coeff, ops in items:
            if not ops:
                qop += float(coeff)
            else:
                term = tuple((int(q), str(P).upper()) for (P, q) in ops)
                qop += QubitOperator(term, float(coeff))
        return get_sparse_operator(qop, n_qubits=int(n_qubits))

    def _expect(self, psi: np.ndarray, qop: QubitOperator) -> float:
        H = self._sparse_from_qop_key(self._qop_key())
        e = np.vdot(psi, H.dot(psi))
        return float(np.real(e))

    def energy(self, params: Sequence[float], get_circuit) -> float:
        c = self._build(params, get_circuit)
        qop = self._to_qubit_operator()
        eng = StatevectorEngine()
        return float(eng.expval(c, qop))

    def energy_and_grad(self, params: Sequence[float], get_circuit) -> Tuple[float, np.ndarray]:
        def _f(x: np.ndarray) -> float:
            return self.energy(x, get_circuit)
        vag = nb.value_and_grad(_f, argnums=0)
        e, g = vag(np.asarray(params, dtype=np.float64))
        return float(e), np.asarray(g, dtype=np.float64)


