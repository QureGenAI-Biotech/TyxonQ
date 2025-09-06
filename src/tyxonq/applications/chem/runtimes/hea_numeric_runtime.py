from __future__ import annotations

from typing import List, Tuple, Sequence
import numpy as np

from tyxonq.core.ir.circuit import Circuit
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine


class HEANumericRuntime:
    def __init__(self, n: int, layers: int, hamiltonian: List[Tuple[float, List[Tuple[str, int]]]], numeric_engine: str | None = None):
        self.n = int(n)
        self.layers = int(layers)
        self.hamiltonian = list(hamiltonian)
        self.numeric_engine = (numeric_engine or "statevector").lower()

    def _build(self, params: Sequence[float], get_circuit) -> Circuit:
        return get_circuit(params)

    def _state(self, c: Circuit) -> np.ndarray:
        if self.numeric_engine == "statevector":
            eng = StatevectorEngine()
            return np.asarray(eng.state(c), dtype=np.complex128)
        elif self.numeric_engine == "mps":
            # TODO: Add MPS numeric path; fallback to exact for now
            eng = StatevectorEngine()
            return np.asarray(eng.state(c), dtype=np.complex128)
        else:
            eng = StatevectorEngine()
            return np.asarray(eng.state(c), dtype=np.complex128)

    def _expect(self, psi: np.ndarray) -> float:
        val = 0.0
        for coeff, term in self.hamiltonian:
            if not term:
                val += float(coeff)
                continue
            # term as [(P,q), ...]
            # For simplicity, reuse circuit measurement path via basis rotations might be costly; here use Z-basis only.
            # Numeric exact path can be extended later.
            # Placeholder: fall back to 0 contribution for non-Z bases to keep API shape; proper implementation later.
            is_all_z = all(p.upper() == "Z" for (p, q) in term)
            if not is_all_z:
                continue
            phase = 0.0
            # fast expectation of Z-strings via bit operations is possible; keep simple for now
            val += float(coeff) * phase
        return float(val)

    def energy(self, params: Sequence[float], get_circuit) -> float:
        c = self._build(params, get_circuit)
        psi = self._state(c)
        return self._expect(psi)


