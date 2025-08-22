from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING
import numpy as np
from ....numerics.api import get_backend

if TYPE_CHECKING:  # pragma: no cover
    from ....core.ir import Circuit


class WavefunctionEngine:
    name = "wavefunction"
    capabilities = {"supports_shots": True}

    def __init__(self, backend_name: str | None = None) -> None:
        # Pluggable numerics backend (numpy/pytorch/cupynumeric)
        self.backend = get_backend(backend_name)

    def run(self, circuit: "Circuit", shots: int | None = None, **kwargs: Any) -> Dict[str, Any]:
        shots = int(shots or 0)
        num_qubits = int(getattr(circuit, "num_qubits", 0))
        state = self._init_state(num_qubits)
        measures: list[int] = []
        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                continue
            name = op[0]
            if name == "h":
                q = int(op[1]); state = self._apply_single_qubit_gate(state, self._H(), q, num_qubits)
            elif name == "rz":
                q = int(op[1]); theta = float(op[2]); state = self._apply_single_qubit_gate(state, self._RZ(theta), q, num_qubits)
            elif name == "rx":
                q = int(op[1]); theta = float(op[2]); state = self._apply_single_qubit_gate(state, self._RX(theta), q, num_qubits)
            elif name == "cx":
                c = int(op[1]); t = int(op[2]); state = self._apply_cx(state, c, t, num_qubits)
            elif name == "measure_z":
                measures.append(int(op[1]))
            else:
                # unsupported ops ignored in this minimal engine
                continue

        expectations: Dict[str, float] = {}
        for q in measures:
            expectations[f"Z{q}"] = float(self._expectation_z(state, q, num_qubits))
        return {"expectations": expectations, "metadata": {"shots": shots, "backend": self.backend.name}}

    def expval(self, circuit: "Circuit", obs: Any, **kwargs: Any) -> float:
        # Not implemented; placeholder for future
        return 0.0

    # --- linear algebra helpers ---
    def _init_state(self, num_qubits: int) -> np.ndarray:
        if num_qubits <= 0:
            return np.array([1.0 + 0.0j])
        state = np.zeros(1 << num_qubits, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        return state

    def _H(self) -> np.ndarray:
        return (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)

    def _RZ(self, theta: float) -> np.ndarray:
        e = np.exp(-0.5j * theta)
        return np.array([[e, 0.0], [0.0, np.conj(e)]], dtype=np.complex128)

    def _RX(self, theta: float) -> np.ndarray:
        c = np.cos(theta / 2.0)
        s = -1j * np.sin(theta / 2.0)
        return np.array([[c, s], [s, c]], dtype=np.complex128)

    def _apply_single_qubit_gate(self, state: np.ndarray, gate: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
        dim = 1 << num_qubits
        new_state = state.copy()
        step = 1 << qubit
        mask = step
        for i in range(dim):
            if (i & mask) == 0:
                j = i | mask
                a0 = state[i]
                a1 = state[j]
                new_state[i] = gate[0, 0] * a0 + gate[0, 1] * a1
                new_state[j] = gate[1, 0] * a0 + gate[1, 1] * a1
        return new_state

    def _apply_cx(self, state: np.ndarray, control: int, target: int, num_qubits: int) -> np.ndarray:
        if control == target:
            return state
        dim = 1 << num_qubits
        new_state = state.copy()
        ctrl_mask = 1 << control
        tgt_mask = 1 << target
        for i in range(dim):
            if (i & ctrl_mask) != 0 and (i & tgt_mask) == 0:
                j = i | tgt_mask
                new_state[i], new_state[j] = state[j], state[i]
        return new_state

    def _expectation_z(self, state: np.ndarray, qubit: int, num_qubits: int) -> float:
        dim = 1 << num_qubits
        mask = 1 << qubit
        exp = 0.0
        for i in range(dim):
            z = 1.0 if (i & mask) == 0 else -1.0
            p = (state[i].real * state[i].real) + (state[i].imag * state[i].imag)
            exp += z * p
        return float(exp)


