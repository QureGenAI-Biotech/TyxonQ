from __future__ import annotations

from typing import List, Tuple, Sequence
from math import pi

import numpy as np
from openfermion import QubitOperator

from tyxonq.core.ir.circuit import Circuit
# Use simulator engine for exact statevector
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
from tyxonq.libs.circuits_library.ucc import build_ucc_circuit


class UCCNumericRuntime:
    def __init__(
        self,
        n_qubits: int,
        n_elec_s: Tuple[int, int],
        h_qubit_op: QubitOperator,
        *,
        ex_ops: List[Tuple] | None = None,
        param_ids: List[int] | None = None,
        init_state: Sequence[float] | Circuit | None = None,
        mode: str = "fermion",
        trotter: bool = False,
        decompose_multicontrol: bool = False,
    ):
        self.n_qubits = int(n_qubits)
        self.n_elec_s = (int(n_elec_s[0]), int(n_elec_s[1]))
        self.h_qubit_op = h_qubit_op
        self.ex_ops = list(ex_ops) if ex_ops is not None else None
        self.param_ids = list(param_ids) if param_ids is not None else None
        self.init_state = init_state
        self.mode = str(mode)
        self.trotter = bool(trotter)
        self.decompose_multicontrol = bool(decompose_multicontrol)

        if self.ex_ops is not None:
            self.n_params = (max(self.param_ids) + 1) if (self.param_ids and len(self.param_ids) > 0) else len(self.ex_ops)
        else:
            self.n_params = 0

    def _build(self, params: Sequence[float]) -> Circuit:
        if self.ex_ops is None or self.n_params == 0:
            return Circuit(self.n_qubits, ops=[])
        return build_ucc_circuit(
            params,
            self.n_qubits,
            self.n_elec_s,
            tuple(self.ex_ops),
            tuple(self.param_ids) if self.param_ids is not None else None,
            mode=self.mode,
            init_state=self.init_state,
            decompose_multicontrol=self.decompose_multicontrol,
            trotter=self.trotter,
        )

    def _state(self, params: Sequence[float]) -> np.ndarray:
        c = self._build(params)
        eng = StatevectorEngine()
        return np.asarray(eng.state(c), dtype=np.complex128)

    def _expect(self, psi: np.ndarray) -> float:
        val = 0.0
        for term, coeff in self.h_qubit_op.terms.items():
            if term == ():
                val += float(getattr(coeff, "real", float(coeff)))
                continue
            phi = psi
            for q, p in term:
                if p == "X":
                    # X = |0><1| + |1><0|
                    phi = self._apply_x(phi, q)
                elif p == "Y":
                    phi = self._apply_y(phi, q)
                else:
                    phi = self._apply_z(phi, q)
            val += float(np.vdot(psi, phi) * coeff)
        return float(val)

    def _apply_x(self, psi: np.ndarray, q: int) -> np.ndarray:
        n = self.n_qubits
        stride = 1 << q
        out = psi.copy()
        for i in range(0, 1 << n, 2 * stride):
            for j in range(stride):
                a = i + j
                b = a + stride
                out[a], out[b] = psi[b], psi[a]
        return out

    def _apply_z(self, psi: np.ndarray, q: int) -> np.ndarray:
        n = self.n_qubits
        stride = 1 << q
        out = psi.copy()
        for i in range(0, 1 << n, 2 * stride):
            for j in range(stride):
                idx = i + j + stride
                out[idx] = -out[idx]
        return out

    def _apply_y(self, psi: np.ndarray, q: int) -> np.ndarray:
        # Y = i|1><0| - i|0><1|
        n = self.n_qubits
        stride = 1 << q
        out = psi.copy()
        for i in range(0, 1 << n, 2 * stride):
            for j in range(stride):
                a = i + j
                b = a + stride
                out[a] = -1j * psi[b]
                out[b] = 1j * psi[a]
        return out

    def energy(self, params: Sequence[float] | None = None) -> float:
        base = np.zeros(self.n_params, dtype=np.float64) if (params is None and self.n_params > 0) else np.asarray(params or [], dtype=np.float64)
        psi = self._state(base)
        return self._expect(psi)

    def energy_and_grad(self, params: Sequence[float] | None = None) -> Tuple[float, np.ndarray]:
        base = np.zeros(self.n_params, dtype=np.float64) if (params is None and self.n_params > 0) else np.asarray(params or [], dtype=np.float64)
        e0 = self.energy(base)
        if self.n_params == 0:
            return float(e0), np.zeros(0, dtype=np.float64)
        g = np.zeros_like(base)
        s = 0.5 * pi
        for i in range(len(base)):
            p_plus = base.copy(); p_plus[i] += s
            p_minus = base.copy(); p_minus[i] -= s
            e_plus = self.energy(p_plus)
            e_minus = self.energy(p_minus)
            g[i] = 0.5 * (e_plus - e_minus)
        return float(e0), g

# Compatibility helper for tests (replaces legacy engine_ucc.apply_excitation)
def apply_excitation(state: np.ndarray, n_qubits: int, n_elec_s, ex_op: tuple, mode: str, engine: str | None = None) -> np.ndarray:
    from tyxonq.applications.chem.chem_libs.quantum_chem_library.statevector_ops import apply_excitation_statevector
    if isinstance(n_elec_s, (tuple, list)):
        n_elec = int(sum(n_elec_s))
    else:
        n_elec = int(n_elec_s)
    return apply_excitation_statevector(state, n_qubits, n_elec, ex_op, mode)



