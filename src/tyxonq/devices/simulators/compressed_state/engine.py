from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING
import numpy as np
from ....numerics.api import get_backend, ArrayBackend
from ..gates import (
    gate_h, gate_rz, gate_rx, gate_cx_4x4,
)
from .matrix_product_state import (
    MPSState,
    init_product_state,
    apply_1q as mps_apply_1q,
    apply_2q as mps_apply_2q,
    to_statevector as mps_to_statevector,
)

if TYPE_CHECKING:  # pragma: no cover
    from ....core.ir import Circuit


class CompressedStateEngine:
    name = "compressed_state"
    capabilities = {"supports_shots": True}

    def __init__(self, backend: ArrayBackend | None = None, backend_name: str | None = None, *, max_bond: int | None = None) -> None:
        # Use global numerics backend; default to numpy if not specified
        self.backend: ArrayBackend = backend or get_backend(backend_name)
        # Optional MPS bond truncation (hard cap)
        self.max_bond: int | None = max_bond

    def _init_state(self, num_qubits: int):
        return init_product_state(num_qubits)

    def _apply_1q(self, state: Any, gate2: Any, qubit: int, num_qubits: int):
        # MPS in-place update
        mps_apply_1q(state, gate2, qubit)
        return state

    def _apply_2q(self, state: Any, gate4: Any, q1: int, q2: int, num_qubits: int):
        # General 2-qubit with SWAP routing
        mps_apply_2q(state, gate4, q1, q2, max_bond=self.max_bond)
        return state

    def _gate_h(self):
        return gate_h()

    def _gate_rz(self, theta: float):
        return gate_rz(theta)

    def _gate_rx(self, theta: float):
        return gate_rx(theta)

    def _gate_cx(self):
        return gate_cx_4x4()

    def run(self, circuit: "Circuit", shots: int | None = None, **kwargs: Any) -> Dict[str, Any]:
        shots = int(shots or 0)
        n = int(getattr(circuit, "num_qubits", 0))
        state = self._init_state(n)
        # unified noise interface (explicit switch)
        use_noise = bool(kwargs.get("use_noise", False))
        noise = kwargs.get("noise") if use_noise else None
        z_atten = [1.0] * n if use_noise else None
        measures: list[int] = []
        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                continue
            name = op[0]
            if name == "h":
                q = int(op[1])
                state = self._apply_1q(state, gate_h(), q, n)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "rz":
                q = int(op[1]); theta = float(op[2])
                state = self._apply_1q(state, gate_rz(theta), q, n)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "rx":
                q = int(op[1]); theta = float(op[2])
                state = self._apply_1q(state, gate_rx(theta), q, n)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "cx":
                q1, q2 = int(op[1]), int(op[2])
                state = self._apply_2q(state, gate_cx_4x4(), q1, q2, n)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q1, q2])
            elif name == "measure_z":
                measures.append(int(op[1]))
        expectations: Dict[str, float] = {}
        # Compute expectations by reconstructing statevector for now (small n tests)
        psi = mps_to_statevector(state)
        # reuse existing kernel's exp-z on statevector logic inline
        for q in measures:
            s_perm = np.moveaxis(psi.reshape([2] * n), q, 0)
            s2 = np.abs(s_perm.reshape(2, -1)) ** 2
            probs = np.sum(s2, axis=1)
            val = float(probs[0] - probs[1])
            if use_noise and z_atten is not None:
                val *= z_atten[q]
            expectations[f"Z{q}"] = val
        return {"expectations": expectations, "metadata": {"shots": shots, "backend": getattr(self.backend, 'name', 'unknown')}}

    def expval(self, circuit: "Circuit", obs: Any, **kwargs: Any) -> float:
        return 0.0

    def _attenuate(self, noise: Any, z_atten: list[float], wires: list[int]) -> None:
        ntype = str(noise.get("type", "")).lower() if noise else ""
        if ntype == "depolarizing":
            p = float(noise.get("p", 0.0))
            factor = max(0.0, 1.0 - 4.0 * p / 3.0)
            for q in wires:
                z_atten[q] *= factor


