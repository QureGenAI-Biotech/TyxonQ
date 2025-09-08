"""Matrix Product State (MPS) simulator engine.

This engine represents the quantum state as a Matrix Product State (MPS) and
applies gates with local updates, enabling simulation of larger systems when the
entanglement is limited.
Characteristics:
- Complexity: memory/time scale with bond dimension chi rather than 2^n
- Control: optional `max_bond` clamps the bond dimension (truncation)
- Features: supports h/rz/rx/cx with SWAP routing; measure_z via reconstruction
- Numerics: uses unified gate kernels plus MPS operations in this package,
  respecting the ArrayBackend abstraction for arrays.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING
import numpy as np
from ....numerics.api import get_backend, ArrayBackend
from ....libs.quantum_library.kernels.gates import (
    gate_h, gate_rz, gate_rx, gate_cx_4x4,
    gate_x, gate_ry, gate_cz_4x4, gate_s, gate_sd, gate_cry_4x4,
)
from ....libs.quantum_library.kernels.matrix_product_state import (
    init_product_state,
    apply_1q as mps_apply_1q,
    apply_2q as mps_apply_2q,
    MPSState,
    to_statevector as mps_to_statevector,
)

if TYPE_CHECKING:  # pragma: no cover
    from ....core.ir import Circuit


class MatrixProductStateEngine:
    name = "matrix_product_state"
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
            elif name == "ry":
                q = int(op[1]); theta = float(op[2])
                state = self._apply_1q(state, gate_ry(theta), q, n)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "cx":
                q1, q2 = int(op[1]), int(op[2])
                state = self._apply_2q(state, gate_cx_4x4(), q1, q2, n)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q1, q2])
            elif name == "cry":
                q1, q2 = int(op[1]), int(op[2])
                theta = float(op[3])
                state = self._apply_2q(state, gate_cry_4x4(theta), q1, q2, n)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q1, q2])
            elif name == "cz":
                q1, q2 = int(op[1]), int(op[2])
                state = self._apply_2q(state, gate_cz_4x4(), q1, q2, n)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q1, q2])
            elif name == "x":
                q = int(op[1])
                state = self._apply_1q(state, gate_x(), q, n)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "s":
                q = int(op[1])
                state = self._apply_1q(state, gate_s(), q, n)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "sdg":
                q = int(op[1])
                state = self._apply_1q(state, gate_sd(), q, n)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "measure_z":
                measures.append(int(op[1]))
            elif name == "barrier":
                # no-op
                continue
        # Barrier 不是量子门（非幺正、无矩阵表示），它是编译/调度指令，用来限制优化重排或做分段同步，对量子态本身不产生物理作用。
        # If shots requested and there are measurements, return sampled counts via reconstructed probabilities
        if shots > 0 and len(measures) > 0:
            nb = self.backend
            psi = mps_to_statevector(state)
            p = nb.square(nb.abs(psi)) if hasattr(nb, 'square') else (np.abs(psi) ** 2)
            p_np = np.asarray(nb.to_numpy(p), dtype=float)
            dim = int(p_np.size)
            # Optional noise mixing via kwargs
            if bool(kwargs.get("use_noise", False)):
                noise = kwargs.get("noise", {}) or {}
                ntype = str(noise.get("type", "")).lower()
                if ntype == "readout":
                    A = None
                    cals = noise.get("cals", {}) or {}
                    for q in range(n):
                        m = cals.get(q)
                        if m is None:
                            m = nb.eye(2)
                        m = nb.asarray(m)
                        A = m if A is None else nb.kron(A, m)
                    p_np = np.asarray(nb.to_numpy(A), dtype=float) @ p_np
                elif ntype == "depolarizing":
                    pval = float(noise.get("p", 0.0))
                    alpha = max(0.0, min(1.0, 4.0 * pval / 3.0))
                    p_np = (1.0 - alpha) * p_np + alpha * (1.0 / dim)
                p_np = np.clip(p_np, 0.0, 1.0)
                s = float(np.sum(p_np))
                p_np = p_np / (s if s > 1e-12 else 1.0)
            else:
                if p_np.sum() > 0:
                    p_np = p_np / float(p_np.sum())
                else:
                    p_np = np.full((dim,), 1.0 / dim, dtype=float)
            rng = nb.rng(None)
            idx_samples = nb.choice(rng, dim, size=shots, p=p_np)
            counts_arr = nb.bincount(nb.asarray(idx_samples), minlength=dim)
            results: Dict[str, int] = {}
            nz = nb.nonzero(counts_arr)[0]
            for idx in nz:
                ii = int(idx)
                bitstr = ''.join('1' if (ii >> (n - 1 - k)) & 1 else '0' for k in range(n))
                results[bitstr] = int(nb.to_numpy(counts_arr)[ii])
            return {"result": results, "metadata": {"shots": shots, "backend": getattr(self.backend, 'name', 'unknown')}}

        expectations: Dict[str, float] = {}
        # Compute expectations by reconstructing statevector for now (small n tests)
        psi = mps_to_statevector(state)
        nb = self.backend
        psi_b = nb.asarray(psi)
        for q in measures:
            s = nb.reshape(psi_b, (2,) * n)
            s_perm = nb.moveaxis(s, q, 0)
            s2 = nb.abs(nb.reshape(s_perm, (2, -1))) ** 2  # type: ignore[operator]
            probs = nb.sum(s2, axis=1)
            probs_np = nb.to_numpy(probs)
            val = float(probs_np[0] - probs_np[1])
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


