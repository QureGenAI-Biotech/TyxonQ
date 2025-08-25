"""Statevector simulator engine.

This engine simulates the pure state |psi> with a dense statevector of size 2^n.
Characteristics:
- Complexity: memory O(2^n), time ~O(poly(gates)*2^n)
- Noise: optional, approximate attenuation on Z expectations when use_noise=True
- Features: supports h/rz/rx/cx, measure_z expectations, and helpers
  (state, probability, amplitude, perfect_sampling)
- Numerics: uses unified kernels in devices.simulators.gates with ArrayBackend.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING
import numpy as np
from ....numerics.api import get_backend
from ....libs.quantum_library.kernels.gates import (
    gate_h, gate_rz, gate_rx, gate_cx_4x4,
)
from ....libs.quantum_library.kernels.statevector import (
    init_statevector,
    apply_1q_statevector,
    apply_2q_statevector,
    expect_z_statevector,
)

if TYPE_CHECKING:  # pragma: no cover
    from ....core.ir import Circuit


class StatevectorEngine:
    name = "statevector"
    capabilities = {"supports_shots": True}

    def __init__(self, backend_name: str | None = None) -> None:
        # Pluggable numerics backend (numpy/pytorch/cupynumeric)
        self.backend = get_backend(backend_name)

    def run(self, circuit: "Circuit", shots: int | None = None, **kwargs: Any) -> Dict[str, Any]:
        shots = int(shots or 0)
        num_qubits = int(getattr(circuit, "num_qubits", 0))
        state = init_statevector(num_qubits)
        # optional noise parameters controlled by explicit switch
        use_noise = bool(kwargs.get("use_noise", False))
        noise = kwargs.get("noise") if use_noise else None
        z_atten = [1.0] * num_qubits if use_noise else None
        measures: list[int] = []
        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                continue
            name = op[0]
            if name == "h":
                q = int(op[1]); state = apply_1q_statevector(self.backend, state, gate_h(), q, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "rz":
                q = int(op[1]); theta = float(op[2]); state = apply_1q_statevector(self.backend, state, gate_rz(theta), q, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "rx":
                q = int(op[1]); theta = float(op[2]); state = apply_1q_statevector(self.backend, state, gate_rx(theta), q, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "cx":
                c = int(op[1]); t = int(op[2]); state = apply_2q_statevector(self.backend, state, gate_cx_4x4(), c, t, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [c, t])
            elif name == "measure_z":
                measures.append(int(op[1]))
            elif name == "barrier":
                # no-op for simulation
                continue
            elif name == "project_z":
                q = int(op[1]); keep = int(op[2])
                state = self._project_z(state, q, keep, num_qubits)
            elif name == "reset":
                q = int(op[1])
                state = self._project_z(state, q, 0, num_qubits)
            else:
                # unsupported ops ignored in this minimal engine
                continue

        expectations: Dict[str, float] = {}
        for q in measures:
            val = float(expect_z_statevector(state, q, num_qubits))
            if use_noise and z_atten is not None:
                val *= z_atten[q]
            expectations[f"Z{q}"] = val
        return {"expectations": expectations, "metadata": {"shots": shots, "backend": self.backend.name}}

    def expval(self, circuit: "Circuit", obs: Any, **kwargs: Any) -> float:
        # Not implemented; placeholder for future
        return 0.0

    # helpers removed; using gates kernels

    def _attenuate(self, noise: Any, z_atten: list[float], wires: list[int]) -> None:
        ntype = str(noise.get("type", "").lower()) if noise else ""
        if ntype == "depolarizing":
            p = float(noise.get("p", 0.0))
            factor = max(0.0, 1.0 - 4.0 * p / 3.0)
            for q in wires:
                z_atten[q] *= factor

    # ---- New public helpers ----
    def state(self, circuit: "Circuit") -> np.ndarray:
        """Return final statevector after applying circuit ops."""
        n = int(getattr(circuit, "num_qubits", 0))
        state = init_statevector(n)
        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                continue
            name = op[0]
            if name == "h":
                q = int(op[1]); state = apply_1q_statevector(self.backend, state, gate_h(), q, n)
            elif name == "rz":
                q = int(op[1]); theta = float(op[2]); state = apply_1q_statevector(self.backend, state, gate_rz(theta), q, n)
            elif name == "rx":
                q = int(op[1]); theta = float(op[2]); state = apply_1q_statevector(self.backend, state, gate_rx(theta), q, n)
            elif name == "cx":
                c = int(op[1]); t = int(op[2]); state = apply_2q_statevector(self.backend, state, gate_cx_4x4(), c, t, n)
            elif name == "project_z":
                q = int(op[1]); keep = int(op[2]); state = self._project_z(state, q, keep, n)
            elif name == "reset":
                q = int(op[1]); state = self._project_z(state, q, 0, n)
        return state

    def probability(self, circuit: "Circuit") -> np.ndarray:
        """Return probability vector over computational basis."""
        s = self.state(circuit)
        return np.abs(s) ** 2

    def amplitude(self, circuit: "Circuit", bitstring: str) -> complex:
        """Return amplitude <bitstring|psi> using big-endian convention (q0 is left)."""
        n = int(getattr(circuit, "num_qubits", 0))
        if len(bitstring) != n:
            raise ValueError("bitstring length must equal num_qubits")
        # Map bitstring to basis index; |00..0> -> 0, |00..1> -> 1, ... big-endian
        idx = 0
        for ch in bitstring:
            idx = (idx << 1) | (1 if ch == '1' else 0)
        s = self.state(circuit)
        return complex(s[idx])

    def perfect_sampling(self, circuit: "Circuit", *, rng: np.random.Generator | None = None) -> tuple[str, float]:
        """Sample a single bitstring from exact probabilities with optional RNG."""
        n = int(getattr(circuit, "num_qubits", 0))
        p = self.probability(circuit)
        if rng is None:
            rng = np.random.default_rng()
        dim = 1 << n
        idx = rng.choice(dim, p=p)
        prob = float(p[idx])
        # index to bitstring (big-endian)
        bits = ''.join('1' if (idx >> (n - 1 - k)) & 1 else '0' for k in range(n))
        return bits, prob

    # internal: projection on Z-basis
    def _project_z(self, state: np.ndarray, qubit: int, keep: int, n: int) -> np.ndarray:
        t = state.reshape([2] * n)
        t = np.moveaxis(t, qubit, 0)
        if keep == 0:
            t[1, ...] = 0
        else:
            t[0, ...] = 0
        t = np.moveaxis(t, 0, qubit)
        out = t.reshape(-1)
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
        return out


