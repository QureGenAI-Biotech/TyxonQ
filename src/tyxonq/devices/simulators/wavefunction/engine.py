from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING
import numpy as np
from ....numerics.api import get_backend
from ..gates import (
    gate_h, gate_rz, gate_rx, gate_cx_4x4,
    init_statevector, apply_1q_statevector, apply_2q_statevector, expect_z_statevector,
)

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
        ntype = str(noise.get("type", "")).lower() if noise else ""
        if ntype == "depolarizing":
            p = float(noise.get("p", 0.0))
            factor = max(0.0, 1.0 - 4.0 * p / 3.0)
            for q in wires:
                z_atten[q] *= factor


