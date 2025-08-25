"""Density matrix simulator engine.

This engine simulates the mixed state rho with a dense 2^n x 2^n matrix.
Characteristics:
- Complexity: memory O(4^n), time ~O(poly(gates)*4^n) (more expensive than statevector)
- Noise: native Kraus channel application via devices.simulators.noise.channels
- Features: supports h/rz/rx/cx, measure_z expectations; best suited for noise studies
- Numerics: uses unified kernels in devices.simulators.gates with ArrayBackend.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING
import numpy as np
from ....numerics.api import get_backend
from ..noise import channels as noise_channels
from ..gates import (
    gate_h, gate_rz, gate_rx, gate_cx_4x4,
    init_density, apply_1q_density, apply_2q_density, exp_z_density,
)

if TYPE_CHECKING:  # pragma: no cover
    from ....core.ir import Circuit


class DensityMatrixEngine:
    name = "density_matrix"
    capabilities = {"supports_shots": True}

    def __init__(self, backend_name: str | None = None) -> None:
        self.backend = get_backend(backend_name)

    def run(self, circuit: "Circuit", shots: int | None = None, **kwargs: Any) -> Dict[str, Any]:
        shots = int(shots or 0)
        n = int(getattr(circuit, "num_qubits", 0))
        rho = init_density(n)
        noise = kwargs.get("noise") if kwargs.get("use_noise") else None

        measures: list[int] = []
        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                continue
            name = op[0]
            if name == "h":
                q = int(op[1]); rho = apply_1q_density(self.backend, rho, gate_h(), q, n)
                rho = self._apply_noise_if_any(rho, noise, [q], n)
            elif name == "rz":
                q = int(op[1]); theta = float(op[2]); rho = apply_1q_density(self.backend, rho, gate_rz(theta), q, n)
                rho = self._apply_noise_if_any(rho, noise, [q], n)
            elif name == "rx":
                q = int(op[1]); theta = float(op[2]); rho = apply_1q_density(self.backend, rho, gate_rx(theta), q, n)
                rho = self._apply_noise_if_any(rho, noise, [q], n)
            elif name == "cx":
                c = int(op[1]); t = int(op[2]); rho = apply_2q_density(self.backend, rho, gate_cx_4x4(), c, t, n)
                rho = self._apply_noise_if_any(rho, noise, [c, t], n)
            elif name == "measure_z":
                measures.append(int(op[1]))
            elif name == "barrier":
                # no-op for simulation
                continue
            elif name == "project_z":
                q = int(op[1]); keep = int(op[2])
                rho = self._project_z(rho, q, keep, n)
            elif name == "reset":
                q = int(op[1]); rho = self._project_z(rho, q, 0, n)

        expectations: Dict[str, float] = {}
        for q in measures:
            e = exp_z_density(self.backend, rho, q, n)
            expectations[f"Z{q}"] = float(e)
        return {"expectations": expectations, "metadata": {"shots": shots, "backend": self.backend.name}}

    def expval(self, circuit: "Circuit", obs: Any, **kwargs: Any) -> float:
        return 0.0

    # helpers removed; using gates kernels

    def _apply_noise_if_any(self, rho: np.ndarray, noise: Any, wires: list[int], n: int) -> np.ndarray:
        if not noise:
            return rho
        ntype = str(noise.get("type", "")).lower()
        try:
            if ntype == "depolarizing":
                p = float(noise.get("p", 0.0))
                Ks = noise_channels.depolarizing(p)
                for q in wires:
                    rho = noise_channels.apply_to_density_matrix(rho, Ks, q, n)
            elif ntype == "amplitude_damping":
                g = float(noise.get("gamma", noise.get("g", 0.0)))
                Ks = noise_channels.amplitude_damping(g)
                for q in wires:
                    rho = noise_channels.apply_to_density_matrix(rho, Ks, q, n)
            elif ntype == "phase_damping":
                lmbda = float(noise.get("lambda", noise.get("l", 0.0)))
                Ks = noise_channels.phase_damping(lmbda)
                for q in wires:
                    rho = noise_channels.apply_to_density_matrix(rho, Ks, q, n)
            elif ntype == "pauli":
                Ks = noise_channels.pauli_channel(float(noise.get("px", 0.0)), float(noise.get("py", 0.0)), float(noise.get("pz", 0.0)))
                for q in wires:
                    rho = noise_channels.apply_to_density_matrix(rho, Ks, q, n)
        except Exception:
            return rho
        return rho
    
    def _project_z(self, rho: np.ndarray, qubit: int, keep: int, n: int) -> np.ndarray:
        # Projector |0><0| or |1><1| on `qubit` using apply_1q_density
        if int(keep) == 0:
            P = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        else:
            P = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        rho2 = apply_1q_density(self.backend, rho, P, qubit, n)
        tr = np.trace(rho2)
        if abs(tr) > 0:
            rho2 = rho2 / tr
        return rho2


