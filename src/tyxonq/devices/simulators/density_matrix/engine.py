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
from ....libs.quantum_library.kernels.gates import (
    gate_h, gate_rz, gate_rx, gate_cx_4x4,
    gate_x, gate_ry, gate_cz_4x4, gate_s, gate_sd, gate_cry_4x4,
)
from ....libs.quantum_library.kernels.density_matrix import (
    init_density,
    apply_1q_density,
    apply_2q_density,
    exp_z_density,
    apply_kraus_density,  # Use new kernel implementation
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
            elif name == "ry":
                q = int(op[1]); theta = float(op[2]); rho = apply_1q_density(self.backend, rho, gate_ry(theta), q, n)
                rho = self._apply_noise_if_any(rho, noise, [q], n)
            elif name == "cx":
                c = int(op[1]); t = int(op[2]); rho = apply_2q_density(self.backend, rho, gate_cx_4x4(), c, t, n)
                rho = self._apply_noise_if_any(rho, noise, [c, t], n)
            elif name == "cz":
                c = int(op[1]); t = int(op[2]); rho = apply_2q_density(self.backend, rho, gate_cz_4x4(), c, t, n)
                rho = self._apply_noise_if_any(rho, noise, [c, t], n)
            elif name == "cry":
                c = int(op[1]); t = int(op[2]); theta = float(op[3]); rho = apply_2q_density(self.backend, rho, gate_cry_4x4(theta), c, t, n)
                rho = self._apply_noise_if_any(rho, noise, [c, t], n)
            elif name == "x":
                q = int(op[1]); rho = apply_1q_density(self.backend, rho, gate_x(), q, n)
                rho = self._apply_noise_if_any(rho, noise, [q], n)
            elif name == "s":
                q = int(op[1]); rho = apply_1q_density(self.backend, rho, gate_s(), q, n)
                rho = self._apply_noise_if_any(rho, noise, [q], n)
            elif name == "sdg":
                q = int(op[1]); rho = apply_1q_density(self.backend, rho, gate_sd(), q, n)
                rho = self._apply_noise_if_any(rho, noise, [q], n)
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
            elif name == "kraus":
                # Handle Kraus channel: ("kraus", qubit, kraus_key) or ("kraus", qubit, kraus_key, status)
                # Note: status is ignored in density matrix simulation (exact evolution)
                q = int(op[1])
                kraus_key = str(op[2])
                kraus_ops = getattr(circuit, "_kraus_cache", {}).get(kraus_key)
                if kraus_ops is not None:
                    rho = apply_kraus_density(rho, kraus_ops, q, n, backend=self.backend)
                    # Note: Kraus channels inherently model noise, no additional noise application needed

        # If shots requested and there are measurements, return sampled counts from diagonal of rho
        if shots > 0 and len(measures) > 0:
            nb = self.backend
            diag_b = nb.diag(rho)
            p_np = np.asarray(nb.real(diag_b), dtype=float).copy()
            p_np[p_np < 0.0] = 0.0
            s = float(np.sum(p_np))
            dim = int(p_np.size)
            # Optional noise injection
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
                    p_np = (np.asarray(nb.to_numpy(A), dtype=float) @ p_np).copy()
                elif ntype == "depolarizing":
                    pval = float(noise.get("p", 0.0))
                    alpha = max(0.0, min(1.0, 4.0 * pval / 3.0))
                    p_np = (1.0 - alpha) * p_np + alpha * (1.0 / dim)
                p_np = np.clip(p_np, 0.0, 1.0)
                s = float(np.sum(p_np))
            if s > 0:
                p_np = p_np / s
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
            return {"result": results, "metadata": {"shots": shots, "backend": self.backend.name}}

        expectations: Dict[str, float] = {}
        for q in measures:
            e = exp_z_density(self.backend, rho, q, n)
            expectations[f"Z{q}"] = float(e)
        return {"expectations": expectations, "metadata": {"shots": shots, "backend": self.backend.name}}

    def expval(self, circuit: "Circuit", obs: Any, **kwargs: Any) -> float:
        try:
            from openfermion.linalg import get_sparse_operator  # type: ignore
        except Exception:
            raise ImportError("expval requires openfermion installed")
        n = int(getattr(circuit, "num_qubits", 0))
        # Build rho via run(shots=0)
        _ = self.run(circuit, shots=0, **kwargs)
        # Recompute rho explicitly for expectation (simple, consistent)
        rho = init_density(n)
        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                continue
            name = op[0]
            if name == "h":
                q = int(op[1]); rho = apply_1q_density(self.backend, rho, gate_h(), q, n)
            elif name == "rz":
                q = int(op[1]); theta = float(op[2]); rho = apply_1q_density(self.backend, rho, gate_rz(theta), q, n)
            elif name == "rx":
                q = int(op[1]); theta = float(op[2]); rho = apply_1q_density(self.backend, rho, gate_rx(theta), q, n)
            elif name == "cx":
                c = int(op[1]); t = int(op[2]); rho = apply_2q_density(self.backend, rho, gate_cx_4x4(), c, t, n)
        H = get_sparse_operator(obs, n_qubits=n).toarray()
        e = np.trace(rho @ H)
        return float(np.real(e))

    # helpers removed; using gates kernels

    def _apply_noise_if_any(self, rho: np.ndarray, noise: Any, wires: list[int], n: int) -> np.ndarray:
        """Apply noise channel to density matrix using new kernel implementation."""
        if not noise:
            return rho
        ntype = str(noise.get("type", "")).lower()
        try:
            if ntype == "depolarizing":
                p = float(noise.get("p", 0.0))
                Ks = noise_channels.depolarizing(p)
                for q in wires:
                    rho = apply_kraus_density(rho, Ks, q, n, backend=self.backend)
            elif ntype == "amplitude_damping":
                g = float(noise.get("gamma", noise.get("g", 0.0)))
                Ks = noise_channels.amplitude_damping(g)
                for q in wires:
                    rho = apply_kraus_density(rho, Ks, q, n, backend=self.backend)
            elif ntype == "phase_damping":
                lmbda = float(noise.get("lambda", noise.get("l", 0.0)))
                Ks = noise_channels.phase_damping(lmbda)
                for q in wires:
                    rho = apply_kraus_density(rho, Ks, q, n, backend=self.backend)
            elif ntype == "pauli":
                Ks = noise_channels.pauli_channel(
                    float(noise.get("px", 0.0)),
                    float(noise.get("py", 0.0)),
                    float(noise.get("pz", 0.0))
                )
                for q in wires:
                    rho = apply_kraus_density(rho, Ks, q, n, backend=self.backend)
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


