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
    gate_x, gate_ry, gate_cz_4x4, gate_s, gate_sd, gate_cry_4x4,
    gate_rxx, gate_ryy, gate_rzz,
)
from ....libs.quantum_library.kernels.statevector import (
    init_statevector,
    apply_1q_statevector,
    apply_2q_statevector,
    expect_z_statevector,
    apply_kqubit_unitary,
    apply_kraus_statevector,
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
        state = init_statevector(num_qubits, backend=self.backend)
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
                q = int(op[1]); theta = op[2]; state = apply_1q_statevector(self.backend, state, gate_rz(theta, backend=self.backend), q, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "rx":
                q = int(op[1]); theta = op[2]; state = apply_1q_statevector(self.backend, state, gate_rx(theta, backend=self.backend), q, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "ry":
                q = int(op[1]); theta = op[2]; state = apply_1q_statevector(self.backend, state, gate_ry(theta, backend=self.backend), q, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "cx":
                c = int(op[1]); t = int(op[2]); state = apply_2q_statevector(self.backend, state, gate_cx_4x4(), c, t, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [c, t])
            elif name == "cry":
                c = int(op[1]); t = int(op[2]); theta = op[3]; state = apply_2q_statevector(self.backend, state, gate_cry_4x4(theta, backend=self.backend), c, t, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [c, t])
            elif name == "cz":
                c = int(op[1]); t = int(op[2]); state = apply_2q_statevector(self.backend, state, gate_cz_4x4(), c, t, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [c, t])
            elif name == "rxx":
                c = int(op[1]); t = int(op[2]); theta = op[3]; state = apply_2q_statevector(self.backend, state, gate_rxx(theta, backend=self.backend), c, t, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [c, t])
            elif name == "ryy":
                c = int(op[1]); t = int(op[2]); theta = op[3]; state = apply_2q_statevector(self.backend, state, gate_ryy(theta, backend=self.backend), c, t, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [c, t])
            elif name == "rzz":
                c = int(op[1]); t = int(op[2]); theta = op[3]; state = apply_2q_statevector(self.backend, state, gate_rzz(theta, backend=self.backend), c, t, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [c, t])
            elif name == "x":
                q = int(op[1]); state = apply_1q_statevector(self.backend, state, gate_x(backend=self.backend), q, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "s":
                q = int(op[1]); state = apply_1q_statevector(self.backend, state, gate_s(), q, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
            elif name == "sdg":
                q = int(op[1]); state = apply_1q_statevector(self.backend, state, gate_sd(), q, num_qubits)
                if use_noise and z_atten is not None:
                    self._attenuate(noise, z_atten, [q])
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
            elif name == "unitary":
                # Handle custom unitary gate
                if len(op) == 3:  # 1-qubit unitary: ("unitary", qubit, matrix_key)
                    q = int(op[1])
                    mat_key = str(op[2])
                    matrix = getattr(circuit, "_unitary_cache", {}).get(mat_key)
                    if matrix is not None:
                        state = apply_kqubit_unitary(state, matrix, [q], num_qubits, self.backend)
                        if use_noise and z_atten is not None:
                            self._attenuate(noise, z_atten, [q])
                elif len(op) == 4:  # 2-qubit unitary: ("unitary", q0, q1, matrix_key)
                    q0, q1 = int(op[1]), int(op[2])
                    mat_key = str(op[3])
                    matrix = getattr(circuit, "_unitary_cache", {}).get(mat_key)
                    if matrix is not None:
                        state = apply_kqubit_unitary(state, matrix, [q0, q1], num_qubits, self.backend)
                        if use_noise and z_atten is not None:
                            self._attenuate(noise, z_atten, [q0, q1])
            elif name == "kraus":
                # Handle Kraus channel: ("kraus", qubit, kraus_key) or ("kraus", qubit, kraus_key, status)
                q = int(op[1])
                kraus_key = str(op[2])
                status_val = float(op[3]) if len(op) > 3 else None
                kraus_ops = getattr(circuit, "_kraus_cache", {}).get(kraus_key)
                if kraus_ops is not None:
                    state = apply_kraus_statevector(
                        state, kraus_ops, q, num_qubits, status_val, self.backend
                    )
                    # Note: Kraus channels inherently model noise, no additional attenuation needed
            else:
                # unsupported ops ignored in this minimal engine
                continue

        # If shots requested and there are measurements, return sampled counts over computational basis
        if shots > 0 and len(measures) > 0:
            nb = self.backend
            probs = nb.square(nb.abs(state)) if hasattr(nb, 'square') else nb.abs(state) ** 2  # type: ignore[operator]
            # Sample indices according to probabilities
            rng = nb.rng(None)
            p_np = np.asarray(nb.to_numpy(probs), dtype=float)
            dim = int(p_np.size)
            # Optional noise mixing / readout channel application
            if bool(kwargs.get("use_noise", False)):
                noise = kwargs.get("noise", {}) or {}
                ntype = str(noise.get("type", "")).lower()
                if ntype == "readout":
                    # Apply full calibration matrix A = kron(A0, A1, ...)
                    A = None
                    cals = noise.get("cals", {}) or {}
                    for q in range(num_qubits):
                        m = cals.get(q)
                        if m is None:
                            m = nb.eye(2)
                        m = nb.asarray(m)
                        A = m if A is None else nb.kron(A, m)
                    p_np = np.asarray(nb.to_numpy(A), dtype=float) @ p_np
                elif ntype == "depolarizing":
                    p = float(noise.get("p", 0.0))
                    alpha = max(0.0, min(1.0, 4.0 * p / 3.0))
                    p_np = (1.0 - alpha) * p_np + alpha * (1.0 / dim)
                # Clamp and renormalize
                p_np = np.clip(p_np, 0.0, 1.0)
                s = float(np.sum(p_np))
                p_np = p_np / (s if s > 1e-12 else 1.0)
            if p_np.sum() > 0:
                p_np = p_np / float(p_np.sum())
            else:
                p_np = np.full((dim,), 1.0 / dim, dtype=float)
            idx_samples = nb.choice(rng, dim, size=shots, p=p_np)
            # Bin counts
            idx_samples_backend = nb.asarray(idx_samples)
            counts_arr = nb.bincount(idx_samples_backend, minlength=dim)
            # Build bitstrings in big-endian order (q0 is left)
            n = num_qubits
            results: Dict[str, int] = {}
            nz = nb.nonzero(counts_arr)[0]
            for idx in nz:
                ii = int(idx)
                bitstr = ''.join('1' if (ii >> (n - 1 - k)) & 1 else '0' for k in range(n))
                results[bitstr] = int(nb.to_numpy(counts_arr)[ii])
            return {"result": results, "metadata": {"shots": shots, "backend": self.backend.name}}

        expectations: Dict[str, float] = {}
        for q in measures:
            val = float(expect_z_statevector(state, q, num_qubits))
            if use_noise and z_atten is not None:
                val *= z_atten[q]
            expectations[f"Z{q}"] = val
        return {"expectations": expectations, "metadata": {"shots": shots, "backend": self.backend.name}}

    def expval(self, circuit: "Circuit", obs: Any, **kwargs: Any) -> float:
        try:
            from openfermion.linalg import get_sparse_operator  # type: ignore
        except Exception:
            raise ImportError("expval requires openfermion installed")
        n = int(getattr(circuit, "num_qubits", 0))
        psi = np.asarray(self.state(circuit), dtype=np.complex128).reshape(-1)
        H = get_sparse_operator(obs, n_qubits=n)
        e = np.vdot(psi, H.dot(psi))
        return float(np.real(e))

    # helpers removed; using gates kernels

    def _attenuate(self, noise: Any, z_atten: list[float], wires: list[int]) -> None:
        ntype = str(noise.get("type", "").lower()) if noise else ""
        if ntype == "depolarizing":
            p = float(noise.get("p", 0.0))
            factor = max(0.0, 1.0 - 4.0 * p / 3.0)
            for q in wires:
                z_atten[q] *= factor

    # ---- New public helpers ----
    def state(self, circuit: "Circuit") -> Any:
        """Return final statevector after applying circuit ops.
        
        Returns backend tensor (preserves autograd for PyTorch backend).
        Supports custom initial state via circuit._initial_state.
        """
        n = int(getattr(circuit, "num_qubits", 0))
        
        # Check if circuit has a custom initial state
        initial_state = getattr(circuit, "_initial_state", None)
        if initial_state is not None:
            # Convert initial state to backend tensor (complex128)
            state = self.backend.array(initial_state, dtype=self.backend.complex128)
        else:
            # Default: initialize to |00...0⟩
            state = init_statevector(n, backend=self.backend)
        
        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                continue
            name = op[0]
            if name == "h":
                q = int(op[1]); state = apply_1q_statevector(self.backend, state, gate_h(), q, n)
            elif name == "rz":
                q = int(op[1]); theta = op[2]; state = apply_1q_statevector(self.backend, state, gate_rz(theta, backend=self.backend), q, n)
            elif name == "rx":
                q = int(op[1]); theta = op[2]; state = apply_1q_statevector(self.backend, state, gate_rx(theta, backend=self.backend), q, n)
            elif name == "ry":
                q = int(op[1]); theta = op[2]; state = apply_1q_statevector(self.backend, state, gate_ry(theta, backend=self.backend), q, n)
            elif name == "cx":
                c = int(op[1]); t = int(op[2]); state = apply_2q_statevector(self.backend, state, gate_cx_4x4(), c, t, n)
            elif name == "cz":
                c = int(op[1]); t = int(op[2]); state = apply_2q_statevector(self.backend, state, gate_cz_4x4(), c, t, n)
            elif name == "rxx":
                c = int(op[1]); t = int(op[2]); theta = op[3]; state = apply_2q_statevector(self.backend, state, gate_rxx(theta, backend=self.backend), c, t, n)
            elif name == "ryy":
                c = int(op[1]); t = int(op[2]); theta = op[3]; state = apply_2q_statevector(self.backend, state, gate_ryy(theta, backend=self.backend), c, t, n)
            elif name == "rzz":
                c = int(op[1]); t = int(op[2]); theta = op[3]; state = apply_2q_statevector(self.backend, state, gate_rzz(theta, backend=self.backend), c, t, n)
            elif name == "x":
                q = int(op[1]); state = apply_1q_statevector(self.backend, state, gate_x(backend=self.backend), q, n)
            elif name == "s":
                q = int(op[1]); state = apply_1q_statevector(self.backend, state, gate_s(), q, n)
            elif name == "sdg":
                q = int(op[1]); state = apply_1q_statevector(self.backend, state, gate_sd(), q, n)
            elif name == "project_z":
                q = int(op[1]); keep = int(op[2]); state = self._project_z(state, q, keep, n)
            elif name == "reset":
                q = int(op[1]); state = self._project_z(state, q, 0, n)
            elif name == "unitary":
                # Handle custom unitary gate
                if len(op) == 3:  # 1-qubit: ("unitary", qubit, matrix_key)
                    q = int(op[1])
                    mat_key = str(op[2])
                    matrix = getattr(circuit, "_unitary_cache", {}).get(mat_key)
                    if matrix is not None:
                        state = apply_kqubit_unitary(state, matrix, [q], n, self.backend)
                elif len(op) == 4:  # 2-qubit: ("unitary", q0, q1, matrix_key)
                    q0, q1 = int(op[1]), int(op[2])
                    mat_key = str(op[3])
                    matrix = getattr(circuit, "_unitary_cache", {}).get(mat_key)
                    if matrix is not None:
                        state = apply_kqubit_unitary(state, matrix, [q0, q1], n, self.backend)
            elif name == "kraus":
                # Handle Kraus channel
                q = int(op[1])
                kraus_key = str(op[2])
                status_val = float(op[3]) if len(op) > 3 else None
                kraus_ops = getattr(circuit, "_kraus_cache", {}).get(kraus_key)
                if kraus_ops is not None:
                    state = apply_kraus_statevector(
                        state, kraus_ops, q, n, status_val, self.backend
                    )
        return state

    def probability(self, circuit: "Circuit") -> Any:
        """Return probability vector over computational basis.
        
        Returns backend tensor (numpy array or torch tensor depending on backend).
        """
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
    def _project_z(self, state: Any, qubit: int, keep: int, n: int) -> Any:
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


