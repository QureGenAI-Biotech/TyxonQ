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
# 单一真相源：权威 op 词汇表 + 门矩阵解析（与 statevector/mps 引擎共用同一门表）
from ....libs.quantum_library.kernels.gate_table import resolve_unitary
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
            # 单一真相源分发：全部幺正门（1q/2q）经权威门表 gate_table.resolve_unitary
            # 解析为 (arity, qubits, matrix)，ρ → G ρ G†。此前这里是手写门分支，缺
            # y/z/t/tdg/cy/iswap/swap/rxx/ryy/rzz/unitary，且对未知 op 静默跳过。
            res = resolve_unitary(name, op, self.backend)
            if res is not None:
                kind, qubits, mat = res
                if kind == "1q":
                    q = int(qubits[0]); rho = apply_1q_density(self.backend, rho, mat, q, n)
                    rho = self._apply_noise_if_any(rho, noise, [q], n)
                else:
                    q0, q1 = int(qubits[0]), int(qubits[1]); rho = apply_2q_density(self.backend, rho, mat, q0, q1, n)
                    rho = self._apply_noise_if_any(rho, noise, [q0, q1], n)
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
            elif name == "unitary":
                # 自定义 k 比特幺正：ρ → U ρ U†（1q 用 apply_1q_density，2q 用 apply_2q_density）
                if len(op) == 3:  # ("unitary", qubit, matrix_key)
                    q = int(op[1]); mat_key = str(op[2])
                    matrix = getattr(circuit, "_unitary_cache", {}).get(mat_key)
                    if matrix is not None:
                        rho = apply_1q_density(self.backend, rho, matrix, q, n)
                elif len(op) == 4:  # ("unitary", q0, q1, matrix_key)
                    q0, q1 = int(op[1]), int(op[2]); mat_key = str(op[3])
                    matrix = getattr(circuit, "_unitary_cache", {}).get(mat_key)
                    if matrix is not None:
                        rho = apply_2q_density(self.backend, rho, matrix, q0, q1, n)
            elif name == "kraus":
                # Handle Kraus channel: ("kraus", qubit, kraus_key) or ("kraus", qubit, kraus_key, status)
                # Note: status is ignored in density matrix simulation (exact evolution)
                q = int(op[1])
                kraus_key = str(op[2])
                kraus_ops = getattr(circuit, "_kraus_cache", {}).get(kraus_key)
                if kraus_ops is not None:
                    rho = apply_kraus_density(rho, kraus_ops, q, n, backend=self.backend)
                    # Note: Kraus channels inherently model noise, no additional noise application needed
            elif name in ("pulse", "pulse_inline"):
                # 密度矩阵表示不提供脉冲级（含 3-level/哈密顿量）演化路径
                raise NotImplementedError(
                    f"DensityMatrixEngine 不支持 op '{name}'：密度矩阵引擎不做脉冲级演化，"
                    f"请改用 statevector 引擎，或先把脉冲编译为幺正门再模拟。"
                )
            else:
                # 单一真相源：未知 op 必须 loudly raise，绝不静默跳过
                raise ValueError(
                    f"DensityMatrixEngine: unsupported op '{name}'. Known ops are "
                    f"defined in libs.quantum_library.kernels.gate_table "
                    f"(unitary/control/special); refusing to silently skip."
                )

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
        # shots=0 解析档：顺带返回精确 probabilities（rho 对角）/ density_matrix，
        # 供 driver 单一源消费（消除对 eng.state() 的第二次独立分发调用）。
        nb = self.backend
        probs_np = np.asarray(nb.real(nb.diag(rho)), dtype=float)
        return {
            "expectations": expectations,
            "probabilities": probs_np,
            "density_matrix": rho,
            "metadata": {"shots": shots, "backend": self.backend.name},
        }

    def expval(self, circuit: "Circuit", obs: Any, **kwargs: Any) -> float:
        try:
            from openfermion.linalg import get_sparse_operator  # type: ignore
        except Exception:
            raise ImportError("expval requires openfermion installed")
        n = int(getattr(circuit, "num_qubits", 0))
        # 单一真相源：复用 state() 的唯一 op 分发求 rho，消除第三套局部循环
        # （此前 expval 自己重算且只覆盖 h/rz/rx/cx 四门，会静默算错含其他门的电路）
        rho = np.asarray(self.state(circuit, **kwargs))
        H = get_sparse_operator(obs, n_qubits=n).toarray()
        e = np.trace(rho @ H)
        return float(np.real(e))

    def state(self, circuit: "Circuit", **kwargs: Any) -> Any:
        """返回密度矩阵 rho（2^n x 2^n）。

        单一真相源：委托 run(shots=0) 的唯一 op 分发循环，避免 run()/expval()/state()
        各维护一套分发而分叉（y/z/t/tdg/cy 等曾因此被静默丢弃）。
        """
        return self.run(circuit, shots=0, **kwargs)["density_matrix"]

    def probabilities(self, circuit: "Circuit", **kwargs: Any) -> Any:
        """返回计算基概率分布（rho 对角，实数、归一）。"""
        return self.run(circuit, shots=0, **kwargs)["probabilities"]

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


