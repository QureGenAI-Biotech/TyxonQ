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
# 单一真相源：门矩阵与权威 op 词汇表 resolve_unitary 同源于 kernels.gates
# （与 statevector/density_matrix 引擎共用同一门表）
from ....libs.quantum_library.kernels.gates import (
    gate_h, gate_rz, gate_rx, gate_cx_4x4,
    gate_x, gate_ry, gate_cz_4x4, gate_s, gate_sd, gate_cry_4x4,
    resolve_unitary,
)
from ....libs.quantum_library.kernels.matrix_product_state import (
    init_product_state,
    apply_1q as mps_apply_1q,
    apply_2q as mps_apply_2q,
    MPSState,
    to_statevector as mps_to_statevector,
    expectation_pauli_native,
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

    def _evolve(
        self,
        circuit: "Circuit",
        *,
        collect_measures: bool = False,
        noise: Any = None,
        z_atten: list[float] | None = None,
    ):
        """唯一 op 分发循环：把 circuit 演化到 MPS 末态（单一真相源）。

        run() / state() / expectation_pauli() 全部经此，杜绝多套分发彼此分叉
        （y/z/t/tdg/cy/iswap/swap/rxx/ryy/rzz 曾在 run()/state() 两处都被静默丢弃）。
        幺正门经权威门表 gates.resolve_unitary 解析为 (arity, qubits, matrix)，
        用 MPS apply_1q/apply_2q 原地施加；measure_z/barrier 为控制 op；MPS 不支持的
        特殊 op（unitary/kraus/project_z/reset/pulse/pulse_inline）与任何未知 op 一律
        loudly raise，绝不静默跳过。

        返回 (state, measures)：state 为 MPSState（不重构态矢，保持 O(nχ) 内存）；
        measures 仅在 collect_measures=True 时收集 measure_z 比特。
        """
        n = int(getattr(circuit, "num_qubits", 0))
        state = self._init_state(n)
        use_noise = z_atten is not None
        measures: list[int] = []
        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                continue
            name = op[0]
            res = resolve_unitary(name, op, self.backend)
            if res is not None:
                kind, qubits, mat = res
                if kind == "1q":
                    q = int(qubits[0])
                    state = self._apply_1q(state, mat, q, n)
                    if use_noise:
                        self._attenuate(noise, z_atten, [q])
                else:
                    q0, q1 = int(qubits[0]), int(qubits[1])
                    state = self._apply_2q(state, mat, q0, q1, n)
                    if use_noise:
                        self._attenuate(noise, z_atten, [q0, q1])
            elif name == "measure_z":
                if collect_measures:
                    measures.append(int(op[1]))
            elif name == "barrier":
                # Barrier 非量子门（编译/调度指令），对量子态无物理作用
                continue
            elif name in ("unitary", "kraus", "project_z", "reset", "pulse", "pulse_inline"):
                # MPS 表示不支持这些特殊 op（无通用 k 比特幺正 / Kraus / 投影 / 脉冲路径）
                raise NotImplementedError(
                    f"MatrixProductStateEngine 不支持 op '{name}'：MPS 引擎仅提供 1q/2q "
                    f"幺正门与 measure_z/barrier；请改用 statevector 引擎"
                    f"（Kraus 噪声可用 density_matrix 引擎）。"
                )
            else:
                # 单一真相源：未知 op 必须 loudly raise，绝不静默跳过
                raise ValueError(
                    f"MatrixProductStateEngine: unsupported op '{name}'. Known ops are "
                    f"defined in libs.quantum_library.kernels.gates "
                    f"(unitary/control/special); refusing to silently skip."
                )
        return state, measures

    def run(self, circuit: "Circuit", shots: int | None = None, **kwargs: Any) -> Dict[str, Any]:
        shots = int(shots or 0)
        n = int(getattr(circuit, "num_qubits", 0))
        # unified noise interface (explicit switch)
        use_noise = bool(kwargs.get("use_noise", False))
        noise = kwargs.get("noise") if use_noise else None
        z_atten = [1.0] * n if use_noise else None
        # 单一真相源：唯一 op 分发在 _evolve（run()/state()/expectation_pauli() 共用），
        # 覆盖权威门表全部幺正门 + measure_z/barrier；MPS 不支持的特殊 op 与未知 op loudly raise。
        state, measures = self._evolve(
            circuit, collect_measures=True, noise=noise, z_atten=z_atten
        )
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
        # shots=0 解析档：顺带返回精确 probabilities / mps / statevector，供 driver 单一源
        # 消费（state() 则直接取 "mps"，不重构态矢，保持 O(nχ) 内存）。
        p_t = nb.square(nb.abs(psi_b)) if hasattr(nb, "square") else nb.abs(psi_b) ** 2
        probs_full = np.asarray(nb.to_numpy(p_t), dtype=float)
        return {
            "expectations": expectations,
            "probabilities": probs_full,
            "mps": state,
            "statevector": psi_b,
            "metadata": {"shots": shots, "backend": getattr(self.backend, 'name', 'unknown')},
        }

    def expval(self, circuit: "Circuit", obs: Any, **kwargs: Any) -> float:
        # 单一真相源：复用 state()（经 _evolve 的唯一分发）求 MPS，再重构态矢算 <psi|H|psi>。
        # 此前是 `return 0.0` 桩——driver.expval()/device_base 走 MPS 时会静默返回错误期望值。
        try:
            from openfermion.linalg import get_sparse_operator  # type: ignore
        except Exception:
            raise ImportError("expval requires openfermion installed")
        n = int(getattr(circuit, "num_qubits", 0))
        psi = np.asarray(
            mps_to_statevector(self.state(circuit, **kwargs)), dtype=np.complex128
        ).reshape(-1)
        H = get_sparse_operator(obs, n_qubits=n)
        e = np.vdot(psi, H.dot(psi))
        return float(np.real(e))

    def state(self, circuit: "Circuit", **kwargs: Any) -> MPSState:
        """Execute circuit and return MPS representation directly.
        
        Returns the MPS state object without converting to statevector,
        preserving O(nχ) memory complexity.
        
        Args:
            circuit: Circuit to execute
            **kwargs: Additional options
            
        Returns:
            MPSState object with list of site tensors
        """
        # 单一真相源：委托 _evolve 的唯一 op 分发循环，直接返回 MPS 表示（不重构态矢，
        # 保持 O(nχ) 内存）。避免 run()/state() 各维护一套分发而分叉。
        return self._evolve(circuit)[0]

    def expectation_pauli(self, circuit: "Circuit", pauli_ops: list, **kwargs: Any) -> Any:
        """Compute Pauli expectation value directly on MPS (O(nχ³)).
        
        This avoids converting MPS to statevector, maintaining efficient
        memory scaling for large systems with low entanglement.
        
        Args:
            circuit: Circuit to execute
            pauli_ops: List of (gate_matrix, [qubits]) tuples
            **kwargs: Additional options (use_native=True to force native MPS computation)
            
        Returns:
            Complex expectation value ⟨ψ|O|ψ⟩
            
        Example:
            >>> from tyxonq.libs.quantum_library.kernels.gates import gate_x, gate_z
            >>> # Compute ⟨X_0⟩
            >>> exp = eng.expectation_pauli(circuit, [(gate_x(), [0])])
            >>> # Compute ⟨Z_0 Z_1⟩ (requires two separate entries)
            >>> exp = eng.expectation_pauli(circuit, [(gate_z(), [0]), (gate_z(), [1])])
        """
        # Get MPS representation
        mps_state = self.state(circuit, **kwargs)
        
        # Use native MPS computation if requested or for large systems
        use_native = kwargs.get("use_native", True)
        n = len(mps_state.tensors)
        
        if use_native or n > 15:  # Use native for n>15 to save memory
            return expectation_pauli_native(mps_state, pauli_ops)
        else:
            # Fallback to statevector for small systems (easier debugging)
            psi = mps_to_statevector(mps_state)
            # Apply Pauli operators and compute ⟨ψ|O|ψ⟩
            # (This is the old statevector path, kept for compatibility)
            from ....numerics.api import get_backend
            nb = get_backend(None)
            psi_transformed = nb.copy(psi) if hasattr(nb, 'copy') else nb.asarray(psi)
            
            for gate, qubits in pauli_ops:
                if len(qubits) == 1:
                    q = qubits[0]
                    from ....libs.quantum_library.kernels.statevector import apply_1q_statevector
                    psi_transformed = apply_1q_statevector(nb, psi_transformed, gate, q, n)
                elif len(qubits) == 2:
                    q1, q2 = qubits
                    from ....libs.quantum_library.kernels.statevector import apply_2q_statevector
                    # Reshape to 4x4 matrix for 2-qubit gate
                    gate_matrix = nb.asarray(gate)
                    if gate_matrix.shape == (2, 2):
                        # Single Pauli, need to kron with identity for second qubit
                        # This branch shouldn't happen with proper API usage
                        pass
                    psi_transformed = apply_2q_statevector(nb, psi_transformed, gate_matrix, q1, q2, n)
            
            result = nb.tensordot(nb.conj(psi), psi_transformed, axes=([0], [0]))
            return nb.real(result)

    def _attenuate(self, noise: Any, z_atten: list[float], wires: list[int]) -> None:
        ntype = str(noise.get("type", "")).lower() if noise else ""
        if ntype == "depolarizing":
            p = float(noise.get("p", 0.0))
            factor = max(0.0, 1.0 - 4.0 * p / 3.0)
            for q in wires:
                z_atten[q] *= factor


