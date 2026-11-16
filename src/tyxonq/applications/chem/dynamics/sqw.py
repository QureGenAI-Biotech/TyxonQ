"""S(q,omega) 动态自旋结构因子引擎（磁材料 Tier 1a：M1 ED 基准 + M2 芯片定标）。

物理口径
--------
零温关联 C_{ij}^{ab}(t) = <psi0| S^a_i(t) S^b_j |psi0>（S^a 取 Pauli
算符 sigma^a，无量纲；1/2 因子由调用方按实验强度约定缩放）。
C(q,t) = sum_{ij} e^{iq(r_i-r_j)} C_{ij}(t)；S(q,omega)：窗 + DFT。

执行档（provider 直切，无运行时抽象）：

- compute_spin_correlations：provider=None 数值精确档（对角化一次 +
  谱投影批量）；provider="simulator" 线路档（初态 + S^b_j Pauli 门 +
  Trotter 块 + Pauli 期望测量）；
- sqw_from_correlations：纯后处理（默认 Z-Z 纵向投影）；
- compute_sqw：直观顶层入口。

M1/M2 验收门（tests/test_qd_sqw.py）：XX 链解析色散 omega(q)=2J|cos q|
对照；数值 vs 线路（simulator）S(q,omega) 峰位一致。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .observables import PauliItems, build_pauli_matrix
from .spectroscopy import apply_window

#: 结果 schema（agent/AI 友好：JSON 可序列化 + 版本化）
SQW_RESULT_SCHEMA = "tyxonq.qd.sqw.v1"

_SPIN_SYMBOLS = ("X", "Y", "Z")
_SPIN_GATES = {"X": "x", "Y": "y", "Z": "z"}


def _site_pauli_items(alpha: str, i: int) -> PauliItems:
    return [(1.0, [(alpha, i)])]


def compute_spin_correlations(
    ham_items: PauliItems,
    psi0,
    times: Sequence[float],
    n_qubits: int,
    *,
    ops: Sequence[str] = ("Z",),
    provider: Optional[str] = None,
    shots: int = 4096,
    init_bitstring: Optional[str] = None,
    **device_kwargs,
) -> Dict[str, Any]:
    """C_{ij}^{ab}(t)（corr[t, i, j, a, b]）。provider=None 数值档。"""
    t_arr = np.asarray(times, dtype=float)
    n = int(n_qubits)
    alphas = tuple(ops)
    for a in alphas:
        if a not in _SPIN_SYMBOLS:
            raise ValueError(f"自旋分量 {a!r} 须为 X/Y/Z。")
    if provider is None:
        return _correlations_exact(ham_items, psi0, t_arr, n, alphas)
    raise ValueError(
        "线路档关联函数须用踢动线性响应（compute_sqw_kicked，provider='simulator'）："
        "直接测 ⟨ψ(t)|S_i S_j|ψ0'⟩ 的线路方案需辅助比特（Hadamard/SWAP test），"
        "当前只开放数值精确档 correlation 与踢动两档。"
    )


def _correlations_exact(ham_items, psi0, t_arr, n, alphas):
    """数值精确档：对角化一次，谱投影批量演化所有 S^b_j|psi0>。"""
    H = build_pauli_matrix(ham_items, n)
    psi = np.asarray(psi0, dtype=np.complex128)
    dim = 1 << n
    if psi.shape != (dim,):
        raise ValueError(f"psi0 形状 {psi.shape} != ({dim},)。")
    evals, evecs = np.linalg.eigh(H)
    phase = np.exp(-1j * t_arr[:, None] * evals[None, :])  # (T, m)
    psi_t = (phase * (evecs.conj().T @ psi)[None, :]) @ evecs.T  # (T, dim)
    # |phi_{jb}(t)> = e^{-iHt} S^b_j|psi0>（谱系数一次批算）
    seeds = np.stack([
        build_pauli_matrix(_site_pauli_items(beta, j), n) @ psi
        for beta in alphas for j in range(n)
    ])  # (nb*n, dim)
    c_seeds = evecs.conj().T @ seeds.T  # (m, nb*n)
    phi_t = np.empty((len(t_arr), n, len(alphas), dim), dtype=np.complex128)
    for col in range(c_seeds.shape[1]):
        b, j = divmod(col, n)
        phi_t[:, j, b, :] = (phase * c_seeds[None, :, col]) @ evecs.T
    S_mats = [[build_pauli_matrix(_site_pauli_items(alpha, i), n)
               for i in range(n)] for alpha in alphas]
    corr = np.zeros((len(t_arr), n, n, len(alphas), len(alphas)),
                    dtype=np.complex128)
    for a in range(len(alphas)):
        for b in range(len(alphas)):
            for j in range(n):
                for i in range(n):
                    corr[:, i, j, a, b] = np.einsum(
                        "td,de,te->t", psi_t.conj(), S_mats[a][i], phi_t[:, j, b]
                    )
    return {"corr": corr, "times": t_arr, "ops": alphas}


def sqw_from_correlations(
    result: Dict[str, Any],
    *,
    n_momentum: int = 64,
    n_frequency: int = 200,
    window: str = "hann",
) -> Dict[str, Any]:
    """C_{ij}(t) -> S(q,omega)（等距格点；默认 Z-Z 纵向分量投影）。"""
    corr = np.asarray(result["corr"])
    t = np.asarray(result["times"], dtype=float)
    if corr.ndim != 5:
        raise ValueError("corr 应为 (T, n, n, n_alpha, n_beta)。")
    ops = list(result.get("ops") or ["Z"])
    idx = ops.index("Z") if "Z" in ops else 0
    zz = corr[:, :, :, idx, idx]  # (T, i, j)
    n = zz.shape[1]
    if len(t) < 2:
        raise ValueError("times 至少两个点。")
    dt = float(t[1] - t[0])
    if dt <= 0 or not np.isclose(t[0], 0) or not np.allclose(np.diff(t), dt):
        raise ValueError("times 必须均匀、从 0 开始。")
    q_grid = np.linspace(0.0, 2.0 * np.pi, int(n_momentum), endpoint=False)
    freqs = np.linspace(0.0, np.pi / dt, int(n_frequency), endpoint=False)
    r = np.arange(n, dtype=float)
    diff = r[:, None] - r[None, :]
    spatial = np.exp(1j * np.outer(q_grid, diff.ravel()))  # (nq, i*j)
    Cq = spatial @ zz.reshape(len(t), -1).T  # (nq, T)
    Cqw = apply_window(Cq.T, t, window=window)  # (T, nq)
    time_phase = np.exp(1j * np.outer(freqs, t))  # (nf, T)
    Sqw = np.abs(dt * (time_phase @ Cqw))  # (nf, nq)
    return {"schema": SQW_RESULT_SCHEMA, "q": q_grid, "omega": freqs, "Sqw": Sqw}


def compute_sqw(
    model,
    times: Sequence[float],
    *,
    psi0=None,
    init_bitstring: Optional[str] = None,
    ops: Sequence[str] = ("Z",),
    provider: Optional[str] = None,
    shots: int = 4096,
    n_momentum: int = 64,
    n_frequency: int = 200,
    window: str = "hann",
    **device_kwargs,
) -> Dict[str, Any]:
    """直观顶层入口：模型 -> S(q,omega) 谱字典。

    provider=None 数值精确档（model.initial_state 或显式 psi0）；
    线路档请用 ``compute_sqw_kicked(provider="simulator")``（踢动线性响应，
    设备友好路线；M2 定标在真机统一验收时启用）。
    """
    if provider is not None:
        return compute_sqw_kicked(
            model, times, provider=provider, shots=shots,
            psi0=psi0, init_bitstring=init_bitstring,
            n_frequency=n_frequency, window=window, **device_kwargs
        )
    ham_items = model.pauli_items()
    n = int(model.n_qubits)
    if psi0 is None:
        psi0 = model.initial_state("dw")
    res = compute_spin_correlations(
        ham_items, psi0, times, n, ops=ops, **device_kwargs
    )
    out = sqw_from_correlations(
        res, n_momentum=n_momentum, n_frequency=n_frequency, window=window
    )
    out["times"] = np.asarray(times, dtype=float)
    return out


__all__ = [
    "compute_spin_correlations",
    "sqw_from_correlations",
    "compute_sqw",
    "SQW_RESULT_SCHEMA",
]


# ---------------- 踢动线性响应（设备友好的线路档路线） ---------------- #

def _kick_items(q: float, positions, n_qubits: int) -> PauliItems:
    """踢动/测量算符 K_q = O_q = sum_j cos(q r_j) X_j 的 Pauli items。"""
    return [(float(np.cos(q * float(positions[j]))), [("X", int(j))])
            for j in range(int(n_qubits))]


def compute_sqw_kicked(
    model,
    times,
    *,
    q_points=None,
    h_kick: float = 0.05,
    provider: Optional[str] = None,
    shots: int = 2048,
    psi0=None,
    init_bitstring: Optional[str] = None,
    n_frequency: int = 200,
    window: str = "hann",
    **device_kwargs,
):
    """踢动线性响应 S(q,omega)（设备友好路线；M2 定标的推荐形式）。

    物理：正负踢动 ``K_q``（每比特 rx 门 ``∓2 h cos(q r_j)``）后的
    期望差分 -> 零温 retarded 响应 -> ``S(q,omega) = -Im G^R(q,omega)``
    （FDT；与 spin/spectra.compute_dsf 同口径，系数因子统一缩放）。

    provider=None 数值精确档（exact.kicked_evolution 谱投影）；
    provider="simulator" 线路档（初态 + rx 踢动门 + Trotter 块 +
    Pauli 期望测量——全部标准门，真机可迁移）。
    """
    from .exact import kicked_evolution
    from .observables import build_pauli_matrix, pauli_expectation
    from .circuits import measure_pauli_expectation, trotter_ops
    from tyxonq.core.ir.circuit import Circuit

    n = int(model.n_qubits)
    ham_items = model.pauli_items()
    positions = list(model.positions) if model.positions is not None else list(range(n))
    t_arr = np.asarray(times, dtype=float)
    if len(t_arr) < 2:
        raise ValueError("times 至少两个点。")
    dt = float(t_arr[1] - t_arr[0])
    q_grid = (np.asarray(q_points, dtype=float) if q_points is not None
              else np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False))
    nq = len(q_grid)
    response = np.zeros((len(t_arr), nq), dtype=np.float64)

    if provider is None:
        if psi0 is None:
            psi0 = model.initial_state("dw")
        for k in range(nq):
            K_items = _kick_items(float(q_grid[k]), positions, n)
            states_plus, states_minus = kicked_evolution(
                ham_items, K_items, float(h_kick), psi0, t_arr, n
            )
            for it in range(len(t_arr)):
                e_p = pauli_expectation(states_plus[it], K_items, n)
                e_m = pauli_expectation(states_minus[it], K_items, n)
                response[it, k] = (float(np.real(e_p)) - float(np.real(e_m))) / 2.0
    elif str(provider) == "simulator":
        if init_bitstring is None:
            init_bitstring = model.initial_bitstring("dw")
        block = trotter_ops(ham_items, dt)
        for k in range(nq):
            q = float(q_grid[k])
            K_items = _kick_items(q, positions, n)
            for it, t in enumerate(t_arr):
                steps = int(round(float(t) / dt))
                vals = []
                for sign in (+1.0, -1.0):
                    circ = Circuit(n)
                    for qubit, ch in enumerate(str(init_bitstring)):
                        if ch == "1":
                            circ.ops.append(("x", qubit))
                    for j in range(n):
                        # rx(theta) = e^{-i theta X / 2}；sign=+1 对应 e^{-i h c X}
                        # （与数值档 kicked_evolution 的 U_plus 同号）
                        ang = sign * 2.0 * float(h_kick) * float(np.cos(q * float(positions[j])))
                        if abs(ang) > 1e-15:
                            circ.ops.append(("rx", j, ang))
                    for _ in range(steps):
                        circ.ops.extend(block)
                    vals.append(measure_pauli_expectation(
                        circ, K_items, n, provider="simulator",
                        shots=shots, **device_kwargs
                    ))
                response[it, k] = (vals[0] - vals[1]) / 2.0
    else:
        raise ValueError(
            f"provider 须为 None（数值）或 'simulator'（线路档），收到 {provider!r}。"
        )

    freqs = np.linspace(0.0, np.pi / dt, int(n_frequency), endpoint=False)
    Cqw = apply_window(response, t_arr, window=window)
    time_phase = np.exp(1j * np.outer(freqs, t_arr))
    Sqw = -(dt * (time_phase @ Cqw)).imag  # -Im G^R（吸收谱口径）
    return {
        "schema": SQW_RESULT_SCHEMA,
        "q": q_grid,
        "omega": freqs,
        "Sqw": Sqw,
        "response": response,
        "times": t_arr,
    }
