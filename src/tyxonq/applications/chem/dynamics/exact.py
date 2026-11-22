"""数值精确档演化与关联函数（ED 基准 / 验证门参照）。

复用原则（无运行时抽象）：本模块只是 ``scipy.linalg.expm`` +
``observables.build_pauli_matrix`` 的薄封装，服务三类调用方——

1. ``models/spin_lattice`` / ``models/exciton`` 的精确参照（M1 ED 基准）；
2. ``circuits.py`` 线路档与 ``pulse_dynamics.py`` 波形档的交叉验证门；
3. ``sqw.py`` / ``exciton_dynamics.py`` 的关联函数数值档。

单位：原子单位 ħ=1；``times`` 均匀网格且含 t=0（谱学 FFT 前提）。
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

from .observables import PauliItems, build_pauli_matrix


def dense_hamiltonian(ham_items: Iterable[Tuple[float, List[Tuple[str, int]]]],
                      n_qubits: int) -> np.ndarray:
    """Pauli sum H → 稠密矩阵。"""
    return build_pauli_matrix(ham_items, n_qubits)


def evolve_exact(ham_items, psi0, times: Sequence[float], n_qubits: int) -> np.ndarray:
    """精确演化 ``|ψ(t)⟩ = e^{-iHt}|ψ0⟩``，返回 shape (len(times), 2^n)。

    Parameters
    ----------
    ham_items :
        Pauli sum 哈密顿量。
    psi0 :
        初始态矢（长度 2^n）。
    times :
        时间网格（不必均匀；谱学用途需均匀且从 0 开始）。
    """
    H = dense_hamiltonian(ham_items, n_qubits)
    psi = np.asarray(psi0, dtype=np.complex128)
    dim = 1 << int(n_qubits)
    if psi.shape != (dim,):
        raise ValueError(f"psi0 形状 {psi.shape} != ({dim},)。")
    # 对角化一次 + 谱投影（对多时间点远快于逐点 expm）
    evals, evecs = np.linalg.eigh(H)
    coeffs = evecs.conj().T @ psi
    phases = np.exp(-1j * np.asarray(times, dtype=float)[:, None] * evals[None, :])
    states = (phases * coeffs[None, :]) @ evecs.T
    return np.asarray(states, dtype=np.complex128)


def expectation_trajectory(ham_items, obs_items, psi0, times: Sequence[float],
                            n_qubits: int) -> np.ndarray:
    """⟨O(t)⟩ = ⟨ψ0| e^{+iHt} O e^{-iHt} |ψ0⟩``，返回 shape (len(times),)。

    Hermitian 观测取实部；非 Hermitian（如关联函数中的 B）保留复数。
    """
    O = build_pauli_matrix(obs_items, n_qubits)
    states = evolve_exact(ham_items, psi0, times, n_qubits)
    vals = np.einsum("ti,ij,tj->t", states.conj(), O, states)
    if np.allclose(O, O.conj().T):
        return vals.real
    return vals


def correlation_exact(A_items, B_items, ham_items, psi0,
                      times: Sequence[float], n_qubits: int) -> np.ndarray:
    """零温关联函数 ``C(t) = ⟨ψ0| e^{+iHt} A e^{-iHt} B |ψ0⟩``。

    这是谱学引擎（``spectroscopy.py`` → ``sqw.py`` / 激子动力学）的
    数值档真值来源。
    """
    A = build_pauli_matrix(A_items, n_qubits)
    B = build_pauli_matrix(B_items, n_qubits)
    psi = np.asarray(psi0, dtype=np.complex128)
    H = dense_hamiltonian(ham_items, n_qubits)
    evals, evecs = np.linalg.eigh(H)
    coeffs = evecs.conj().T @ psi
    Bc = evecs.conj().T @ (B @ psi)
    t = np.asarray(times, dtype=float)
    # C(t) = Σ_{mn} c_m^* A_mn Bc_n e^{i t (ε_m - ε_n)}
    A_mat = evecs.conj().T @ A @ evecs
    phase = np.exp(1j * (t[:, None, None])
                   * (evals[None, :, None] - evals[None, None, :]))
    return np.einsum("m,tmn,n->t", coeffs.conj(), phase * A_mat[None, :, :], Bc)


def kicked_evolution(ham_items, kick_items, kick_strength: float, psi0,
                     times: Sequence[float], n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
    """±K 踢动演化（线性响应 device 档的数值参照）。

    ``ψ_±(t) = e^{-iHt} e^{∓ i h K} |ψ0⟩``（K 为踢动算符的 Pauli sum），
    差分 ``(⟨O⟩_+ - ⟨O⟩_-)/2h → ∂⟨O⟩/∂h``（h→0 外推，QD-RB6 门）。
    返回 ``(states_plus, states_minus)``，各 shape (len(times), 2^n)。
    """
    K = build_pauli_matrix(kick_items, n_qubits)
    psi = np.asarray(psi0, dtype=np.complex128)
    U_plus = expm(-1j * kick_strength * K)
    U_minus = expm(+1j * kick_strength * K)
    states_plus = evolve_exact(ham_items, U_plus @ psi, times, n_qubits)
    states_minus = evolve_exact(ham_items, U_minus @ psi, times, n_qubits)
    return states_plus, states_minus


__all__ = [
    "dense_hamiltonian",
    "evolve_exact",
    "expectation_trajectory",
    "correlation_exact",
    "kicked_evolution",
    "PauliItems",
]
