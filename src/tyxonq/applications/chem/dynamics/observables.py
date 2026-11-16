"""可观测量纯函数（数值档，轻量、无注册表）。

Pauli sum 表示与 chem 惯例一致（见 ``hamiltonian_builders.py``）::

    pauli_items = [(coeff, [("X", 0), ("Z", 1)]), ...]

态矢 bit 约定（与线路档交叉验证的自洽基准）：
数值档采用 LSB = qubit 0（即 ``np.kron(Q_{n-1}, ..., Q_0)`` 的标准顺序，
整数指标 ``x`` 的第 ``q`` 个二进制位 = qubit ``q`` 的 Z 值）。
线路档（``circuits.py``）的 counts 聚合经 ``expval_pauli_sum`` 后处理
单一出处对齐到同一约定（QD-RB3 bit-order 门）。

单位：原子单位 ħ=1。
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

#: Pauli sum 条目类型：coeff 与 (symbol, qubit) 作用列表
PauliItems = List[Tuple[float, List[Tuple[str, int]]]]

_SINGLES = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def _validate_symbol(symbol: str) -> np.ndarray:
    try:
        return _SINGLES[symbol]
    except KeyError as exc:
        raise ValueError(
            f"非法 Pauli 符号 {symbol!r}；合法值 I/X/Y/Z。"
        ) from exc


def build_pauli_matrix(items: Iterable[Tuple[float, List[Tuple[str, int]]]],
                       n_qubits: int) -> np.ndarray:
    """把 Pauli sum 编译为 ``2^n`` 稠密矩阵（小体系数值档用）。"""
    dim = 1 << int(n_qubits)
    out = np.zeros((dim, dim), dtype=np.complex128)
    for coeff, ops in items:
        if not ops:
            out += float(np.real(coeff)) * np.eye(dim, dtype=np.complex128)
            continue
        # qubit 0 为 LSB（整数指标 x 的第 0 位）：kron 从高比特向低比特
        # 迭代，使 qubit 0 成为最右因子（与 IR/counts 聚合约定一致，已由
        # 「x@q0 → ⟨Z0⟩=-1」线路/数值交叉自检校准）
        occupied = {q: _validate_symbol(s) for s, q in ops}
        mat = np.array([[1.0]], dtype=np.complex128)
        for q in reversed(range(int(n_qubits))):
            mat = np.kron(mat, occupied.get(q, _SINGLES["I"]))
        out += coeff * mat
    return out


def pauli_expectation(state, items: Iterable[Tuple[float, List[Tuple[str, int]]]],
                      n_qubits: int) -> complex:
    """⟨ψ| Σ c_k P_k |ψ⟩（不归一化）。"""
    psi = np.asarray(state)
    if psi.ndim != 1 or psi.shape[0] != (1 << int(n_qubits)):
        raise ValueError("state 必须是一维 2^n 态矢。")
    mat = build_pauli_matrix(items, n_qubits)
    return complex(np.vdot(psi, mat @ psi))


def site_population_items(n_qubits: int) -> List[PauliItems]:
    """每个 site 的占据数布居 |1⟩⟨1|_q = (I - Z_q)/2 的 Pauli items。"""
    out: List[PauliItems] = []
    for q in range(int(n_qubits)):
        out.append([(0.5, []), (-0.5, [("Z", q)])])
    return out


def site_populations(state, n_qubits: int) -> np.ndarray:
    """各 site 布居 ⟨n_q⟩ 数组（长度 n）。"""
    psi = np.asarray(state)
    probs = np.abs(psi) ** 2
    n = int(n_qubits)
    out = np.zeros(n, dtype=np.float64)
    for x in range(len(probs)):
        if probs[x] <= 0:
            continue
        for q in range(n):
            if (x >> q) & 1:
                out[q] += probs[x]
    return out


def bitstring_to_state(bitstring: str, n_qubits: int) -> np.ndarray:
    """从位串构造态矢（LSB = qubit 0：位串右端字符是 qubit 0）。

    例：``"10"``（n=2）→ qubit0=0、qubit1=1 → 整数指标 2。
    """
    s = str(bitstring).replace(" ", "")
    if len(s) != int(n_qubits):
        raise ValueError(f"位串长度 {len(s)} != n_qubits {n_qubits}。")
    x = 0
    for q, ch in enumerate(reversed(s)):
        if ch not in ("0", "1"):
            raise ValueError(f"位串含非二进制字符 {ch!r}。")
        if ch == "1":
            x |= 1 << q
    state = np.zeros(1 << int(n_qubits), dtype=np.complex128)
    state[x] = 1.0
    return state


def state_to_bitstring(state, n_qubits: int, *, tol: float = 1e-12) -> str:
    """基矢态的逆映射（非基矢态报错；用于测试与调试）。"""
    psi = np.asarray(state)
    if np.max(np.abs(psi.imag)) > tol:
        raise ValueError("态矢含虚部，不是基矢态。")
    probs = psi.real
    idx = [i for i, p in enumerate(probs) if abs(p) > tol]
    if len(idx) != 1 or abs(abs(probs[idx[0]]) - 1.0) > 1e-9:
        raise ValueError("态矢不是基矢态（|amp|=1）。")
    x = idx[0]
    return "".join(str((x >> q) & 1) for q in range(int(n_qubits) - 1, -1, -1))


__all__ = [
    "PauliItems",
    "build_pauli_matrix",
    "pauli_expectation",
    "site_population_items",
    "site_populations",
    "bitstring_to_state",
    "state_to_bitstring",
]
