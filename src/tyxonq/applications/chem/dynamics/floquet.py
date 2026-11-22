"""最近邻 Floquet Ising 线路与 Z 基自关联观测量。

模型对应 Mi 等 2021 年 MBL-DTC 实验：
U_F = exp(-i Σ h_i Z_i / 2) exp(-i Σ φ_i Z_iZ_j / 4)
      exp(-i π g Σ X_i / 2)。
初态位串沿用 QD 的 little-endian 文本约定，硬件映射由示例显式提供。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from tyxonq.core.ir.circuit import Circuit


def nearest_neighbor_edges(n_qubits: int) -> list[tuple[int, int]]:
    n = int(n_qubits)
    if n <= 0:
        raise ValueError("n_qubits 必须为正整数")
    return [(i, i + 1) for i in range(n - 1)]


def neel_bitstring(n_qubits: int) -> str:
    """逻辑 site i 的初态为 |i mod 2>，返回 IR 所需的反序文本。"""
    n = int(n_qubits)
    if n <= 0:
        raise ValueError("n_qubits 必须为正整数")
    return "".join(str(i % 2) for i in reversed(range(n)))


def sample_dtc_disorder(
    n_qubits: int,
    seed: int,
    *,
    edges: Optional[Sequence[tuple[int, int]]] = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """论文范围内的固定 PCG64 disorder；保存 seed 即可完全重建。"""
    n = int(n_qubits)
    coupled_edges = list(edges or nearest_neighbor_edges(n))
    rng = np.random.default_rng(int(seed))
    h = rng.uniform(-np.pi, np.pi, size=n)
    phi = rng.uniform(-1.5 * np.pi, -0.5 * np.pi, size=len(coupled_edges))
    return h, phi, coupled_edges


def build_floquet_dtc_circuit(
    n_qubits: int,
    n_cycles: int,
    g: float,
    h: Sequence[float] | np.ndarray,
    phi: Sequence[float] | np.ndarray,
    *,
    edges: Optional[Sequence[tuple[int, int]]] = None,
    init_bitstring: Optional[str] = None,
) -> Circuit:
    """构建初态制备 + 指定周期数的 Floquet 线路，不自动附加测量。"""
    n = int(n_qubits)
    cycles = int(n_cycles)
    if n <= 0 or cycles < 0:
        raise ValueError("n_qubits 必须为正，n_cycles 必须非负")
    if not np.isfinite(g):
        raise ValueError("g 必须为有限值")
    coupled_edges = list(edges or nearest_neighbor_edges(n))
    h_arr = np.asarray(h, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    if h_arr.shape != (n,) or phi_arr.shape != (len(coupled_edges),):
        raise ValueError("h/phi 长度与比特数或耦合边数不一致")
    if not np.all(np.isfinite(h_arr)) or not np.all(np.isfinite(phi_arr)):
        raise ValueError("disorder 参数必须为有限值")
    for a, b in coupled_edges:
        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise ValueError(f"非法耦合边 {(a, b)}")

    bitstring = init_bitstring or neel_bitstring(n)
    if len(bitstring) != n or set(bitstring) - {"0", "1"}:
        raise ValueError("init_bitstring 必须是长度等于 n_qubits 的二进制串")
    circuit = Circuit(n)
    for q, ch in enumerate(reversed(bitstring)):
        if ch == "1":
            circuit.ops.append(("x", q))
    for _ in range(cycles):
        for q in range(n):
            circuit.ops.append(("rx", q, float(np.pi * g)))
        for (a, b), angle in zip(coupled_edges, phi_arr):
            circuit.ops.append(("rzz", a, b, float(angle / 2.0)))
        for q in range(n):
            circuit.ops.append(("rz", q, float(h_arr[q])))
    return circuit


def autocorrelator_items(init_bitstring: str) -> list[tuple[float, list[tuple[str, int]]]]:
    """site/disorder 平均前的 Z_i(0)Z_i(t)/L Pauli sum。"""
    bits = str(init_bitstring)
    n = len(bits)
    if n == 0 or set(bits) - {"0", "1"}:
        raise ValueError("init_bitstring 必须为非空二进制串")
    logical_bits = list(reversed(bits))
    return [
        ((1.0 if logical_bits[q] == "0" else -1.0) / n, [("Z", q)])
        for q in range(n)
    ]


def autocorrelator_from_payload(payload: dict[str, Any], init_bitstring: str) -> float:
    """从统一 counts payload 计算自关联；位序只由公共后处理层解释。"""
    from tyxonq.libs.hamiltonian_encoding.hamiltonian_grouping import (
        group_hamiltonian_pauli_terms,
    )
    from tyxonq.postprocessing import apply_postprocessing

    identity, groups = group_hamiltonian_pauli_terms(
        autocorrelator_items(init_bitstring), len(init_bitstring),
    )
    total = float(identity)
    for _, items in groups.items():
        post = apply_postprocessing(payload, {
            "method": "expval_pauli_sum",
            "identity_const": 0.0,
            "items": items,
        })
        result = post.get("result") or {}
        if "energy" not in result:
            raise ValueError("payload 无法生成自关联期望")
        total += float(result["energy"])
    return total


__all__ = [
    "autocorrelator_from_payload",
    "autocorrelator_items",
    "build_floquet_dtc_circuit",
    "nearest_neighbor_edges",
    "neel_bitstring",
    "sample_dtc_disorder",
]
