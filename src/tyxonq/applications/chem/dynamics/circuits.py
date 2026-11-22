"""线路档时间演化（Trotter → Circuit 链式 ``.run()`` → counts 聚合）。

- 演化 = Trotter 块线路化：Pauli 项 ``exp(-i c P dt)`` 经
  ``libs/circuits_library/utils.evolve_pauli_ops``（实现 ``exp(-i φ P/2)``，
  故角度取 ``2*c*dt``）。
- 测量 = 复用 chem 的 Pauli 分组测量惯例：按测量基分组 →
  基旋转前缀（X→H、Y→S†H）+ ``measure_z`` → 统一链式 ``.run()`` →
  ``apply_postprocessing(method="expval_pauli_sum")``。counts→Pauli 期望
  的 bit-order 聚合单一出处即后处理层（QD-RB3 门）。
- 真机等待：``wait_async_result`` setdefault 仅真机（HANDOFF §2.4）。

单位：原子单位 ħ=1；Trotter 步长 ``dt`` 与数值档同口径。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tyxonq.core.ir.circuit import Circuit
from tyxonq.libs.circuits_library.utils import evolve_pauli_ops
from tyxonq.libs.hamiltonian_encoding.hamiltonian_grouping import (
    group_hamiltonian_pauli_terms,
)

from .observables import PauliItems


def trotter_ops(ham_items: Iterable[Tuple[float, List[Tuple[str, int]]]],
                dt: float, *, order: int = 1) -> List[Tuple]:
    """Pauli sum → Trotter 块 ops（单个时间步 dt；order=1 一阶 / 2 对称二阶）。"""
    ops: List[Tuple] = []
    items = [(float(c), ops_list) for c, ops_list in ham_items]
    # utils.evolve_pauli_ops 期望 pauli_string 元素为 (idx, symbol)；
    # chem Pauli items 是 (symbol, idx)——此处做一次显式转换
    to_is = lambda pl: tuple((q, s) for s, q in pl)  # noqa: E731
    if order == 2:
        for coeff, pauli_string in items:
            if pauli_string:
                ops.extend(evolve_pauli_ops(to_is(pauli_string), coeff * dt))
        for coeff, pauli_string in reversed(items):
            if pauli_string:
                ops.extend(evolve_pauli_ops(to_is(pauli_string), coeff * dt))
    else:
        for coeff, pauli_string in items:
            if pauli_string:
                # evolve_pauli_ops(P, φ) = exp(-i φ P / 2) → φ = 2 c dt
                ops.extend(evolve_pauli_ops(to_is(pauli_string), 2.0 * coeff * dt))
    return ops


def build_trotter_circuit(
    ham_items,
    n_qubits: int,
    dt: float,
    n_steps: int,
    *,
    order: int = 1,
    init_bitstring: Optional[str] = None,
) -> Circuit:
    """「初态制备 + n_steps × Trotter 块」演化线路。

    init_bitstring 字符位置 = IR 比特索引（与 counts 聚合同一约定）。
    """
    n = int(n_qubits)
    if n_steps < 0:
        raise ValueError("n_steps 必须非负。")
    circ = Circuit(n)
    if init_bitstring is not None:
        s = str(init_bitstring).replace(" ", "")
        if len(s) != n:
            raise ValueError(f"init_bitstring 长度 {len(s)} != n_qubits {n}。")
        # 位串约定 = little-endian（右端 = qubit 0），与 counts /
        # expval_pauli_sum / bitstring_to_state 同一约定（QD-RB3 门）
        for q, ch in enumerate(reversed(s)):
            if ch == "1":
                circ.ops.append(("x", q))
            elif ch != "0":
                raise ValueError(f"位串含非二进制字符 {ch!r}。")
    block = trotter_ops(ham_items, dt, order=order)
    for _ in range(int(n_steps)):
        circ.ops.extend(list(block))
    return circ


def _basis_prefix_ops(bases: Tuple[str, ...], n_qubits: int) -> List[Tuple]:
    """测量基旋转前缀（X→H、Y→S†H）+ 全比特 measure_z。"""
    ops: List[Tuple] = []
    for q, p in enumerate(bases):
        if p == "X":
            ops.append(("h", q))
        elif p == "Y":
            ops.append(("sdg", q))
            ops.append(("h", q))
    for q in range(int(n_qubits)):
        ops.append(("measure_z", q))
    return ops


def measure_pauli_expectation(
    circuit: Circuit,
    pauli_items: PauliItems,
    n_qubits: int,
    *,
    provider: str = "simulator",
    device: str = "statevector",
    shots: int = 1024,
    noise: Optional[dict] = None,
    **device_kwargs,
) -> float:
    """在已演化线路后测量一个 Pauli sum 的期望（统一链式 ``.run()``）。"""
    from tyxonq.postprocessing import apply_postprocessing

    identity_const, groups = group_hamiltonian_pauli_terms(
        list(pauli_items), int(n_qubits)
    )
    total = float(identity_const)
    for bases, items in groups.items():
        circ = circuit.extended(_basis_prefix_ops(bases, n_qubits))
        # 真机（异步硬件）必须等待结果就绪再取计数（HANDOFF §4.1 教训）
        run_kwargs = dict(device_kwargs)
        run_kwargs.setdefault(
            "wait_async_result", provider not in (None, "simulator", "local", "default")
        )
        out = circ.run(provider=provider, device=device, shots=shots,
                       noise=noise, **run_kwargs)
        payloads = out if isinstance(out, (list, tuple)) else [out]
        for rr in payloads:
            post = apply_postprocessing(rr, {
                "method": "expval_pauli_sum",
                "identity_const": 0.0,
                "items": items,
            })
            total += float((post.get("result") or {}).get("energy", 0.0))
    return total


def trotter_expectation_trajectory(
    ham_items,
    obs_items,
    times: Sequence[float],
    n_qubits: int,
    *,
    dt: float,
    init_bitstring: Optional[str] = None,
    provider: str = "simulator",
    device: str = "statevector",
    shots: int = 1024,
    noise: Optional[dict] = None,
    **device_kwargs,
) -> np.ndarray:
    """线路档 ⟨O(t)⟩ 轨迹（固定 Trotter 步长 dt，每点独立整段演化线路）。

    时间点步数取 ``round(t/dt)``；演化线路按步数缓存、按测量基分组执行。
    """
    t_arr = np.asarray(times, dtype=float)
    steps = np.rint(t_arr / float(dt)).astype(int)
    if np.any(steps < 0):
        raise ValueError("times 含负值。")
    traj = np.zeros(len(t_arr), dtype=np.float64)
    circuits: Dict[int, Circuit] = {}
    for i, k in enumerate(steps):
        if int(k) not in circuits:
            circuits[int(k)] = build_trotter_circuit(
                ham_items, n_qubits, dt, int(k), init_bitstring=init_bitstring
            )
        traj[i] = measure_pauli_expectation(
            circuits[int(k)],
            list(obs_items),
            n_qubits,
            provider=provider,
            device=device,
            shots=shots,
            noise=noise,
            **device_kwargs,
        )
    return traj


__all__ = [
    "trotter_ops",
    "build_trotter_circuit",
    "measure_pauli_expectation",
    "trotter_expectation_trajectory",
]
