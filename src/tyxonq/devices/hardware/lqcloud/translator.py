"""将 TyxonQ 门线路转换为官方 LQCloud 线路对象。"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple


_INSTALL_HINT = "请执行 `pip install 'tyxonq[lqcloud]'`。"


def _quantum_circuit_class():
    """延迟导入可选 SDK，避免未使用 LQCloud 时增加启动依赖。"""
    try:
        from lqcloud import QuantumCircuit
    except ImportError as exc:
        raise RuntimeError(f"LQCloud provider 需要 lqcloud==0.4.2；{_INSTALL_HINT}") from exc
    return QuantumCircuit


def _normalised_ops(circuit: Any) -> List[Tuple[Any, ...]]:
    """合并 TyxonQ 的门操作和末尾 instruction 操作。"""
    ops = [tuple(op) for op in circuit.ops]
    for name, qubits in circuit.instructions:
        if name == "measure":
            ops.extend(("measure_z", int(q)) for q in qubits)
        elif name == "reset":
            ops.extend(("reset", int(q)) for q in qubits)
        elif name == "barrier":
            ops.append(("barrier", *(int(q) for q in qubits)))
        else:
            raise NotImplementedError(f"LQCloud 暂不支持 TyxonQ instruction: {name}")
    return ops


def _split_terminal_measurements(
    ops: Sequence[Tuple[Any, ...]],
) -> Tuple[List[Tuple[Any, ...]], List[int]]:
    """提取线路末尾唯一的一组显式测量。"""
    first_measure = next(
        (index for index, op in enumerate(ops) if op and op[0] == "measure_z"),
        None,
    )
    if first_measure is None:
        raise ValueError(
            "LQCloud 线路必须显式调用 add_measure(...) 或 measure_z(...)。"
        )

    measurement_ops = ops[first_measure:]
    if any(not op or op[0] != "measure_z" for op in measurement_ops):
        raise ValueError("LQCloud 首版只支持位于线路末尾的一组测量。")

    measured_qubits = [int(op[1]) for op in measurement_ops]
    if len(set(measured_qubits)) != len(measured_qubits):
        raise ValueError("LQCloud 首版不支持重复测量同一个逻辑比特。")
    return list(ops[:first_measure]), measured_qubits


def _angle(value: Any, gate: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"LQCloud {gate} 的角度必须是可转换为 float 的数值。") from exc


def _apply_gate(qc: Any, op: Tuple[Any, ...]) -> None:
    """把一个受支持的 TyxonQ 门写入 LQCloud QuantumCircuit。"""
    name = str(op[0]).lower()
    if name in {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "reset"}:
        getattr(qc, name)(int(op[1]))
        return
    if name in {"rx", "ry", "rz"}:
        getattr(qc, name)(_angle(op[2], name), int(op[1]))
        return
    if name in {"cx", "cnot", "cy", "cz", "swap", "iswap"}:
        method = qc.cx if name == "cnot" else getattr(qc, name)
        method(int(op[1]), int(op[2]))
        return
    if name == "barrier":
        qubits: Iterable[int] = (int(q) for q in op[1:])
        qc.barrier(*qubits)
        return
    raise NotImplementedError(f"LQCloud 暂不支持 TyxonQ 门: {name}")


def to_lqcloud(circuit: Any) -> Any:
    """转换为官方 ``lqcloud.QuantumCircuit``，不产生网络请求。"""
    QuantumCircuit = _quantum_circuit_class()
    gate_ops, measured_qubits = _split_terminal_measurements(_normalised_ops(circuit))
    qc = QuantumCircuit(int(circuit.num_qubits), len(measured_qubits))

    for op in gate_ops:
        _apply_gate(qc, op)

    # LQCloud 要求每次测量前有覆盖对应比特的 barrier。
    qc.barrier(*measured_qubits)
    for classical_bit, qubit in enumerate(measured_qubits):
        qc.measure(qubit, classical_bit)
    return qc


__all__ = ["to_lqcloud"]
