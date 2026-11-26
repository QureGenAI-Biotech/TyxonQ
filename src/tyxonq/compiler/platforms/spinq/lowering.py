"""TyxonQ IR 到 SpinQit 原生线路的直接 lowering。

本模块不经过 Qiskit 或 QASM。设备无关分解与路由由 ``GateCompiler`` 完成后，
这里只把编译后的 IR 翻译为 ``spinqit.Circuit``，再由 SpinQit native compiler
生成云后端可执行对象。SDK 仅在实际调用时懒加载。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

_INSTALL_HINT = (
    "请在隔离环境安装 spinqit==0.3.0；其 numpy<2.0.0 依赖与 TyxonQ 主环境冲突，"
    "不要直接降级主环境 numpy。"
)


def _spinqit_api() -> Any:
    try:
        import spinqit
    except ImportError as exc:
        raise RuntimeError(f"SpinQ provider 需要 SpinQit SDK。{_INSTALL_HINT}") from exc
    return spinqit


def _normalised_ops(circuit: Any) -> List[Tuple[Any, ...]]:
    ops = [tuple(op) for op in circuit.ops]
    for name, qubits in circuit.instructions:
        if name == "measure":
            ops.extend(("measure_z", int(q)) for q in qubits)
        elif name == "barrier":
            ops.append(("barrier", *(int(q) for q in qubits)))
        else:
            raise NotImplementedError(f"SpinQ 暂不支持 TyxonQ instruction: {name}")
    return ops


def _split_terminal_measurements(
    ops: Sequence[Tuple[Any, ...]],
) -> Tuple[List[Tuple[Any, ...]], List[int]]:
    first_measure = next(
        (index for index, op in enumerate(ops) if op and op[0] == "measure_z"),
        None,
    )
    if first_measure is None:
        raise ValueError("SpinQ 线路必须显式调用 add_measure(...) 或 measure_z(...)。")
    measurement_ops = ops[first_measure:]
    if any(
        not op or (op[0] != "measure_z" and op[0] != "barrier")
        for op in measurement_ops
    ):
        raise ValueError("SpinQ 首版只支持位于线路末尾的一组测量。")
    measured_qubits = [int(op[1]) for op in measurement_ops if op[0] == "measure_z"]
    if len(set(measured_qubits)) != len(measured_qubits):
        raise ValueError("SpinQ 首版不支持重复测量同一个逻辑比特。")
    return list(ops[:first_measure]), measured_qubits


def _physical_mapping(num_qubits: int, physical_qubits: Any) -> List[int]:
    if physical_qubits is None:
        return list(range(num_qubits))
    if isinstance(physical_qubits, (str, bytes)) or not isinstance(
        physical_qubits, Sequence
    ):
        raise TypeError("physical_qubits 必须是整数序列。")
    if len(physical_qubits) != num_qubits:
        raise ValueError(
            f"physical_qubits 长度必须等于线路比特数 {num_qubits}，"
            f"当前为 {len(physical_qubits)}。"
        )
    mapping: List[int] = []
    for value in physical_qubits:
        if isinstance(value, bool):
            raise TypeError("physical_qubits 只能包含非负整数。")
        try:
            qubit = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("physical_qubits 只能包含非负整数。") from exc
        if qubit < 0:
            raise ValueError("physical_qubits 不能包含负数。")
        mapping.append(qubit)
    if len(set(mapping)) != len(mapping):
        raise ValueError("physical_qubits 不能包含重复物理比特。")
    return mapping


def _angle(value: Any, gate: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"SpinQ {gate} 的角度必须是可转换为 float 的数值。") from exc


def _emit(circ: Any, gate: Any, targets: Sequence[Any], angle: float | None = None) -> None:
    if len(targets) == 1:
        instruction = (gate, targets[0]) if angle is None else (gate, targets[0], angle)
    else:
        qubits = tuple(targets)
        instruction = (gate, qubits) if angle is None else (gate, qubits, angle)
    circ << instruction


def _apply_gate(circ: Any, api: Any, q: Sequence[Any], op: Tuple[Any, ...]) -> List[str]:
    name = str(op[0]).lower()
    single = {
        "i": "I",
        "id": "I",
        "h": "H",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "s": "S",
        "sdg": "Sd",
        "t": "T",
        "tdg": "Td",
    }
    rotations = {"rx": "Rx", "ry": "Ry", "rz": "Rz"}
    two_qubit = {"cx": "CX", "cnot": "CX", "cy": "CY", "cz": "CZ", "swap": "SWAP"}
    three_qubit = {"ccx": "CCX", "cswap": "CSWAP"}

    if name in single:
        _emit(circ, getattr(api, single[name]), [q[int(op[1])]])
        return [single[name]]
    if name in rotations:
        _emit(
            circ,
            getattr(api, rotations[name]),
            [q[int(op[1])]],
            _angle(op[2], name),
        )
        return [rotations[name]]
    if name in two_qubit:
        native_name = two_qubit[name]
        _emit(circ, getattr(api, native_name), [q[int(op[1])], q[int(op[2])]])
        return [native_name]
    if name in three_qubit:
        native_name = three_qubit[name]
        _emit(
            circ,
            getattr(api, native_name),
            [q[int(op[1])], q[int(op[2])], q[int(op[3])]],
        )
        return [native_name]
    if name == "rzz":
        a, b = int(op[1]), int(op[2])
        theta = _angle(op[3], name)
        _emit(circ, api.CX, [q[a], q[b]])
        _emit(circ, api.Rz, [q[b]], theta)
        _emit(circ, api.CX, [q[a], q[b]])
        return ["CX", "Rz", "CX"]
    if name == "barrier":
        _emit(circ, api.BARRIER, [q[int(index)] for index in op[1:]])
        return ["BARRIER"]
    if name == "unitary":
        raise NotImplementedError(
            "SpinQ lowering 不直接接受自定义 unitary；请先由 gate_compiler 核心分解为原生门。"
        )
    raise NotImplementedError(f"SpinQ 暂不支持 TyxonQ 门: {name}")


def to_spinq_circuit(
    circuit: Any,
    *,
    physical_qubits: Sequence[int] | None = None,
    logical_qubit_map: Sequence[int] | None = None,
    apply_physical_mapping: bool = True,
    api: Any = None,
) -> Tuple[Any, Dict[str, Any]]:
    """把 TyxonQ IR 直译为 SpinQit Circuit，并返回映射元数据。

    ``physical_qubits[i]`` 是编译后 IR 中本地比特 ``i`` 对应的物理比特。
    默认在 lowering 烘焙映射；SQC-25 设置 ``apply_physical_mapping=False``，保持
    紧凑 native qnum，并把同一布局交给 ``configure_keep_layout``。测量列表此时
    也必须保持本地逻辑下标，避免在提交前重复使用物理下标。
    """
    api = api or _spinqit_api()
    gate_ops, measured_qubits = _split_terminal_measurements(_normalised_ops(circuit))
    num_qubits = int(circuit.num_qubits)
    mapping = _physical_mapping(num_qubits, physical_qubits)
    logical_map = (
        list(range(num_qubits))
        if logical_qubit_map is None
        else _physical_mapping(num_qubits, logical_qubit_map)
    )
    native_mapping = mapping if apply_physical_mapping else list(range(num_qubits))
    allocated_qubits = max(num_qubits, max(native_mapping, default=-1) + 1)

    circ = api.Circuit()
    q = circ.allocateQubits(allocated_qubits)
    # SpinQ Cloud 拒绝显式 MEASURE 门，云端会在线路末尾自动测量。
    circ.allocateClbits(0)
    emitted: List[str] = []
    mapped_q = [q[native] for native in native_mapping]
    for op in gate_ops:
        emitted.extend(_apply_gate(circ, api, mapped_q, op))

    metadata = {
        "target": "spinqit.Circuit",
        "num_qubits": num_qubits,
        "allocated_qubits": allocated_qubits,
        "gate_stats": dict(Counter(emitted)),
        "physical_qubits": mapping,
        "native_qubits": native_mapping,
        "measured_qubits": measured_qubits,
        "measured_logical_qubits": [logical_map[q] for q in measured_qubits],
        "cloud_measure_qubits": [native_mapping[q] for q in measured_qubits],
        "runtime_keep_layout": None if apply_physical_mapping else mapping,
        "measurement_mode": "cloud_automatic",
        "measurement_map": [
            {
                "classical_bit": classical_bit,
                "ir_qubit": ir_qubit,
                "logical_qubit": logical_map[ir_qubit],
                "physical_qubit": mapping[ir_qubit],
            }
            for classical_bit, ir_qubit in enumerate(measured_qubits)
        ],
        "bit_order": "little_endian",
        "mapping_stage": "compile_time_lowering" if apply_physical_mapping else "runtime_keep_layout",
    }
    return circ, metadata


def to_spinq_exe(
    circuit: Any,
    *,
    physical_qubits: Sequence[int] | None = None,
    logical_qubit_map: Sequence[int] | None = None,
    apply_physical_mapping: bool = True,
    optimization_level: int = 0,
    api: Any = None,
) -> Tuple[Any, Dict[str, Any]]:
    """IR → SpinQit 原生 Circuit → native compiler 可执行对象。"""
    api = api or _spinqit_api()
    level = int(optimization_level)
    if level not in (0, 1, 2, 3):
        raise ValueError("SpinQit native optimization_level 必须为 0、1、2 或 3。")
    native, metadata = to_spinq_circuit(
        circuit,
        physical_qubits=physical_qubits,
        logical_qubit_map=logical_qubit_map,
        apply_physical_mapping=apply_physical_mapping,
        api=api,
    )
    executable = api.get_compiler("native").compile(native, level)
    metadata["target"] = "spinqit.native.executable"
    metadata["native_optimization_level"] = level
    return executable, metadata


__all__ = ["to_spinq_circuit", "to_spinq_exe"]
