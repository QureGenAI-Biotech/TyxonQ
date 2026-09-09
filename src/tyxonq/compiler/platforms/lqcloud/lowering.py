"""LQCloud 直译降门（TyxonQ IR dialect -> LQCloud SDK 对象 dialect）。

设计定位：这是一个"方言编译器"的 lowering——源方言是 TyxonQ IR 的 op/instruction，
目标方言是官方 ``lqcloud.QuantumCircuit`` **对象**（纯对象方言，无 qasm/文本中间层）。

R1 起 ``LQCloudCompiler`` **先委托 gate_compiler 公用核心**（分解 matchgate/rotations +
路由），本模块只负责把核心产出的 IR 做 **1:1 直译**：
  - guodun 目标是门级 QCIS **文本**，客户端必须把 IR 降门到 {X,H,RZ,CZ,M}；
  - lqcloud 目标是 SDK **对象**，原生支持丰富高层门（cx/cz/iswap/... 由 SDK 方法承接），
    降门/转译交给云端，故此处对受支持的门做 1:1 直译（角度以 Python float 直接进 SDK，
    零文本序列化、零精度损耗）。
  - SDK 与我方都未提供的门（如 rxx/ryy/rzz）会在本地 loudly raise，绝不提交。
  - **2-qubit 自定义酉不再由本模块分解**：数守恒 matchgate 已由核心 ``MatchgateDecompose``
    分解为原生 rz/rx/h/cx（单一真相源，消除重复实现）；若仍有 2-qubit 酉走到这里，说明它
    不是 matchgate、核心无法分解，本模块清晰报错而非提交。

约定：不产生任何网络请求；仅在调用 ``to_lqcloud`` 时延迟导入可选 SDK。
"""

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
    # barrier 是编译指令（对量子态无物理作用）。由于 IR 把 barrier 存在 instructions、
    # measure_z 存在 ops，_normalised_ops 合并后 barrier 可能落到测量之后（如 HEA 的
    # RY 层间 barrier）。允许并丢弃这些 barrier；只有真正的门/操作出现在测量之后才拒绝。
    if any(not op or (op[0] != "measure_z" and op[0] != "barrier") for op in measurement_ops):
        raise ValueError("LQCloud 首版只支持位于线路末尾的一组测量。")

    measured_qubits = [int(op[1]) for op in measurement_ops if op[0] == "measure_z"]
    if len(set(measured_qubits)) != len(measured_qubits):
        raise ValueError("LQCloud 首版不支持重复测量同一个逻辑比特。")
    return list(ops[:first_measure]), measured_qubits


def _angle(value: Any, gate: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"LQCloud {gate} 的角度必须是可转换为 float 的数值。") from exc


def _apply_gate(qc: Any, op: Tuple[Any, ...], unitary_cache: Any = None) -> None:
    """把一个受支持的 TyxonQ 门写入 LQCloud QuantumCircuit。"""
    name = str(op[0]).lower()
    if name in {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "reset"}:
        getattr(qc, name)(int(op[1]))
        return
    if name in {"rx", "ry", "rz"}:
        getattr(qc, name)(_angle(op[2], name), int(op[1]))
        return
    if name in {"crx", "cry", "crz"}:
        # SDK 原生受控旋转，签名 cry(theta, control, target)：角度先于比特。
        # TyxonQ op = (name, control, target, theta)。
        getattr(qc, name)(_angle(op[3], name), int(op[1]), int(op[2]))
        return
    if name in {"cx", "cnot", "cy", "cz", "swap", "iswap"}:
        method = qc.cx if name == "cnot" else getattr(qc, name)
        method(int(op[1]), int(op[2]))
        return
    if name == "rzz":
        # SDK 无 rzz；用标准分解 RZZ(t)=CX(a,b)·RZ(b,t)·CX(a,b)（与 guodun lowering 一致）。
        # 约定：rzz(t)=exp(-i·t/2·Z⊗Z)、rz(t)=exp(-i·t/2·Z)，CX 把 Z_b 共轭成 Z_a·Z_b。
        a = int(op[1]); b = int(op[2]); th = _angle(op[3], "rzz")
        qc.cx(a, b)
        qc.rz(th, b)
        qc.cx(a, b)
        return
    if name == "unitary":
        # 矩阵不内联在 op 里，而存于 circuit._unitary_cache[mat_key]（op[-1]），比特在 op[1:-1]。
        import numpy as _np
        mat_key = op[-1]
        qubits = [int(q) for q in op[1:-1]]
        matrix = (unitary_cache or {}).get(mat_key)
        if matrix is None:
            raise ValueError(f"LQCloud unitary 缺少矩阵缓存: {mat_key}")
        if len(qubits) == 1:
            # SDK su2/unitary 原生支持单比特 2x2 自定义酉（1:1 直译）。
            mat = _np.asarray(
                matrix.tolist() if hasattr(matrix, "tolist") else matrix, dtype=complex
            )
            qc.su2(mat, qubits[0])
            return
        # 2-qubit 及以上：lowering 不再自带分解（消除与核心 MatchgateDecompose 的重复实现）。
        # 数守恒 matchgate（LUCJ/SQD 的 two_mode_fock 块）必须已由核心分解为原生 rz/rx/h/cx；
        # 若仍有 2-qubit 酉走到这里，说明它不是 matchgate、核心按设计原样保留，无法提交 LQCloud。
        raise NotImplementedError(
            "LQCloud lowering 只直译单比特自定义酉（su2）；2-qubit 及以上酉必须先由 "
            "gate_compiler 核心分解——数守恒 matchgate 经 DecomposeSpec(matchgate=True)"
            "（默认已开启）分解为原生 rz/rx/h/cx。走到这里通常意味着该 2-qubit 酉不是 "
            "matchgate，核心无法分解，故无法提交 LQCloud。"
        )
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

    unitary_cache = getattr(circuit, "_unitary_cache", {}) or {}
    for op in gate_ops:
        _apply_gate(qc, op, unitary_cache)

    # LQCloud 要求每次测量前有覆盖对应比特的 barrier。
    qc.barrier(*measured_qubits)
    for classical_bit, qubit in enumerate(measured_qubits):
        qc.measure(qubit, classical_bit)
    return qc


__all__ = ["to_lqcloud"]
