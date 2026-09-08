"""LQCloud 直译降门（TyxonQ IR dialect -> LQCloud SDK 对象 dialect）。

设计定位：这是一个"方言编译器"的 lowering pass——源方言是 TyxonQ IR 的 op/instruction，
目标方言是官方 ``lqcloud.QuantumCircuit`` **对象**（纯对象方言，无 qasm/文本中间层）。

与 guodun lowering 的关键区别：
  - guodun 目标是门级 QCIS **文本**，客户端必须把 IR 降门到 {X,H,RZ,CZ,M}；
  - lqcloud 目标是 SDK **对象**，原生支持丰富高层门（cx/cz/iswap/... 由 SDK 方法承接），
    降门/转译交给云端，故此处对受支持的门做 1:1 直译（角度以 Python float 直接进 SDK，
    零文本序列化、零精度损耗）。
  - SDK 与我方都未提供的门（如 rxx/ryy/rzz）会在本地 loudly raise，绝不提交。

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


# 物理依据：fSim 门可分解为 iSWAP 型相互作用 + CPhase。
# LUCJ/SQD 的 two_mode_fock_matrix 正是 fSim 家族的 2-qubit matchgate（比裸 fSim
# 多出的单比特相位自由度用两侧 Rz 吸收），故能按此思路降到 lqcloud 原生门。
def _decompose_two_qubit_unitary(matrix: Any, q0: int, q1: int) -> List[Tuple[Any, ...]]:
    """把 2-qubit 数守恒 matchgate 分解成 lqcloud 原生门序列（TyxonQ op 词汇）。

    适用对象：LUCJ/SQD 的 ``two_mode_fock_matrix(v)``——fSim 家族的 2-qubit matchgate，
    4x4 结构为 |00>→1、单激发子空间 {|01>,|10>} 上作用 W=XvX、|11>→det(v)。

    分解原理（忽略全局相位，全部用 lqcloud 原生 rz/rx/h/cx 实现）：
      1. |11> 相位 det(v)=e^{2iμ} 由 e^{iμN}（N=总粒子数）承载 = 两比特各一个 Rz(μ)；
      2. 去掉该相位后 S=e^{-iμ}W∈SU(2)，在单激发子空间按 ZXZ 欧拉角分解 S=Rz(a)Rx(b)Rz(c)；
      3. 子空间 Rz = 两比特 Rz 之差（单激发基 (|01>,|10>) 下 σ_z^sub=(Z0-Z1)/2）；子空间
         Rx(b)=exp(-i(b/2)X_sub) = partial-iSWAP = RXX(b/2)·RYY(b/2)（XX 与 YY 对易），
         再用 RXX=(H⊗H)RZZ(H⊗H)、RYY=(Rx⊗Rx)RZZ(Rx†⊗Rx†)、RZZ=cx·rz·cx 落到原生门
         （lqcloud 无 rxx/ryy/rzz/fsim）。

    返回按“时间顺序”（先作用者在前）排列的 op 列表，可回放给 lqcloud SDK 或 TyxonQ Circuit。
    """
    import numpy as _np
    from math import atan2, pi

    mat = _np.asarray(
        matrix.tolist() if hasattr(matrix, "tolist") else matrix, dtype=complex
    )
    if mat.shape != (4, 4):
        raise ValueError("2-qubit unitary 必须是 4x4 矩阵。")

    tol = 1e-8
    # 校验数守恒 matchgate 结构：|00>/|11> 与单激发子空间之间不得有耦合。
    leakage = [
        mat[0, 1], mat[0, 2], mat[0, 3],
        mat[1, 0], mat[2, 0], mat[3, 0],
        mat[1, 3], mat[2, 3], mat[3, 1], mat[3, 2],
    ]
    if max(abs(x) for x in leakage) > tol:
        raise NotImplementedError(
            "LQCloud 2-qubit unitary 分解仅支持数守恒 matchgate（two_mode_fock 形式）；"
            "检测到跨粒子数子空间耦合，无法按 fSim 家族分解。"
        )

    # |00> 相位归一到 1（全局相位，物理无关），使分解对一般对角 A 也成立。
    a00 = complex(mat[0, 0])
    if abs(a00) > tol:
        mat = mat * (abs(a00) / a00)

    w_block = mat[1:3, 1:3]
    delta = complex(mat[3, 3])
    mu = float(_np.angle(delta)) / 2.0
    # S = e^{-iμ}·W ∈ SU(2)（det S = e^{-2iμ}·det W = e^{-2iμ}·e^{2iμ} = 1）。
    su2 = _np.exp(-1j * mu) * w_block
    s00, s10 = complex(su2[0, 0]), complex(su2[1, 0])

    # ZXZ 欧拉角：S = Rz(a)·Rx(b)·Rz(c)，b∈[0,π]（退化情形未定角乘以~0 项，重建仍精确）。
    b = 2.0 * atan2(abs(s10), abs(s00))
    a_plus_c = -2.0 * float(_np.angle(s00)) if abs(s00) > 1e-12 else 0.0
    a_minus_c = 2.0 * float(_np.angle(s10)) + pi if abs(s10) > 1e-12 else 0.0
    a = (a_plus_c + a_minus_c) / 2.0
    c = (a_plus_c - a_minus_c) / 2.0
    half_b = b / 2.0

    ops: List[Tuple[Any, ...]] = []
    eps = 1e-12
    # (1) c 端 ZXZ（最先作用）：exp(-i(c/2)σ_z^sub) = Rz_q0(c/2)·Rz_q1(-c/2)
    #     （单激发子空间基 (|01>,|10>) 下 σ_z^sub = |01><01|-|10><10| = (Z0-Z1)/2）
    if abs(c) > eps:
        ops.append(("rz", q0, c / 2.0))
        ops.append(("rz", q1, -c / 2.0))
    # (2)(3) 子空间 Rx(b) = partial-iSWAP = RYY(b/2)·RXX(b/2)；b≈0 时整块为单位元，
    #        跳过以省真机门数（每个冗余 cx 都会引入噪声）。
    if abs(half_b) > eps:
        # RYY(b/2) = (Rx(-π/2)⊗Rx(-π/2))·RZZ(b/2)·(Rx(π/2)⊗Rx(π/2))
        ops.append(("rx", q0, -pi / 2))
        ops.append(("rx", q1, -pi / 2))
        ops.append(("cx", q0, q1))
        ops.append(("rz", q1, half_b))
        ops.append(("cx", q0, q1))
        ops.append(("rx", q0, pi / 2))
        ops.append(("rx", q1, pi / 2))
        # RXX(b/2) = (H⊗H)·RZZ(b/2)·(H⊗H)
        ops.append(("h", q0))
        ops.append(("h", q1))
        ops.append(("cx", q0, q1))
        ops.append(("rz", q1, half_b))
        ops.append(("cx", q0, q1))
        ops.append(("h", q0))
        ops.append(("h", q1))
    # (4) a 端 ZXZ：exp(-i(a/2)σ_z^sub) = Rz_q0(a/2)·Rz_q1(-a/2)
    if abs(a) > eps:
        ops.append(("rz", q0, a / 2.0))
        ops.append(("rz", q1, -a / 2.0))
    # (5) e^{iμN} = Rz_q0(μ)·Rz_q1(μ)（承载 |11> 相位 det(v)，全局相位 e^{iμ} 忽略）
    if abs(mu) > eps:
        ops.append(("rz", q0, mu))
        ops.append(("rz", q1, mu))
    return ops


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
        mat = _np.asarray(
            matrix.tolist() if hasattr(matrix, "tolist") else matrix, dtype=complex
        )
        if len(qubits) == 1:
            # SDK su2/unitary 原生支持单比特 2x2 自定义酉（1:1 直译）。
            qc.su2(mat, qubits[0])
            return
        if len(qubits) == 2:
            # 2-qubit 数守恒 matchgate（LUCJ/SQD 的 two_mode_fock 轨道旋转块）：
            # SDK 的 su2 只吃单比特 2x2，故按 fSim 家族结构分解成原生 rz/rx/h/cx 后回放。
            for sub_op in _decompose_two_qubit_unitary(mat, qubits[0], qubits[1]):
                _apply_gate(qc, sub_op, unitary_cache)
            return
        raise NotImplementedError("LQCloud 暂不支持 3 比特及以上的自定义酉。")
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
