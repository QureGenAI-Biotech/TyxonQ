"""国盾 QCIS 直译降门（TyxonQ IR dialect -> QCIS gate-level dialect）。

设计定位：这是一个"方言编译器"的 lowering pass——源方言是 TyxonQ IR 的 op 元组，
目标方言是国盾 QCIS 的门级指令 {X, H, RZ, CZ, M}。

职责边界（C 阶段起）：GuodunCompiler.compile 先委托 gate_compiler 核心做设备无关分解
（matchgate + rotations），本 lowering 只把编译后的 IR **纯翻译**为 QCIS。故参数化旋转
（rx/ry/rxx/ryy/rzz）与 2-qubit matchgate 的分解恒等式已移至核心（stages/decompose/），
此处不再重复；只保留核心不覆盖的翻译：1q fixed(y/z/s/sdg/t/tdg)→rz/x、cx→H·CZ·H、
以及具名双比特门 swap/iswap/cy/cry（内部所需旋转已就地内联为 h/rz/cx，不回调已删分支）。

为什么不经过 QASM(2 或 3) 中间层：
  - QASM 是第三种方言，IR -> QASM -> QCIS 需要先 serialize 成文本再 parse 回来，
    引入无谓的文本往返与浮点十进制截断，且分解策略受制于 Qiskit / cqlib。
  - 直译全程在 IR op 元组上做恒等式展开：零 serialize/parse 往返、零精度损耗、
    目标方言完全可控，与既有 gd_sim1 验证过的门级 QCIS 方言保持一致。

正确性保证：本模块只使用标准门恒等式（见每条规则注释），并由离线数值校验
（tests 中对 UNITARY_1Q/2Q 逐门比对分解前后 unitary，允许全局相位差）兜底。

约定（与 gates.py 权威 op 词汇表一致）：
  - 1q op: ("name", q) 或 ("name", q, theta)
  - 2q op: ("name", q0, q1) 或 ("name", q0, q1, theta)，q0 为高位/控制位
  - 测量: ops 里的 ("measure_z", q)；instructions 里的 ("measure", (q, ...))
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

_PI = math.pi

# 目标 QCIS 门级 basis（measure_z 会在 emit 时写成 M）。
_QCIS_BASIS = frozenset({"x", "h", "rz", "cz", "measure_z"})

# 非幺正/需专门路径的 op：门级直译不支持，loudly raise（脉冲请走 pulse-QCIS 路径）。
_SPECIAL_OPS = frozenset(
    {"unitary", "kraus", "project_z", "reset", "pulse", "pulse_inline"}
)

# emit 时 basis op -> QCIS 指令助记符。
_QCIS_MNEMONIC = {"x": "X", "h": "H", "rz": "RZ", "cz": "CZ", "measure_z": "M"}


def _theta(op: Sequence[Any]) -> float:
    """取双比特参数门的 theta，兼容 (name,q0,q1,theta) 与 (name,q0,theta) 两种历史写法。"""
    if len(op) > 3:
        return float(op[3])
    return float(op[2])


def _expand(op: Sequence[Any]) -> List[Tuple[Any, ...]]:
    """把单个非 basis 门做**一层**恒等式展开（结果可能仍含非 basis 门，交由递归继续降）。

    所有单比特非 basis 门最终归约到 {x, h, rz}；双比特门归约到 {cz} + 单比特。
    允许全局相位差（对测量计数无影响）。
    """
    name = str(op[0]).lower()

    # ---- 单比特门 ----
    if name == "y":
        # Y = RZ(pi)·X（精确，无相位差）
        q = int(op[1])
        return [("x", q), ("rz", q, _PI)]
    if name == "z":
        # Z ~ RZ(pi)（差全局相位 -i）
        q = int(op[1])
        return [("rz", q, _PI)]
    if name == "s":
        # S ~ RZ(pi/2)
        q = int(op[1])
        return [("rz", q, _PI / 2)]
    if name == "sdg":
        q = int(op[1])
        return [("rz", q, -_PI / 2)]
    if name == "t":
        # T ~ RZ(pi/4)
        q = int(op[1])
        return [("rz", q, _PI / 4)]
    if name == "tdg":
        q = int(op[1])
        return [("rz", q, -_PI / 4)]
    # rx/ry/rxx/ryy/rzz 的参数化旋转分解已由核心 stages/decompose/rotations.py 统一接管
    # （见 GuodunCompiler.compile 委托 DecomposeSpec(rotations=True)），此处不再重复实现。

    # ---- 双比特门 ----
    if name == "cx":
        # CX(c,t) = H(t)·CZ(c,t)·H(t)
        c = int(op[1]); t = int(op[2])
        return [("h", t), ("cz", c, t), ("h", t)]
    if name == "cy":
        # CY(c,t) = RX(pi/2,t)·CZ(c,t)·RX(-pi/2,t)（共轭把 Z 转到 Y）；
        # RX(θ)=H·RZ(θ)·H 就地内联，避免依赖已被核心接管的 rx 分支。
        c = int(op[1]); t = int(op[2])
        return [
            ("h", t), ("rz", t, _PI / 2), ("h", t),
            ("cz", c, t),
            ("h", t), ("rz", t, -_PI / 2), ("h", t),
        ]
    if name == "swap":
        # SWAP = CX(a,b)·CX(b,a)·CX(a,b)
        a = int(op[1]); b = int(op[2])
        return [("cx", a, b), ("cx", b, a), ("cx", a, b)]
    if name == "iswap":
        # iSWAP ~ RZZ(pi/2) 后接 SWAP（iSWAP = e^{i pi/4}·SWAP·RZZ(pi/2)）；
        # RZZ(θ)=CX·RZ(θ,目标)·CX 就地内联，避免依赖已被核心接管的 rzz 分支。
        a = int(op[1]); b = int(op[2])
        return [
            ("cx", a, b), ("rz", b, _PI / 2), ("cx", a, b),
            ("cx", a, b), ("cx", b, a), ("cx", a, b),
        ]
    if name == "cry":
        # CRY(t) = RY(t/2)·CX·RY(-t/2)·CX（目标位）；
        # RY(θ)=RZ(-pi/2)·H·RZ(θ)·H·RZ(pi/2) 就地内联，避免依赖已被核心接管的 ry 分支。
        c = int(op[1]); t = int(op[2]); th = _theta(op)
        half = th / 2
        return [
            ("rz", t, -_PI / 2), ("h", t), ("rz", t, half), ("h", t), ("rz", t, _PI / 2),
            ("cx", c, t),
            ("rz", t, -_PI / 2), ("h", t), ("rz", t, -half), ("h", t), ("rz", t, _PI / 2),
            ("cx", c, t),
        ]
    # rxx/ryy/rzz 的参数化旋转分解已由核心接管（同 rx/ry），此处不再重复实现。
    # swap/iswap/cy/cry 是核心不分解的具名双比特门，故保留在此做方言翻译。

    if name in _SPECIAL_OPS:
        raise ValueError(
            f"Guodun 门级直译不支持特殊 op {name!r}；"
            "脉冲请走 pulse-QCIS 路径，自定义幺正/噪声/复位暂不支持门级提交。"
        )
    raise ValueError(f"Guodun 直译遇到未知门 {name!r}，无法降门到 QCIS basis。")


def _lower_op(op: Sequence[Any]) -> List[Tuple[Any, ...]]:
    """递归把单个 op 降到 QCIS 门级 basis，返回 basis op 列表。"""
    if not isinstance(op, (list, tuple)) or not op:
        return []
    name = str(op[0]).lower()
    if name in _QCIS_BASIS:
        return [tuple(op)]
    if name == "barrier":
        return []  # barrier 对 QCIS 无意义，丢弃
    lowered: List[Tuple[Any, ...]] = []
    for sub in _expand(op):
        lowered += _lower_op(sub)
    return lowered


def _normalize_measures(circuit: Any) -> List[Tuple[Any, ...]]:
    """把 instructions 里的 ("measure", (q,...)) 归一到 ops 里的 ("measure_z", q)。

    与既有 GuodunCompiler 行为一致：已用 measure_z 显式测量的比特不重复添加。
    """
    ops: List[Tuple[Any, ...]] = [tuple(op) for op in getattr(circuit, "ops", []) or []]
    measured = {int(op[1]) for op in ops if op and str(op[0]).lower() == "measure_z"}
    for instr in getattr(circuit, "instructions", []) or []:
        if not instr or str(instr[0]).lower() != "measure":
            continue
        for q in instr[1]:
            if int(q) not in measured:
                ops.append(("measure_z", int(q)))
                measured.add(int(q))
    return ops


def _emit_qcis(basis_ops: Sequence[Sequence[Any]], physical_qubits: Sequence[int]) -> str:
    """把 basis op 序列写成门级 QCIS 文本，逻辑比特 i -> physical_qubits[i]。"""
    lines: List[str] = []
    for op in basis_ops:
        name = str(op[0]).lower()
        mnemonic = _QCIS_MNEMONIC.get(name)
        if mnemonic is None:
            raise ValueError(f"降门后仍出现非 basis 指令 {name!r}")
        if name == "cz":
            q0 = physical_qubits[int(op[1])]
            q1 = physical_qubits[int(op[2])]
            lines.append(f"{mnemonic} Q{q0} Q{q1}")
        elif name == "rz":
            q = physical_qubits[int(op[1])]
            lines.append(f"{mnemonic} Q{q} {float(op[2]):.9g}")
        else:  # x / h / measure_z
            q = physical_qubits[int(op[1])]
            lines.append(f"{mnemonic} Q{q}")
    return "\n".join(lines)


def _gate_stats(basis_ops: Sequence[Sequence[Any]]) -> Dict[str, int]:
    """统计各 QCIS 门出现次数（供 metadata / 校验用）。"""
    stats: Dict[str, int] = {}
    for op in basis_ops:
        key = _QCIS_MNEMONIC[str(op[0]).lower()]
        stats[key] = stats.get(key, 0) + 1
    return stats


def lower_circuit_to_gate_qcis(
    circuit: Any, physical_qubits: Sequence[int]
) -> Tuple[str, Dict[str, int], List[Tuple[Any, ...]]]:
    """把 TyxonQ Circuit 直译降门为门级 QCIS。

    Args:
        circuit: TyxonQ Circuit IR。
        physical_qubits: 逻辑比特 i 对应的物理比特号列表。

    Returns:
        (qcis_text, gate_stats, basis_ops)

    Raises:
        ValueError: 线路缺少显式测量，或含不支持的特殊 op。
    """
    ops = _normalize_measures(circuit)
    if not any(op and str(op[0]).lower() == "measure_z" for op in ops):
        raise ValueError(
            "Guodun 线路必须显式调用 add_measure(...) 或 measure_z(...)"
        )

    basis_ops: List[Tuple[Any, ...]] = []
    for op in ops:
        basis_ops += _lower_op(op)

    qcis = _emit_qcis(basis_ops, physical_qubits)
    return qcis, _gate_stats(basis_ops), basis_ops


__all__ = [
    "lower_circuit_to_gate_qcis",
    "_QCIS_BASIS",
    "_lower_op",
    "_expand",
]
