"""权威 op 词汇表与门矩阵解析（模拟器引擎分发的单一真相源）。

背景：三个模拟器引擎（statevector / density_matrix / matrix_product_state）曾各自
维护 run() 与 state() 两套 op 分发循环，门集互不一致且对未知 op 静默 ``continue``，
导致 ``cry`` / ``y`` / ``z`` / ``t`` / ``tdg`` / ``cy`` 等门被静默丢弃而算错。

本模块把"哪些 op 是幺正门、各自对应哪个门矩阵、参数在 op 元组的哪个位置"集中定义
**一次**。各引擎的单一 ``_evolve`` 循环通过 :func:`resolve_unitary` 取矩阵，用各自
表示的 apply 内核施加；非幺正的特殊/控制 op 由引擎显式处理；两者都不认识的 op 必须
loudly ``raise``，绝不静默跳过。

约定：
  - 比特序 qubit 0 = 最高有效位（MSB，左端），与 apply_*_statevector / 采样 bitstr 一致。
  - 2q 门的 4x4 矩阵基序为 |q0 q1>，其中第一个参数 q0 为该对的高位（控制位）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from .gates import (
    gate_h,
    gate_x,
    gate_y,
    gate_z,
    gate_s,
    gate_sd,
    gate_t,
    gate_td,
    gate_rx,
    gate_ry,
    gate_rz,
    gate_cx_4x4,
    gate_cy_4x4,
    gate_cz_4x4,
    gate_cry_4x4,
    gate_rxx,
    gate_ryy,
    gate_rzz,
    gate_iswap_4x4,
    gate_swap_4x4,
)

# 1q 幺正门：name -> (has_param, builder(theta, backend) -> 2x2)
# 无参门的 builder 忽略 theta；有参门（rx/ry/rz）从 op[2] 取 theta。
UNITARY_1Q: Dict[str, Tuple[bool, Callable[[Any, Any], Any]]] = {
    "h": (False, lambda th, bk: gate_h()),
    "x": (False, lambda th, bk: gate_x(backend=bk)),
    "y": (False, lambda th, bk: gate_y()),
    "z": (False, lambda th, bk: gate_z()),
    "s": (False, lambda th, bk: gate_s()),
    "sdg": (False, lambda th, bk: gate_sd()),
    "t": (False, lambda th, bk: gate_t()),
    "tdg": (False, lambda th, bk: gate_td()),
    "rx": (True, lambda th, bk: gate_rx(th, backend=bk)),
    "ry": (True, lambda th, bk: gate_ry(th, backend=bk)),
    "rz": (True, lambda th, bk: gate_rz(th, backend=bk)),
}

# 2q 幺正门：name -> (has_param, builder(theta, backend) -> 4x4)
# 无参门从 op[1],op[2] 取 (q0,q1)；有参门（cry/rxx/ryy/rzz）另从 op[3] 取 theta。
UNITARY_2Q: Dict[str, Tuple[bool, Callable[[Any, Any], Any]]] = {
    "cx": (False, lambda th, bk: gate_cx_4x4()),
    "cy": (False, lambda th, bk: gate_cy_4x4()),
    "cz": (False, lambda th, bk: gate_cz_4x4()),
    "iswap": (False, lambda th, bk: gate_iswap_4x4()),
    "swap": (False, lambda th, bk: gate_swap_4x4()),
    "cry": (True, lambda th, bk: gate_cry_4x4(th, backend=bk)),
    "rxx": (True, lambda th, bk: gate_rxx(th, backend=bk)),
    "ryy": (True, lambda th, bk: gate_ryy(th, backend=bk)),
    "rzz": (True, lambda th, bk: gate_rzz(th, backend=bk)),
}

# 控制 op：非幺正、不改变量子态，全引擎接受（measure_z 收集测量比特，barrier 空操作）。
CONTROL_OPS = frozenset({"measure_z", "barrier"})

# 特殊 op：非纯幺正门矩阵，需引擎按各自表示专门处理（不支持则 loudly raise）。
SPECIAL_OPS = frozenset(
    {"unitary", "kraus", "project_z", "reset", "pulse", "pulse_inline"}
)

UNITARY_OPS = frozenset(UNITARY_1Q) | frozenset(UNITARY_2Q)

# 全引擎公认"应当支持或明确处置"的 op 全集（护栏测试据此断言分发完备性）。
KNOWN_OPS = UNITARY_OPS | CONTROL_OPS | SPECIAL_OPS


def resolve_unitary(
    name: str, op: Tuple, backend: Any = None
) -> Optional[Tuple[str, Tuple[int, ...], Any]]:
    """把一个幺正 op 解析为 (arity, qubits, matrix)。

    Returns:
        ("1q", (q,), mat2)  或 ("2q", (q0, q1), mat4)；若 name 不是纯幺正门则返回
        None（交由引擎处理控制/特殊 op 或 raise）。
    """
    if name in UNITARY_1Q:
        has_param, build = UNITARY_1Q[name]
        q = int(op[1])
        th = op[2] if has_param else None
        return ("1q", (q,), build(th, backend))
    if name in UNITARY_2Q:
        has_param, build = UNITARY_2Q[name]
        q0, q1 = int(op[1]), int(op[2])
        th = op[3] if has_param else None
        return ("2q", (q0, q1), build(th, backend))
    return None
