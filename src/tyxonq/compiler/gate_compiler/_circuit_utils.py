"""编译步骤重建线路的共享小工具。

背景（为什么需要它）
--------------------
``Circuit`` 是 dataclass，字段只有 ``num_qubits/ops/metadata/instructions``；而
``_unitary_cache``（自定义酉矩阵缓存，形如 ``{mat_key: 4x4 ndarray}``）**不是** dataclass
字段。因此无论用 ``dataclasses.replace`` 还是重新构造 ``Circuit(...)``，都不会把它带过来。

各编译步骤都要“改写 ops 后重建线路”，若忘记回挂 ``_unitary_cache``，后续
``MatchgateDecompose`` 会因 ``cache.get(mat_key) is None`` 抛“matchgate 分解缺少矩阵缓存”。
这正是 lqcloud 方言委托核心分解 LUCJ/SQD 线路的前置阻塞点：核心步骤链
（AutoMeasure → GatesTransform → MatchgateDecompose → RotationsDecompose → Routing → …）
中任何一环丢缓存，委托就会失败。本模块把“重建 + 保留缓存”集中一处，消除各步骤重复的
同一 idiom，并保证缓存贯穿整条链。

架构红线：源 IR 一字不改——本函数只返回**新**线路，绝不原地修改传入的 ``circuit``。
"""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ...core.ir import Circuit


def rebuild_circuit(
    circuit: "Circuit",
    *,
    ops: Optional[List[Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> "Circuit":
    """以新的 ``ops``/``metadata`` 重建线路，并保留非 dataclass 的 ``_unitary_cache``。

    Args:
        circuit: 原线路（不会被修改；本函数只返回新线路）。
        ops: 新的 op 列表；``None`` 表示沿用原 ``ops``（dataclass 字段原样带过来）。
        metadata: 新的 metadata；``None`` 表示沿用原 ``metadata``。

    Returns:
        与 ``circuit`` 同类型的新线路；若原线路带非空 ``_unitary_cache``，则复制到新线路。

    Notes:
        - dataclass 线路走 ``dataclasses.replace``（保留其余字段），非 dataclass 的测试替身
          走类型构造器兜底。
        - ``_initial_state`` 等其它非 dataclass 私有属性沿用既有行为（不在此保留）：本工具
          只针对编译链真正依赖的 ``_unitary_cache``，以免扩大改动面。
    """
    changes: Dict[str, Any] = {}
    if ops is not None:
        changes["ops"] = ops
    if metadata is not None:
        changes["metadata"] = metadata

    if is_dataclass(circuit):
        new_circuit = replace(circuit, **changes)
    else:
        # 兜底：非 dataclass 线路（测试替身等）用类型构造器重建。
        new_circuit = type(circuit)(
            getattr(circuit, "num_qubits", 0),
            ops=changes.get("ops", list(getattr(circuit, "ops", []) or [])),
            metadata=changes.get("metadata", dict(getattr(circuit, "metadata", {}) or {})),
            instructions=list(getattr(circuit, "instructions", []) or []),
        )

    # _unitary_cache 不是 dataclass 字段，replace()/重建都不会带过来，需显式保留，
    # 否则后续 matchgate 分解会因缺少矩阵缓存而抛错。
    cache = getattr(circuit, "_unitary_cache", None)
    if cache:
        new_circuit._unitary_cache = dict(cache)
    return new_circuit


__all__ = ["rebuild_circuit"]
