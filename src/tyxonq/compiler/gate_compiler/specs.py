"""类型化编译规格（spec）对象。

``compile()`` 的分解 / 路由行为由这里的小型类型化 spec 描述，取代旧的字符串列表
DSL（``compile_plan=["decompose/rotations", ...]``）。布尔值会被自动升级为对应 spec
的默认实例，使 ``decompose=True`` 与 ``decompose=DecomposeSpec(...)`` 等价可用，
既保留“一键开关”的简洁，又提供细粒度控制。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


@dataclass
class DecomposeSpec:
    """门分解规格：把高层 / 参数化门降到原生门集合。

    字段:
      matchgate: 分解 2-qubit 数守恒 matchgate（two_mode_fock / fSim 家族）到原生
          ``{rz, rx, h, cx}``。这是核心编译器的旗舰分解特色，默认开启。
      rotations: 分解参数化旋转 ``rx/ry/rxx/ryy/rzz`` 到 ``{h, rz, cx}``。默认关闭
          （许多平台原生支持旋转门，无谓展开反而增加门数）。
      two_qubit_unitary: 分解一般 2-qubit unitary。当前经 matchgate 覆盖数守恒子集；
          通用 KAK 分解列为后续增量。
    """

    matchgate: bool = True
    rotations: bool = False
    two_qubit_unitary: bool = False


@dataclass
class RoutingSpec:
    """布局 / 路由规格：把逻辑比特映射到物理比特并按耦合图插入 SWAP。

    字段:
      coupling_map: 物理比特耦合图 ``[[i, j], ...]``；为 None 时由路由步骤从
          ``device_rule["coupling_map"]`` 读取。
      initial_layout: 初始逻辑→物理布局（物理比特下标列表，长度=线路比特数）；
          为 None 时优先用恒等布局（物理 0..k-1），其诱导子图不连通时再按耦合图
          BFS 自动挑选。
      algorithm: 路由算法，目前委托 ``"qiskit"``（qiskit.transpile）。
      optimization_level: 传给 qiskit.transpile 的优化级别（0-3）。默认 **0**：
          路由步骤只负责布局与 SWAP 插入，不做门集合重写（重写是 ``GatesTransform`` /
          分解步骤的职责）；实测 qiskit 2.1 在 optimization_level>=1 时会把部分单比特门
          合成为 ``u2`` 等 IR 词汇表外的门，导致 ``from_qiskit`` 无法读回。
      seed: 传给 qiskit.transpile 的 ``seed_transpiler``。默认 **0** 而非 qiskit 的
          None（随机）：qiskit 的布局/SWAP 搜索带随机性，同一线路两次编译可能得到
          不同的 SWAP 插入位置，会让真机结果与化学能量无法复现、测试也不稳定。
          显式传 ``None`` 可恢复随机搜索（用于对比多次路由质量）。
    """

    coupling_map: Optional[List[List[int]]] = None
    initial_layout: Optional[Any] = None
    algorithm: str = "qiskit"
    optimization_level: int = 0
    seed: Optional[int] = 0


def coerce_decompose(decompose: Any) -> Optional[DecomposeSpec]:
    """把 ``bool | DecomposeSpec | None`` 归一为 ``DecomposeSpec | None``。

    - ``None`` / ``False`` → ``None``（不追加任何分解步骤）；
    - ``True`` → ``DecomposeSpec()``（默认：matchgate 开、rotations 关）；
    - ``DecomposeSpec`` → 原样返回。
    """
    if decompose is None or decompose is False:
        return None
    if decompose is True:
        return DecomposeSpec()
    if isinstance(decompose, DecomposeSpec):
        return decompose
    raise TypeError(
        f"decompose 必须是 bool 或 DecomposeSpec，收到 {type(decompose).__name__}"
    )


def coerce_route(route: Any) -> Optional[RoutingSpec]:
    """把 ``bool | RoutingSpec | None`` 归一为 ``RoutingSpec | None``。

    - ``None`` / ``False`` → ``None``（不追加路由步骤）；
    - ``True`` → ``RoutingSpec()``（耦合图等从 device_rule 读取）；
    - ``RoutingSpec`` → 原样返回。
    """
    if route is None or route is False:
        return None
    if route is True:
        return RoutingSpec()
    if isinstance(route, RoutingSpec):
        return route
    raise TypeError(
        f"route 必须是 bool 或 RoutingSpec，收到 {type(route).__name__}"
    )


__all__ = [
    "DecomposeSpec",
    "RoutingSpec",
    "coerce_decompose",
    "coerce_route",
]
