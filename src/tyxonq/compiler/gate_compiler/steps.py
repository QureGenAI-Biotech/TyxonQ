"""核心编译步骤的装配与执行（取代旧的字符串 DSL build_plan/_resolve_stage）。

设计要点：
  - ``build_compile_steps`` 按**类型化参数**（DecomposeSpec/RoutingSpec/optimize/simplify）
    直接追加步骤对象；未启用的能力不追加占位步骤（不再有 NoOp* 占位类）。
  - ``apply_steps`` 是一个朴素函数：顺序对线路施加每个 ``CompileStep``，返回最终线路。
  - 步骤协议为 ``CompileStep``（方法 ``apply``），定义见 ``compiler.api``。
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, TYPE_CHECKING

from .specs import (
    DecomposeSpec,
    RoutingSpec,
    coerce_decompose,
    coerce_route,
)

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.compiler import CompileStep
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule


def build_compile_steps(
    *,
    basis_gates: Optional[Sequence[str]] = None,
    decompose: "bool | DecomposeSpec" = True,
    route: "bool | RoutingSpec" = False,
    optimize: int = 0,
    simplify: bool = False,
    device_rule: Optional["DeviceRule"] = None,
) -> List["CompileStep"]:
    """按类型化参数装配编译步骤列表。

    固定顺序：
      1. 规范化：``AutoMeasure``（无测量时补 Z 测量）→ ``GatesTransform``（按 basis_gates 重写门）；
      2. 分解：``MatchgateDecompose`` / ``RotationsDecompose``（分解**先于**路由，因 to_qiskit 不认 unitary op）；
      3. 路由：``Routing``（route 为真时追加，委托 qiskit.transpile）；
      4. 优化：``MergePrune``（optimize>=1 时追加）；
      5. 化简：``LightconeSimplify``（simplify 为真时追加）。

    Args:
        basis_gates: 目标原生门集合，透传给 ``GatesTransform``（None 时用其内置默认）。
        decompose: ``bool | DecomposeSpec``，控制分解步骤（True→默认 matchgate 开）。
        route: ``bool | RoutingSpec``，控制是否追加路由步骤（默认 False，不追加）。
        optimize: 优化级别；>=1 时追加 ``MergePrune``。
        simplify: 为真时追加 ``LightconeSimplify``。
        device_rule: 设备能力描述；``RoutingSpec.coupling_map`` 缺省时从中读取耦合图。

    Returns:
        按执行顺序排列的 ``CompileStep`` 列表。
    """
    from .stages.rewrite.auto_measure import AutoMeasure
    from .stages.rewrite.gates_transform import GatesTransform

    steps: List["CompileStep"] = [AutoMeasure(), GatesTransform()]

    dspec = coerce_decompose(decompose)
    if dspec is not None:
        # matchgate 覆盖数守恒 2-qubit unitary；two_qubit_unitary 目前亦经 matchgate 落地
        # （通用 KAK 分解为后续增量），故两者任一为真都追加同一个 MatchgateDecompose。
        if dspec.matchgate or dspec.two_qubit_unitary:
            from .stages.decompose.matchgate import MatchgateDecompose

            steps.append(MatchgateDecompose())
        if dspec.rotations:
            from .stages.decompose.rotations import RotationsDecompose

            steps.append(RotationsDecompose())

    rspec = coerce_route(route)
    if rspec is not None:
        # 路由步骤延迟导入：route=False（默认）时不触发，保持核心装配零额外依赖。
        from .stages.layout.routing import Routing

        steps.append(Routing(rspec, device_rule=device_rule))

    if int(optimize) >= 1:
        from .stages.rewrite.merge_prune import MergePrune

        steps.append(MergePrune())

    if simplify:
        from .stages.simplify.lightcone import LightconeSimplify

        steps.append(LightconeSimplify())

    return steps


def apply_steps(
    circuit: "Circuit", steps: Sequence["CompileStep"], **opts: Any
) -> "Circuit":
    """顺序对 ``circuit`` 施加每个编译步骤，返回最终线路。

    每个步骤的 ``apply(circuit, **opts)`` 自行从 ``opts`` 取所需参数（如 basis_gates、
    device_rule、assume_measure_all 等），未识别的键被 ``**opts`` 吸收，互不干扰。
    """
    current = circuit
    for step in steps:
        current = step.apply(current, **opts)
    return current


__all__ = ["build_compile_steps", "apply_steps"]
