"""布局 / 路由步骤包（layout）。

包含把逻辑比特映射到物理比特并按耦合图插入 SWAP 的编译步骤：
  - ``routing.Routing``：委托 qiskit.transpile 完成布局与路由。

步骤由 ``gate_compiler.steps.build_compile_steps`` 按类型化 ``RoutingSpec`` 装配；
``route=False``（默认）时**不追加路由步骤**（不再有 NoOp 占位类）。
"""

from __future__ import annotations

from .routing import Routing

__all__ = ["Routing"]
