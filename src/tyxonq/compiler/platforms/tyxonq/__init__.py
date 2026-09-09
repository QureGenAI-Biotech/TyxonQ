"""TyxonQ 原生平台适配器（gate 编译输出 ir/qasm2/qasm3/qiskit）。

薄适配器：本身不含编译逻辑，统一委托门电路公用核心编译器 `GateCompiler`。
保留为独立平台目录，以与 lqcloud/guodun/qiskit 等平台适配器保持对称结构。
"""

from __future__ import annotations

from ...gate_compiler import GateCompiler


class TyxonQCompiler(GateCompiler):
    """TyxonQ 原生平台编译器：直接委托 gate_compiler 公用核心。"""

    name = "tyxonq"


__all__ = ["TyxonQCompiler", "GateCompiler"]
