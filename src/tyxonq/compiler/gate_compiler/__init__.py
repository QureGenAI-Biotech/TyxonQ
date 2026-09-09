"""门电路公用核心编译器（gate_compiler）。

设备无关的统一编译核心：收集分解（decompose）、重写（rewrite）、布局/路由（layout）、
化简（simplify）等最先进编译功能。各硬件平台适配器（platforms/*）统一委托本核心完成
编译后，再翻译为自身方言，避免编译逻辑在各平台重复实现。
"""

from __future__ import annotations

from .compiler import GateCompiler
from .specs import DecomposeSpec, RoutingSpec
from .steps import apply_steps, build_compile_steps

__all__ = [
    "GateCompiler",
    "DecomposeSpec",
    "RoutingSpec",
    "build_compile_steps",
    "apply_steps",
]
