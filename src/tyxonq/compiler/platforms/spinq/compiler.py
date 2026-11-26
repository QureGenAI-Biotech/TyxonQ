"""SpinQ 平台编译器：核心 IR 编译后直译为 SpinQit native 可执行对象。"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .lowering import to_spinq_exe

if TYPE_CHECKING:
    from ...api import CompileResult
    from ....core.ir import Circuit
    from ....devices import DeviceRule
    from ...gate_compiler import DecomposeSpec, RoutingSpec

_SUPPORTED_OUTPUTS = ("spinq", "spinq_exe", "native", "object")


class SpinQCompiler:
    """TyxonQ IR → 核心编译 IR → SpinQit native executable。"""

    name = "spinq"

    def compile(
        self,
        circuit: "Circuit",
        *,
        decompose: "bool | DecomposeSpec" = True,
        route: "bool | RoutingSpec" = False,
        device_rule: "DeviceRule | None" = None,
        options: Dict[str, Any] | None = None,
    ) -> "CompileResult":
        opts = dict(options or {})
        output = str(opts.pop("output", "spinq")).lower()
        if output not in _SUPPORTED_OUTPUTS:
            raise ValueError("SpinQ 编译器只支持 output='spinq'（native 可执行对象）。")
        physical_qubits = opts.pop("physical_qubits", None)
        runtime_layout = bool(opts.pop("spinq_runtime_layout", False))
        native_optimization_level = int(opts.pop("spinq_optimization_level", 0))

        from ...gate_compiler import GateCompiler

        core_opts = dict(opts)
        core_opts["output"] = "ir"
        core_result = GateCompiler().compile(
            circuit,
            decompose=decompose,
            route=route,
            device_rule=device_rule,
            options=core_opts,
        )
        compiled_ir = core_result["compiled_source"]

        # 用户显式映射优先；否则使用核心路由回填的 initial_layout。
        # SQC-25 使用运行期 keep_layout，其他平台仍在 lowering 烘焙映射。
        mapping = physical_qubits
        if mapping is None and isinstance(device_rule, dict):
            mapping = device_rule.get("initial_layout")
        routing = getattr(compiled_ir, "metadata", {}).get("routing") or {}
        executable, native_meta = to_spinq_exe(
            compiled_ir,
            physical_qubits=mapping,
            logical_qubit_map=routing.get("final_layout"),
            apply_physical_mapping=not runtime_layout,
            optimization_level=native_optimization_level,
        )

        core_meta = core_result.get("metadata", {}) or {}
        metadata = {
            "output": "spinq",
            "conversion": "core_compile_then_ir_to_spinqit_native",
            "core_steps": core_meta.get("steps"),
            "core": core_meta,
            "routing": routing or None,
            **native_meta,
        }
        return {
            "circuit": circuit,
            "compiled_source": executable,
            "metadata": metadata,
        }


__all__ = ["SpinQCompiler"]
