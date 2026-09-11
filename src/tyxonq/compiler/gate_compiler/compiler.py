from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from .specs import DecomposeSpec, RoutingSpec
from .steps import apply_steps, build_compile_steps

if TYPE_CHECKING:
    from ..api import CompileResult
    from ...core.ir import Circuit
    from ...devices import DeviceRule


class GateCompiler:
    """门电路公用核心编译器（设备无关）。

    统一收集分解（decompose）、重写（rewrite）、布局/路由（layout）、化简（simplify）
    等最先进编译功能。各硬件平台适配器（platforms/*）均委托本核心完成编译后，
    再翻译为自身方言，避免逻辑在各平台重复实现。

    编译能力由**类型化参数**（basis_gates/decompose/route/optimize/simplify）驱动：
    ``build_compile_steps`` 据此装配一串 ``CompileStep``，``apply_steps`` 顺序施加。
    """

    name = "tyxonq"

    def compile(
        self,
        circuit: "Circuit",
        *,
        basis_gates: Any = None,
        decompose: "bool | DecomposeSpec" = True,
        route: "bool | RoutingSpec" = False,
        optimize: int = 0,
        simplify: bool = False,
        device_rule: "DeviceRule | None" = None,
        options: Dict[str, Any] | None = None,
    ) -> "CompileResult":  # type: ignore[override]
        opts = dict(options or {})
        # device_rule 是独立参数，避免经 options 重复传入而与显式关键字冲突。
        opts.pop("device_rule", None)
        output: str = str(opts.get("output", "ir")).lower()

        # basis_gates：显式参数优先，其次 options；均无则 None（GatesTransform 用自身默认门集合）。
        if basis_gates is None:
            basis_gates = opts.get("basis_gates")
        if basis_gates:
            basis_gates = list(basis_gates)
            opts["basis_gates"] = basis_gates

        # qasm2 输出：qelib1.inc 门集有限，若未显式指定 basis_gates，默认先降级到 qelib1
        # 兼容基，保证 GateQASM2Exporter 不会遇到 iswap/rxx/ryy/rzz 等非 qelib1 门。
        if output in ("qasm", "qasm2") and not basis_gates:
            basis_gates = ["cx", "h", "rz", "rx", "cz"]
            opts["basis_gates"] = basis_gates

        # optimize 与 options.optimization_level 归一（options 显式值优先）。
        optimization_level = int(opts.get("optimization_level", optimize))
        opts["optimization_level"] = optimization_level

        if not device_rule:
            device_rule = {}

        # 按类型化参数装配编译步骤并顺序施加（规范化 → 分解 → 路由 → 优化 → 化简）。
        steps = build_compile_steps(
            basis_gates=basis_gates,
            decompose=decompose,
            route=route,
            optimize=optimization_level,
            simplify=simplify,
            device_rule=device_rule,
        )
        lowered = apply_steps(circuit, steps, device_rule=device_rule, **opts)
        step_names = [str(getattr(s, "name", type(s).__name__)) for s in steps]

        metadata: Dict[str, Any] = {
            "output": output,
            "options": dict(opts),
            "device_rule": dict(device_rule),
            "steps": step_names,
            "basis_gates": list(basis_gates or []),
            "optimization_level": optimization_level,
        }
        # 输出格式选择：
        if output in ("ir","tyxonq"):
            return {"circuit": circuit, "compiled_source": lowered, "metadata": metadata}
        if output in ("qasm3", "qasm3.0", "openqasm3", "openqasm3.0"):
            # Export gate-level circuit to OpenQASM 3.0 format
            from .exporters.gate_qasm3_exporter import GateQASM3Exporter
            exporter = GateQASM3Exporter()
            qasm3_code = exporter.export(lowered)
            return {"circuit": circuit, "compiled_source": qasm3_code, "metadata": metadata}
        if output in ("qasm", "qasm2"):
            # 纯 QASM2 exporter（对齐 qasm3，零外部依赖，不再绕道 qiskit）。
            # 上面已保证降到 qelib1 兼容基，故此处直译即可拿到真正的 QASM2 字符串。
            from .exporters.gate_qasm2_exporter import GateQASM2Exporter

            qasm2_code = GateQASM2Exporter().export(lowered)
            return {"circuit": circuit, "compiled_source": qasm2_code, "metadata": metadata}
        if output == "qiskit":
            # qiskit 互操作输出目标：纯 IR → qiskit.QuantumCircuit 转换（不做 transpile，
            # 布局布线是 routing 步骤的职责）。qiskit 不可用时 to_qiskit 会抛清晰错误。
            from .interop.qiskit_interop import to_qiskit

            add_measures = bool(opts.get("add_measures", True))
            qc = to_qiskit(lowered, add_measures=add_measures)
            return {"circuit": qc, "compiled_source": qc, "metadata": metadata}
        # 未识别的 output：返回 IR
        return {"circuit": circuit, "compiled_source": lowered, "metadata": metadata}
