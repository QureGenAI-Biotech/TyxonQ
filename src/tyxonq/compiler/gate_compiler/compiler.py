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
            # 若本地未实现 QASM 降级，薄转发到 qiskit 实现
            # output = tyxonq equals to qasm2
            try:
                from ..platforms.qiskit import QiskitCompiler  # type: ignore

                qk_opts = dict(opts)
                # 若未显式指定，使用 qiskit 偏好门集合
                if not qk_opts.get("basis_gates"):
                    qk_opts["basis_gates"] = ["cx", "h", "rz", "rx", "cz"]
                qk_opts["output"] = "qasm2"
                # 需要将 lowered 作为输入传给 qiskit 方言适配
                return QiskitCompiler().compile({"circuit": lowered, "options": qk_opts})  # type: ignore[arg-type]
            except Exception:
                # 降级失败则仍返回 IR
                return {"circuit": circuit, "compiled_source": lowered, "metadata": metadata}
        if output == "qiskit":
            try:
                from ..platforms.qiskit import QiskitCompiler  # type: ignore

                qk_opts = dict(opts)
                # 若未显式指定，使用 qiskit 偏好门集合
                if not qk_opts.get("basis_gates"):
                    qk_opts["basis_gates"] = ["cx", "h", "rz", "rx", "cz"]
                qk_opts["output"] = "qiskit"
                return QiskitCompiler().compile({"circuit": lowered, "options": qk_opts})  # type: ignore[arg-type]
            except Exception:
                return {"circuit": circuit, "compiled_source": lowered, "metadata": metadata}
        # 未识别的 output：返回 IR
        return {"circuit": circuit, "compiled_source": lowered, "metadata": metadata}
