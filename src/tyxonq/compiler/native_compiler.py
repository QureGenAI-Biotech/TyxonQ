from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .api import CompileRequest, CompileResult
    from ..core.ir import Circuit
    from ..devices import DeviceCapabilities


class NativeCompiler:
    name = "default"

    def compile(self, request: "CompileRequest") -> "CompileResult":  # type: ignore[override]
        circuit: "Circuit" = request["circuit"]
        target: "DeviceCapabilities" = request.get("target", {})  # type: ignore[assignment]
        options: Dict[str, Any] = request.get("options", {})

        from .pipeline import build_pipeline

        pipeline_names = options.get("pipeline", [
            "rewrite/measurement",
            "scheduling/shot_scheduler",
        ])

        # Auto-derive measurement items when not provided, based on IR ops
        if "measurements" not in options:
            try:
                from types import SimpleNamespace

                derived = []
                for op in getattr(circuit, "ops", []) or []:
                    if isinstance(op, (list, tuple)) and op:
                        if op[0] == "measure_z" and len(op) >= 2:
                            derived.append(SimpleNamespace(wires=(int(op[1]),), obs="Z"))
                if derived:
                    options = dict(options)
                    options["measurements"] = derived
            except Exception:
                pass

        pipe = build_pipeline(pipeline_names)
        lowered = pipe.run(circuit, target, **options)

        # Optional: emit an execution plan when shot_plan/total_shots is provided
        execution_plan = None
        try:
            from .stages.scheduling.shot_scheduler import schedule  # lazy import

            if "shot_plan" in options or "total_shots" in options:
                execution_plan = schedule(
                    lowered,
                    options.get("shot_plan"),
                    total_shots=options.get("total_shots"),
                    caps=target,
                )
        except Exception:
            execution_plan = None

        metadata: Dict[str, Any] = {
            "target": "tyxonq",
            "options": dict(options),
            "caps": dict(target),
            "pipeline": list(pipeline_names),
            "execution_plan": execution_plan,
        }
        return {"circuit": lowered, "metadata": metadata}


