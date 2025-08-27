from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING
import warnings

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

        # If circuit contains no explicit measurements, auto-insert Z measurements on all qubits.
        # Emit a non-fatal warning to inform users.
        try:
            has_meas = any(
                (op and isinstance(op, (list, tuple)) and str(op[0]).lower() == "measure_z")
                for op in getattr(circuit, "ops", []) or []
            )
            if not has_meas:
                nq = int(getattr(circuit, "num_qubits", 0))
                if nq > 0:
                    circuit = circuit.extended([("measure_z", q) for q in range(nq)])
                    warnings.warn(
                        "No explicit measurements found; auto-added Z measurements on all qubits during compilation.",
                        UserWarning,
                    )
        except Exception:
            # Best-effort; keep compilation robust
            pass

        # Auto-derive measurement items when not provided, based on IR ops
        if "measurements" not in options:
            # Use core.measurements types to describe intent for downstream stages
            try:
                from ..core.measurements import Expectation  # type: ignore
            except Exception:  # pragma: no cover
                Expectation = None  # type: ignore

            derived = []
            for op in getattr(circuit, "ops", []) or []:
                if isinstance(op, (list, tuple)) and op:
                    if str(op[0]).lower() == "measure_z" and len(op) >= 2:
                        if Expectation is not None:
                            derived.append(Expectation(obs="Z", wires=(int(op[1]),)))
                        else:
                            # Lightweight fallback if core.measurements is unavailable
                            derived.append({"obs": "Z", "wires": (int(op[1]),)})
            if derived:
                options = dict(options)
                options["measurements"] = derived

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


