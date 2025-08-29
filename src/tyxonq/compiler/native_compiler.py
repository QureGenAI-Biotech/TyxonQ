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

        # Default basis_gates for native pipeline (can be overridden by options)
        basis_gates = list(options.get("basis_gates", ["h", "rx", "rz", "cx", "cz"]))
        optimization_level = int(options.get("optimization_level", 0))

        pipeline_names = options.get("pipeline", [
            "rewrite/measurement",
            # Enable simple merge/prune when opt level >= 1
            *( ["rewrite/merge_prune"] if optimization_level >= 1 else [] ),
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
        # Basis rewrite: map a few common gates into chosen basis before pipeline
        def _rewrite_to_basis(c: "Circuit") -> "Circuit":
            new_ops = []
            for op in getattr(c, "ops", []) or []:
                if not (isinstance(op, (list, tuple)) and op):
                    new_ops.append(op)
                    continue
                name = str(op[0]).lower()
                if name == "x" and "rx" in basis_gates:
                    q = int(op[1]); new_ops.append(("rx", q, 3.141592653589793))
                elif name == "y" and "ry" in basis_gates:
                    q = int(op[1]); new_ops.append(("ry", q, 3.141592653589793))
                elif name in ("cx", "cz", "h", "rx", "ry", "rz"):
                    new_ops.append(tuple(op))
                elif name == "rxx" and "rxx" in basis_gates:
                    new_ops.append(tuple(op))
                elif name == "rzz" and "rzz" in basis_gates:
                    new_ops.append(tuple(op))
                elif name == "cy" and "cy" in basis_gates:
                    new_ops.append(tuple(op))
                else:
                    # Unknown or unsupported for rewrite: keep as-is
                    new_ops.append(tuple(op))
            return type(c)(c.num_qubits, ops=new_ops, metadata=dict(getattr(c, "metadata", {})), instructions=list(getattr(c, "instructions", [])))

        circuit = _rewrite_to_basis(circuit)
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
            "basis_gates": list(basis_gates),
            "optimization_level": optimization_level,
            "execution_plan": execution_plan,
        }
        return {"circuit": lowered, "metadata": metadata}


