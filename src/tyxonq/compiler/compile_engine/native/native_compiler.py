from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING
from .compile_plan import build_plan
from ...stages.scheduling.shot_scheduler import schedule 

if TYPE_CHECKING:
    from ...api import CompileResult
    from ....core.ir import Circuit
    from ....devices import DeviceRule


class NativeCompiler:
    name = "default"

    def compile(self,circuit: "Circuit", options:Dict[str, Any] = {},compile_plan=None,device_rule=None) -> "CompileResult":  # type: ignore[override]
        output: str = str(options.get("output", "ir")).lower()
        # Default basis_gates for native pipeline (can be overridden by options)
        if not options.get("basis_gates",None):
            basis_gates = ["h", "rx", "rz", "cx", "cz"]
            options['basis_gates'] = basis_gates
        
        optimization_level = int(options.get("optimization_level", 0))
        options['optimization_level'] = optimization_level

        # Resolve compile_plan: prefer  compile_plan → request.compile_plan → []
        
        if compile_plan:
            request_compile_plan = compile_plan
        else:
            request_compile_plan = []
        if type(request_compile_plan) is not list:
            request_compile_plan = [request_compile_plan]
        # Always preprend essential normalization passes
        final_pipeline = [
            "rewrite/auto_measure",
            "rewrite/gates_transform",
            *request_compile_plan,
        ]
        plan = build_plan(final_pipeline)
        # If Hamiltonian/QubitOperator provided in options, pass through to rewrite pass

        lowered = plan.execute_plan(circuit, **options)

        if not device_rule:
            device_rule = {}

        job_plan = None
        if "shot_plan" in options or "total_shots" in options:
            job_plan = schedule(
                lowered,
                options.get("shot_plan"),
                total_shots=options.get("total_shots"),
                device_rule=device_rule,
            )


        metadata: Dict[str, Any] = {
            "output": output,
            "options": dict(options),
            "device_rule": dict(device_rule),
            "compile_plan": list(final_pipeline),
            "basis_gates": list(basis_gates),
            "optimization_level": optimization_level,
            "job_plan": job_plan
        }
        # 输出格式选择：
        if output in ("ir","tyxonq"):
            return {"circuit": lowered, "metadata": metadata}
        if output in ("qasm", "qasm2"):
            # 若本地未实现 QASM 降级，薄转发到 qiskit 实现
            # output = tyxonq equals to qasm2
            try:
                from ..qiskit import QiskitCompiler  # type: ignore

                qk_opts = dict(options)
                # 若未显式指定，使用 qiskit 偏好门集合
                if not qk_opts.get("basis_gates"):
                    qk_opts["basis_gates"] = ["cx", "h", "rz", "rx", "cz"]
                qk_opts["output"] = "qasm2"
                # 需要将 lowered 作为输入传给 qiskit 方言适配
                return QiskitCompiler().compile({"circuit": lowered, "options": qk_opts})  # type: ignore[arg-type]
            except Exception:
                # 降级失败则仍返回 IR
                return {"circuit": lowered, "metadata": metadata}
        if output == "qiskit":
            try:
                from ..qiskit import QiskitCompiler  # type: ignore

                qk_opts = dict(options)
                # 若未显式指定，使用 qiskit 偏好门集合
                if not qk_opts.get("basis_gates"):
                    qk_opts["basis_gates"] = ["cx", "h", "rz", "rx", "cz"]
                qk_opts["output"] = "qiskit"
                return QiskitCompiler().compile({"circuit": lowered, "options": qk_opts})  # type: ignore[arg-type]
            except Exception:
                return {"circuit": lowered, "metadata": metadata}
        # 未识别：返回 IR
        return {"circuit": lowered, "metadata": metadata}


