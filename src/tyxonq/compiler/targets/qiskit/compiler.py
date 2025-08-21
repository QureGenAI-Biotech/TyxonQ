from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.compiler import CompileRequest, CompileResult, Compiler
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceCapabilities


class QiskitCompiler:
    """Target compiler for Qiskit/IBMQ backends.

    Skeleton implementation aligning with the refactor plan:
    - Accepts an IR `Circuit` and a `DeviceCapabilities` specification
    - Returns the same circuit with minimal metadata for now
    - Future versions will add:
        * Gate set mapping (dialect)
        * Layout/routing strategies
        * Optimization levels mapping to passes
        * Pulse/Schedule lowering when applicable
    """

    def compile(self, request: "CompileRequest") -> "CompileResult":  # type: ignore[override]
        # Minimal refactor preserving old behavior shape: pass-through with metadata
        from .dialect import normalize_transpile_options

        circuit: "Circuit" = request["circuit"]
        target: "DeviceCapabilities" = request.get("target", {})  # type: ignore[assignment]
        options: Dict[str, Any] = request.get("options", {})
        norm_opts = normalize_transpile_options(options)
        metadata: Dict[str, Any] = {"target": "qiskit", "options": dict(norm_opts), "caps": dict(target)}
        return {"circuit": circuit, "metadata": metadata}


