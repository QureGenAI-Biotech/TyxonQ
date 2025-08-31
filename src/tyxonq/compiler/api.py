from __future__ import annotations

from typing import Any, Dict, Protocol, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceCapabilities


class CompileRequest(TypedDict):
    """Structured compile request.

    Fields:
        circuit: IR circuit to compile.
        target: Device capability description for capability negotiation.
        options: Additional compile options (optimization level, layout hints).
    """

    circuit: "Circuit"
    target: Any
    options: Dict[str, Any]


class CompileResult(TypedDict):
    """Result of compilation containing the compiled circuit and metadata."""

    circuit: "Circuit"
    metadata: Dict[str, Any]


class Pass(Protocol):
    """Compilation pass that transforms a circuit for a given target."""

    def run(self, circuit: "Circuit", caps: "DeviceCapabilities", **opts: Any) -> "Circuit": ...


class Compiler(Protocol):
    """Compiler interface that transforms IR for a target device/backend."""

    def compile(self, request: CompileRequest) -> CompileResult: ...


def compile(
    circuit: "Circuit",
    *,
    compile_engine: str = "default",
    output: str = "ir",
    target: str | None = None,
    options: Dict[str, Any] | None = None,
) -> CompileResult:
    """Unified compile entry.

    Parameters:
        circuit: IR circuit to compile
        compile_engine: 'tyxonq' | 'qiskit'|'default' | 'native'
        output: 'ir' | 'qasm2' | 'qiskit'  # 'ir' accepted as alias of 'tyxonq'
        target: device capabilities for provider-aware compilation
        options: compile_engine-specific compile options
    """

    # Parse target string to capability dict if provided
    def _parse_target(target_str: str) -> Dict[str, Any]:
        # expected forms: "simulator::statevector", "simulator::density_matrix", "hardware::ibm::ibm_oslo"
        parts = [p for p in (target_str or "").split("::") if p]
        obj: Dict[str, Any] = {}
        if not parts:
            return {"scope": "simulator", "variant": "statevector"}
        if len(parts) == 1:
            obj["scope"] = parts[0]
            obj["variant"] = "statevector" if parts[0] == "simulator" else parts[0]
        elif len(parts) == 2:
            obj["scope"], obj["variant"] = parts
        else:
            obj["scope"], obj["variant"], obj["detail"] = parts[0], parts[1], "::".join(parts[2:])
        return obj

    cap_target: Dict[str, Any] = _parse_target(target) if isinstance(target, str) else {}
    options = options or {}
    # Map generic output to provider-specific
    prov = compile_engine.lower()
    if prov in ("tyxonq", "native"):
        prov = "default"
    out = output.lower()

    # Auto-switch to qiskit compile engine for TyxonQ cloud hardware targets
    try:
        scope = str(cap_target.get("scope", "")).lower()
        variant = str(cap_target.get("variant", "")).lower()
        detail = str(cap_target.get("detail", "")).lower()
        is_tyxonq_hw = scope in ("hardware", "cloud") and ("tyxonq" in variant or "tyxonq" in detail)
        if is_tyxonq_hw and prov != "qiskit":
            print("Info: target requires TyxonQ cloud hardware; switching compile_engine to 'qiskit'.")
            prov = "qiskit"
    except Exception:
        pass

    if prov == "qiskit":
        from .providers.qiskit import QiskitCompiler

        if out == "qiskit":
            out_opt = "qiskit"
        elif out == "qasm2":
            out_opt = "qasm2"
        elif out in ("ir", "tyxonq"):
            # Request original IR
            out_opt = "ir"
        else:
            out_opt = "ir"
        opts = dict(options)
        # fill default basis_gates if missing/empty for qiskit path
        if not opts.get("basis_gates"):
            opts["basis_gates"] = ["cx", "h", "rz", "rx", "cz"]
        opts["output"] = out_opt
        return QiskitCompiler().compile({"circuit": circuit, "target": cap_target, "options": opts})  # type: ignore[arg-type]

    # default native provider (TyxonQ)
    from .native_compiler import NativeCompiler

    # Run native pipeline (stages list resolved inside NativeCompiler)
    return NativeCompiler().compile({"circuit": circuit, "target": cap_target, "options": dict(options)})  # type: ignore[arg-type]


