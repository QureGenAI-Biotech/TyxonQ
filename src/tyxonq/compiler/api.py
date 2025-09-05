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

    def run(self, circuit: "Circuit", **opts: Any) -> "Circuit": ...


class Compiler(Protocol):
    """Compiler interface that transforms IR for a target device/backend."""

    def compile(self, request: CompileRequest) -> CompileResult: ...


def compile(
    circuit: "Circuit",
    *,
    compile_engine: str = "default",
    output: str = "ir",
    options: Dict[str, Any] | None = None,
) -> CompileResult:
    """Unified compile entry.

    Parameters:
        circuit: IR circuit to compile
        compile_engine: 'tyxonq' | 'qiskit'|'default' | 'native'
        output: 'ir' | 'qasm2' | 'qiskit'  # 'ir' accepted as alias of 'tyxonq'
        options: compile_engine-specific compile options
    """



    # cap_target: Dict[str, Any] = _parse_target(target_device) if isinstance(target_device, str) else {}
    opts = dict(options or {})

    if circuit._device_opts.get("provider") == "tyxonq" and circuit._device_opts.get('device') == 'homebrew_s2': 
        output = "qasm2"
    if output:
        opts["output"] = output

    prov = (compile_engine or "default").lower()
    if prov in ("default", "tyxonq", "native"):
        from .compile_engine.native.native_compiler import NativeCompiler

        return NativeCompiler().compile({"circuit": circuit,  "options": opts})  # type: ignore[arg-type]
    if prov == "qiskit":
        from .compile_engine.qiskit import QiskitCompiler

        return QiskitCompiler().compile({"circuit": circuit, "options": opts})  # type: ignore[arg-type]
    # Fallback to native
    from .compile_engine.native.native_compiler import NativeCompiler
    return NativeCompiler().compile({"circuit": circuit,"options": opts})  # type: ignore[arg-type]


