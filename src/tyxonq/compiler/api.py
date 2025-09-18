from __future__ import annotations

from typing import Any, Dict, Protocol, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule



class CompileResult(TypedDict):
    """Result of compilation containing the compiled circuit and metadata."""

    circuit: "Circuit"
    metadata: Dict[str, Any]


class Pass(Protocol):
    """Compilation pass that transforms a circuit for a given target."""

    def execute_plan(self, circuit: "Circuit", **opts: Any) -> "Circuit": ...


def compile(
    circuit: "Circuit",
    *,
    compile_engine: str = "default",
    output: str = "ir",
    compile_plan: list[str,Any] | None = None,
    device_rule: Dict[str, Any] | None = None,
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

    compile_engine = (compile_engine or "default").lower()
    if compile_engine in ("default", "tyxonq", "native"):
        from .compile_engine.native.native_compiler import NativeCompiler

        return NativeCompiler().compile(circuit = circuit,compile_plan= compile_plan, device_rule=device_rule, options = opts)  # type: ignore[arg-type]
    if compile_engine == "qiskit":
        from .compile_engine.qiskit import QiskitCompiler

        return QiskitCompiler().compile(circuit= circuit, options = opts)  # type: ignore[arg-type]
    # Fallback to native
    from .compile_engine.native.native_compiler import NativeCompiler
    return NativeCompiler().compile(circuit = circuit,compile_plan=compile_plan, device_rule=device_rule,options = opts)  # type: ignore[arg-type]


