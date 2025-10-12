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
    """Unified compilation entry point for quantum circuits.

    This function provides a high-level interface for compiling TyxonQ circuits
    to various target formats and backends. It supports multiple compilation
    engines and output formats while providing a consistent API.

    Args:
        circuit (Circuit): Input TyxonQ circuit to compile.
        
        compile_engine (str, optional): Compilation backend to use.
            Supported values:
            - "default" or "tyxonq" or "native": TyxonQ native compiler
            - "qiskit": Qiskit-based compilation (requires qiskit)
            Defaults to "default".
            
        output (str, optional): Target output format.
            Supported values:
            - "ir": TyxonQ intermediate representation (default)
            - "qasm2": OpenQASM 2.0 format
            - "qiskit": Qiskit QuantumCircuit object
            Defaults to "ir".
            
        compile_plan (list, optional): Custom compilation pipeline stages.
            List of pass names to execute in order. If None, uses default
            pipeline with essential normalization passes.
            
        device_rule (Dict[str, Any], optional): Device-specific constraints
            and optimization rules. Used for hardware-aware compilation.
            
        options (Dict[str, Any], optional): Engine-specific compilation options.
            Common options:
            - "basis_gates" (list): Target gate set for decomposition
            - "optimization_level" (int): Optimization level (0-3)
            - "shot_plan" or "total_shots": Shot allocation planning
            
    Returns:
        CompileResult: Dictionary containing compiled circuit and metadata:
            {
                "circuit": Circuit | QuantumCircuit | str,  # Compiled circuit
                "metadata": Dict[str, Any]  # Compilation metadata
            }

    Examples:
        >>> # Basic compilation to TyxonQ IR
        >>> c = Circuit(2)
        >>> c.h(0).cx(0, 1)
        >>> result = compile(c)
        >>> compiled_circuit = result["circuit"]
        
        >>> # Compile to OpenQASM with custom basis gates
        >>> result = compile(c, 
        ...     compile_engine="native",
        ...     output="qasm2",
        ...     options={"basis_gates": ["cx", "h", "rz"]})
        >>> qasm_code = result["circuit"]
        
        >>> # Hardware-aware compilation with device constraints
        >>> device_rule = {"coupling_map": [[0, 1], [1, 2]], "max_qubits": 3}
        >>> result = compile(c,
        ...     device_rule=device_rule,
        ...     options={"optimization_level": 2})
        
        >>> # Custom compilation pipeline
        >>> custom_plan = ["rewrite/gate_fusion", "optimize/commute_cnots"]
        >>> result = compile(c, compile_plan=custom_plan)

    Notes:
        - The "homebrew_s2" device automatically forces "qasm2" output
        - Native compiler includes essential normalization passes by default
        - Qiskit engine requires qiskit installation for QASM/Qiskit output
        - Compilation falls back to native engine if requested engine unavailable
        - Shot planning enables optimized measurement scheduling
        
    Raises:
        RuntimeError: If required compilation engine is not available.
        ValueError: If invalid compilation options are provided.
        
    See Also:
        Circuit.compile: Instance method for circuit compilation.
        CompileResult: Type definition for compilation results.
        tyxonq.compiler.passes: Available compilation passes.
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


