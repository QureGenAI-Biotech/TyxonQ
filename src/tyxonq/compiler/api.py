from __future__ import annotations

from typing import Any, Dict, Protocol, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.core.ir.pulse import PulseProgram
    from tyxonq.devices import DeviceRule



class CompileResult(TypedDict):
    """Result of compilation containing the compiled circuit and metadata."""

    circuit: Any  # Circuit | str (for TQASM/QASM output)
    metadata: Dict[str, Any]


class PulseCompileResult(TypedDict):
    """Result of pulse compilation containing the compiled pulse schedule and metadata."""

    pulse_schedule: Any  # PulseSchedule or compiled Circuit with pulse ops
    metadata: Dict[str, Any]


class Pass(Protocol):
    """Compilation pass that transforms a circuit for a given target."""

    def execute_plan(self, circuit: "Circuit", **opts: Any) -> "Circuit": ...


def compile(
    circuit: "Circuit",
    *,
    compile_engine: str = "default",
    output: str = "ir",
    compile_plan: list | None = None,
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
            - "pulse": Pulse-level compilation (TyxonQ core feature)
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
            - "device_params" (dict): Device physical parameters (for pulse compilation)
            - "calibrations" (dict): Custom pulse calibrations (for pulse compilation)
            - "mode" (str): Pulse compilation mode - "hybrid", "pulse_only", "auto_lower"
            
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
    
    # 智能推断：output="tqasm" 或 "qasm3" 自动启用 pulse 编译
    if output in ("tqasm", "tqasm0.2", "qasm3", "qasm3.0", "openqasm3", "openqasm3.0") and compile_engine in ("default", "tyxonq", "native"):
        compile_engine = "pulse"
        import warnings
        warnings.warn(
            f"output='{output}' 需要脉冲级编译，自动启用 pulse compiler。"
            "建议显式调用 circuit.use_pulse() 以明确编译意图。",
            UserWarning,
            stacklevel=2
        )
    
    if compile_engine in ("default", "tyxonq", "native"):
        from .compile_engine.native.native_compiler import NativeCompiler

        result = NativeCompiler().compile(circuit = circuit,compile_plan= compile_plan, device_rule=device_rule, options = opts)  # type: ignore[arg-type]
        
        # 缓存字符串结果到 _source，避免重复编译
        if isinstance(result.get("circuit"), str):
            circuit._source = result["circuit"]
        
        return result
    if compile_engine == "pulse":
        # Pulse-level compilation (TyxonQ core feature)
        from .pulse_compile_engine import PulseCompiler
        
        # 关键修复：如果 circuit 没有显式调用 use_pulse()，但指定了 pulse 编译
        # 则自动设置 _compile_engine 和 _compile_opts，以便云端提交时能正确编译
        # Circuit 没有显式调用 use_pulse()，自动设置
        circuit._compile_engine = "pulse"
        circuit._compile_opts = dict(opts)
        
        # 自动补足缺失的 device_params（使用默认测试参数）
        if not opts.get("device_params") and not circuit.metadata.get("pulse_device_params"):
            import warnings
            default_device_params = {
                "qubit_freq": [5.0e9] * circuit.num_qubits,  # 默认 5 GHz
                "anharmonicity": [-330e6] * circuit.num_qubits,  # 默认 -330 MHz
            }
            opts["device_params"] = default_device_params
            warnings.warn(
                f"脉冲编译缺少 device_params，已自动补足默认测试参数：\n"
                f"  - qubit_freq: {default_device_params['qubit_freq']} Hz (5 GHz)\n"
                f"  - anharmonicity: {default_device_params['anharmonicity']} Hz (-330 MHz)\n"
                f"建议通过 circuit.use_pulse(device_params={{...}}) 提供真实设备参数。",
                UserWarning,
                stacklevel=2
            )
        
        # 关键修复: 如果 output="tqasm"，自动设置 inline_pulses=True
        if output in ("tqasm", "tqasm0.2", "qasm3", "qasm3.0", "openqasm3", "openqasm3.0"):
            if "inline_pulses" not in opts:
                opts["inline_pulses"] = True
        
        compiler = PulseCompiler(optimization_level=opts.get("optimization_level", 1))
        
        # 移除 device_params 避免重复传递
        compile_opts = {k: v for k, v in opts.items() if k not in ("device_params", "calibrations")}
        
        compiled_circuit = compiler.compile(
            circuit,
            device_params=opts.get("device_params"),
            calibrations=opts.get("calibrations"),
            **compile_opts
        )
        
        # 如果返回字符串（如 TQASM），包装为标准格式
        if isinstance(compiled_circuit, str):
            # 关键优化：缓存 TQASM 到 circuit._source，避免 .run() 时重复编译
            circuit._source = compiled_circuit
            return {"circuit": compiled_circuit, "metadata": {}}
        else:
            return {"circuit": compiled_circuit, "metadata": getattr(compiled_circuit, "metadata", {})}
    if compile_engine == "qiskit":
        from .compile_engine.qiskit import QiskitCompiler

        result = QiskitCompiler().compile(circuit= circuit, options = opts)  # type: ignore[arg-type]
        
        # 缓存字符串结果到 _source，避免重复编译
        if isinstance(result.get("circuit"), str):
            circuit._source = result["circuit"]
        
        return result
    # Fallback to native
    from .compile_engine.native.native_compiler import NativeCompiler
    result = NativeCompiler().compile(circuit = circuit,compile_plan=compile_plan, device_rule=device_rule,options = opts)  # type: ignore[arg-type]
    
    # 缓存字符串结果到 _source，避免重复编译
    if isinstance(result.get("circuit"), str):
        circuit._source = result["circuit"]
    
    return result


def compile_pulse(
    pulse_program: "PulseProgram",
    *,
    output: str = "pulse_ir",
    device_params: Dict[str, Any] | None = None,
    calibrations: Dict[str, Any] | None = None,
    options: Dict[str, Any] | None = None,
) -> PulseCompileResult:
    """Unified compilation entry point for pulse programs (平级 with compile()).
    
    This function compiles PulseProgram to executable pulse schedules, parallel
    to compile() for Circuit. Both functions follow the same architecture pattern.
    
    Args:
        pulse_program (PulseProgram): Input TyxonQ pulse program to compile.
        
        output (str, optional): Target output format.
            Supported values:
            - "pulse_ir": TyxonQ Native Pulse IR (default)
            - "tqasm": TQASM 0.2 format (for cloud execution)
            - "openqasm3": OpenQASM 3 with pulse extensions (future)
            Defaults to "pulse_ir".
            
        device_params (Dict[str, Any], optional): Device physical parameters:
            - qubit_freq (list): Qubit frequencies (Hz)
            - anharmonicity (list): Anharmonicity values (Hz)
            - T1 (list): Amplitude damping times (s)
            - T2 (list): Dephasing times (s)
            - coupling_map (list): Qubit connectivity
            
        calibrations (Dict[str, Any], optional): Custom pulse calibrations:
            - Format: {"gate_name": {"qubits": [0, 1], "pulse": waveform}}
            
        options (Dict[str, Any], optional): Compilation options:
            - optimization_level (int): Optimization level (0-3)
            - inline_pulses (bool): Inline pulse definitions (for cloud)
            - dt (float): Sample time step (s)
            - max_pulse_duration (float): Max pulse duration (s)
    
    Returns:
        PulseCompileResult: Dictionary containing compiled pulse schedule and metadata:
            {
                "pulse_schedule": PulseSchedule | Circuit | str,
                "metadata": Dict[str, Any]
            }
    
    Examples:
        >>> # Basic pulse compilation
        >>> prog = PulseProgram(1)
        >>> prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2)
        >>> result = compile_pulse(prog, device_params={
        ...     "qubit_freq": [5.0e9],
        ...     "anharmonicity": [-330e6]
        ... })
        >>> pulse_schedule = result["pulse_schedule"]
        
        >>> # Compile to TQASM for cloud submission
        >>> result = compile_pulse(prog,
        ...     output="tqasm",
        ...     device_params={...},
        ...     options={"inline_pulses": True})
        >>> tqasm_code = result["pulse_schedule"]
    
    Notes:
        - This function is parallel to compile() for Circuit
        - PulseProgram and Circuit are peer-level IR representations
        - Pulse compilation follows: PulseProgram → Compiler → Device
        - No .to_circuit() conversion needed for native execution
    
    See Also:
        compile: Compilation entry point for Circuit.
        PulseProgram.compile: Instance method for pulse compilation.
        PulseCompileResult: Type definition for pulse compilation results.
    """
    opts = dict(options or {})
    
    # Import pulse compiler
    from .pulse_compile_engine import PulseCompiler
    
    # 关键改进：PulseProgram 不需要转换为 Circuit！
    # TQASMExporter 直接支持 PulseProgram 和 Circuit 两种输入
    
    # 自动补足 device_params（如果未提供）
    if not device_params:
        device_params = {
            "qubit_freq": [5.0e9] * pulse_program.num_qubits,
            "anharmonicity": [-330e6] * pulse_program.num_qubits,
        }
    
    # 更新 PulseProgram 的 device_params
    pulse_program.device_params.update(device_params)
    
    # 根据output格式决定返回内容
    if output in ("tqasm", "tqasm0.2", "qasm3", "qasm3.0", "openqasm3", "openqasm3.0"):
        # 导出为TQASM（直接从 PulseProgram 导出）
        from .pulse_compile_engine.native.tqasm_exporter import TQASMExporter
        
        # 关键：直接传递 PulseProgram，TQASMExporter 会自动检测类型
        exporter = TQASMExporter(version="tqasm" if "tqasm" in output else "openqasm3")
        tqasm_code = exporter.export(pulse_program)
        
        # 缓存 TQASM 到 pulse_program._source，避免 run() 时重复编译
        pulse_program._source = tqasm_code
        
        return {
            "pulse_schedule": tqasm_code,
            "metadata": {}
        }
    else:
        # 返回pulse_ir（PulseProgram 自身）
        return {
            "pulse_schedule": pulse_program,
            "metadata": {
                "pulse_device_params": device_params,
                "pulse_calibrations": calibrations or {}
            }
        }


