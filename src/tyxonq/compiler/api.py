from __future__ import annotations

from typing import Any, Dict, Protocol, TypedDict, TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.core.ir.pulse import PulseProgram
    from tyxonq.devices import DeviceRule
    from .gate_compiler.specs import DecomposeSpec, RoutingSpec



class CompileResult(TypedDict):
    """Result of compilation containing the compiled circuit and metadata."""

    circuit: Any  # Circuit | str (for TQASM/QASM output)
    compiled_source: Optional[str]  # 编译后的源代码（QASM2/QASM3/TQASM 等）如果有的话
    metadata: Dict[str, Any]


class PulseCompileResult(TypedDict):
    """Result of pulse compilation containing the compiled pulse schedule and metadata."""
    pulse_program: Any  # PulseProgram IR object
    compiled_pulse_schedule: Optional[str]  # Compiled pulse schedule (TQASM/OpenQASM3 with defcal)
    metadata: Dict[str, Any]


class CompileStep(Protocol):
    """编译步骤协议：把一个线路变换为更接近目标平台 / 原生门集合的线路。

    每个步骤实现 ``apply(circuit, **opts)``，从 ``opts`` 自取所需参数（basis_gates、
    device_rule 等）并返回新线路。核心编译器按类型化参数装配一串 ``CompileStep``
    后顺序施加（见 ``gate_compiler.steps.build_compile_steps`` / ``apply_steps``）。
    """

    def apply(self, circuit: "Circuit", **opts: Any) -> "Circuit": ...


def compile(
    circuit: "Circuit",
    *,
    platform: str = "tyxonq",
    output: str = "ir",
    basis_gates: list | None = None,
    decompose: "bool | DecomposeSpec" = True,
    route: "bool | RoutingSpec" = False,
    optimize: int = 0,
    simplify: bool = False,
    device_rule: "DeviceRule | None" = None,
    options: Dict[str, Any] | None = None,
    compile_engine: str | None = None,
) -> CompileResult:
    """量子线路的统一编译入口（类型化关键字参数）。

    编译能力由**类型化参数**描述（取代旧的字符串列表 DSL ``compile_plan``）：核心
    编译器 ``gate_compiler`` 按这些参数装配一串 ``CompileStep`` 顺序施加，各硬件平台
    适配器（``platforms/*``）统一委托核心后再翻译为自身方言。

    Args:
        circuit: 待编译的 TyxonQ 线路（或 PulseProgram）。
        platform: 目标编译平台。取值：
            - "tyxonq"（默认；别名 "default"/"native" 归一到此）：TyxonQ 核心，输出 ir/qasm2/qasm3/qiskit；
            - "pulse"：脉冲级编译；
            - "lqcloud"/"guodun"/"qiskit"：对应硬件平台方言。
        output: 目标输出格式（"ir"/"qasm2"/"qasm3"/"qiskit"/平台专属如 "lqcloud"/"qcis"）。
        basis_gates: 目标原生门集合，透传给规范化/分解步骤与平台方言。
        decompose: ``bool | DecomposeSpec``。True（默认）→ 分解 2-qubit 数守恒 matchgate；
            ``DecomposeSpec(matchgate=, rotations=, two_qubit_unitary=)`` 提供细粒度控制；
            False → 不分解。
        route: ``bool | RoutingSpec``。True → 按 ``device_rule`` 的耦合图委托 qiskit.transpile 路由；
            ``RoutingSpec(coupling_map=, initial_layout=, algorithm=, optimization_level=)`` 细粒度控制；
            False（默认）→ 不路由。
        optimize: 优化级别（0-3）；>=1 时启用相邻门合并/恒等剪枝。
        simplify: 为真时启用光锥化简（裁剪不影响测量结果的算符）。
        device_rule: 设备能力描述（可携带 coupling_map/num_qubits 等），流向所有平台分支。
        options: 平台专属逃生舱（如 pulse 的 device_params/calibrations/mode）。
        compile_engine: **过渡别名**，等价于 ``platform``（供旧代码使用）；非 None 时覆盖 platform。

    Returns:
        CompileResult: ``{"circuit", "compiled_source", "metadata"}``。

    Examples:
        >>> c = Circuit(2); c.h(0).cx(0, 1)
        >>> result = compile(c)                       # 默认 platform="tyxonq", output="ir"
        >>> result = compile(c, platform="qiskit", output="qasm2")
        >>> # 硬件感知：按设备耦合图路由
        >>> result = compile(c, platform="lqcloud", route=True,
        ...                  device_rule={"coupling_map": [[0, 1], [1, 2]]})

    Notes:
        - "homebrew_s2" 设备对门电路强制 qasm2 输出、对脉冲电路强制 tyxonq_homebrew_tqasm 输出。
        - 检测到脉冲操作或显式脉冲模式时，platform=tyxonq 会自动切换到 pulse 编译。

    See Also:
        Circuit.compile: 线路实例方法（转发本函数的类型化关键字）。
        gate_compiler.specs: DecomposeSpec / RoutingSpec 类型化规格。
    """
    
    def _has_pulse_ops(circuit_or_pulse: Any) -> bool:
        """检查 circuit 或 pulse_program 中是否含有脉冲操作
        
        支持 Circuit 和 PulseProgram 两种类型：
        - Circuit 的 ops 格式：(gate_name, ...) 或 ("pulse", qubit, key, params)
        - PulseProgram 的 ops 格式：(qubit, waveform, params) - 有有冲口上就是 pulse
        
        Args:
            circuit_or_pulse: Circuit 或 PulseProgram 对象
        
        Returns:
            True 如果是 PulseProgram 或包含脉冲操作，False 否则
        """
        # 检查是否是 PulseProgram（简单的类型检查）
        # PulseProgram 特迷：ops 的每个元素都是 (int, waveform, dict) 格式
        if hasattr(circuit_or_pulse, '__class__'):
            class_name = circuit_or_pulse.__class__.__name__
            if class_name == "PulseProgram":
                # PulseProgram 本身原就是脉冲程序，只要有 ops 就是 pulse
                return hasattr(circuit_or_pulse, 'ops') and len(circuit_or_pulse.ops) > 0
        
        # 对于 Circuit 对象：检查 ops 中是否有脉冲相关的操作
        if not hasattr(circuit_or_pulse, 'ops'):
            return False
        
        for op in circuit_or_pulse.ops:
            if not isinstance(op, (list, tuple)) or len(op) == 0:
                continue
            
            op_type = op[0]
            # 检查是否为脉冲相关操作
            if op_type in ("pulse", "pulse_inline", "play", "set_phase", "shift_phase", "set_frequency"):
                return True
        
        return False

    # ... existing code ...



    # cap_target: Dict[str, Any] = _parse_target(target_device) if isinstance(target_device, str) else {}
    opts = dict(options or {})

    # 类型化参数落地到 opts：basis_gates 供规范化/分解步骤与平台方言消费。
    # （optimize 不在此注入 optimization_level：它经显式参数流向 GateCompiler 内部归一，
    #   且 qiskit 方言需自行把 opt_level 映射为 optimization_level，提前注入会破坏该映射。）
    if basis_gates is not None:
        opts["basis_gates"] = list(basis_gates)

    # 判断逻辑：
    # 1. 检查 circuit 中是否有脉冲操作（pulse, pulse_inline, play 等）
    # 2. 检查是否显式调用 use_pulse() 或指定了脉冲编译模式
    has_pulse_ops = _has_pulse_ops(circuit)
    has_explicit_pulse_mode = (
        getattr(circuit, "_compile_engine", None) == "pulse"
        or opts.get("mode") in ("pulse_only", "hybrid")
    )
    is_pulse_compilation = has_pulse_ops or has_explicit_pulse_mode
    
    # 检测设备目标
    is_homebrew_s2 = (
        circuit._device_opts.get("provider") == "tyxonq" 
        and circuit._device_opts.get("device") == "homebrew_s2"
    )
    
    # ===================== 编译规则实现 =====================
    # 规则1：门电路 + homebrew_s2 → 自动设为 qasm2
    # 规则3：脉冲电路 + homebrew_s2 → 自动设为 tyxonq_homebrew_tqasm
    
    if is_pulse_compilation:
        if is_homebrew_s2:
            # 规则3：脉冲电路 + homebrew_s2 → tyxonq_homebrew_tqasm
            output = "tyxonq_homebrew_tqasm"
        if output == 'ir':
            output = 'pulse_ir'
        # 其他脉冲编译：保持用户设置或默认
        
        # 自动设置 homebrew_s2 默认的波形限制（如果用户未指定）
        if "supported_waveforms" not in opts:
            opts["supported_waveforms"] = ["drag", "gaussian", "constant"]
    elif is_homebrew_s2:
        # 规则1：门电路 + homebrew_s2 → qasm2
        output = "qasm2"
        # 为 homebrew_s2 设置默认 basis_gates（如果未指定）
        if "basis_gates" not in opts:
            opts["basis_gates"] = ["cx", "h", "rz", "rx", "cz"]
    
    if output:
        opts["output"] = output

    # 过渡别名：compile_engine= 等价于 platform=（同时给出时以 compile_engine 覆盖）。
    if compile_engine is not None:
        platform = compile_engine
    platform = (platform or "tyxonq").lower()
    # 归一化平台别名：default/native → tyxonq（历史 compile_engine 取值）。
    if platform in ("default", "native"):
        platform = "tyxonq"

    # 智能推断：如果检测到脉冲操作或显式脉冲编译模式，自动启用 pulse 编译
    # 注意：output="qasm3" 等不再直接触发脉冲编译，只在有实际脉冲操作时才启用
    if is_pulse_compilation and platform == "tyxonq":
        platform = "pulse"
        import warnings
        warnings.warn(
            f"检测到脉冲操作或脉冲编译模式，自动启用 pulse compiler。"
            "建议显式调用 circuit.use_pulse() 以明确编译意图。",
            UserWarning,
            stacklevel=2
        )

    if platform == "tyxonq":
        from .platforms.tyxonq import TyxonQCompiler

        result = TyxonQCompiler().compile(
            circuit,
            basis_gates=basis_gates,
            decompose=decompose,
            route=route,
            optimize=optimize,
            simplify=simplify,
            device_rule=device_rule,
            options=opts,
        )  # type: ignore[arg-type]

        # 缓存编译源代码到 _source，避免重复编译
        circuit._compiled_source = result["compiled_source"]

        return result
    if platform == "pulse":
        # Pulse-level compilation (TyxonQ core feature)
        from .pulse_compiler import PulseCompiler
        
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
        
        # 关键修复: 如果 output="tqasm"、"qasm3" 等，自动设置 inline_pulses=True
        if output in ("tqasm", "tqasm0.2", "qasm3", "qasm3.0", "openqasm3", "openqasm3.0", "tyxonq_homebrew_tqasm"):
            if "inline_pulses" not in opts:
                opts["inline_pulses"] = True
        
        compiler = PulseCompiler(
            optimization_level=opts.get("optimization_level", 1),
            supported_waveforms=opts.get("supported_waveforms")  # 新增：传递波形约束
        )
        
        # 移除 device_params 避免重复传递
        compile_opts = {k: v for k, v in opts.items() if k not in ("device_params", "calibrations", "supported_waveforms")}
        
        compiled_circuit = compiler.compile(
            circuit,
            device_params=opts.get("device_params"),
            calibrations=opts.get("calibrations"),
            supported_waveforms=opts.get("supported_waveforms"),
            **compile_opts
        )
        
        # PulseCompiler 现在返回统一的 CompileResult 结构
        # 包含 circuit、compiled_source、metadata 三个字段
        # 返回值已经是正确的 CompileResult 格式
        compiled_source = compiled_circuit.get("compiled_source")
        # 缓存编译源代码（TQASM/QASM3 等）
        circuit._compiled_source = compiled_source
        return compiled_circuit
    if platform == "guodun":
        from .platforms.guodun import GuodunCompiler

        # decompose/route/device_rule 流向 GuodunCompiler（C 阶段起委托 gate_compiler 核心，
        # 对齐 lqcloud）。注：guodun 的 QCIS 原生基固定，内部强制 matchgate+rotations 分解；
        # route 默认 False（暂不接拓扑发现，为后续增量）。physical_qubits 经 options 流入，
        # 在编译期烘焙进 QCIS 文本（区别于 lqcloud 的运行期 initial_layout）。
        result = GuodunCompiler().compile(
            circuit=circuit,
            decompose=decompose,
            route=route,
            device_rule=device_rule,
            options=opts,
        )  # type: ignore[arg-type]
        circuit._compiled_source = result["compiled_source"]
        return result
    if platform == "lqcloud":
        # 一等平台：IR -> lqcloud SDK QuantumCircuit 对象（无 qasm/文本中间层）。
        # decompose/route/device_rule 流向 LQCloudCompiler（R1 起委托 gate_compiler 核心）。
        # 注意 physical_qubits 是运行期 backend 参数（映射 SDK initial_layout），随 dev_opts 流向驱动，
        # 不在编译期消费，故此处 options 不含它。
        from .platforms.lqcloud import LQCloudCompiler

        result = LQCloudCompiler().compile(
            circuit=circuit,
            decompose=decompose,
            route=route,
            device_rule=device_rule,
            options=opts,
        )  # type: ignore[arg-type]
        circuit._compiled_source = result["compiled_source"]
        return result
    if platform == "qiskit":
        from .platforms.qiskit import QiskitCompiler

        result = QiskitCompiler().compile(circuit=circuit, device_rule=device_rule, options=opts)  # type: ignore[arg-type]

        # 缓存编译源代码到 _source，避免重复编译
        circuit._compiled_source = result["compiled_source"]

        return result
    # Fallback：委托 gate_compiler 公用核心（TyxonQ 平台）
    from .platforms.tyxonq import TyxonQCompiler
    result = TyxonQCompiler().compile(
        circuit,
        basis_gates=basis_gates,
        decompose=decompose,
        route=route,
        optimize=optimize,
        simplify=simplify,
        device_rule=device_rule,
        options=opts,
    )  # type: ignore[arg-type]

    # 缓存编译源代码到 _source，避免重复编译
    circuit._compiled_source = result["compiled_source"]

    return result


def compile_pulse(
    pulse_program: "PulseProgram",
    *,
    output: str = "pulse_ir",
    device_params: Dict[str, Any] | None = None,
    calibrations: Dict[str, Any] | None = None,
    device: str | None = None,
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
        
        device (str, optional): Target hardware device for TQASM style selection:
            - "homebrew_s2": Uses Qiskit-compatible qreg syntax
            - Others: Uses standard OpenQASM 3.0 qubit syntax
            If not specified, auto-detected from pulse_program._device_opts
            
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
        - This is a thin wrapper around compile() with pulse-specific defaults
    
    See Also:
        compile: Compilation entry point for Circuit.
        PulseProgram.compile: Instance method for pulse compilation.
        PulseCompileResult: Type definition for pulse compilation results.
    """
    opts = dict(options or {})
    
    # 关键：自动补足 device_params（如果未提供）
    if not device_params:
        device_params = {
            "qubit_freq": [5.0e9] * pulse_program.num_qubits,
            "anharmonicity": [-330e6] * pulse_program.num_qubits,
        }
    
    # 更新 PulseProgram 的 device_params
    pulse_program.device_params.update(device_params)
    
    # 设置编译选项
    opts["device_params"] = device_params
    if calibrations:
        opts["calibrations"] = calibrations
    
    # 提取设备信息（用于 TQASM 风格选择）
    device_name = device or pulse_program._device_opts.get("device", "default")
    pulse_program.metadata["device_target"] = device_name
    
    # 关键：调用通用的 compile() 函数，继承其所有逻辑
    # compile() 会自动处理：
    # 1. 云端设备判断（tyxonq + homebrew_s2）
    # 2. 脉冲编译的自动启用
    # 3. output 格式的智能转换
    # 4. supported_waveforms 的自动设置
    # 5. 缓存机制
    result = compile(
        pulse_program,
        platform="pulse",  # 强制使用脉冲编译
        output=output,
        options=opts
    )
    
    # 转换返回格式为 PulseCompileResult
    return {
        "pulse_program": result["circuit"],
        "compiled_pulse_schedule": result["compiled_source"],
        "metadata": result.get("metadata", {})
    }

