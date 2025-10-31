"""TyxonQ native pulse compiler - Main compiler class.

This module implements the main PulseCompiler class that orchestrates the
pulse compilation pipeline: Gate → Pulse → Scheduled Pulse Sequence.

Architecture aligns with Memory 9f31913f (核心架构流程):
    问题 → 哈密顿量 → 电路 → 【编译】 → 执行 → 后处理
                              ↓
                        Pulse Compiler (this module)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit

from .gate_to_pulse import GateToPulsePass
from .pulse_lowering import PulseLoweringPass
from .pulse_scheduling import PulseSchedulingPass


class PulseCompiler:
    """TyxonQ native pulse-level compiler.
    
    This compiler transforms gate-level circuits into pulse sequences suitable
    for hardware execution. It supports calibration-based and automatic gate
    decomposition to pulse primitives.
    
    Compilation Pipeline:
        1. GateToPulsePass: Convert standard gates to pulse sequences
        2. PulseLoweringPass: Inline defcal (calibration) definitions
        3. PulseSchedulingPass: Optimize pulse timing and resource allocation
    
    Example:
        >>> from tyxonq import Circuit
        >>> from tyxonq.compiler.pulse_compile_engine import PulseCompiler
        >>> 
        >>> # Create gate circuit
        >>> c = Circuit(2)
        >>> c.h(0)
        >>> c.cx(0, 1)
        >>> 
        >>> # Compile to pulse
        >>> compiler = PulseCompiler()
        >>> pulse_circuit = compiler.compile(c, device_params={
        ...     "qubit_freq": [5.0e9, 5.1e9],
        ...     "anharmonicity": [-330e6, -320e6]
        ... })
        >>> 
        >>> # Execute pulse circuit
        >>> result = pulse_circuit.run()
    
    Attributes:
        optimization_level (int): Compilation optimization level (0-3)
            - 0: No optimization, basic decomposition only
            - 1: Basic pulse scheduling
            - 2: Advanced scheduling + pulse merging
            - 3: Full optimization with calibration-aware routing
    """
    
    def __init__(self, optimization_level: int = 1):
        """Initialize the pulse compiler.
        
        Args:
            optimization_level: Optimization level (0-3). Default is 1.
        """
        self.optimization_level = optimization_level
        self._gate_to_pulse_pass = GateToPulsePass()
        self._pulse_lowering_pass = PulseLoweringPass()
        self._pulse_scheduling_pass = PulseSchedulingPass()
    
    def compile(
        self,
        circuit: "Circuit",
        device_params: Optional[Dict[str, Any]] = None,
        calibrations: Optional[Dict[str, Any]] = None,
        output: str = "pulse_ir",
        **options: Any
    ) -> Any:
        """Compile a gate-level circuit to pulse-level representation.
        
        Args:
            circuit: Input gate-level circuit
            device_params: Device physical parameters:
                - qubit_freq: List of qubit frequencies (Hz)
                - anharmonicity: List of anharmonicity values (Hz)
                - T1: List of amplitude damping times (s)
                - T2: List of dephasing times (s)
                - coupling_map: Qubit connectivity graph
            calibrations: Custom pulse calibrations for gates:
                - Format: {"gate_name": {"qubits": [0, 1], "pulse": waveform}}
            output: Output format:
                - "pulse_ir": TyxonQ Native Pulse IR (default)
                - "tqasm": TQASM 0.2 format (for cloud execution)
            **options: Additional compilation options:
                - mode: "hybrid" | "pulse_only" | "auto_lower" (default: "hybrid")
                - dt: Sample time step (default: 1e-10 s)
                - max_pulse_duration: Maximum allowed pulse duration (default: 1e-6 s)
                - inline_pulses: Whether to inline pulse definitions (default: False)
        
        Returns:
            Compiled circuit (pulse_ir) or TQASM code string (tqasm)
        
        Raises:
            ValueError: If circuit contains unsupported gates or invalid parameters
        """
        device_params = device_params or {}
        calibrations = calibrations or {}
        mode = options.get("mode", "hybrid")
        
        # Store calibrations and device params in circuit metadata
        circuit = circuit.with_metadata(
            pulse_device_params=device_params,
            pulse_calibrations=calibrations,
            pulse_mode=mode
        )
        
        # 确保 mode 参数被传递给 Gate→Pulse Pass
        options_with_mode = dict(options)
        options_with_mode["mode"] = mode
        
        # Pass 1: Gate → Pulse decomposition
        circuit = self._gate_to_pulse_pass.execute_plan(circuit, **options_with_mode)
        
        # Pass 2: Pulse lowering (defcal inlining)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Pulse 表示的三种形式：
        #
        # 1️⃣ Gate-level (抽象)
        #    ops = [("x", 0)]  ← 高层门操作
        #
        # 2️⃣ Pulse-level with References (符号引用，默认模式)
        #    ops = [("pulse", 0, "rx_q0_123", {params})]
        #    metadata["pulse_library"] = {"rx_q0_123": Drag(...)}
        #    优点：保持 Python 对象，支持 PyTorch autograd
        #    缺点：依赖 metadata 传递
        #    用途：本地模拟、TyxonQ Native IR
        #
        # 3️⃣ Pulse-level Inlined (完全展开，序列化友好)
        #    ops = [("pulse_inline", 0, {"type": "drag", "args": [...]}, {params})]
        #    优点：自包含、可序列化、云端兼容
        #    缺点：失去 Python 对象的灵活性
        #    用途：TQASM 导出、云端提交、文件保存
        #
        # inline_pulses 参数控制：
        # - False (默认): 保留符号引用，适合本地模拟和优化
        # - True: 完全内联，适合云端提交和序列化
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if options.get("inline_pulses", False):
            circuit = self._pulse_lowering_pass.execute_plan(circuit, **options)
        
        # Pass 3: Pulse scheduling (if optimization enabled)
        if self.optimization_level >= 1:
            circuit = self._pulse_scheduling_pass.execute_plan(circuit, **options)
        
        # Output format conversion
        if output in ("tqasm", "tqasm0.2"):
            # Export to TQASM 0.2 format (TensorCircuit compatible)
            from .tqasm_exporter import TQASMExporter
            exporter = TQASMExporter(version="tqasm")
            tqasm_code = exporter.export(circuit)
            return tqasm_code
        elif output in ("qasm3", "qasm3.0", "openqasm3", "openqasm3.0"):
            # Export to OpenQASM 3.0 format (IBM/Rigetti compatible)
            from .tqasm_exporter import TQASMExporter
            exporter = TQASMExporter(version="openqasm3")
            tqasm_code = exporter.export(circuit)
            return tqasm_code
        elif output == "pulse_ir":
            # Return TyxonQ Native Pulse IR
            return circuit
        else:
            raise ValueError(
                f"Unsupported output format: {output}. "
                f"Use 'pulse_ir', 'tqasm' (or 'tqasm0.2'), or 'qasm3' (or 'openqasm3')."
            )
    
    def add_calibration(
        self,
        gate_name: str,
        qubits: List[int],
        pulse_waveform: Any,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a custom pulse calibration for a specific gate.
        
        This allows users to define custom pulse implementations for gates,
        which will be used instead of the default decomposition.
        
        Args:
            gate_name: Name of the gate to calibrate (e.g., "x", "cx")
            qubits: Qubit indices this calibration applies to
            pulse_waveform: Pulse waveform object (from tyxonq.waveforms)
            params: Additional parameters:
                - qubit_freq: Qubit frequency (Hz)
                - drive_freq: Drive frequency (Hz)
                - phase: Pulse phase (rad)
        
        Example:
            >>> from tyxonq import waveforms
            >>> compiler = PulseCompiler()
            >>> 
            >>> # Custom X gate calibration
            >>> x_pulse = waveforms.Drag(amp=0.5, duration=160, sigma=40, beta=0.2)
            >>> compiler.add_calibration("x", [0], x_pulse, {
            ...     "qubit_freq": 5.0e9,
            ...     "drive_freq": 5.0e9
            ... })
        """
        if not hasattr(self, "_custom_calibrations"):
            self._custom_calibrations: Dict[str, Dict] = {}
        
        key = f"{gate_name}_{'_'.join(map(str, qubits))}"
        self._custom_calibrations[key] = {
            "gate": gate_name,
            "qubits": list(qubits),
            "pulse": pulse_waveform,
            "params": params or {}
        }
    
    def get_calibrations(self) -> Dict[str, Dict]:
        """Get all registered pulse calibrations.
        
        Returns:
            Dictionary of calibrations keyed by gate_qubits string
        """
        return getattr(self, "_custom_calibrations", {})
