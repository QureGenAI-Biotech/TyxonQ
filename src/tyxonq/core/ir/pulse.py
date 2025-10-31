from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import warnings


@dataclass
class PulseInstruction:
    """A single pulse instruction targeting a hardware channel.

    Fields:
        channel: Hardware channel identifier (e.g., "d0", "u1").
        start: Start time in sample units (integer ticks).
        duration: Duration in sample units (ticks).
        waveform: Real or complex amplitude samples. Concrete dtype/shape is
            backend-specific; a Python list is accepted here for simplicity.
        metadata: Arbitrary metadata describing the pulse (shape, amp, sigma).

    Note:
        The unit convention follows sample counts to remain backend-agnostic.
        Conversion to seconds uses the schedule's sampling_rate_hz.
    """

    channel: str
    start: int
    duration: int
    waveform: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PulseSchedule:
    """A collection of timed pulse instructions with a sampling rate.

    Fields:
        sampling_rate_hz: Sampling frequency in Hertz for time conversion.
        instructions: Ordered list of pulse instructions.
        globals: Optional global parameters for template expansion or backends.
        
    This class represents a low-level pulse schedule for hardware execution.
    For high-level pulse programming, see PulseProgram below.
    """

    sampling_rate_hz: float
    instructions: List[PulseInstruction] = field(default_factory=list)
    globals: Dict[str, Any] = field(default_factory=dict)

    def append(self, instr: PulseInstruction) -> None:
        """Append an instruction to the schedule."""

        self.instructions.append(instr)

    def end_time(self) -> int:
        """Return the schedule end time in sample units.

        Defined as max over `start + duration` across all instructions, or 0
        when the schedule is empty.
        """

        if not self.instructions:
            return 0
        return max(i.start + i.duration for i in self.instructions)

    def duration_seconds(self) -> float:
        """Return the schedule duration in seconds based on sampling_rate_hz."""

        return self.end_time() / float(self.sampling_rate_hz)


@dataclass
class PulseProgram:
    """High-level representation for pure pulse-level quantum programming.
    
    PulseProgram is at the SAME abstraction level as Circuit, providing
    pulse-level control instead of gate-level control.
    
    Key Design Principles:
        1. Independent execution (不依赖 Circuit)
        2. Chain API consistency (与 Circuit 对齐)
        3. Dual-path support (链式调用 + 数值模拟)
        4. Direct compilation (真正的 .compile())
    
    Architectural Alignment:
        Circuit (Gate Programming)     PulseProgram (Pulse Programming)
        ==========================     ===============================
        .h(0).cx(0,1)                 .drag(0, ...).gaussian(1, ...)
        
        双链路方案 A:                    双链路方案 A:
        Circuit                       PulseProgram
          → .device()                   → .device()
          → .run()                      → .run() (直接执行，不转换)
          → 云端/真机                      → 云端/真机
        
        双链路方案 B:                    双链路方案 B:
        Circuit                       PulseProgram
          → .state()                    → .state()
          → 本地数值模拟                    → 本地数值模拟
    
    Attributes:
        num_qubits: Number of qubits in the system
        pulse_ops: List of pulse operations (qubit, waveform, params)
        device_params: Physical device parameters
        metadata: Additional metadata (calibrations, etc.)
    
    Examples:
        >>> from tyxonq.core.ir.pulse import PulseProgram
        >>> 
        >>> # 链式 API (推荐)
        >>> prog = PulseProgram(2)
        >>> prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
        >>> prog.gaussian(1, amp=0.3, duration=400, sigma=100, qubit_freq=5.1e9)
        >>> result = prog.device(provider="tyxonq", device="homebrew_s2").run()
        >>> 
        >>> # 数值模拟
        >>> state = prog.state(backend="numpy")
    
    See Also:
        - Circuit: Gate-level quantum programming (parallel abstraction)
    """
    
    num_qubits: int
    pulse_ops: List[Tuple[int, Any, Dict[str, Any]]] = field(default_factory=list)
    device_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Chainable stage options (similar to Circuit)
    _compile_opts: Dict[str, Any] = field(default_factory=dict)
    _device_opts: Dict[str, Any] = field(default_factory=dict)
    _post_opts: Dict[str, Any] = field(default_factory=dict)
    
    # Compiled output cache (when .compile() is called explicitly)
    _compiled_output: Optional[Any] = field(default=None)
    # 关键：缓存编译后的 TQASM/QASM 字符串，防止重复编译
    _source: Optional[str] = field(default=None)
    
    def __post_init__(self):
        """Validate pulse program structure."""
        if self.num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        
        # Validate pulse operations
        for op in self.pulse_ops:
            if not isinstance(op, (tuple, list)) or len(op) != 3:
                raise ValueError("pulse_op must be (qubit, waveform, params) tuple")
            qubit, waveform, params = op
            if not isinstance(qubit, int) or qubit < 0 or qubit >= self.num_qubits:
                raise ValueError(f"Invalid qubit index: {qubit}")
            if not isinstance(params, dict):
                raise TypeError("params must be a dict")
        
        # Initialize chainable options
        if not hasattr(self, '_compile_opts') or self._compile_opts is None:
            object.__setattr__(self, '_compile_opts', {})
        if not hasattr(self, '_device_opts') or self._device_opts is None:
            object.__setattr__(self, '_device_opts', {})
        if not hasattr(self, '_post_opts') or self._post_opts is None:
            object.__setattr__(self, '_post_opts', {})
    
    def add_pulse(
        self, 
        qubit: int, 
        waveform: Any,
        **params: Any
    ) -> "PulseProgram":
        """Add a pulse operation to the program (advanced API).
        
        Note: For common waveforms, use chain methods like .drag(), .gaussian() instead.
        This method is for custom waveforms.
        
        Args:
            qubit: Target qubit index
            waveform: Pulse waveform object (from tyxonq.waveforms)
            **params: Physical parameters:
                - qubit_freq (float): Qubit frequency (Hz)
                - drive_freq (float): Drive frequency (Hz)
                - anharmonicity (float): Anharmonicity (Hz)
                - phase (float): Pulse phase (radians)
        
        Returns:
            Self for method chaining
        
        Examples:
            >>> from tyxonq import waveforms
            >>> prog = PulseProgram(1)
            >>> # Custom waveform
            >>> prog.add_pulse(0, MyCustomWaveform(...), qubit_freq=5.0e9)
        """
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.num_qubits})")
        
        self.pulse_ops.append((qubit, waveform, dict(params)))
        return self
    
    # ---- 链式方法 (与 Circuit 对齐) ----
    
    def drag(
        self,
        qubit: int,
        amp: float,
        duration: int,
        sigma: float,
        beta: float,
        **params: Any
    ) -> "PulseProgram":
        """Add DRAG pulse (Derivative Removal by Adiabatic Gate).
        
        Args:
            qubit: Target qubit index
            amp: Pulse amplitude (typical: 0.5-1.0)
            duration: Pulse duration in nanoseconds (typical: 80-200 ns)
            sigma: Gaussian width in nanoseconds
            beta: DRAG coefficient (typical: 0.1-0.5)
            **params: Additional parameters (qubit_freq, drive_freq, phase, etc.)
        
        Returns:
            Self for method chaining
        
        Examples:
            >>> prog = PulseProgram(1)
            >>> prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
        """
        from ...waveforms import Drag
        waveform = Drag(amp=amp, duration=int(duration), sigma=sigma, beta=beta)
        return self.add_pulse(qubit, waveform, **params)
    
    def gaussian(
        self,
        qubit: int,
        amp: float,
        duration: int,
        sigma: float,
        **params: Any
    ) -> "PulseProgram":
        """Add Gaussian pulse.
        
        Args:
            qubit: Target qubit index
            amp: Pulse amplitude
            duration: Pulse duration in nanoseconds
            sigma: Gaussian width in nanoseconds
            **params: Additional parameters
        
        Returns:
            Self for method chaining
        
        Examples:
            >>> prog = PulseProgram(1)
            >>> prog.gaussian(0, amp=0.5, duration=200, sigma=50, qubit_freq=5.0e9)
        """
        from ...waveforms import Gaussian
        waveform = Gaussian(amp=amp, duration=int(duration), sigma=sigma)
        return self.add_pulse(qubit, waveform, **params)
    
    def constant(
        self,
        qubit: int,
        amp: float,
        duration: int,
        **params: Any
    ) -> "PulseProgram":
        """Add constant (flat-top) pulse.
        
        Args:
            qubit: Target qubit index
            amp: Pulse amplitude
            duration: Pulse duration in nanoseconds
            **params: Additional parameters
        
        Returns:
            Self for method chaining
        
        Examples:
            >>> prog = PulseProgram(1)
            >>> prog.constant(0, amp=0.3, duration=100, qubit_freq=5.0e9)
        """
        from ...waveforms import Constant
        waveform = Constant(amp=amp, duration=int(duration))
        return self.add_pulse(qubit, waveform, **params)
    
    def cosine_drag(
        self,
        qubit: int,
        amp: float,
        duration: int,
        phase: float,
        alpha: float,
        **params: Any
    ) -> "PulseProgram":
        """Add Cosine DRAG pulse.
        
        Args:
            qubit: Target qubit index
            amp: Pulse amplitude
            duration: Pulse duration in nanoseconds
            phase: Pulse phase in radians
            alpha: DRAG alpha parameter
            **params: Additional parameters
        
        Returns:
            Self for method chaining
        """
        from ...waveforms import CosineDrag
        waveform = CosineDrag(amp=amp, duration=int(duration), phase=phase, alpha=alpha)
        return self.add_pulse(qubit, waveform, **params)
    
    def set_device_params(self, **params: Any) -> "PulseProgram":
        """Set physical device parameters.
        
        Args:
            **params: Device parameters:
                - qubit_freq (list): Qubit frequencies for all qubits (Hz)
                - anharmonicity (list): Anharmonicity values (Hz)
                - T1 (list): Amplitude damping times (s)
                - T2 (list): Dephasing times (s)
        
        Returns:
            Self for method chaining
        
        Examples:
            >>> prog = PulseProgram(2)
            >>> prog.set_device_params(
            ...     qubit_freq=[5.0e9, 5.1e9],
            ...     anharmonicity=[-330e6, -320e6],
            ...     T1=[80e-6, 85e-6],
            ...     T2=[120e-6, 125e-6]
            ... )
        """
        self.device_params.update(params)
        return self
    
    # ---- 链式配置方法 (与 Circuit 对齐) ----
    
    def compile(self, output: str = "pulse_ir", **options: Any) -> Any:
        """Compile pulse program via compile_pulse() (平级 with Circuit.compile()).
        
        Args:
            output (str): Output format ("pulse_ir", "tqasm", "openqasm3")
            **options: Compilation options:
                - device_params (dict): Physical device parameters
                - calibrations (dict): Custom pulse calibrations
                - optimization_level (int): Optimization level (0-3)
                - inline_pulses (bool): Inline pulse definitions
        
        Returns:
            Compiled pulse schedule
        
        Examples:
            >>> prog = PulseProgram(1)
            >>> prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2)
            >>> compiled = prog.compile(output="tqasm", options={"inline_pulses": True})
        """
        # 调用独立的 compile_pulse() 函数 (平级架构)
        from ...compiler.api import compile_pulse
        
        merged_opts = dict(self._compile_opts)
        merged_opts.update(options)
        
        # 传递设备参数
        device_params = merged_opts.pop("device_params", self.device_params)
        calibrations = merged_opts.pop("calibrations", None)
        
        result = compile_pulse(
            self,
            output=output,
            device_params=device_params,
            calibrations=calibrations,
            options=merged_opts
        )
        
        return result["pulse_schedule"]
    
    def device(self, **options: Any) -> "PulseProgram":
        """Configure device execution options.
        
        This method is part of the Chain API (双链路方案 A).
        
        Args:
            **options: Device configuration options:
                provider (str): Execution provider ("simulator", "tyxonq", etc.)
                device (str): Specific device name
                shots (int): Number of measurement shots
        
        Returns:
            Self for method chaining
        
        Examples:
            >>> prog = PulseProgram(1)
            >>> prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2)
            >>> prog.device(provider="tyxonq", device="homebrew_s2", shots=1024)
        """
        self._device_opts.update(options)
        return self
    
    def run(self, **options: Any) -> Any:
        """Execute the pulse program (平级 with Circuit.run()).
        
        Correct execution flow (平级架构):
            PulseProgram.run()
              → compile_pulse() (独立编译器)
              → device_base.run()
              → Driver执行
        
        Args:
            **options: Execution options:
                - shots (int): Number of measurement shots (0 = statevector)
                - backend (str): Numeric backend ("numpy", "pytorch", "cupy")
                - provider (str): Execution provider ("simulator", "tyxonq", etc.)
                - device (str): Specific device name
        
        Returns:
            Execution result (state, counts, or task handle)
        
        Examples:
            >>> # Submit to hardware (Chain API)
            >>> prog = PulseProgram(1)
            >>> prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2)
            >>> result = prog.device(provider="tyxonq", device="homebrew_s2").run()
            >>> 
            >>> # Local simulation
            >>> state = prog.run(backend="numpy", shots=0)
        """
        # Merge options from .device() and .run()
        merged_opts = {**self._device_opts, **options}
        provider = merged_opts.get("provider")
        device_name = merged_opts.get("device")
        shots = merged_opts.get("shots", 0)
        
        # Check if hardware execution requested
        if provider or device_name:
            # Hardware/cloud execution path (平级 with Circuit)
            # 步骤1: 编译 (via compile_pulse)
            from ...compiler.api import compile_pulse
            
            compile_opts = dict(self._compile_opts)
            
            # 调用独立的 compile_pulse() 函数
            result = compile_pulse(
                self,
                output="pulse_ir",
                device_params=self.device_params,
                calibrations=None,
                options=compile_opts
            )
            
            # 步骤2: 提交到设备 (via compiled circuit)
            compiled_circuit = result["pulse_schedule"]
            
            # 复制设备选项
            for key, value in self._device_opts.items():
                compiled_circuit._device_opts[key] = value
            
            # 委托给已编译的 Circuit 执行
            return compiled_circuit.run(**options)
        else:
            # Local simulation path
            backend = options.get("backend")
            
            if shots == 0:
                # Statevector simulation
                return self.state(backend=backend)
            else:
                # Sampling from statevector
                state = self.state(backend=backend)
                import numpy as np
                from ...numerics.api import get_backend
                backend_obj = get_backend()
                
                probs = np.abs(backend_obj.to_numpy(state))**2
                samples = np.random.choice(len(probs), size=shots, p=probs)
                from collections import Counter
                return {"counts": Counter(samples)}

    
    def state(self, backend: Optional[str] = None, form: Optional[str] = None) -> Any:
        """Get the quantum state via numerical simulation (双链路方案 B).
        
        This is the pulse-level equivalent of Circuit.state().
        
        Args:
            backend: Numeric backend ("numpy", "pytorch", "cupy")
            form: Output format ("tensor", "numpy")
        
        Returns:
            Quantum state as backend tensor or numpy array
        
        Examples:
            >>> prog = PulseProgram(1)
            >>> prog.add_pulse(0, waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.2),
            ...                qubit_freq=5.0e9)
            >>> state = prog.state(backend="numpy")
            >>> print(state.shape)  # (2,) for 1-qubit
        """
        try:
            from ...libs.quantum_library import pulse_simulation
            from ...numerics.context import set_backend
            from ...numerics.api import get_backend
        except ImportError as e:
            raise ImportError(
                "Pulse simulation requires tyxonq.libs.quantum_library.pulse_simulation. "
                f"Import error: {e}"
            )
        
        # Set backend
        if backend is not None:
            set_backend(backend)
        backend_obj = get_backend()
        
        # Get initial state
        import numpy as np
        initial_state = np.zeros(2**self.num_qubits, dtype=np.complex128)
        initial_state[0] = 1.0
        initial_state = backend_obj.array(initial_state, dtype=backend_obj.complex128)
        
        # Apply each pulse sequentially
        current_state = initial_state
        for qubit, waveform, params in self.pulse_ops:
            # Extract parameters
            qubit_freq = params.get("qubit_freq")
            if qubit_freq is None:
                freq_list = self.device_params.get("qubit_freq")
                if freq_list and qubit < len(freq_list):
                    qubit_freq = freq_list[qubit]
                else:
                    qubit_freq = 5.0e9  # Default
            
            drive_freq = params.get("drive_freq", qubit_freq)
            anharmonicity = params.get("anharmonicity")
            if anharmonicity is None:
                anharm_list = self.device_params.get("anharmonicity")
                if anharm_list and qubit < len(anharm_list):
                    anharmonicity = anharm_list[qubit]
                else:
                    anharmonicity = -330e6  # Default
            
            # Get T1/T2 if available
            T1 = params.get("T1")
            T2 = params.get("T2")
            if T1 is None:
                t1_list = self.device_params.get("T1")
                if t1_list and qubit < len(t1_list):
                    T1 = t1_list[qubit]
            if T2 is None:
                t2_list = self.device_params.get("T2")
                if t2_list and qubit < len(t2_list):
                    T2 = t2_list[qubit]
            
            # Evolve state under this pulse
            current_state = pulse_simulation.evolve_pulse_hamiltonian(
                initial_state=current_state,
                pulse_waveform=waveform,
                qubit=qubit,
                qubit_freq=qubit_freq,
                drive_freq=drive_freq,
                anharmonicity=anharmonicity,
                T1=T1,
                T2=T2,
                backend=backend_obj
            )
        
        # Handle output format
        if form == "numpy":
            return np.asarray(current_state)
        else:
            return current_state
    
    def to_circuit(self) -> Any:  # Returns Circuit type
        """Convert pulse program to Circuit with pulse operations.
        
        ⚠️  WARNING: This is an OPTIONAL compatibility feature!
        
        PulseProgram should execute directly via .device().run(),
        not through .to_circuit(). This method exists only for:
            - Debugging and inspection
            - Reverse engineering (pulse → gate conversion experiments)
            - Compatibility with Circuit-only workflows
        
        For production pulse programming, use the direct execution path:
            prog.device(...).run()  # ✅ Correct
            prog.to_circuit().device(...).run()  # ❌ Unnecessary
        
        Returns:
            Circuit with pulse operations in metadata
        
        Examples:
            >>> prog = PulseProgram(1)
            >>> prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
            >>> 
            >>> # Optional: Convert for debugging
            >>> circuit = prog.to_circuit()
            >>> print(f"Ops: {circuit.ops}")  # Inspect pulse operations
        """
        import warnings
        warnings.warn(
            "Converting PulseProgram to Circuit. "
            "For native pulse execution, use .device().run() directly. "
            ".to_circuit() is only for compatibility/debugging.",
            UserWarning,
            stacklevel=2
        )
        
        from .circuit import Circuit
        
        c = Circuit(self.num_qubits)
        
        # Initialize pulse library in metadata
        c.metadata["pulse_library"] = {}
        
        # Convert pulse_ops to circuit operations
        for idx, (qubit, waveform, params) in enumerate(self.pulse_ops):
            # Store waveform in library
            pulse_key = f"pulse_{idx}"
            c.metadata["pulse_library"][pulse_key] = waveform
            
            # Add pulse operation
            c.ops.append(("pulse", qubit, pulse_key, params))
        
        # Copy device params
        if self.device_params:
            c.metadata["pulse_device_params"] = dict(self.device_params)
        
        # Copy metadata
        for key, value in self.metadata.items():
            if key not in c.metadata:
                c.metadata[key] = value
        
        return c


__all__ = [
    "PulseInstruction",
    "PulseSchedule",
    "PulseProgram",
]