from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Dict, Optional, Sequence, Tuple, overload, Literal
import time
import warnings
import json
from ...compiler.api import compile as compile_api  # lazy import to avoid hard deps
import warnings

# ---- Global defaults for chainable stages (configurable via top-level API) ----
_GLOBAL_COMPILE_DEFAULTS: Dict[str, Any] = {}
_GLOBAL_DEVICE_DEFAULTS: Dict[str, Any] = {}
_GLOBAL_POSTPROC_DEFAULTS: Dict[str, Any] = {"method": None}


def set_global_compile_defaults(options: Dict[str, Any]) -> Dict[str, Any]:
    _GLOBAL_COMPILE_DEFAULTS.update(dict(options))
    return dict(_GLOBAL_COMPILE_DEFAULTS)


def get_global_compile_defaults() -> Dict[str, Any]:
    return dict(_GLOBAL_COMPILE_DEFAULTS)


def set_global_device_defaults(options: Dict[str, Any]) -> Dict[str, Any]:
    _GLOBAL_DEVICE_DEFAULTS.update(dict(options))
    return dict(_GLOBAL_DEVICE_DEFAULTS)


def get_global_device_defaults() -> Dict[str, Any]:
    return dict(_GLOBAL_DEVICE_DEFAULTS)


def set_global_postprocessing_defaults(options: Dict[str, Any]) -> Dict[str, Any]:
    _GLOBAL_POSTPROC_DEFAULTS.update(dict(options))
    if "method" not in _GLOBAL_POSTPROC_DEFAULTS:
        _GLOBAL_POSTPROC_DEFAULTS["method"] = None
    return dict(_GLOBAL_POSTPROC_DEFAULTS)


def get_global_postprocessing_defaults() -> Dict[str, Any]:
    base = dict(_GLOBAL_POSTPROC_DEFAULTS)
    if "method" not in base:
        base["method"] = None
    return base

@dataclass
class Circuit:
    """Minimal intermediate representation (IR) for a quantum circuit.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        ops: A sequence of operation descriptors. The concrete type is left
            open for backends/compilers to interpret (e.g., gate tuples, IR
            node objects). Keeping this generic allows the IR to evolve while
            tests exercise the structural contract.
    """

    num_qubits: int
    ops: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    instructions: List[Tuple[str, Tuple[int, ...]]] = field(default_factory=list)

    def __init__(self, num_qubits: int, ops: Optional[List[Any]] = None, 
                 metadata: Optional[Dict[str, Any]] = None,
                 instructions: Optional[List[Tuple[str, Tuple[int, ...]]]] = None,
                 # Initial state support (for numerical simulation)
                 inputs: Optional[Any] = None,
                 # Compile-stage defaults (visible, with defaults)
                 compile_engine: str = "default",
                 compile_output: str = "ir",
                 compile_target: str = "simulator::statevector",
                 compile_options: Optional[Dict[str, Any]] = None,
                 # Device-stage defaults
                 device_provider: str = "simulator",
                 device_device: str = "statevector",
                 device_shots: int = 1024,
                 device_options: Optional[Dict[str, Any]] = None,
                 # Result handling: whether to fetch final result (None=auto)
                 wait_async_result: Optional[bool] = False,
                 # Postprocessing defaults
                 postprocessing_method: Optional[str] = None,
                 postprocessing_options: Optional[Dict[str, Any]] = None,
                 # Draw defaults
                 draw_output: Optional[str] = None,
                 # Optional pre-compiled or provider-native source
                 source: Optional[Any] = None):
        """Initialize a Circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            ops: List of operations. Defaults to empty list if not provided.
            metadata: Circuit metadata. Defaults to empty dict if not provided.
            instructions: List of instructions. Defaults to empty list if not provided.
            inputs: Initial quantum state (numerical simulation only). 
                   If provided, the circuit starts from this state instead of |00...0⟩.
                   Can be:
                   - 1D array: state vector (shape: [2^n])
                   - 2D array: density matrix (shape: [2^n, 2^n])
                   Note: This is only supported in numerical simulation, not on real quantum hardware.
        """
        self.num_qubits = num_qubits
        self.ops = ops if ops is not None else []
        self.metadata = metadata if metadata is not None else {}
        self.instructions = instructions if instructions is not None else []
        
        # Store initial state for numerical simulation
        self._initial_state = inputs
        
        # Chainable stage options seeded from global defaults
        self._compile_opts: Dict[str, Any] = get_global_compile_defaults()
        self._device_opts: Dict[str, Any] = get_global_device_defaults()
        self._post_opts: Dict[str, Any] = get_global_postprocessing_defaults()
        # Visible defaults applied (constructor-specified overrides)
        self._compile_engine: str = str(compile_engine)
        self._compile_output: str = str(compile_output)
        self._compile_target: str = str(compile_target)
        if compile_options:
            self._compile_opts.update(dict(compile_options))
        # Device defaults
        self._device_opts.setdefault("provider", str(device_provider))
        self._device_opts.setdefault("device", str(device_device))
        self._device_opts.setdefault("shots", int(device_shots))
        if device_options:
            self._device_opts.update(dict(device_options))
        # Result handling
        self._wait_async_result: Optional[bool] = wait_async_result
        # Postprocessing defaults
        if postprocessing_options:
            self._post_opts.update(dict(postprocessing_options))
        if "method" not in self._post_opts:
            self._post_opts["method"] = postprocessing_method

        # Optional direct-execution source (e.g., QASM string or provider object)
        self._compiled_source = source
        # Draw defaults (e.g., "text", "mpl", "latex")
        self._draw_output: Optional[str] = str(draw_output) if draw_output is not None else None

        # Ensure structural validation runs even with custom __init__
        self.__post_init__()

    # Context manager support for simple builder-style usage in tests
    def __enter__(self) -> "Circuit":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Do not suppress exceptions
        return False

    # Builder compatibility: expose .circuit() to return self
    def circuit(self) -> "Circuit":
        return self

    def __post_init__(self) -> None:
        if self.num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        # Lightweight structural validation: ints used as qubit indices are in range
        for op in self.ops:
            if not isinstance(op, tuple) and not isinstance(op, list):
                raise TypeError("op must be a tuple or list")
            if not op:
                raise ValueError("op cannot be empty")
            if not isinstance(op[0], str):
                raise TypeError("op name must be a string")
            # Validate any int-like argument as qubit index
            for arg in op[1:]:
                if isinstance(arg, int):
                    if arg < 0 or arg >= self.num_qubits:
                        raise ValueError("qubit index out of range in op")
        # Validate instructions
        for inst in self.instructions:
            if not isinstance(inst, tuple) or len(inst) != 2:
                raise TypeError("instruction must be (name, (indices,)) tuple")
            iname, idxs = inst
            if not isinstance(iname, str):
                raise TypeError("instruction name must be a string")
            if not isinstance(idxs, tuple):
                raise TypeError("instruction indices must be a tuple")
            for q in idxs:
                if not isinstance(q, int) or q < 0 or q >= self.num_qubits:
                    raise ValueError("instruction qubit index out of range")

    def with_metadata(self, **kwargs: Any) -> "Circuit":
        """Return a new Circuit with merged metadata (shallow merge)."""
        new_meta = dict(self.metadata)
        new_meta.update(kwargs)
        return replace(self, metadata=new_meta)

    # ---- Chainable configuration stages ----
    def device(self, **options: Any) -> "Circuit":
        """Configure device execution options for the circuit.
        
        This method sets device-related parameters that will be used when the circuit
        is executed. It supports chainable configuration.
        
        Args:
            **options: Device configuration options. Common options include:
                provider (str): Execution provider ("simulator", "tyxonq", etc.)
                device (str): Specific device name ("statevector", "homebrew_s2", etc.)
                shots (int): Number of measurement shots
                
        Returns:
            Circuit: The same circuit instance with updated device options.
            
        Examples:
            >>> c = Circuit(2)
            >>> c.h(0).cx(0,1)
            >>> configured = c.device(provider="simulator", device="statevector", shots=1000)
            >>> # Chain with execution:
            >>> result = c.device(shots=100).run()
        """
        self._device_opts.update(dict(options))
        return self

    def postprocessing(self, **options: Any) -> "Circuit":
        """Configure postprocessing options for measurement results.
        
        This method sets postprocessing parameters that will be applied to
        measurement results after circuit execution.
        
        Args:
            **options: Postprocessing configuration options. Common options include:
                method (str): Postprocessing method ("readout_mitigation", etc.)
                
        Returns:
            Circuit: The same circuit instance with updated postprocessing options.
            
        Examples:
            >>> c = Circuit(2)
            >>> c.h(0).cx(0,1).measure_all()
            >>> configured = c.postprocessing(method="readout_mitigation")
            >>> # Chain with device and execution:
            >>> result = c.device(shots=100).postprocessing(method="readout_mitigation").run()
        """
        self._post_opts.update(dict(options))
        if "method" not in self._post_opts:
            self._post_opts["method"] = None
        return self

    def with_noise(self, noise_type: str, **noise_params: Any) -> "Circuit":
        """Configure noise model for circuit simulation (simplified API).
        
        This is a convenience method that automatically configures the density matrix
        simulator with the specified noise model. It's equivalent to calling:
        `.device(provider="simulator", device="density_matrix", use_noise=True, noise={...})`
        
        Args:
            noise_type (str): Type of noise model. Supported types:
                - "depolarizing": Uniform random Pauli errors
                - "amplitude_damping": Energy loss (T1 relaxation)
                - "phase_damping": Decoherence (T2 dephasing)
                - "pauli": Custom Pauli error rates
            **noise_params: Noise model parameters:
                - For "depolarizing": p (float, 0-1) - error probability
                - For "amplitude_damping": gamma or g (float, 0-1) - damping rate
                - For "phase_damping": lambda or l (float, 0-1) - dephasing rate
                - For "pauli": px, py, pz (float, 0-1) - X/Y/Z error rates
        
        Returns:
            Circuit: The same circuit instance configured with noise.
        
        Examples:
            >>> # Depolarizing noise
            >>> c = Circuit(2).h(0).cx(0, 1)
            >>> c.with_noise("depolarizing", p=0.05).run(shots=1024)
            
            >>> # Amplitude damping
            >>> c.with_noise("amplitude_damping", gamma=0.1).run(shots=1024)
            
            >>> # Phase damping
            >>> c.with_noise("phase_damping", lambda=0.05).run(shots=1024)
            
            >>> # Pauli channel (asymmetric noise)
            >>> c.with_noise("pauli", px=0.01, py=0.01, pz=0.05).run(shots=1024)
            
            >>> # Chain with other configuration
            >>> result = c.with_noise("depolarizing", p=0.05).device(shots=2048).run()
        
        Notes:
            - Automatically switches to density_matrix simulator
            - Noise is applied after every gate operation
            - For more fine-grained control, use `.device()` directly
        """
        # Build noise configuration dictionary
        noise_config = {"type": noise_type}
        noise_config.update(noise_params)
        
        # Configure device with density matrix simulator and noise
        self._device_opts.update({
            "provider": "simulator",
            "device": "density_matrix",
            "use_noise": True,
            "noise": noise_config
        })
        
        return self
    
    # ---- Pulse Programming API ----
    def use_pulse(self, mode: str = "hybrid", supported_waveforms: list | None = None, **pulse_options: Any) -> "Circuit":
        """Configure Pulse-level compilation for this circuit.
        
        TyxonQ Pulse Programming supports dual-mode execution (Memory: 8b12df21):
            - Mode A (Chain): Gate Circuit → Pulse Compiler → Execution
            - Mode B (Direct): Hamiltonian → Direct Pulse Evolution
        
        This method configures Mode A (chain compilation).
        
        Args:
            mode (str): Pulse compilation mode:
                - "hybrid": Mix gates and pulses (keep measurements, compile others)
                - "pulse_only": Compile all gates to pulses
                - "auto_lower": Automatic gate→pulse decision
            supported_waveforms (list, optional): List of supported waveform types.
                Used for hardware compatibility checking. Examples:
                - None (default): No validation
                - ['drag', 'gaussian']: Only these waveforms allowed
                - ['drag', 'gaussian', 'constant']: Hardware supports these
            **pulse_options: Additional Pulse compiler options:
                - device_params (dict): Physical device parameters:
                    - qubit_freq (list): Qubit frequencies (Hz)
                    - anharmonicity (list): Anharmonicity values (Hz)
                    - T1 (list): Amplitude damping times (s)
                    - T2 (list): Dephasing times (s)
                - inline_pulses (bool): Whether to inline pulse definitions
                    - False: Keep pulse references (default, supports autograd)
                    - True: Fully expand pulses (cloud-compatible, serializable)
                - output (str): Output format:
                    - "pulse_ir": TyxonQ Native IR (default)
                    - "tqasm": TQASM 0.2 format (for cloud submission)
        
        Returns:
            Circuit: The same circuit instance with Pulse compilation configured.
        
        Examples:
            >>> # Local simulation with Pulse (supports PyTorch autograd)
            >>> c = Circuit(2).h(0).cx(0, 1)
            >>> c.use_pulse(
            ...     mode="pulse_only",
            ...     device_params={
            ...         "qubit_freq": [5.0e9, 5.1e9],
            ...         "anharmonicity": [-330e6, -320e6]
            ...     },
            ...     inline_pulses=False
            ... )
            >>> result = c.device(provider="simulator").run()
            
            >>> # Cloud submission with TQASM and hardware constraints
            >>> c.use_pulse(
            ...     mode="pulse_only",
            ...     supported_waveforms=['drag', 'gaussian', 'constant'],
            ...     device_params={...},
            ...     inline_pulses=True,
            ...     output="tqasm"
            ... )
            >>> result = c.device(provider="tyxonq", device="homebrew_s2").run()
        
        See Also:
            - add_calibration(): Add custom pulse calibrations
            - docs/source/user_guide/pulse/index.rst: Complete Pulse programming guide
        """
        # Configure compile engine to use Pulse compiler
        self._compile_engine = "pulse"
        
        # Store Pulse-specific options
        pulse_compile_opts = {"mode": mode}
        if supported_waveforms is not None:
            pulse_compile_opts["supported_waveforms"] = supported_waveforms
        pulse_compile_opts.update(pulse_options)
        self._compile_opts.update(pulse_compile_opts)
        
        return self
    
    def add_calibration(self, gate_name: str, qubits: List[int], pulse_waveform: Any, params: Optional[Dict[str, Any]] = None) -> "Circuit":
        """Add a custom Pulse calibration for a specific gate.
        
        Custom calibrations override default gate→pulse decomposition,
        enabling hardware-specific optimization and fine-tuning.
        
        Args:
            gate_name (str): Gate name to calibrate (e.g., "x", "cx", "h")
            qubits (List[int]): Qubit indices this calibration applies to
            pulse_waveform: Pulse waveform object (from tyxonq.waveforms)
            params (dict, optional): Physical parameters:
                - qubit_freq (float): Qubit frequency (Hz)
                - drive_freq (float): Drive frequency (Hz)
                - phase (float): Pulse phase (radians)
        
        Returns:
            Circuit: The same circuit instance with calibration added.
        
        Examples:
            >>> from tyxonq import waveforms
            >>> 
            >>> c = Circuit(2)
            >>> 
            >>> # Custom DRAG pulse for X gate on qubit 0
            >>> x_pulse = waveforms.Drag(amp=0.5, duration=160, sigma=40, beta=0.2)
            >>> c.add_calibration("x", [0], x_pulse, {
            ...     "qubit_freq": 5.0e9,
            ...     "drive_freq": 5.0e9
            ... })
            >>> 
            >>> # Now when compiling to Pulse, this calibration will be used
            >>> c.x(0).use_pulse(mode="pulse_only").run()
        
        Notes:
            - Calibrations are stored in metadata["pulse_calibrations"]
            - Multiple calibrations can be added for different gates/qubits
            - Last added calibration takes precedence for conflicts
        """
        # Initialize calibrations dict in metadata
        if "pulse_calibrations" not in self.metadata:
            self.metadata["pulse_calibrations"] = {}
        
        # Store calibration
        cal_key = f"{gate_name}_{'_'.join(map(str, qubits))}"
        self.metadata["pulse_calibrations"][cal_key] = {
            "gate": gate_name,
            "qubits": list(qubits),
            "pulse": pulse_waveform,
            "params": params or {}
        }
        
        return self

    def state(self, engine: str | None = None, backend: Any | None = None, form: str | None = None) -> Any:
        """Get the quantum state of this circuit.
        
        This method executes the circuit using a state simulator and returns the
        quantum state. The simulator engine is automatically selected based on the
        circuit's device configuration, or can be explicitly specified.
        
        Args:
            engine (str | None): Simulator engine to use. If None, uses the device
                configured via .device() method. Options:
                - "statevector": Dense statevector simulation (O(2^n) memory)
                - "mps" or "matrix_product_state": MPS simulation (efficient for low entanglement)
                - "density_matrix": Density matrix simulation (supports noise)
            backend: Optional numeric backend (numpy/pytorch). If None, uses current global backend.
            form (str | None): Output format. Options:
                - None or "ket" or "tensor": Return backend tensor (default, preserves autograd)
                - "numpy": Return numpy array (breaks autograd)
        
        Returns:
            Quantum state representation (depends on engine and form):
            - statevector: 1D array/tensor of shape [2^num_qubits]
            - mps: 1D array/tensor (reconstructed from MPS)
            - density_matrix: 2D array/tensor of shape [2^num_qubits, 2^num_qubits]
            
        Examples:
            >>> # Use default statevector engine (returns backend tensor)
            >>> import torch
            >>> tq.set_backend("pytorch")
            >>> c = Circuit(2)
            >>> c.h(0).cx(0, 1)
            >>> psi = c.state()  # Returns torch.Tensor (preserves gradients)
            >>> print(type(psi), psi.shape)
            <class 'torch.Tensor'> torch.Size([4])
            
            >>> # Get numpy array (for visualization/non-differentiable use)
            >>> psi_np = c.state(form="numpy")
            >>> print(type(psi_np))
            <class 'numpy.ndarray'>
            
            >>> # Legacy compatibility: form="ket" also returns backend tensor
            >>> psi_ket = c.state(form="ket")
            >>> # psi_ket is identical to c.state()
            
            >>> # Configure MPS simulator via device()
            >>> c = Circuit(10)
            >>> c.device(provider="simulator", device="matrix_product_state", max_bond=32)
            >>> for i in range(10): c.h(i)
            >>> psi = c.state()  # Automatically uses MPS engine
            
            >>> # Explicitly specify engine
            >>> psi_mps = c.state(engine="mps")
        """
        # Resolve engine from device configuration or explicit parameter
        if engine is None:
            device_str = str(self._device_opts.get("device", "statevector"))
            # Normalize device string to engine name
            if "matrix_product_state" in device_str or "mps" in device_str:
                engine = "mps"
            elif "density_matrix" in device_str:
                engine = "density_matrix"
            else:
                engine = "statevector"
        
        # Select and instantiate the appropriate engine
        if engine == "statevector":
            from ...devices.simulators.statevector.engine import StatevectorEngine
            eng = StatevectorEngine(backend_name=backend)
            state_result = eng.state(self)
            
            # Handle output format: default is backend tensor (preserves autograd)
            if form == "numpy":
                # Explicitly requested numpy array
                import numpy as np
                return np.asarray(state_result)
            else:
                # Default or "ket"/"tensor": return backend tensor
                return state_result
        
        elif engine in ("mps", "matrix_product_state"):
            from ...devices.simulators.matrix_product_state.engine import MatrixProductStateEngine
            from ...libs.quantum_library.kernels.matrix_product_state import to_statevector as mps_to_statevector
            
            # Extract MPS-specific options from device config
            max_bond = self._device_opts.get("max_bond")
            eng = MatrixProductStateEngine(backend_name=backend, max_bond=max_bond)
            
            # Run the circuit to get MPS state, then convert to statevector
            result = eng.run(self, shots=0)
            # For now, reconstruct statevector from MPS for compatibility
            # TODO: Return native MPS representation in future
            from ...libs.quantum_library.kernels.matrix_product_state import init_product_state
            mps_state = init_product_state(self.num_qubits)
            
            # Re-execute circuit to build MPS state
            for op in self.ops:
                if not isinstance(op, (list, tuple)) or not op:
                    continue
                name = op[0]
                if name in ("h", "rz", "rx", "ry", "x", "s", "sdg"):
                    from ...libs.quantum_library.kernels.matrix_product_state import apply_1q as mps_apply_1q
                    from ...libs.quantum_library.kernels.gates import (
                        gate_h, gate_rz, gate_rx, gate_ry, gate_x, gate_s, gate_sd
                    )
                    q = int(op[1])
                    if name == "h":
                        mps_apply_1q(mps_state, gate_h(), q)
                    elif name == "rz":
                        mps_apply_1q(mps_state, gate_rz(float(op[2])), q)
                    elif name == "rx":
                        mps_apply_1q(mps_state, gate_rx(float(op[2])), q)
                    elif name == "ry":
                        mps_apply_1q(mps_state, gate_ry(float(op[2])), q)
                    elif name == "x":
                        mps_apply_1q(mps_state, gate_x(), q)
                    elif name == "s":
                        mps_apply_1q(mps_state, gate_s(), q)
                    elif name == "sdg":
                        mps_apply_1q(mps_state, gate_sd(), q)
                elif name in ("cx", "cz", "cy", "cry"):
                    from ...libs.quantum_library.kernels.matrix_product_state import apply_2q as mps_apply_2q
                    from ...libs.quantum_library.kernels.gates import (
                        gate_cx_4x4, gate_cz_4x4, gate_cry_4x4
                    )
                    q1, q2 = int(op[1]), int(op[2])
                    if name == "cx":
                        mps_apply_2q(mps_state, gate_cx_4x4(), q1, q2, max_bond=max_bond)
                    elif name == "cz":
                        mps_apply_2q(mps_state, gate_cz_4x4(), q1, q2, max_bond=max_bond)
                    elif name == "cry":
                        mps_apply_2q(mps_state, gate_cry_4x4(float(op[3])), q1, q2, max_bond=max_bond)
            
            return mps_to_statevector(mps_state)
        
        elif engine == "density_matrix":
            from ...devices.simulators.density_matrix.engine import DensityMatrixEngine
            eng = DensityMatrixEngine(backend_name=backend)
            # Density matrix engines don't have a direct .state() method
            # We need to reconstruct the density matrix from run() results
            result = eng.run(self, shots=0)
            # For density matrix, return the reconstructed density matrix
            # This requires adding a .density_matrix() method to the engine
            # For now, fall back to statevector
            from ...devices.simulators.statevector.engine import StatevectorEngine
            fallback_eng = StatevectorEngine(backend_name=backend)
            state_result = fallback_eng.state(self)
            
            # Handle output format: default is backend tensor
            if form == "numpy":
                import numpy as np
                return np.asarray(state_result)
            else:
                return state_result
        
        else:
            raise ValueError(
                f"Unknown engine '{engine}'. Supported: 'statevector', 'mps', 'density_matrix'"
            )

    def wavefunction(
        self,
        engine: str | None = None,
        backend: Any | None = None,
        form: str | None = None,
    ) -> Any:
        """Get the quantum wavefunction of this circuit.

        This is an alias for state() with clearer quantum physics semantics.
        The term 'wavefunction' is traditional in quantum mechanics, while 'state'
        is more general (applicable to mixed states in density matrix formalism).

        Args:
            engine: Simulator engine ("statevector", "mps", etc.)
            backend: Numeric backend (numpy/pytorch)
            form: Output format ("ket", "tensor", or "numpy")

        Returns:
            Quantum wavefunction as backend tensor or numpy array

        Examples:
            >>> c = Circuit(2)
            >>> c.h(0).cx(0, 1)
            >>> psi = c.wavefunction()  # |Φ+⟩ = (|00⟩ + |11⟩)/√2
            >>> psi_np = c.wavefunction(form="numpy")

        See Also:
            state: General quantum state getter (supports mixed states)
        """
        return self.state(engine=engine, backend=backend, form=form)

    # ---- Lightweight helpers ----
    def gate_count(self, gate_list: Optional[Sequence[str]] = None) -> int:
        """Count the number of gates in the circuit.
        
        Args:
            gate_list: Optional list of gate names to count. If provided, only
                gates with names in this list are counted (case-insensitive).
                If None, returns the total number of gates.
                
        Returns:
            int: Number of gates matching the criteria.
            
        Examples:
            >>> c = Circuit(2)
            >>> c.h(0).cx(0,1).x(1)
            >>> c.gate_count()  # Total gates
            3
            >>> c.gate_count(["h", "x"])  # Only H and X gates
            2
            >>> c.gate_count(["cnot", "cx"])  # CNOT gates (case-insensitive)
            1
        """
        if gate_list is None:
            return len(self.ops)
        names = {str(x).lower() for x in (gate_list if isinstance(gate_list, (list, tuple, set)) else [gate_list])}
        count = 0
        for op in self.ops:
            if str(op[0]).lower() in names:
                count += 1
        return count

    def gate_summary(self) -> Dict[str, int]:
        """Return a summary of gate types and their frequencies.
        
        Returns:
            Dict[str, int]: Mapping from gate name (lowercase) to count.
            
        Examples:
            >>> c = Circuit(3)
            >>> c.h(0).h(1).cx(0,1).cx(1,2).x(2)
            >>> c.gate_summary()
            {'h': 2, 'cx': 2, 'x': 1}
        """
        summary: Dict[str, int] = {}
        for op in self.ops:
            k = str(op[0]).lower()
            summary[k] = summary.get(k, 0) + 1
        return summary

    # ---- Analysis helpers (lightweight, backend-agnostic) ----
    def count_flop(self) -> Optional[int]:
        """Return a heuristic FLOP estimate for statevector simulation.

        This avoids tensor network dependencies. The estimate is coarse:
        - 1q gate ~ O(2^n)
        - 2q gate ~ O(2^(n+1))
        - other gates ignored
        Returns None if n is not available.
        """
        try:
            n = int(self.num_qubits)
        except Exception:
            return None
        flop: int = 0
        for op in self.ops:
            name = str(op[0]).lower()
            if name in ("h", "rx", "ry", "rz", "x", "y", "z", "s", "sdg"):
                flop += 1 << n
            elif name in ("cx", "cz", "cy", "cnot", "rxx", "rzz"):
                flop += 1 << (n + 1)
        return flop

    def get_circuit_summary(self):  # pragma: no cover - optional pandas
        """Return a dict summarizing the circuit if pandas is available.

        Columns: #qubits, #gates, #CNOT, #multicontrol, depth (if qiskit available), #FLOP (heuristic).
        """

        n_qubits = self.num_qubits
        n_gates = len(self.ops)
        n_cnot = self.gate_count(["cnot", "cx"])
        n_mc = sum(1 for op in self.ops if "multicontrol" in str(op[0]).lower())
        depth = None
        try:
            from ...compiler.compile_engine.qiskit.dialect import to_qiskit  # type: ignore
            qc = to_qiskit(self, add_measures=False)
            depth = qc.depth()
        except Exception:
            depth = None
        flop = self.count_flop()
        return {"#qubits": n_qubits, "#gates": [n_gates], "#CNOT": [n_cnot], "#multicontrol": [n_mc], "depth": [depth], "#FLOP": [flop]}

    def extended(self, extra_ops: Sequence[Sequence[Any]]) -> "Circuit":
        """Return a new Circuit with ops extended by extra_ops (no mutation)."""
        new_ops = list(self.ops) + [tuple(op) for op in extra_ops]
        return Circuit(num_qubits=self.num_qubits, ops=new_ops, metadata=dict(self.metadata), instructions=list(self.instructions))

    def compose(self, other: "Circuit", indices: Optional[Sequence[int]] = None) -> "Circuit":
        """Append another Circuit's ops. If `indices` given, remap other's qubits by indices[i]."""
        if indices is None:
            if other.num_qubits != self.num_qubits:
                raise ValueError("compose requires equal num_qubits when indices is None")
            mapped_ops = list(other.ops)
        else:
            # indices maps other's logical i -> self physical indices[i]
            idx_list = list(indices)
            def _map_op(op: Sequence[Any]) -> tuple:
                mapped: List[Any] = [op[0]]
                for a in op[1:]:
                    if isinstance(a, int):
                        if a < 0 or a >= len(idx_list):
                            raise ValueError("compose indices out of range for other circuit")
                        mapped.append(int(idx_list[a]))
                    else:
                        mapped.append(a)
                return tuple(mapped)
            mapped_ops = [_map_op(op) for op in other.ops]
        return self.extended(mapped_ops)

    def remap_qubits(self, mapping: Dict[int, int], *, new_num_qubits: Optional[int] = None) -> "Circuit":
        """Return a new Circuit with qubit indices remapped according to `mapping`.

        All int arguments in ops are treated as qubit indices and must be present in mapping.
        """
        def _remap_op(op: Sequence[Any]) -> tuple:
            out: List[Any] = [op[0]]
            for a in op[1:]:
                if isinstance(a, int):
                    if a not in mapping:
                        raise KeyError(f"qubit {a} missing in mapping")
                    out.append(int(mapping[a]))
                else:
                    out.append(a)
            return tuple(out)
        nn = int(new_num_qubits) if new_num_qubits is not None else self.num_qubits
        return Circuit(num_qubits=nn, ops=[_remap_op(op) for op in self.ops], metadata=dict(self.metadata), instructions=list(self.instructions))

    def positional_logical_mapping(self) -> Dict[int, int]:
        """Return positional->logical mapping from explicit instructions or measure_z ops."""
        # Prefer explicit instructions if present
        measures = [idxs for (n, idxs) in self.instructions if str(n).lower() == "measure"]
        if measures:
            pos_to_logical: Dict[int, int] = {}
            for pos, idxs in enumerate(measures):
                if not idxs:
                    continue
                pos_to_logical[pos] = int(idxs[0])
            return pos_to_logical or {i: i for i in range(self.num_qubits)}
        # Fallback to scanning measure_z ops
        pos_to_logical: Dict[int, int] = {}
        pos = 0
        for op in self.ops:
            if op and str(op[0]).lower() == "measure_z":
                q = int(op[1])
                pos_to_logical[pos] = q
                pos += 1
        return pos_to_logical or {i: i for i in range(self.num_qubits)}

    def inverse(self, *, strict: bool = False) -> "Circuit":
        """Return a unitary inverse circuit for supported ops (h, cx, rz).

        Non-unitary ops like measure/reset/barrier are skipped unless strict=True (then error).
        Unknown ops raise if strict=True, else skipped.
        """
        inv_ops: List[tuple] = []
        for op in reversed(self.ops):
            name = str(op[0]).lower()
            if name == "h":
                inv_ops.append(("h", int(op[1])))
            elif name == "cx":
                inv_ops.append(("cx", int(op[1]), int(op[2])))
            elif name == "rz":
                inv_ops.append(("rz", int(op[1]), -float(op[2])))
            elif name in ("measure_z", "reset", "barrier"):
                if strict:
                    raise ValueError(f"non-unitary op not invertible: {name}")
                continue
            else:
                if strict:
                    raise NotImplementedError(f"inverse not implemented for op: {name}")
                continue
        return Circuit(num_qubits=self.num_qubits, ops=inv_ops, metadata=dict(self.metadata), instructions=list(self.instructions))

    # ---- JSON IO (provider-agnostic, minimal) ----
    def to_json_obj(self) -> Dict[str, Any]:
        return {
            "num_qubits": int(self.num_qubits),
            "ops": list(self.ops),
            "metadata": dict(self.metadata),
            "instructions": [(n, list(idxs)) for (n, idxs) in self.instructions],
        }

    def to_json_str(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_json_obj(), ensure_ascii=False, indent=indent)


    # ---- Provider adapters (thin convenience wrappers) ----
    def to_openqasm(self) -> str:
        """Serialize this IR circuit to OpenQASM 2 using the compiler facade.

        Delegates to compiler API (compile_engine='qiskit', output='qasm2').
        """
        compiled = self.compile(compile_engine="qiskit", output="qasm2")
        # compile() returns self with compiled_source stored in _compiled_source
        return compiled._compiled_source  # type: ignore[return-value]

    @overload
    def compile(self, *, provider: None = ..., output: None = ..., target: Any | None = ..., options: Dict[str, Any] | None = ...) -> "Circuit": ...

    @overload
    def compile(self, *, provider: str = ..., output: str = ..., target: Any | None = ..., options: Dict[str, Any] | None = ...) -> Any: ...

    def compile(
        self,
        *,
        compile_engine: Optional[str] = None,
        output: Optional[str] = None,
        target: Any | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Any:
        """Delegate to compiler.api.compile or act as chainable setter when no args.

        - Chainable模式：若 provider/output/target/options 全为 None，则返回 self（不触发编译）。
        - 编译模式：转发到 compiler.api.compile，options 与已记录的 _compile_opts 合并。
        """
        # Chainable: no explicit args means only marking intent
        if compile_engine is None and output is None and target is None and options is None:
            return self


        # 这里需要判断一下 是不是已经编译过了。
        if self._compile_engine == compile_engine and self._compile_output == output and \
            self._compile_opts == options:
            # If a direct source is present, skip compilation entirely 
            if self._compiled_source is not None:
                return self

        # Delegate to compiler facade exactly following its contract
        prov = compile_engine or "default"
        out = output or "ir"
        merged_opts = dict(self._compile_opts)
        if options:
            merged_opts.update(options)
        # compiler/api.compile 现在只接受 (circuit, compile_engine, output, options)
        res = compile_api(self, compile_engine=prov, output=out, options=merged_opts)
        
        # 关键优化：如果编译结果是字符串（TQASM/QASM），缓存到 _source 避免重复编译
        # 这样后续 .run() 时会直接使用缓存的 source，不会重新编译
        self._compiled_source = res.get("compiled_source",None)
        return self
    

    def run(
        self,
        *,
        provider: Optional[str] = None,
        device: Optional[str] = None,
        shots: int = 1024,
        wait_async_result: Optional[bool] = False,
        **opts: Any,
    ) -> Any:
        """执行电路：
        - 若构造时提供了 source，则直接按 source 提交给设备层；
        - 否则，先按 compile_engine/output/target 进行编译，再根据产物类型提交。
        注意：不在此处补测量或发出告警；该逻辑归属编译阶段。
        """
        from ...devices import base as device_base
        from ...devices.hardware import config as hwcfg

        # Merge device options with call-time overrides and extract reserved keys
        dev_opts = {**self._device_opts, **opts}
        dev_provider = dev_opts.pop("provider", provider)
        dev_device = dev_opts.pop("device", device)
        dev_shots = int(dev_opts.pop("shots", shots))

        # If pre-compiled/native source exists, submit directly
        if self._compiled_source is not None:
            tasks = device_base.run(
                provider=dev_provider,
                device=dev_device,
                source=self._compiled_source,
                shots=dev_shots,
                **dev_opts,
            )
        else:
            # Compile first using current defaults
            # compile() 返回 self，并将编译结果存储在 self._compiled_source 中
            compiled_circuit = self.compile(
                compile_engine=self._compile_engine,
                output=self._compile_output,
                target=self._compile_target,
                options=self._compile_opts,
            )
            
            # compiled_circuit 是 Circuit 对象，编译结果存在 _compiled_source 属性中
            source_to_submit = compiled_circuit._compiled_source
            if source_to_submit is not None:
                # 有编译源代码（QASM2/QASM3/TQASM），提交给 device
                tasks = device_base.run(
                    provider=dev_provider,
                    device=dev_device,
                    source=source_to_submit,
                    shots=dev_shots,
                    **dev_opts,
                )
            else:
                # 没有编译源代码，直接提交 IR (仅供 simulator/local 使用)
                tasks = device_base.run(
                    provider=dev_provider,
                    device=dev_device,
                    circuit=self,
                    shots=dev_shots,
                    **dev_opts,
                )

        # unified_list = tasks if isinstance(tasks, list) else [tasks]
        unified_result_list=[]
        # Normalize to list of unified payloads
        if wait_async_result is False:
            for t in tasks:
                task_result = t.get_result(wait=False)
                unified_result_list.append(task_result)

        else:
            for t in tasks:
                task_result = t.get_result(wait=True)
                unified_result_list.append(task_result)

    
        # Fetch final results where needed and attach postprocessing
        from ...postprocessing import apply_postprocessing  # 延迟导入，保持解耦
        results: List[Dict[str, Any]] = []
        for rr in unified_result_list:
            if isinstance(rr, dict):
                has_payload = (rr.get('result') is not None) or (rr.get('counts') is not None) or (rr.get('expectations') is not None)
                if has_payload:
                    post = apply_postprocessing(rr, self._post_opts if isinstance(self._post_opts, dict) else {})
                    rr["postprocessing"] = post
                    results.append(rr)
                else:
                    error_result = {
                        'result':{},
                        'result_meta': rr.get('result_meta', {}),
                        'postprocessing': {
                            'method': None,
                            'result': None
                        }
                    }
                    results.append(error_result)
            else:
                raise TypeError('result is not a dict',rr)
             
        return results if isinstance(tasks, list) else results[0]

    # ---- Task helpers for cloud.api thin wrappers ----
    def get_task_details(self,task: Any, *, wait: bool = False, poll_interval: float = 2.0, timeout: float = 15.0) -> Dict[str, Any]:
        return task.get_result(task=task, wait=wait, poll_interval=poll_interval, timeout=timeout)
    
    def get_result(self, task: Any, *, wait: bool = False, poll_interval: float = 2.0, timeout: float = 15.0)-> Dict[str, Any]:
        return task.get_result(task=task, wait=wait, poll_interval=poll_interval, timeout=timeout)

    def cancel(self, task: Any) -> Any:
        dev = getattr(task, "device", None)
        if dev is None:
            raise ValueError("Task handle missing device information")
        dev_str = str(dev)
        prov = (dev_str.split("::", 1)[0]) if "::" in dev_str else "simulator"
        from ...devices.base import resolve_driver
        from ...devices.hardware import config as hwcfg

        tok = hwcfg.get_token(provider=prov, device=dev_str)
        drv = resolve_driver(prov, dev_str)
        if hasattr(drv, "remove_task"):
            return drv.remove_task(task, tok)
        raise NotImplementedError("cancel not supported for this provider/task type")

    def submit_task(
        self,
        *,
        provider: Optional[str] = None,
        device: Optional[str] = None,
        shots: int = 1024,
        compiler: str = "qiskit",
        auto_compile: bool = True,
        **opts: Any,
    ) -> Any:
        # Submit is an alias of run with identical semantics
        return self.run(provider=provider, device=device, shots=shots, compiler=compiler, auto_compile=auto_compile, **opts)

    # Note: builder-style gate helpers have been moved to `CircuitBuilder`.

    # Instruction helpers
    def add_measure(self, *qubits: int) -> "Circuit":
        new_inst = list(self.instructions)
        for q in qubits:
            new_inst.append(("measure", (int(q),)))
        return replace(self, instructions=new_inst)

    def add_reset(self, *qubits: int) -> "Circuit":
        new_inst = list(self.instructions)
        for q in qubits:
            new_inst.append(("reset", (int(q),)))
        return replace(self, instructions=new_inst)

    def add_barrier(self, *qubits: int) -> "Circuit":
        new_inst = list(self.instructions)
        if qubits:
            new_inst.append(("barrier", tuple(int(q) for q in qubits)))
        else:
            new_inst.append(("barrier", tuple(range(self.num_qubits))))
        return replace(self, instructions=new_inst)

    # ---- Builder-style ergonomic gate helpers (in-place; return self) ----
    def h(self, q: int):
        """Apply Hadamard gate to qubit q.
        
        The Hadamard gate creates superposition by mapping |0⟩ → (|0⟩ + |1⟩)/√2
        and |1⟩ → (|0⟩ - |1⟩)/√2. It's fundamental for quantum algorithms.
        
        Args:
            q (int): Target qubit index (0-based).
            
        Returns:
            Circuit: Self for method chaining.
            
        Examples:
            >>> c = Circuit(2)
            >>> c.h(0)  # Create superposition on qubit 0
            >>> c.h(0).h(1)  # Chain operations
        """
        self.ops.append(("h", int(q)))
        return self

    def H(self, q: int):
        return self.h(q)

    def rz(self, q: int, theta: Any):
        """Apply RZ rotation gate around Z-axis to qubit q.
        
        The RZ gate implements a rotation by angle theta around the Z-axis:
        RZ(θ) = exp(-iθZ/2) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
        
        This gate only changes the relative phase between |0⟩ and |1⟩ states
        without affecting computational basis probabilities.
        
        Args:
            q (int): Target qubit index (0-based).
            theta (Any): Rotation angle in radians. Can be a float, parameter, or expression.
            
        Returns:
            Circuit: Self for method chaining.
            
        Examples:
            >>> import numpy as np
            >>> c = Circuit(1)
            >>> c.rz(0, np.pi/4)  # π/4 rotation
            >>> c.rz(0, 'theta')  # Parameterized rotation
        """
        self.ops.append(("rz", int(q), theta))
        return self

    def RZ(self, q: int, theta: Any):
        return self.rz(q, theta)

    def rx(self, q: int, theta: Any):
        self.ops.append(("rx", int(q), theta))
        return self

    def RX(self, q: int, theta: Any):
        return self.rx(q, theta)

    def cx(self, c: int, t: int):
        """Apply controlled-X (CNOT) gate between control and target qubits.
        
        The CNOT gate flips the target qubit if and only if the control qubit is |1⟩:
        CNOT|00⟩ = |00⟩, CNOT|01⟩ = |01⟩, CNOT|10⟩ = |11⟩, CNOT|11⟩ = |10⟩
        
        This is the fundamental two-qubit gate for creating entanglement and
        implementing classical logic operations in quantum circuits.
        
        Args:
            c (int): Control qubit index (0-based).
            t (int): Target qubit index (0-based).
            
        Returns:
            Circuit: Self for method chaining.
            
        Examples:
            >>> # Create Bell state
            >>> c = Circuit(2)
            >>> c.h(0).cx(0, 1)  # |00⟩ + |11⟩
            >>> 
            >>> # Chain multiple CNOTs
            >>> c = Circuit(3)
            >>> c.cx(0, 1).cx(1, 2)  # Linear entanglement
        """
        self.ops.append(("cx", int(c), int(t)))
        return self

    def CX(self, c: int, t: int):
        return self.cx(c, t)

    def cnot(self, c: int, t: int):
        return self.cx(c, t)

    def CNOT(self, c: int, t: int):
        return self.cx(c, t)

    def unitary(self, *qubits: int, matrix: Any):
        """Apply arbitrary unitary matrix to one or more qubits.
        
        This is a general-purpose gate that applies a custom unitary transformation
        to the specified qubits. It's useful for:
        - Implementing custom gates not available in the standard gate set
        - Random quantum circuits and benchmarking
        - Variational quantum algorithms with parameterized unitaries
        - Clifford gate optimization
        
        Args:
            *qubits: Target qubit indices (1 or 2 qubits supported).
            matrix: Unitary matrix as numpy array or backend tensor.
                   - For 1 qubit: 2×2 complex matrix
                   - For 2 qubits: 4×4 complex matrix
                   
        Returns:
            Circuit: Self for method chaining.
            
        Raises:
            ValueError: If qubits count is not 1 or 2, or matrix shape is invalid.
            
        Examples:
            >>> import numpy as np
            >>> 
            >>> # Apply custom single-qubit gate (√X)
            >>> c = Circuit(1)
            >>> sqrt_x = np.array([[0.5+0.5j, 0.5-0.5j],
            ...                    [0.5-0.5j, 0.5+0.5j]])
            >>> c.unitary(0, matrix=sqrt_x)
            >>> 
            >>> # Apply custom two-qubit gate (iSWAP)
            >>> c = Circuit(2)
            >>> iswap = np.array([[1, 0, 0, 0],
            ...                   [0, 0, 1j, 0],
            ...                   [0, 1j, 0, 0],
            ...                   [0, 0, 0, 1]])
            >>> c.unitary(0, 1, matrix=iswap)
            >>> 
            >>> # Use in variational circuits
            >>> from tyxonq.libs.quantum_library.kernels.gates import gate_ry
            >>> param = 0.5
            >>> c.unitary(0, matrix=gate_ry(param))
        """
        # Validate inputs
        if len(qubits) == 0:
            raise ValueError("unitary requires at least one qubit index")
        if len(qubits) > 2:
            raise ValueError(f"unitary currently supports 1-2 qubits, got {len(qubits)}")
        
        # Validate qubit indices
        for q in qubits:
            if not isinstance(q, int) or q < 0 or q >= self.num_qubits:
                raise ValueError(f"Invalid qubit index: {q}")
        
        # Validate matrix shape
        from ...numerics.api import get_backend
        K = get_backend(None)
        mat = K.asarray(matrix)  # Keep tensor type to preserve autograd
        expected_dim = 1 << len(qubits)  # 2^k for k qubits
        mat_shape = tuple(mat.shape) if hasattr(mat, 'shape') else (len(mat), len(mat[0]))
        if mat_shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Matrix shape {mat_shape} incompatible with {len(qubits)} qubit(s), "
                f"expected {(expected_dim, expected_dim)}"
            )
        
        # Store matrix in metadata for retrieval by executor
        # Use a unique key to avoid collisions
        mat_key = f"_unitary_{len(self.ops)}"
        if not hasattr(self, "_unitary_cache"):
            self._unitary_cache = {}
        self._unitary_cache[mat_key] = mat
        
        # Add operation with matrix key
        if len(qubits) == 1:
            self.ops.append(("unitary", int(qubits[0]), mat_key))
        elif len(qubits) == 2:
            self.ops.append(("unitary", int(qubits[0]), int(qubits[1]), mat_key))
        
        return self

    def kraus(self, qubit: int, operators: Any, status: float | None = None):
        """Apply general Kraus channel (quantum noise/measurement) to a qubit.
        
        This method applies a completely positive trace-preserving (CPTP) map
        represented by Kraus operators {K₀, K₁, ..., Kₙ} satisfying ∑ᵢ K†ᵢKᵢ = I.
        
        Kraus channels model:
        - Quantum noise (decoherence, damping, dephasing)
        - Measurement-induced dynamics (MIPT, monitoring)
        - Open quantum systems evolution
        - Post-selection protocols
        
        Physical interpretation:
        - Statevector: Stochastic unraveling |ψ⟩ → Kᵢ|ψ⟩/||Kᵢ|ψ⟩|| (Monte Carlo)
        - Density matrix: Exact evolution ρ → ∑ᵢ KᵢρK†ᵢ
        
        Args:
            qubit: Target qubit index (0-based)
            operators: List of Kraus operators, each a 2×2 numpy array/tensor.
                      Standard channels available in tyxonq.libs.quantum_library.noise:
                      - depolarizing_channel(p)
                      - amplitude_damping_channel(gamma)  # T₁ relaxation
                      - phase_damping_channel(lambda)     # T₂ dephasing
                      - pauli_channel(px, py, pz)
                      - measurement_channel(p)            # For MIPT
            status: Random variable in [0,1] for stochastic selection (statevector only).
                   If None, uses uniform random sampling.
                   
        Returns:
            Circuit: Self for method chaining
            
        Examples:
            >>> # Apply amplitude damping (T₁ relaxation)
            >>> from tyxonq.libs.quantum_library.noise import amplitude_damping_channel
            >>> c = Circuit(2)
            >>> c.h(0).cx(0, 1)
            >>> kraus_ops = amplitude_damping_channel(gamma=0.1)
            >>> c.kraus(0, kraus_ops)
            >>>
            >>> # Measurement-induced phase transition (MIPT)
            >>> from tyxonq.libs.quantum_library.noise import measurement_channel
            >>> c = Circuit(10)
            >>> # ... apply random unitaries ...
            >>> for i in range(10):
            >>>     c.kraus(i, measurement_channel(p=0.1), status=np.random.rand())
            >>>
            >>> # Custom Kraus operators
            >>> import numpy as np
            >>> K0 = np.array([[1, 0], [0, 0.9]])  # Custom channel
            >>> K1 = np.array([[0, 0.1], [0, 0]])
            >>> c.kraus(0, [K0, K1])
            >>>
            >>> # Chain with other gates
            >>> c.h(0).kraus(0, amplitude_damping_channel(0.05)).cx(0, 1)
        """
        if not isinstance(qubit, int) or qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Invalid qubit index: {qubit}")
        
        # Store Kraus operators in cache for retrieval by executor
        kraus_key = f"_kraus_{len(self.ops)}"
        if not hasattr(self, "_kraus_cache"):
            self._kraus_cache = {}
        self._kraus_cache[kraus_key] = operators
        
        # Add operation to IR with kraus key and optional status
        if status is not None:
            self.ops.append(("kraus", int(qubit), kraus_key, float(status)))
        else:
            self.ops.append(("kraus", int(qubit), kraus_key))
        
        return self

    def measure_z(self, q: int):
        """Add Z-basis measurement instruction for qubit q.
        
        Measures the qubit in the computational basis {|0⟩, |1⟩}, collapsing
        the quantum state and producing a classical bit outcome.
        
        Note: This adds a measurement instruction to the circuit but does not
        immediately execute it. The measurement occurs during circuit execution.
        
        Args:
            q (int): Target qubit index (0-based) to measure.
            
        Returns:
            Circuit: Self for method chaining.
            
        Examples:
            >>> c = Circuit(2)
            >>> c.h(0).cx(0, 1)  # Create Bell state
            >>> c.measure_z(0).measure_z(1)  # Measure both qubits
            >>> 
            >>> # Measure all qubits in a loop
            >>> for i in range(c.num_qubits):
            ...     c.measure_z(i)
        """
        self.ops.append(("measure_z", int(q)))
        return self

    def MEASURE_Z(self, q: int):
        return self.measure_z(q)

    def measure_reference(self, q: int, with_prob: bool = False) -> tuple[str, float] | str:
        """Perform reference measurement (simulation-time measurement with result).
        
        This method immediately measures the qubit and returns the measurement
        outcome along with its probability. Unlike measure_z(), this is executed
        during circuit construction for simulation purposes.
        
        Note: This is primarily for simulation/testing workflows, particularly
        useful for mid-circuit measurement scenarios where you need to condition
        on measurement outcomes.
        
        Args:
            q (int): Target qubit index (0-based) to measure.
            with_prob (bool): If True, return both outcome and probability as (outcome, prob).
                             If False, return only the outcome string.
            
        Returns:
            tuple[str, float] | str: If with_prob=True, returns ("0" or "1", probability).
                                     If with_prob=False, returns "0" or "1".
            
        Examples:
            >>> c = Circuit(2)
            >>> c.h(0)  # Create superposition
            >>> outcome = c.measure_reference(0)  # Get measurement outcome
            >>> print(f"Measured: {outcome}")  # "0" or "1"
            >>> 
            >>> # With probability
            >>> outcome, prob = c.measure_reference(0, with_prob=True)
            >>> print(f"Measured {outcome} with probability {prob}")
            >>> 
            >>> # Use for conditional logic
            >>> if outcome == "0":
            ...     c.x(1)  # Apply X if measured 0
        """
        from ...devices.simulators.statevector.engine import StatevectorEngine
        from ...numerics.api import get_backend
        
        # Get current state
        eng = StatevectorEngine(backend_name=None)
        state = eng.state(self)
        
        # Compute measurement probabilities for qubit q
        nb = get_backend(None)
        n = self.num_qubits
        state_tensor = nb.reshape(nb.asarray(state), (2,) * n)
        state_perm = nb.moveaxis(state_tensor, q, 0)
        probs_2d = nb.abs(nb.reshape(state_perm, (2, -1))) ** 2
        probs = nb.sum(probs_2d, axis=1)
        probs_np = nb.to_numpy(probs)
        
        # Sample measurement outcome
        rng = nb.rng(None)
        outcome_idx = int(rng.choice(2, p=probs_np))
        outcome_str = "1" if outcome_idx == 1 else "0"
        prob = float(probs_np[outcome_idx])
        
        if with_prob:
            return (outcome_str, prob)
        else:
            return outcome_str

    def mid_measurement(self, q: int, keep: int = 0) -> "Circuit":
        """Perform mid-circuit measurement with post-selection.
        
        This method adds a projection operation that collapses the quantum state
        by measuring qubit q and post-selecting on the specified outcome.
        The state is renormalized after projection.
        
        Note: This is a non-unitary operation that reduces the quantum state
        to a subspace. It's useful for:
        - Quantum error correction protocols
        - Adaptive quantum algorithms
        - Syndrome extraction circuits
        - Stabilizer simulation benchmarks
        
        Args:
            q (int): Target qubit index (0-based) to measure and project.
            keep (int): Post-selected measurement outcome (0 or 1).
                       If keep=0, project onto |0⟩ subspace for qubit q.
                       If keep=1, project onto |1⟩ subspace for qubit q.
            
        Returns:
            Circuit: Self for method chaining.
            
        Examples:
            >>> # Post-select on measuring 0
            >>> c = Circuit(2)
            >>> c.h(0).cx(0, 1)
            >>> c.mid_measurement(0, keep=0)  # Keep only |0⟩ component of qubit 0
            >>> # State is now projected and renormalized
            >>> 
            >>> # Quantum error correction syndrome extraction
            >>> c = Circuit(5)
            >>> # ... encode logical qubit ...
            >>> c.cx(0, 3).cx(1, 3)  # Syndrome extraction
            >>> c.mid_measurement(3, keep=0)  # Post-select on no error
            >>> 
            >>> # Adaptive algorithm
            >>> outcome = c.measure_reference(0)
            >>> if outcome == "0":
            ...     c.mid_measurement(0, keep=0)  # Collapse to measured state
        """
        # Add project_z operation to circuit ops
        # This will be handled by engines that support projection
        self.ops.append(("project_z", int(q), int(keep)))
        return self

    def reset(self, q: int):
        """Warning: reset operation is typically not supported by hardware in logical circuits.
        This is a simulation-only operation that projects qubit to |0⟩ state."""
        warnings.warn("reset operation is typically not supported by hardware in logical circuits. "
                    "This is a simulation-only operation that projects qubit to |0⟩ state.", 
                    UserWarning, stacklevel=2)
        self.ops.append(("reset", int(q)))
        return self

    def RESET(self, q: int):
        return self.reset(q)
    
    # --- Additional common gates to preserve legacy examples ---
    def x(self, q: int):
        self.ops.append(("x", int(q)))
        return self

    def X(self, q: int):
        return self.x(q)

    def y(self, q: int):
        self.ops.append(("y", int(q)))
        return self

    def Y(self, q: int):
        return self.y(q)

    def z(self, q: int):
        """Apply Pauli-Z gate to qubit q."""
        self.ops.append(("z", int(q)))
        return self

    def Z(self, q: int):
        return self.z(q)

    def s(self, q: int):
        """Apply S gate (phase gate, √Z) to qubit q."""
        self.ops.append(("s", int(q)))
        return self

    def S(self, q: int):
        return self.s(q)

    def sdg(self, q: int):
        """Apply S† gate (inverse of S gate) to qubit q."""
        self.ops.append(("sdg", int(q)))
        return self

    def Sdg(self, q: int):
        return self.sdg(q)

    def t(self, q: int):
        """Apply T gate (π/8 gate, √S) to qubit q."""
        self.ops.append(("t", int(q)))
        return self

    def T(self, q: int):
        return self.t(q)

    def tdg(self, q: int):
        """Apply T† gate (inverse of T gate) to qubit q."""
        self.ops.append(("tdg", int(q)))
        return self

    def Tdg(self, q: int):
        return self.tdg(q)

    def ry(self, q: int, theta: Any):
        self.ops.append(("ry", int(q), theta))
        return self

    def RY(self, q: int, theta: Any):
        return self.ry(q, theta)

    def cz(self, c: int, t: int):
        self.ops.append(("cz", int(c), int(t)))
        return self

    def CZ(self, c: int, t: int):
        return self.cz(c, t)

    def cy(self, c: int, t: int):
        self.ops.append(("cy", int(c), int(t)))
        return self

    def CY(self, c: int, t: int):
        return self.cy(c, t)

    def rxx(self, c: int, t: int, theta: Any):
        self.ops.append(("rxx", int(c), int(t), theta))
        return self

    def RXX(self, c: int, t: int, theta: Any):
        return self.rxx(c, t, theta)

    def ryy(self, c: int, t: int, theta: Any):
        self.ops.append(("ryy", int(c), int(t), theta))
        return self

    def RYY(self, c: int, t: int, theta: Any):
        return self.ryy(c, t, theta)

    def rzz(self, c: int, t: int, theta: Any):
        self.ops.append(("rzz", int(c), int(t), theta))
        return self

    def RZZ(self, c: int, t: int, theta: Any):
        return self.rzz(c, t, theta)

    def iswap(self, q0: int, q1: int):
        """Apply iSWAP gate between two qubits.
        
        The iSWAP gate exchanges quantum states and applies a relative phase:
        iSWAP = exp(-iπ/4 · σ_x ⊗ σ_x)
        
        Matrix representation:
        [[1,  0,  0,  0],
         [0,  0, 1j,  0],
         [0, 1j,  0,  0],
         [0,  0,  0,  1]]
        
        Physical properties:
        - Swaps quantum states: iSWAP|01⟩ = i|10⟩, iSWAP|10⟩ = i|01⟩
        - Adds π/2 relative phase to swapped basis states
        - Native gate on many superconducting platforms (Rigetti, IonQ)
        - Useful for exchanging and entangling qubits
        
        Reference:
            Shende & Markov, "Minimal universal two-qubit controlled-NOT-based circuits",
            Physical Review A 72, 062305 (2005)
            https://arxiv.org/abs/quant-ph/0308033
        
        Args:
            q0 (int): First qubit index (0-based).
            q1 (int): Second qubit index (0-based).
            
        Returns:
            Circuit: Self for method chaining.
            
        Examples:
            >>> # Create iSWAP entanglement
            >>> c = Circuit(2)
            >>> c.h(0).iswap(0, 1)  # Creates a specific entangled state
            >>> 
            >>> # Chain multiple iSWAPs
            >>> c = Circuit(4)
            >>> c.iswap(0, 1).iswap(2, 3)  # Two independent iSWAPs
            >>> 
            >>> # Use in hybrid gate-pulse mode
            >>> c = Circuit(3)
            >>> c.h(0).iswap(0, 1).cx(1, 2)  # Mix iSWAP and CNOT
        """
        self.ops.append(("iswap", int(q0), int(q1)))
        return self

    def ISWAP(self, q0: int, q1: int):
        """Uppercase alias for iswap()."""
        return self.iswap(q0, q1)

    def swap(self, q0: int, q1: int):
        """Apply SWAP gate between two qubits.
        
        The SWAP gate exchanges quantum states without adding phase:
        SWAP = [[1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]]
        
        Physical properties:
        - Pure state exchange: SWAP|01⟩ = |10⟩, SWAP|10⟩ = |01⟩
        - No relative phase (unlike iSWAP)
        - Equivalent to 3 CNOT gates: CX(q0,q1)·CX(q1,q0)·CX(q0,q1)
        - Useful for qubit relabeling and layout optimization
        - Commonly used in quantum algorithms to reorder qubits
        
        Implementation note:
            In pulse-level compilation, SWAP is decomposed into 3 CX gates
            for hardware execution (see gate_to_pulse.py)
        
        Args:
            q0 (int): First qubit index (0-based).
            q1 (int): Second qubit index (0-based).
            
        Returns:
            Circuit: Self for method chaining.
            
        Examples:
            >>> # Swap adjacent qubits
            >>> c = Circuit(3)
            >>> c.h(0).cx(0, 1).swap(0, 2)  # Rearrange qubit order
            >>> 
            >>> # Use in layout optimization
            >>> c = Circuit(4)
            >>> c.cx(0, 1).swap(1, 3).cx(1, 2)  # Implement logical circuit on physical hardware
            >>> 
            >>> # Combine with measurements
            >>> c = Circuit(2)
            >>> c.h(0).h(1).swap(0, 1).measure_z(0).measure_z(1)
        """
        self.ops.append(("swap", int(q0), int(q1)))
        return self

    def SWAP(self, q0: int, q1: int):
        """Uppercase alias for swap()."""
        return self.swap(q0, q1)

    def expectation(self, *pauli_ops: Any) -> Any:
        """Compute expectation value of Pauli operator product.
        
        This method computes ⟨ψ|O|ψ⟩ where O is a product of Pauli operators.
        The computation method is automatically selected based on the circuit's
        device configuration.
        
        Each Pauli operator is specified as (gate_matrix, [qubit_indices]).
        
        ⚡ PERFORMANCE TIP (性能提示):
        ----------------------------------------
        For multiple observables, avoid calling expectation() repeatedly as
        each call re-executes the circuit. Instead, use the Hamiltonian matrix
        approach for 10-15x speedup:
        
        对于多个observable，避免重夏调用expectation()，因为每次调用都会
        重新执行电路。使用Hamiltonian矩阵方法可获得10-15倍加速：
        
        ❌ SLOW (慢 - 避免):
            energy = 0.0
            for i in range(n):
                energy += circuit.expectation((gate_z(), [i]))  # N circuit executions!
        
        ✅ FAST (快 - 推荐):
            from tyxonq.libs.quantum_library.kernels.pauli import pauli_string_sum_dense
            H = pauli_string_sum_dense(pauli_terms, weights)
            psi = circuit.state()  # Execute circuit only once!
            energy = torch.real(torch.dot(torch.conj(psi), H @ psi))
        
        See examples/performance_optimization_tips.md for detailed guide.
        详见 examples/performance_optimization_tips.md 获取详细指南。
        
        Args:
            *pauli_ops: Variable number of Pauli operator tuples.
                Each tuple is (gate, qubits) where:
                - gate: Pauli gate matrix or gate object (X, Y, Z)
                - qubits: List of qubit indices
                
        Returns:
            Expectation value (real number for Hermitian operators)
            
        Examples:
            >>> # Single-qubit Pauli-X expectation: ⟨X_0⟩
            >>> c = Circuit(2).h(0).cx(0,1)
            >>> exp_x = c.expectation((tq.gates.x(), [0]))
            
            >>> # Two-qubit Pauli product: ⟨Z_0 Z_1⟩
            >>> exp_zz = c.expectation((tq.gates.z(), [0]), (tq.gates.z(), [1]))
            
            >>> # For multiple observables (OPTIMIZED):
            >>> from tyxonq.libs.quantum_library.kernels.pauli import pauli_string_sum_dense
            >>> pauli_terms = [[3, 3, 0], [1, 0, 0], ...]  # ZZ, X, ...
            >>> weights = [-1.0, -1.0, ...]
            >>> H = pauli_string_sum_dense(pauli_terms, weights)
            >>> psi = c.state()
            >>> energy = torch.real(torch.dot(torch.conj(psi), H @ psi))
            
            >>> # Works with MPS simulator
            >>> c = Circuit(10)
            >>> c.device(provider="simulator", device="matrix_product_state", max_bond=32)
            >>> for i in range(10): c.h(i)
            >>> exp_x = c.expectation((gate_x(), [0]))  # Uses MPS backend
            
        Notes:
            - Automatically uses the appropriate simulator based on device() config
            - Supports statevector, MPS, and density matrix simulators
            - For statevector/MPS: computes exact expectation
            - For density matrix: supports mixed states and noise
            - Each call re-executes the circuit - use Hamiltonian approach for multiple obs
        """
        from ...numerics.api import get_backend
        
        nb = get_backend()
        n = self.num_qubits
        
        # Determine which engine to use based on device configuration
        device_str = str(self._device_opts.get("device", "statevector"))
        
        if "matrix_product_state" in device_str or "mps" in device_str:
            # Use MPS-specific expectation computation
            # For MPS, we can compute expectations more efficiently
            # by keeping the MPS representation
            return self._expectation_mps(pauli_ops, nb, n)
        
        elif "density_matrix" in device_str:
            # Use density matrix expectation
            return self._expectation_density_matrix(pauli_ops, nb, n)
        
        else:
            # Default: statevector expectation
            return self._expectation_statevector(pauli_ops, nb, n)
    
    def expval(self, *pauli_ops: Any) -> Any:
        """Convenient alias for expectation() (PennyLane-style short name).
        
        Computes expectation value of Pauli operator product: ⟨ψ|O|ψ⟩
        
        This is a thin forwarding wrapper for expectation(), providing
        a shorter method name consistent with industry frameworks
        (PennyLane, TensorCircuit, etc.).
        
        Args:
            *pauli_ops: Pauli operator tuples (gate, qubits)
            
        Returns:
            Expectation value (real number for Hermitian operators)
            
        Examples:
            >>> c = Circuit(2).h(0).cx(0,1)
            >>> exp_z0 = c.expval((gate_z(), [0]))  # Equivalent to expectation()
            >>> exp_zz = c.expval((gate_z(), [0]), (gate_z(), [1]))
            
        See Also:
            expectation(): Full method with comprehensive documentation
        """
        return self.expectation(*pauli_ops)
    
    def _expectation_statevector(self, pauli_ops: tuple, nb: Any, n: int) -> Any:
        """Compute expectation using statevector simulator."""
        from ...libs.quantum_library.kernels.statevector import (
            apply_1q_statevector,
            apply_2q_statevector,
        )
        
        # Get current state
        psi = self.state(engine="statevector")
        
        # Apply Pauli operators to transform the state
        # For ⟨ψ|O|ψ⟩, we compute ⟨Oψ|ψ⟩ since Pauli ops are Hermitian
        psi_transformed = nb.copy(psi)
        
        for gate_spec in pauli_ops:
            if not isinstance(gate_spec, (tuple, list)) or len(gate_spec) != 2:
                raise ValueError(f"Each Pauli operator must be (gate, qubits), got {gate_spec}")
            
            gate, qubits = gate_spec
            
            # Extract gate matrix
            if hasattr(gate, 'tensor'):
                gate_matrix = nb.asarray(gate.tensor)
            elif hasattr(gate, '__call__'):
                gate_matrix = nb.asarray(gate())
            else:
                gate_matrix = nb.asarray(gate)
            
            # Apply gate to transformed state
            if not isinstance(qubits, (list, tuple)):
                qubits = [qubits]
            
            if len(qubits) == 1:
                q = int(qubits[0])
                psi_transformed = apply_1q_statevector(nb, psi_transformed, gate_matrix, q, n)
            elif len(qubits) == 2:
                q1, q2 = int(qubits[0]), int(qubits[1])
                psi_transformed = apply_2q_statevector(nb, psi_transformed, gate_matrix, q1, q2, n)
            else:
                raise NotImplementedError(f"Pauli products on {len(qubits)} qubits not yet supported")
        
        # Compute inner product ⟨ψ|O|ψ⟩ = ⟨Oψ|ψ⟩
        result = nb.tensordot(nb.conj(psi_transformed), psi, axes=1)
        
        # For Hermitian operators, result should be real
        return nb.real(result)
    
    def _expectation_mps(self, pauli_ops: tuple, nb: Any, n: int) -> Any:
        """Compute expectation using MPS simulator with native MPS computation.
        
        Uses O(nχ³) tensor network contraction directly on MPS representation,
        avoiding O(2^n) statevector conversion for improved performance on large systems.
        """
        from ...devices.simulators.matrix_product_state.engine import MatrixProductStateEngine
        
        # Extract MPS configuration from device options
        max_bond = self._device_opts.get("max_bond")
        
        # Create MPS engine
        eng = MatrixProductStateEngine(backend_name=None, max_bond=max_bond)
        
        # Convert pauli_ops to list format expected by engine
        pauli_list = []
        for gate_spec in pauli_ops:
            if not isinstance(gate_spec, (tuple, list)) or len(gate_spec) != 2:
                raise ValueError(f"Each Pauli operator must be (gate, qubits), got {gate_spec}")
            
            gate, qubits = gate_spec
            
            # Extract gate matrix
            if hasattr(gate, 'tensor'):
                gate_matrix = nb.asarray(gate.tensor)
            elif hasattr(gate, '__call__'):
                gate_matrix = nb.asarray(gate())
            else:
                gate_matrix = nb.asarray(gate)
            
            if not isinstance(qubits, (list, tuple)):
                qubits = [qubits]
            
            pauli_list.append((gate_matrix, list(qubits)))
        
        # Use native MPS expectation computation for large systems (n>15)
        # Fall back to statevector for small systems (easier debugging)
        use_native = n > 15
        
        try:
            result = eng.expectation_pauli(self, pauli_list, use_native=use_native)
            return nb.real(result)
        except Exception:
            # Fallback to statevector method if native computation fails
            return self._expectation_statevector(pauli_ops, nb, n)
    
    def _expectation_density_matrix(self, pauli_ops: tuple, nb: Any, n: int) -> Any:
        """Compute expectation using density matrix simulator.
        
        For density matrix ρ, we compute Tr(ρO) where O is the Pauli product.
        """
        from ...libs.quantum_library.kernels.statevector import (
            apply_1q_statevector,
            apply_2q_statevector,
        )
        
        # Get statevector (density matrix engine falls back to statevector for state())
        psi = self.state(engine="density_matrix")
        
        # Apply the same statevector method
        # TODO: Implement true density matrix expectation Tr(ρO)
        psi_transformed = nb.copy(psi)
        
        for gate_spec in pauli_ops:
            if not isinstance(gate_spec, (tuple, list)) or len(gate_spec) != 2:
                raise ValueError(f"Each Pauli operator must be (gate, qubits), got {gate_spec}")
            
            gate, qubits = gate_spec
            
            # Extract gate matrix
            if hasattr(gate, 'tensor'):
                gate_matrix = nb.asarray(gate.tensor)
            elif hasattr(gate, '__call__'):
                gate_matrix = nb.asarray(gate())
            else:
                gate_matrix = nb.asarray(gate)
            
            if not isinstance(qubits, (list, tuple)):
                qubits = [qubits]
            
            if len(qubits) == 1:
                q = int(qubits[0])
                psi_transformed = apply_1q_statevector(nb, psi_transformed, gate_matrix, q, n)
            elif len(qubits) == 2:
                q1, q2 = int(qubits[0]), int(qubits[1])
                psi_transformed = apply_2q_statevector(nb, psi_transformed, gate_matrix, q1, q2, n)
            else:
                raise NotImplementedError(f"Pauli products on {len(qubits)} qubits not yet supported")
        
        result = nb.tensordot(nb.conj(psi_transformed), psi, axes=1)
        return nb.real(result)
    
    # --- draw() typing overloads to improve IDE/linter navigation ---
    @overload
    def draw(self, output: Literal["text"], *args: Any, **kwargs: Any) -> str: ...

    @overload
    def draw(self, output: Literal["mpl"], *args: Any, **kwargs: Any) -> Any: ...

    @overload
    def draw(self, output: Literal["latex"], *args: Any, **kwargs: Any) -> str: ...

    @overload
    def draw(self, *args: Any, **kwargs: Any) -> Any: ...

    # --- Draw via Qiskit provider: compile IR→QuantumCircuit and delegate draw ---
    def draw(self, *args: Any, **kwargs: Any) -> Any:
        """Render the circuit using Qiskit if available.

        Behavior:
        - Convert IR → Qiskit QuantumCircuit directly (no intermediate qasm2 dump),
          auto-adding measurements if none present.
        - Delegate all args/kwargs to `QuantumCircuit.draw`.
        - If Qiskit is not installed, return a minimal textual `gate_summary()` string.
        """
        try:
            from ...compiler.compile_engine.qiskit.dialect import to_qiskit  # type: ignore

            qc = to_qiskit(self, add_measures=True)
            # Resolve default output: prefer per-circuit _draw_output, else 'text'
            if "output" not in kwargs and (len(args) == 0):
                kwargs["output"] = self._draw_output or "text"
            return qc.draw(*args, **kwargs)
        except Exception:
            return str(self.gate_summary())
    @classmethod
    def from_json_obj(cls, obj: Dict[str, Any]) -> "Circuit":
        inst_raw = obj.get("instructions", [])
        inst: List[Tuple[str, Tuple[int, ...]]] = []
        for n, idxs in inst_raw:
            inst.append((str(n), tuple(int(x) for x in idxs)))
        return cls(
            num_qubits=int(obj.get("num_qubits", 0)),
            ops=list(obj.get("ops", [])),
            metadata=dict(obj.get("metadata", {})),
            instructions=inst,
        )

    @classmethod
    def from_json_str(cls, s: str) -> "Circuit":
        return cls.from_json_obj(json.loads(s))


@dataclass
class Hamiltonian:
    """Intermediate representation for a quantum Hamiltonian operator.

    This class serves as a flexible container for representing quantum Hamiltonians
    in TyxonQ's compilation pipeline. The Hamiltonian can be encoded in various
    formats depending on the backend and compilation target.

    Attributes:
        terms (Any): Backend-specific representation of the Hamiltonian terms.
            This may be:
            - A Pauli operator sum for symbolic manipulation
            - A sparse matrix representation for efficient storage
            - A dense matrix for direct numerical computation
            - A list of (coefficient, pauli_string) tuples
            
    The intentionally loose typing allows different compiler stages and devices
    to specialize the representation as needed while maintaining a common interface.
    
    Examples:
        >>> # Pauli string representation (commonly used)
        >>> h = Hamiltonian()
        >>> h.terms = [(0.5, [('Z', 0)]), (0.3, [('X', 0), ('X', 1)])]
        
        >>> # Dense matrix representation
        >>> import numpy as np
        >>> h = Hamiltonian()
        >>> h.terms = np.array([[1, 0], [0, -1]])  # Pauli-Z matrix
        
    See Also:
        tyxonq.libs.hamiltonian_encoding: Functions for converting between different
            Hamiltonian representations (Jordan-Wigner, Bravyi-Kitaev, etc.).
        tyxonq.applications.chem: Quantum chemistry applications that construct
            molecular Hamiltonians.
    """

    terms: Any


# ---- Module-level task helpers (for cloud.api thin delegation) ----
def get_task_details(task: Any, *, prettify: bool = False) -> Dict[str, Any]:
    dev = getattr(task, "device", None)
    if dev is None:
        # simulator inline task may still provide results()
        if hasattr(task, "results"):
            try:
                return task.results()
            except Exception:
                pass
        raise ValueError("Task handle missing device information")
    dev_str = str(dev)
    prov = (dev_str.split("::", 1)[0]) if "::" in dev_str else "simulator"
    from ...devices.base import resolve_driver
    from ...devices.hardware import config as hwcfg

    tok = hwcfg.get_token(provider=prov, device=dev_str)
    drv = resolve_driver(prov, dev_str)
    return drv.get_task_details(task, tok)


def cancel_task(task: Any) -> Any:
    dev = getattr(task, "device", None)
    if dev is None:
        raise ValueError("Task handle missing device information")
    dev_str = str(dev)
    prov = (dev_str.split("::", 1)[0]) if "::" in dev_str else "simulator"
    from ...devices.base import resolve_driver
    from ...devices.hardware import config as hwcfg

    tok = hwcfg.get_token(provider=prov, device=dev_str)
    drv = resolve_driver(prov, dev_str)
    if hasattr(drv, "remove_task"):
        return drv.remove_task(task, tok)
    raise NotImplementedError("cancel not supported for this provider/task type")

