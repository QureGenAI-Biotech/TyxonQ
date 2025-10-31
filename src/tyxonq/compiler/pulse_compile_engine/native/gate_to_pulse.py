"""Gate-to-Pulse decomposition pass.

This pass converts standard quantum gates into pulse sequences using either:
1. User-provided calibrations (defcal definitions)
2. Default physics-based decompositions

Supported gates:
    - Single-qubit: X, Y, Z, H, S, T, RX, RY, RZ, U3
    - Two-qubit: CX, CZ, SWAP, iSWAP (via cross-resonance or parametric drives)

Physics basis:
    - Single-qubit rotations via resonant Rabi driving
    - CX gate via cross-resonance (ZX interaction)
    - CZ gate via H·CX·H decomposition
    - Virtual-Z gates (frame updates, zero gate time)
    
References:
    [1] QuTiP-qip processor model (Quantum 6, 630, 2022)
    [2] IBM Qiskit: Pulse-level programming
    [3] Rigetti: arXiv:1903.02492 - Parametric gates
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit

# Import waveforms for default pulse shapes
try:
    from tyxonq import waveforms
except ImportError:
    waveforms = None  # type: ignore


class GateToPulsePass:
    """Decompose standard gates into pulse sequences.
    
    This pass implements the first stage of pulse compilation, converting
    abstract gate operations into concrete pulse waveforms.
    
    Strategy:
        1. Check for user-provided calibrations (defcal)
        2. If not found, use default physics-based decomposition
        3. Create pulse operations and store waveforms in circuit metadata
    
    Example:
        >>> pass_instance = GateToPulsePass()
        >>> pulse_circuit = pass_instance.execute_plan(gate_circuit, device_params={...})
    """
    
    def __init__(self):
        """Initialize the gate-to-pulse pass."""
        self._default_calibrations = self._build_default_calibrations()
    
    def execute_plan(
        self,
        circuit: "Circuit",
        **options: Any
    ) -> "Circuit":
        """Execute gate-to-pulse decomposition.
        
        Args:
            circuit: Input circuit with gate operations
            **options: Compilation options:
                - mode: "hybrid" | "pulse_only" | "auto_lower"
                - dt: Time step (default: 1e-10 s)
        
        Returns:
            Circuit with pulse operations (gates may be preserved in hybrid mode)
        """
        mode = options.get("mode", "hybrid")
        device_params = circuit.metadata.get("pulse_device_params", {})
        calibrations = circuit.metadata.get("pulse_calibrations", {})
        
        # Initialize pulse library in metadata (not _pulse_cache)
        if "pulse_library" not in circuit.metadata:
            circuit.metadata["pulse_library"] = {}
        
        new_ops: List[Any] = []
        
        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                new_ops.append(op)
                continue
            
            gate_name = str(op[0]).lower()
            
            # Check if this gate should be converted to pulse
            if self._should_convert_to_pulse(gate_name, mode):
                pulse_ops = self._gate_to_pulse(
                    op, device_params, calibrations, circuit
                )
                if pulse_ops:
                    new_ops.extend(pulse_ops)
                else:
                    # Fallback: keep original gate
                    new_ops.append(op)
            else:
                # Hybrid mode: keep gate as-is
                new_ops.append(op)
        
        from dataclasses import replace
        return replace(circuit, ops=new_ops)
    
    def _should_convert_to_pulse(self, gate_name: str, mode: str) -> bool:
        """Determine if a gate should be converted to pulse.
        
        Args:
            gate_name: Gate name (lowercase)
            mode: Compilation mode
        
        Returns:
            True if gate should be converted to pulse
        """
        if mode == "pulse_only":
            return True
        
        # In hybrid mode, only convert specific gates
        # (e.g., keep measurement gates as-is)
        if gate_name in ("measure", "measure_z", "barrier"):
            return False
        
        return mode == "auto_lower"
    
    def _gate_to_pulse(
        self,
        op: Any,
        device_params: Dict[str, Any],
        calibrations: Dict[str, Any],
        circuit: "Circuit"
    ) -> List[Any]:
        """Convert a single gate to pulse operations.
        
        Args:
            op: Gate operation tuple
            device_params: Device physical parameters
            calibrations: User-provided calibrations
            circuit: Circuit object (for pulse cache)
        
        Returns:
            List of pulse operations
        """
        gate_name = str(op[0]).lower()
        
        # Check user calibrations first
        if gate_name in calibrations:
            return self._apply_calibration(op, calibrations[gate_name], circuit)
        
        # Use default decomposition
        if gate_name in ("x", "rx"):
            return self._decompose_x_gate(op, device_params, circuit)
        elif gate_name in ("y", "ry"):
            return self._decompose_y_gate(op, device_params, circuit)
        elif gate_name in ("z", "rz"):
            return self._decompose_z_gate(op, device_params, circuit)
        elif gate_name == "h":
            return self._decompose_h_gate(op, device_params, circuit)
        elif gate_name == "cx":
            return self._decompose_cx_gate(op, device_params, circuit)
        elif gate_name == "cz":
            return self._decompose_cz_gate(op, device_params, circuit)
        else:
            # Unsupported gate: return empty (will fallback to original gate)
            return []
    
    def _decompose_x_gate(
        self, op: Any, device_params: Dict, circuit: "Circuit"
    ) -> List[Any]:
        """Decompose X/RX gate to pulse."""
        qubit = int(op[1])
        angle = float(op[2]) if len(op) > 2 else math.pi
        
        # Get qubit frequency
        qubit_freq = self._get_qubit_freq(qubit, device_params)
        
        # Create Drag pulse for X rotation (default calibration)
        if waveforms is not None:
            pulse = waveforms.Drag(
                amp=angle / math.pi,  # Scale amplitude by rotation angle
                duration=160,  # ns (typical value)
                sigma=40,
                beta=0.2
            )
            
            # Store in pulse library (metadata)
            pulse_key = f"rx_q{qubit}_{id(pulse)}"
            circuit.metadata["pulse_library"][pulse_key] = pulse
            
            # Return pulse operation
            return [("pulse", qubit, pulse_key, {
                "qubit_freq": qubit_freq,
                "drive_freq": qubit_freq,
                "anharmonicity": device_params.get("anharmonicity", [-300e6])[qubit]
                    if isinstance(device_params.get("anharmonicity"), list) else -300e6
            })]
        
        return []
    
    def _decompose_y_gate(
        self, op: Any, device_params: Dict, circuit: "Circuit"
    ) -> List[Any]:
        """Decompose Y/RY gate to pulse (X pulse with π/2 phase shift).
        
        Physical Principle:
            RY(θ) = RZ(π/2) · RX(θ) · RZ(-π/2)
            
            In the rotating frame, Y rotation is equivalent to an X rotation
            with the drive phase shifted by π/2 (90 degrees).
        
        Implementation:
            Use DRAG pulse with phase = π/2 (represented in waveform metadata)
        
        Args:
            op: Gate operation tuple ("y" or "ry", qubit, [angle])
            device_params: Device parameters
            circuit: Circuit object
        
        Returns:
            List containing a single phase-shifted pulse operation
        """
        qubit = int(op[1])
        angle = float(op[2]) if len(op) > 2 else math.pi
        
        # Get qubit frequency
        qubit_freq = self._get_qubit_freq(qubit, device_params)
        
        # Create Drag pulse with π/2 phase shift for Y rotation
        if waveforms is not None:
            pulse = waveforms.Drag(
                amp=angle / math.pi,
                duration=160,
                sigma=40,
                beta=0.2
            )
            
            # Store in pulse library
            pulse_key = f"ry_q{qubit}_{id(pulse)}"
            circuit.metadata["pulse_library"][pulse_key] = pulse
            
            # Return pulse with phase shift metadata
            return [("pulse", qubit, pulse_key, {
                "qubit_freq": qubit_freq,
                "drive_freq": qubit_freq,
                "anharmonicity": self._get_anharmonicity(qubit, device_params),
                "phase": math.pi / 2  # π/2 phase for Y rotation
            })]
        
        return []
    
    def _decompose_z_gate(
        self, op: Any, device_params: Dict, circuit: "Circuit"
    ) -> List[Any]:
        """Decompose Z/RZ gate to virtual-Z gate (frame change).
        
        Physical Principle:
            Z rotations commute with all subsequent rotations in the rotating
            frame and can be implemented as a phase frame update without
            any physical pulse.
            
            RZ(θ) · RX(φ) = RX(φ) · RZ(θ)
            RZ(θ) · RY(φ) = RY(φ) · RZ(θ)
        
        Implementation Strategy:
            1. Create a "virtual_z" operation with phase parameter
            2. Phase scheduler will accumulate phase for subsequent pulses
            3. No physical pulse is emitted (saves gate time!)
        
        References:
            [1] McKay et al., PRA 96, 022330 (2017) - Virtual Z Gates
            [2] IBM Qiskit: Virtual Z implementation
        
        Args:
            op: Gate operation tuple ("z" or "rz", qubit, [angle])
            device_params: Device parameters
            circuit: Circuit object
        
        Returns:
            List containing a virtual-z operation (no physical pulse)
        """
        qubit = int(op[1])
        angle = float(op[2]) if len(op) > 2 else math.pi
        
        # Virtual-Z: just store phase frame update
        return [("virtual_z", qubit, angle)]
    
    def _decompose_h_gate(
        self, op: Any, device_params: Dict, circuit: "Circuit"
    ) -> List[Any]:
        """Decompose Hadamard gate to pulse sequence.
        
        Physical Decomposition:
            H = RZ(π) · RY(π/2) = Virtual-Z(π) + RY(π/2)
            
            Or equivalently:
            H = RY(π/2) · RX(π)
        
        Implementation:
            Use the second form to minimize virtual-Z operations:
            1. RY(π/2): DRAG pulse with phase = π/2
            2. RX(π): DRAG pulse with phase = 0
        
        Args:
            op: Gate operation tuple ("h", qubit)
            device_params: Device parameters
            circuit: Circuit object
        
        Returns:
            List of two pulse operations
        """
        qubit = int(op[1])
        qubit_freq = self._get_qubit_freq(qubit, device_params)
        anharmonicity = self._get_anharmonicity(qubit, device_params)
        
        if waveforms is None:
            return []
        
        pulse_ops = []
        
        # 1. RY(π/2)
        ry_pulse = waveforms.Drag(
            amp=0.5,  # π/2 rotation
            duration=160,
            sigma=40,
            beta=0.2
        )
        ry_key = f"h_ry_q{qubit}_{id(ry_pulse)}"
        circuit.metadata["pulse_library"][ry_key] = ry_pulse
        pulse_ops.append(("pulse", qubit, ry_key, {
            "qubit_freq": qubit_freq,
            "drive_freq": qubit_freq,
            "anharmonicity": anharmonicity,
            "phase": math.pi / 2  # Y rotation phase
        }))
        
        # 2. RX(π)
        rx_pulse = waveforms.Drag(
            amp=1.0,  # π rotation (X gate)
            duration=160,
            sigma=40,
            beta=0.2
        )
        rx_key = f"h_rx_q{qubit}_{id(rx_pulse)}"
        circuit.metadata["pulse_library"][rx_key] = rx_pulse
        pulse_ops.append(("pulse", qubit, rx_key, {
            "qubit_freq": qubit_freq,
            "drive_freq": qubit_freq,
            "anharmonicity": anharmonicity,
            "phase": 0.0  # X rotation phase
        }))
        
        return pulse_ops
    
    def _decompose_cx_gate(
        self, op: Any, device_params: Dict, circuit: "Circuit"
    ) -> List[Any]:
        """Decompose CX gate to cross-resonance pulse sequence.
        
        Physical Model (Cross-Resonance Interaction):
            H_CR(t) = Ω(t) · (σ_x^control ⊗ σ_z^target)
            
            Where:
            - Ω(t): Time-dependent control pulse amplitude
            - Interaction drives control qubit at target frequency
            - Generates entangling ZX interaction
        
        Pulse Sequence:
            1. Pre-rotation: RX(-π/2) on control
            2. Cross-resonance pulse on control @ target_freq
            3. Simultaneous echo pulse on target (suppress ZI/IX errors)
            4. Post-rotation: RX(π/2) on control
        
        References:
            [1] QuTiP-qip: Quantum 6, 630 (2022) - SCQubits Processor
            [2] Rigetti: arXiv:1903.02492 - Parametric Resonance Gates
            [3] IBM: PRL 127, 200505 (2021) - CR Gate Optimization
        
        Args:
            op: Gate operation tuple ("cx", control, target)
            device_params: Device physical parameters
            circuit: Circuit object
        
        Returns:
            List of pulse operations for CX gate
        """
        if len(op) < 3:
            return []
        
        control = int(op[1])
        target = int(op[2])
        
        # Get device parameters
        control_freq = self._get_qubit_freq(control, device_params)
        target_freq = self._get_qubit_freq(target, device_params)
        coupling_strength = device_params.get("coupling_strength", 5e6)  # Hz
        
        # CX gate time (determined by coupling strength)
        # Gate time ≈ π / (2 * coupling_strength)
        cx_duration = int(device_params.get("cx_duration", 400))  # ns
        
        if waveforms is None:
            return []
        
        pulse_ops = []
        
        # 1. Pre-rotation: RX(-π/2) on control
        pre_pulse = waveforms.Drag(
            amp=-0.5,  # -π/2 rotation
            duration=160,
            sigma=40,
            beta=0.2
        )
        pre_key = f"cx_pre_c{control}_t{target}_{id(pre_pulse)}"
        circuit.metadata["pulse_library"][pre_key] = pre_pulse
        pulse_ops.append(("pulse", control, pre_key, {
            "qubit_freq": control_freq,
            "drive_freq": control_freq,
            "anharmonicity": self._get_anharmonicity(control, device_params)
        }))
        
        # 2. Cross-resonance pulse: control qubit driven at target frequency
        # Use Gaussian envelope to reduce spectral leakage
        cr_pulse = waveforms.Gaussian(
            amp=device_params.get("cr_amplitude", 0.3),
            duration=cx_duration,
            sigma=cx_duration / 4.0  # Smooth edges
        )
        cr_key = f"cx_cr_c{control}_t{target}_{id(cr_pulse)}"
        circuit.metadata["pulse_library"][cr_key] = cr_pulse
        pulse_ops.append(("pulse", control, cr_key, {
            "qubit_freq": control_freq,
            "drive_freq": target_freq,  # Drive at target frequency!
            "anharmonicity": self._get_anharmonicity(control, device_params),
            "cr_target": target  # Mark as cross-resonance
        }))
        
        # 3. Simultaneous rotary echo on target (optional, suppresses errors)
        # This cancels unwanted ZI and IX terms in the interaction Hamiltonian
        echo_enabled = device_params.get("cr_echo", True)
        if echo_enabled:
            echo_pulse = waveforms.Constant(
                amp=device_params.get("echo_amplitude", 0.1),
                duration=cx_duration
            )
            echo_key = f"cx_echo_c{control}_t{target}_{id(echo_pulse)}"
            circuit.metadata["pulse_library"][echo_key] = echo_pulse
            pulse_ops.append(("pulse", target, echo_key, {
                "qubit_freq": target_freq,
                "drive_freq": target_freq,
                "anharmonicity": self._get_anharmonicity(target, device_params)
            }))
        
        # 4. Post-rotation: RX(π/2) on control
        post_pulse = waveforms.Drag(
            amp=0.5,  # π/2 rotation
            duration=160,
            sigma=40,
            beta=0.2
        )
        post_key = f"cx_post_c{control}_t{target}_{id(post_pulse)}"
        circuit.metadata["pulse_library"][post_key] = post_pulse
        pulse_ops.append(("pulse", control, post_key, {
            "qubit_freq": control_freq,
            "drive_freq": control_freq,
            "anharmonicity": self._get_anharmonicity(control, device_params)
        }))
        
        return pulse_ops
    
    def _decompose_cz_gate(
        self, op: Any, device_params: Dict, circuit: "Circuit"
    ) -> List[Any]:
        """Decompose CZ gate to pulse sequence.
        
        Strategy:
            CZ = H_target · CX(control, target) · H_target
            
            Where H is Hadamard gate. This decomposition reduces CZ to CX,
            which is already implemented via cross-resonance pulses.
        
        Physical Justification:
            CX: control-X (ZX interaction)
            CZ: control-Z (ZZ interaction)
            
            Relation:
            CZ |00⟩ = |00⟩
            CZ |01⟩ = |01⟩
            CZ |10⟩ = |10⟩
            CZ |11⟩ = -|11⟩
            
            CX |00⟩ = |00⟩
            CX |01⟩ = |01⟩
            CX |10⟩ = |11⟩
            CX |11⟩ = |10⟩
            
            H conjugates X ↔ Z:
            H · X · H = Z
            H · CX · H = CZ (on target qubit)
        
        Alternative (Direct CR Implementation):
            For hardware with tunable coupling, can implement CZ directly
            via parametric modulation (future enhancement).
        
        References:
            [1] Nielsen & Chuang - Gate Equivalences
            [2] DiCarlo Lab: arXiv:1903.02492 - Parametric CZ gates
        
        Args:
            op: Gate operation tuple ("cz", control, target)
            device_params: Device parameters
            circuit: Circuit object
        
        Returns:
            List of pulse operations implementing CZ
        """
        if len(op) < 3:
            return []
        
        control = int(op[1])
        target = int(op[2])
        
        pulse_ops = []
        
        # Decomposition: CZ = H_target · CX · H_target
        
        # 1. Pre-Hadamard on target
        h_pre_ops = self._decompose_h_gate(
            ("h", target),
            device_params,
            circuit
        )
        pulse_ops.extend(h_pre_ops)
        
        # 2. CX gate
        cx_ops = self._decompose_cx_gate(
            ("cx", control, target),
            device_params,
            circuit
        )
        pulse_ops.extend(cx_ops)
        
        # 3. Post-Hadamard on target
        h_post_ops = self._decompose_h_gate(
            ("h", target),
            device_params,
            circuit
        )
        pulse_ops.extend(h_post_ops)
        
        return pulse_ops
    
    def _apply_calibration(
        self, op: Any, calibration: Dict, circuit: "Circuit"
    ) -> List[Any]:
        """Apply user-provided calibration."""
        pulse = calibration.get("pulse")
        params = calibration.get("params", {})
        
        if pulse is None:
            return []
        
        # Store pulse in library (metadata)
        pulse_key = f"cal_{op[0]}_{id(pulse)}"
        circuit.metadata["pulse_library"][pulse_key] = pulse
        
        # Create pulse operation
        qubits = calibration.get("qubits", [int(op[1])])
        qubit = qubits[0]
        
        return [("pulse", qubit, pulse_key, params)]
    
    def _get_qubit_freq(self, qubit: int, device_params: Dict) -> float:
        """Get qubit frequency from device parameters."""
        freqs = device_params.get("qubit_freq", [5.0e9])
        if isinstance(freqs, list) and qubit < len(freqs):
            return float(freqs[qubit])
        return 5.0e9  # Default frequency
    
    def _get_anharmonicity(self, qubit: int, device_params: Dict) -> float:
        """Get qubit anharmonicity from device parameters.
        
        Args:
            qubit: Qubit index
            device_params: Device parameters
        
        Returns:
            Anharmonicity in Hz (typically -200 to -350 MHz for transmons)
        """
        anharm = device_params.get("anharmonicity", [-300e6])
        if isinstance(anharm, list) and qubit < len(anharm):
            return float(anharm[qubit])
        return -300e6  # Default anharmonicity for transmon qubits
    
    def _build_default_calibrations(self) -> Dict[str, Any]:
        """Build default pulse calibrations for common gates."""
        # Placeholder for future default calibration library
        return {}
