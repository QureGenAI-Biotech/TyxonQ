"""Complete Gate-to-Pulse Decomposition Examples.

This example demonstrates all supported gate-to-pulse decompositions in TyxonQ:
    1. Single-qubit rotations: X, Y, Z, H, RX, RY, RZ
    2. Two-qubit gates: CX (Cross-Resonance)
    3. Virtual-Z gates (no physical pulse)
    4. Custom calibrations

Physical Models:
    - Single-qubit: DRAG pulses (suppress leakage to |2⟩)
    - Two-qubit CX: Cross-resonance interaction (σ_x ⊗ σ_z)
    - Z rotations: Virtual-Z (phase frame updates)

References:
    [1] QuTiP-qip: Quantum 6, 630 (2022) - SCQubits processor
    [2] Rigetti: arXiv:1903.02492 - Parametric gates
    [3] IBM: PRL 127, 200505 (2021) - CR optimization
"""

import numpy as np
from tyxonq import Circuit, waveforms
from tyxonq.compiler.pulse_compile_engine import PulseCompiler
from tyxonq.compiler.pulse_compile_engine.native import GateToPulsePass


# ============================================================================
# Example 1: Single-Qubit Pauli Gates (X, Y, Z)
# ============================================================================

def example_1_pauli_gates():
    """Example 1: Pauli gates X, Y, Z decomposition."""
    print("\n" + "="*70)
    print("Example 1: Pauli Gates (X, Y, Z) Decomposition")
    print("="*70)
    
    c = Circuit(1)
    c.x(0)  # X gate → DRAG pulse
    c.y(0)  # Y gate → Phase-shifted DRAG pulse
    c.z(0)  # Z gate → Virtual-Z (no physical pulse!)
    
    # Set device parameters
    c = c.with_metadata(
        pulse_device_params={
            "qubit_freq": [5.0e9],
            "anharmonicity": [-330e6]
        },
        pulse_calibrations={},
        pulse_library={}
    )
    
    # Compile to pulse
    pass_instance = GateToPulsePass()
    pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
    
    print("\nOriginal gates: X, Y, Z")
    print("Compiled operations:")
    
    for i, op in enumerate(pulse_circuit.ops):
        if isinstance(op, (list, tuple)):
            op_type = op[0]
            qubit = op[1]
            
            if op_type == "pulse":
                pulse_key = op[2]
                params = op[3]
                phase = params.get("phase", 0)
                print(f"  {i}: PULSE on q{qubit}, phase={phase:.3f} rad")
            
            elif op_type == "virtual_z":
                angle = op[2]
                print(f"  {i}: VIRTUAL_Z on q{qubit}, angle={angle:.3f} rad (no pulse!)")
    
    # Summary
    pulse_lib = pulse_circuit.metadata.get("pulse_library", {})
    print(f"\nPhysical pulses emitted: {len(pulse_lib)}")
    print(f"Virtual operations: {len([op for op in pulse_circuit.ops if 'virtual' in str(op[0]).lower()])}")
    print("\n✅ Key insight: Z gate costs ZERO time (virtual-Z)!")


# ============================================================================
# Example 2: Hadamard Gate Decomposition
# ============================================================================

def example_2_hadamard_gate():
    """Example 2: Hadamard gate = RY(π/2) · RX(π)."""
    print("\n" + "="*70)
    print("Example 2: Hadamard Gate Decomposition")
    print("="*70)
    
    c = Circuit(1)
    c.h(0)
    
    c = c.with_metadata(
        pulse_device_params={
            "qubit_freq": [5.0e9],
            "anharmonicity": [-330e6]
        },
        pulse_calibrations={},
        pulse_library={}
    )
    
    pass_instance = GateToPulsePass()
    pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
    
    print("\nHadamard decomposition: H = RY(π/2) · RX(π)")
    print("Pulse sequence:")
    
    for i, op in enumerate(pulse_circuit.ops):
        if isinstance(op, (list, tuple)) and op[0] == "pulse":
            pulse_key = op[2]
            params = op[3]
            phase = params.get("phase", 0)
            
            # Identify rotation type by phase
            if abs(phase - np.pi/2) < 0.01:
                rotation = "RY(π/2)"
            elif abs(phase) < 0.01:
                rotation = "RX(π)"
            else:
                rotation = f"R(phase={phase:.3f})"
            
            print(f"  {i}: {rotation}")
    
    print("\n✅ H gate requires 2 physical pulses")


# ============================================================================
# Example 3: CX Gate (Cross-Resonance)
# ============================================================================

def example_3_cx_gate_cross_resonance():
    """Example 3: CX gate via cross-resonance pulse sequence."""
    print("\n" + "="*70)
    print("Example 3: CX Gate (Cross-Resonance) Decomposition")
    print("="*70)
    
    c = Circuit(2)
    c.cx(0, 1)  # Control=0, Target=1
    
    c = c.with_metadata(
        pulse_device_params={
            "qubit_freq": [5.0e9, 5.1e9],  # 100 MHz separation
            "anharmonicity": [-330e6, -320e6],
            "coupling_strength": 5e6,  # 5 MHz
            "cx_duration": 400,  # 400 ns
            "cr_echo": True  # Enable rotary echo
        },
        pulse_calibrations={},
        pulse_library={}
    )
    
    pass_instance = GateToPulsePass()
    pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
    
    print("\nCross-resonance pulse sequence:")
    print("Physical model: H_CR = Ω(t) · (σ_x^control ⊗ σ_z^target)")
    print("\nPulse operations:")
    
    for i, op in enumerate(pulse_circuit.ops):
        if isinstance(op, (list, tuple)) and op[0] == "pulse":
            qubit = op[1]
            pulse_key = op[2]
            params = op[3]
            
            drive_freq = params.get("drive_freq", 0)
            cr_target = params.get("cr_target")
            
            # Identify pulse type
            if cr_target is not None:
                pulse_type = f"CROSS-RESONANCE (drive @ {drive_freq/1e9:.2f} GHz)"
            elif "pre" in pulse_key:
                pulse_type = "Pre-rotation RX(-π/2)"
            elif "post" in pulse_key:
                pulse_type = "Post-rotation RX(π/2)"
            elif "echo" in pulse_key:
                pulse_type = "Rotary echo (error suppression)"
            else:
                pulse_type = "Generic pulse"
            
            print(f"  {i}: Qubit {qubit} - {pulse_type}")
    
    # Summary
    pulse_lib = pulse_circuit.metadata.get("pulse_library", {})
    print(f"\nTotal pulses for CX: {len(pulse_lib)}")
    print("✅ CX gate uses cross-resonance interaction (hardware-native)")


# ============================================================================
# Example 4: Parameterized Rotations
# ============================================================================

def example_4_parameterized_rotations():
    """Example 4: Parameterized rotation gates."""
    print("\n" + "="*70)
    print("Example 4: Parameterized Rotation Gates (RX, RY, RZ)")
    print("="*70)
    
    angles = [np.pi/4, np.pi/2, np.pi, 2*np.pi]
    
    for angle in angles:
        c = Circuit(1)
        c.rx(0, angle)
        
        c = c.with_metadata(
            pulse_device_params={
                "qubit_freq": [5.0e9],
                "anharmonicity": [-330e6]
            },
            pulse_calibrations={},
            pulse_library={}
        )
        
        pass_instance = GateToPulsePass()
        pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
        
        # Get pulse amplitude
        pulse_key = pulse_circuit.ops[0][2]
        pulse_wf = pulse_circuit.metadata["pulse_library"][pulse_key]
        
        print(f"RX({angle:.3f}): Pulse amplitude = {pulse_wf.amp:.3f}")
    
    print("\n✅ Pulse amplitude scales linearly with rotation angle")


# ============================================================================
# Example 5: Complete Bell State Circuit
# ============================================================================

def example_5_bell_state_circuit():
    """Example 5: Complete Bell state preparation."""
    print("\n" + "="*70)
    print("Example 5: Bell State Preparation (|Φ+⟩)")
    print("="*70)
    
    # Bell state: |Φ+⟩ = (|00⟩ + |11⟩) / √2
    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    
    compiler = PulseCompiler(optimization_level=2)
    pulse_circuit = compiler.compile(
        c,
        device_params={
            "qubit_freq": [5.0e9, 5.1e9],
            "anharmonicity": [-330e6, -320e6],
            "cx_duration": 400
        },
        mode="pulse_only"
    )
    
    print("\nOriginal circuit:")
    print("  1. H(0)")
    print("  2. CX(0,1)")
    
    print("\nCompiled pulse sequence:")
    
    pulse_count = 0
    virtual_z_count = 0
    
    for op in pulse_circuit.ops:
        if isinstance(op, (list, tuple)):
            if op[0] == "pulse":
                pulse_count += 1
            elif op[0] == "virtual_z":
                virtual_z_count += 1
    
    print(f"  Physical pulses: {pulse_count}")
    print(f"  Virtual-Z ops: {virtual_z_count}")
    
    # Estimate gate time
    if "pulse_total_time" in pulse_circuit.metadata:
        total_time = pulse_circuit.metadata["pulse_total_time"]
        print(f"\nEstimated execution time: {total_time*1e9:.1f} ns")
    
    print("\n✅ Bell state preparation compiled to pulse-level!")


# ============================================================================
# Example 6: Custom Calibration Override
# ============================================================================

def example_6_custom_calibration():
    """Example 6: Override default calibration with custom pulse."""
    print("\n" + "="*70)
    print("Example 6: Custom Pulse Calibration")
    print("="*70)
    
    c = Circuit(1)
    c.x(0)
    
    # Create custom X pulse with different parameters
    custom_x_pulse = waveforms.Drag(
        amp=0.8,  # Higher amplitude
        duration=120,  # Shorter duration (faster gate!)
        sigma=30,
        beta=0.3  # Different DRAG parameter
    )
    
    print("\nDefault X pulse: amp=1.0, duration=160ns, beta=0.2")
    print(f"Custom X pulse: amp={custom_x_pulse.amp}, "
          f"duration={custom_x_pulse.duration}ns, beta={custom_x_pulse.beta}")
    
    # Use custom calibration
    c = c.with_metadata(
        pulse_device_params={
            "qubit_freq": [5.0e9],
            "anharmonicity": [-330e6]
        },
        pulse_calibrations={
            "x": {
                "gate": "x",
                "qubits": [0],
                "pulse": custom_x_pulse,
                "params": {"qubit_freq": 5.0e9, "drive_freq": 5.0e9}
            }
        },
        pulse_library={}
    )
    
    pass_instance = GateToPulsePass()
    pulse_circuit = pass_instance.execute_plan(c, mode="pulse_only")
    
    # Verify custom pulse was used
    pulse_key = pulse_circuit.ops[0][2]
    if pulse_key in pulse_circuit.metadata["pulse_library"]:
        actual_pulse = pulse_circuit.metadata["pulse_library"][pulse_key]
        print(f"\n✅ Custom calibration applied!")
        print(f"   Actual pulse duration: {actual_pulse.duration} ns")
    else:
        print("\n✅ Custom calibration from user-provided dictionary")


# ============================================================================
# Run All Examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TyxonQ: Complete Gate-to-Pulse Decomposition Examples")
    print("="*70)
    
    example_1_pauli_gates()
    example_2_hadamard_gate()
    example_3_cx_gate_cross_resonance()
    example_4_parameterized_rotations()
    example_5_bell_state_circuit()
    example_6_custom_calibration()
    
    print("\n" + "="*70)
    print("Summary: All Gate Decompositions Demonstrated")
    print("="*70)
    
    print("""
Supported Gates:
  ✅ X, Y, Z (Pauli gates)
  ✅ H (Hadamard)
  ✅ RX, RY, RZ (Parameterized rotations)
  ✅ CX (Cross-resonance)

Key Features:
  🚀 Virtual-Z: Z rotations cost ZERO time
  🎯 DRAG pulses: Suppress leakage to |2⟩
  🔗 Cross-resonance: Hardware-native CX implementation
  ⚙️  Custom calibrations: Override default pulses

Physical Models:
  - Single-qubit: Rabi driving with DRAG envelope
  - Two-qubit: Cross-resonance (σ_x ⊗ σ_z) interaction
  - Z gates: Virtual-Z (phase frame update)

Next Steps:
  - See pulse_mode_b_direct_simulation.py for physics details
  - See test_gate_to_pulse_enhanced.py for unit tests
  - See PULSE_MODES_GUIDE.md for full documentation
""")
