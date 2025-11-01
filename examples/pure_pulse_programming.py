"""Pure Pulse Programming with TyxonQ.

This example demonstrates TyxonQ's dual-mode pulse programming architecture:
    - Mode A (Chain): Gate Circuit → Pulse Compiler → Execution
    - Mode B (Direct): Pure Pulse Program → Direct Execution (THIS FILE)

Mode B (Pure Pulse) enables:
    1. Direct pulse-level control without gate abstractions
    2. Hardware-native programming for pulse-optimized algorithms
    3. Fine-grained pulse sequence design for quantum control

Use Cases:
    - Pulse shape optimization experiments
    - Quantum optimal control
    - Hardware calibration routines
    - Research on pulse-level quantum dynamics

Reference Memory: de2051a3 (Pulse双模式示例规范)
"""

import numpy as np
from tyxonq import waveforms
from tyxonq.core.ir.pulse import PulseProgram


# ==============================================================================
# Example 1: Single-Qubit Pulse Sequence
# ==============================================================================

def example_1_single_qubit_pulse():
    """Example 1: Simple single-qubit pulse program."""
    print("\n" + "="*70)
    print("Example 1: Single-Qubit Pure Pulse Programming")
    print("="*70)
    
    # Create pure pulse program (no gates!)
    prog = PulseProgram(num_qubits=1)
    
    # Set device parameters
    prog.set_device_params(
        qubit_freq=[5.0e9],
        anharmonicity=[-330e6],
        T1=[80e-6],
        T2=[120e-6]
    )
    
    # Define pulse sequence
    # 1. DRAG pulse for X-rotation
    x_pulse = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.2)
    prog.add_pulse(0, x_pulse, qubit_freq=5.0e9, drive_freq=5.0e9)
    
    # 2. Another DRAG pulse for Y-rotation (phase-shifted)
    y_pulse = waveforms.Drag(amp=0.5, duration=160, sigma=40, beta=0.2)
    prog.add_pulse(0, y_pulse, qubit_freq=5.0e9, drive_freq=5.0e9, phase=np.pi/2)
    
    print("\nPulse Program:")
    print(f"  Number of qubits: {prog.num_qubits}")
    print(f"  Number of pulses: {len(prog.pulse_ops)}")
    print(f"  Device params: qubit_freq={prog.device_params['qubit_freq']}")
    
    # Execute directly (Mode B: bypasses gate→pulse compilation)
    print("\nExecuting pure pulse program...")
    final_state = prog.run(backend="numpy")
    
    print(f"\nFinal state (first 4 amplitudes):")
    print(f"  {final_state[:min(len(final_state), 4)]}")
    print(f"  Norm: {np.linalg.norm(final_state):.6f}")
    
    print("\n✅ Pure pulse execution successful!")


# ==============================================================================
# Example 2: Two-Qubit Cross-Resonance Pulse
# ==============================================================================

def example_2_two_qubit_cross_resonance():
    """Example 2: Two-qubit cross-resonance pulse sequence."""
    print("\n" + "="*70)
    print("Example 2: Two-Qubit Cross-Resonance Pulse")
    print("="*70)
    
    prog = PulseProgram(num_qubits=2)
    
    # Device parameters
    prog.set_device_params(
        qubit_freq=[5.0e9, 5.1e9],
        anharmonicity=[-330e6, -320e6]
    )
    
    # CX pulse sequence (Cross-Resonance implementation)
    print("\nBuilding CX pulse sequence:")
    
    # 1. Pre-rotation on control
    pre_pulse = waveforms.Drag(amp=-0.5, duration=160, sigma=40, beta=0.2)
    prog.add_pulse(0, pre_pulse, qubit_freq=5.0e9, drive_freq=5.0e9)
    print("  1. Pre-rotation RX(-π/2) on control qubit")
    
    # 2. Cross-resonance pulse (control driven @ target frequency)
    cr_pulse = waveforms.Gaussian(amp=0.3, duration=400, sigma=100)
    prog.add_pulse(0, cr_pulse, 
                   qubit_freq=5.0e9, 
                   drive_freq=5.1e9)  # Drive at target frequency!
    print("  2. Cross-resonance pulse @ target frequency")
    
    # 3. Echo pulse on target
    echo_pulse = waveforms.Constant(amp=0.1, duration=400)
    prog.add_pulse(1, echo_pulse, qubit_freq=5.1e9, drive_freq=5.1e9)
    print("  3. Rotary echo on target qubit")
    
    # 4. Post-rotation on control
    post_pulse = waveforms.Drag(amp=0.5, duration=160, sigma=40, beta=0.2)
    prog.add_pulse(0, post_pulse, qubit_freq=5.0e9, drive_freq=5.0e9)
    print("  4. Post-rotation RX(π/2) on control qubit")
    
    print(f"\nTotal pulses: {len(prog.pulse_ops)}")
    
    # Execute
    print("\nExecuting two-qubit pulse program...")
    final_state = prog.run(backend="numpy")
    
    print(f"\nFinal state shape: {final_state.shape}")
    print(f"Norm: {np.linalg.norm(final_state):.6f}")
    
    print("\n✅ Cross-resonance pulse execution successful!")


# ==============================================================================
# Example 3: Pulse Shape Optimization
# ==============================================================================

def example_3_pulse_shape_optimization():
    """Example 3: Optimize pulse shapes for gate fidelity."""
    print("\n" + "="*70)
    print("Example 3: Pulse Shape Optimization")
    print("="*70)
    
    print("\nComparing different pulse waveforms for X-gate:")
    print("-" * 70)
    
    waveform_types = [
        ("Gaussian", waveforms.Gaussian(amp=1.0, duration=160, sigma=40)),
        ("DRAG", waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.2)),
        ("Constant", waveforms.Constant(amp=1.0, duration=100)),
        ("CosineDrag", waveforms.CosineDrag(amp=1.0, duration=160, phase=0, alpha=0.5))
    ]
    
    results = []
    
    for name, waveform in waveform_types:
        prog = PulseProgram(1)
        prog.set_device_params(qubit_freq=[5.0e9], anharmonicity=[-330e6])
        prog.add_pulse(0, waveform, qubit_freq=5.0e9, drive_freq=5.0e9)
        
        # Execute
        final_state = prog.run(backend="numpy")
        
        # Check result
        pop_0 = abs(final_state[0])**2
        pop_1 = abs(final_state[1])**2
        
        results.append((name, pop_0, pop_1))
        print(f"{name:15s}: |0⟩={pop_0:.4f}, |1⟩={pop_1:.4f}")
    
    print("\n💡 Insight: Different waveforms produce different state evolution")
    print("   This is expected - pulse calibration is needed for accurate gates")
    print("\n✅ Pulse shape comparison complete!")


# ==============================================================================
# Example 4: Integration with Circuit API
# ==============================================================================

def example_4_pulse_to_circuit_conversion():
    """Example 4: Convert PulseProgram to Circuit for chain API."""
    print("\n" + "="*70)
    print("Example 4: PulseProgram ↔ Circuit Conversion")
    print("="*70)
    
    # Create pure pulse program
    prog = PulseProgram(1)
    prog.set_device_params(qubit_freq=[5.0e9], anharmonicity=[-330e6])
    
    x_pulse = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.2)
    prog.add_pulse(0, x_pulse, qubit_freq=5.0e9)
    
    print("\n1. Created PulseProgram with 1 pulse")
    
    # Convert to Circuit
    circuit = prog.to_circuit()
    
    print(f"2. Converted to Circuit:")
    print(f"   - num_qubits: {circuit.num_qubits}")
    print(f"   - ops: {len(circuit.ops)}")
    print(f"   - pulse_library: {len(circuit.metadata.get('pulse_library', {}))}")
    
    # Now can use Circuit's chain API
    print("\n3. Using Circuit chain API:")
    print("   circuit.device(provider='simulator').run()")
    
    # Execute via Circuit
    from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
    engine = StatevectorEngine()
    result = engine.run(circuit, shots=0)
    
    print("\n✅ PulseProgram ↔ Circuit integration working!")


# ==============================================================================
# Example 5: Hardware-Native Pulse Sequences
# ==============================================================================

def example_5_hardware_native_pulses():
    """Example 5: Hardware-native pulse sequences (no gate abstraction)."""
    print("\n" + "="*70)
    print("Example 5: Hardware-Native Pulse Sequences")
    print("="*70)
    
    print("\nScenario: Quantum optimal control for custom gate")
    print("-" * 70)
    
    prog = PulseProgram(1)
    prog.set_device_params(
        qubit_freq=[5.0e9],
        anharmonicity=[-330e6],
        T1=[80e-6],
        T2=[120e-6]
    )
    
    # Custom pulse sequence for optimal control
    # (In practice, these would be optimized via gradient descent)
    pulses = [
        waveforms.Drag(amp=0.3, duration=100, sigma=25, beta=0.15),
        waveforms.Drag(amp=0.5, duration=120, sigma=30, beta=0.20),
        waveforms.Drag(amp=0.2, duration=80, sigma=20, beta=0.10),
    ]
    
    print(f"\nApplying {len(pulses)} optimized pulses:")
    for i, pulse in enumerate(pulses, 1):
        prog.add_pulse(0, pulse, qubit_freq=5.0e9)
        print(f"  {i}. {pulse.__class__.__name__}: "
              f"amp={pulse.amp}, duration={pulse.duration}ns")
    
    # Execute
    final_state = prog.run(backend="numpy")
    
    print(f"\nFinal state:")
    print(f"  |0⟩: {abs(final_state[0])**2:.6f}")
    print(f"  |1⟩: {abs(final_state[1])**2:.6f}")
    
    print("\n💡 Use case: Pulse-level quantum optimal control")
    print("   - Direct pulse programming bypasses gate abstractions")
    print("   - Enables hardware-native optimization")
    print("   - Supports gradient-based pulse shape tuning")
    
    print("\n✅ Hardware-native pulse sequence executed!")


# ==============================================================================
# Run All Examples
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TyxonQ: Pure Pulse Programming (Mode B)")
    print("="*70)
    print("\nDual-Mode Pulse Architecture:")
    print("  • Mode A (Chain): Gate → Pulse Compiler → Execute")
    print("  • Mode B (Direct): Pure Pulse → Direct Execute (THIS FILE)")
    
    example_1_single_qubit_pulse()
    example_2_two_qubit_cross_resonance()
    example_3_pulse_shape_optimization()
    example_4_pulse_to_circuit_conversion()
    example_5_hardware_native_pulses()
    
    print("\n" + "="*70)
    print("Summary: Pure Pulse Programming Demonstrated")
    print("="*70)
    
    print("""
Key Features:
  ✅ Direct pulse-level control (no gate abstraction)
  ✅ Hardware-native programming
  ✅ Flexible pulse sequence design
  ✅ Integration with Circuit API
  ✅ Support for quantum optimal control

Use Cases:
  🔬 Pulse shape optimization experiments
  🎯 Quantum optimal control algorithms
  🔧 Hardware calibration routines
  📊 Research on pulse-level dynamics

Comparison:
  Mode A (gate_to_pulse.py)     : High-level, automatic compilation
  Mode B (pure_pulse_programming): Low-level, manual control

Next Steps:
  - See pulse_mode_a_chain_compilation.py for Mode A
  - See pulse_gate_decomposition_complete.py for gate→pulse
  - See PULSE_MODES_GUIDE.md for complete documentation
""")
