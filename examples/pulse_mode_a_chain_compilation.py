"""Example: Pulse-level compilation with TyxonQ.

This example demonstrates TyxonQ's pulse-level compilation capabilities,
a core differentiating feature compared to traditional quantum frameworks.

Pulse compilation allows you to:
    1. Convert gate-level circuits to hardware pulse sequences
    2. Customize pulse calibrations for specific gates
    3. Optimize pulse timing and scheduling
    4. Simulate real physics effects (T1/T2, frequency detuning)

TyxonQ supports dual-mode pulse programming (Memory 8b12df21):
    - Mode A (Chain): Gate Circuit → Pulse Compiler → Pulse Sequence → Execution
    - Mode B (Direct): Hamiltonian → Direct Pulse Evolution (see pulse_simulation.py)

This example demonstrates Mode A (chain compilation).
"""

from tyxonq import Circuit, waveforms
from tyxonq.compiler.pulse_compile_engine import PulseCompiler


def example_1_basic_pulse_compilation():
    """Example 1: Basic gate-to-pulse compilation."""
    print("\n" + "="*70)
    print("Example 1: Basic Gate-to-Pulse Compilation")
    print("="*70)
    
    # Create a simple circuit
    c = Circuit(2)
    c.h(0)
    c.x(1)
    c.measure_z(0)
    c.measure_z(1)
    
    print("\nOriginal circuit:")
    print(f"  Num qubits: {c.num_qubits}")
    print(f"  Operations: {c.ops}")
    
    # Create pulse compiler
    compiler = PulseCompiler(optimization_level=1)
    
    # Define device parameters (realistic superconducting transmon)
    device_params = {
        "qubit_freq": [5.0e9, 5.1e9],      # Qubit frequencies (Hz)
        "anharmonicity": [-330e6, -320e6],  # Anharmonicity (Hz)
        "T1": [80e-6, 85e-6],               # Amplitude damping time (s)
        "T2": [120e-6, 125e-6]              # Dephasing time (s)
    }
    
    # Compile to pulse (hybrid mode: gates + pulses)
    pulse_circuit = compiler.compile(
        c,
        device_params=device_params,
        mode="hybrid"  # Keep measurement gates as-is
    )
    
    print("\nCompiled pulse circuit:")
    print(f"  Num qubits: {pulse_circuit.num_qubits}")
    print(f"  Operations: {len(pulse_circuit.ops)} total")
    
    # Count pulse operations
    pulse_ops = [op for op in pulse_circuit.ops 
                 if isinstance(op, (list, tuple)) and 
                 str(op[0]).lower() == "pulse"]
    print(f"  Pulse operations: {len(pulse_ops)}")
    
    # Print metadata
    print("\nCompilation metadata:")
    print(f"  Mode: {pulse_circuit.metadata.get('pulse_mode')}")
    if "pulse_total_time" in pulse_circuit.metadata:
        total_time = pulse_circuit.metadata["pulse_total_time"]
        print(f"  Total execution time: {total_time*1e9:.2f} ns")
    
    return pulse_circuit


def example_2_custom_pulse_calibration():
    """Example 2: Custom pulse calibrations for specific gates."""
    print("\n" + "="*70)
    print("Example 2: Custom Pulse Calibrations")
    print("="*70)
    
    c = Circuit(1)
    c.x(0)
    c.x(0)  # Apply X twice (should return to |0⟩)
    
    # Create custom X gate pulse using DRAG waveform
    # DRAG (Derivative Removal by Adiabatic Gate) reduces leakage errors
    custom_x_pulse = waveforms.Drag(
        amp=0.5,      # Pulse amplitude
        duration=160,  # Pulse duration (ns)
        sigma=40,      # Gaussian width
        beta=0.2       # DRAG parameter (reduces leakage)
    )
    
    print("\nCustom X pulse calibration:")
    print(f"  Waveform: {type(custom_x_pulse).__name__}")
    print(f"  Amplitude: {custom_x_pulse.amp}")
    print(f"  Duration: {custom_x_pulse.duration} ns")
    print(f"  DRAG beta: {custom_x_pulse.beta}")
    
    # Create compiler and add calibration
    compiler = PulseCompiler(optimization_level=2)
    compiler.add_calibration(
        gate_name="x",
        qubits=[0],
        pulse_waveform=custom_x_pulse,
        params={
            "qubit_freq": 5.0e9,
            "drive_freq": 5.0e9
        }
    )
    
    # Compile with custom calibration
    pulse_circuit = compiler.compile(
        c,
        device_params={"qubit_freq": [5.0e9], "anharmonicity": [-330e6]},
        calibrations=compiler.get_calibrations(),
        mode="pulse_only"
    )
    
    print("\nCompiled circuit with custom calibration:")
    print(f"  Operations: {len(pulse_circuit.ops)}")
    pulse_ops = [op for op in pulse_circuit.ops 
                 if isinstance(op, (list, tuple)) and "pulse" in str(op[0]).lower()]
    print(f"  Pulse operations: {len(pulse_ops)}")
    
    return pulse_circuit


def example_3_pulse_only_mode():
    """Example 3: Full pulse-only compilation (no gate abstractions)."""
    print("\n" + "="*70)
    print("Example 3: Pulse-Only Mode (Full Decomposition)")
    print("="*70)
    
    # Create Bell state circuit
    c = Circuit(2)
    c.h(0)
    c.cx(0, 1)
    
    print("\nOriginal circuit (gates):")
    for i, op in enumerate(c.ops):
        print(f"  {i}: {op}")
    
    compiler = PulseCompiler(optimization_level=3)
    
    # Compile to pulse-only (all gates → pulses)
    pulse_circuit = compiler.compile(
        c,
        device_params={
            "qubit_freq": [5.0e9, 5.1e9],
            "anharmonicity": [-330e6, -320e6]
        },
        mode="pulse_only"  # Decompose ALL gates to pulses
    )
    
    print("\nCompiled circuit (pulse-only):")
    for i, op in enumerate(pulse_circuit.ops):
        if isinstance(op, (list, tuple)):
            op_name = op[0]
            op_qubit = op[1] if len(op) > 1 else "?"
            print(f"  {i}: {op_name} on qubit {op_qubit}")
    
    # Show scheduling information
    if "pulse_schedule" in pulse_circuit.metadata:
        schedule = pulse_circuit.metadata["pulse_schedule"]
        print(f"\nScheduling information:")
        print(f"  Total pulse events: {len(schedule)}")
        print(f"  Total execution time: {pulse_circuit.metadata.get('pulse_total_time', 0)*1e9:.2f} ns")
    
    return pulse_circuit


def example_4_waveform_comparison():
    """Example 4: Compare different pulse waveforms for the same gate."""
    print("\n" + "="*70)
    print("Example 4: Waveform Comparison")
    print("="*70)
    
    waveform_types = [
        ("Gaussian", waveforms.Gaussian(amp=1.0, duration=160, sigma=40)),
        ("DRAG", waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.2)),
        ("Constant", waveforms.Constant(amp=1.0, duration=100)),
        ("CosineDrag", waveforms.CosineDrag(amp=1.0, duration=160, phase=0, alpha=0.5))
    ]
    
    print("\nComparing waveforms for X gate:")
    print("-" * 70)
    
    for name, waveform in waveform_types:
        c = Circuit(1)
        c.x(0)
        
        compiler = PulseCompiler()
        compiler.add_calibration("x", [0], waveform, {
            "qubit_freq": 5.0e9,
            "drive_freq": 5.0e9
        })
        
        pulse_circuit = compiler.compile(
            c,
            device_params={"qubit_freq": [5.0e9]},
            calibrations=compiler.get_calibrations()
        )
        
        total_time = pulse_circuit.metadata.get("pulse_total_time", 0)
        print(f"  {name:15s}: {total_time*1e9:6.2f} ns")


def example_5_pulse_scheduling_optimization():
    """Example 5: Pulse scheduling for parallel execution."""
    print("\n" + "="*70)
    print("Example 5: Pulse Scheduling Optimization")
    print("="*70)
    
    # Create circuit with independent operations
    c = Circuit(4)
    c.x(0)  # Can run in parallel
    c.x(1)  # Can run in parallel
    c.x(2)  # Can run in parallel
    c.x(3)  # Can run in parallel
    c.cx(0, 1)  # Requires qubits 0, 1
    c.cx(2, 3)  # Can run parallel with above (different qubits)
    
    print("\nCircuit with parallelizable operations:")
    for i, op in enumerate(c.ops):
        print(f"  {i}: {op}")
    
    # Compile with different optimization levels
    for opt_level in [0, 1, 2, 3]:
        compiler = PulseCompiler(optimization_level=opt_level)
        pulse_circuit = compiler.compile(
            c,
            device_params={
                "qubit_freq": [5.0e9, 5.1e9, 5.2e9, 5.3e9],
                "anharmonicity": [-330e6] * 4
            },
            mode="pulse_only"
        )
        
        total_time = pulse_circuit.metadata.get("pulse_total_time", 0)
        print(f"\nOptimization level {opt_level}: {total_time*1e9:.2f} ns")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("TyxonQ Pulse Compilation Examples")
    print("Demonstrating pulse-level quantum control and compilation")
    print("="*70)
    
    try:
        example_1_basic_pulse_compilation()
        example_2_custom_pulse_calibration()
        example_3_pulse_only_mode()
        example_4_waveform_comparison()
        example_5_pulse_scheduling_optimization()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        
        print("\nKey Takeaways:")
        print("  1. TyxonQ provides pulse-level compilation (not available in many frameworks)")
        print("  2. Supports custom pulse calibrations for hardware-specific optimization")
        print("  3. Offers hybrid mode (gates + pulses) and pulse-only mode")
        print("  4. Includes pulse scheduling for parallel execution")
        print("  5. Compatible with realistic device parameters (T1/T2, anharmonicity)")
        
        print("\nNext Steps:")
        print("  - See pulse_simulation.py for direct Hamiltonian evolution (Mode B)")
        print("  - Check examples/pulse_advanced/ for VQE with pulse optimization")
        print("  - Read docs/pulse_programming_guide.md for detailed documentation")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
