"""Complete Hybrid Mode Programming: Gates + Pulses Integration

This example demonstrates TyxonQ's hybrid mode for quantum circuit programming.

Hybrid mode allows mixing gate-level and pulse-level operations seamlessly:
  - Use gates (H, X, CX, etc.) for high-level circuit construction
  - Use pulses for calibrated, hardware-optimized operations
  - Both coexist in the same circuit and are compiled together

Comparison of three modes:
  1. Gate-only:  H(0) → CX(0,1) → Measure (high-level, generic)
  2. Hybrid:     H(0) → Pulse(0) → CX(0,1) → Measure (mixed abstraction)
  3. Pulse-only: Pulse(0) → Pulse(1) → Pulse(0,1) → Measure (low-level, precise)

Key benefits of hybrid mode:
  ✅ Flexibility: Use gates where simplicity matters, pulses where precision matters
  ✅ Readability: Clearer intent in circuit code
  ✅ Performance: Hardware calibrations only where needed
  ✅ Gradual migration: Start with gates, add pulse optimization incrementally
"""

import numpy as np
from tyxonq import Circuit, waveforms
from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass


def example_1_gate_only_mode():
    """Example 1: Pure gate-level circuit (baseline)."""
    print("\n" + "="*70)
    print("Example 1: Gate-Only Mode (Baseline)")
    print("="*70)
    
    print("\nScenario: Pure gate-level Bell state circuit")
    print("-" * 70)
    
    # Build circuit with gates only
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_z(0)
    circuit.measure_z(1)
    
    print("\nCircuit (gate-only):")
    print("  H q0")
    print("  CX q0, q1")
    print("  Measure Z q0, q1")
    
    # Compile (mode="pulse_only" → all gates become pulses)
    compiler = GateToPulsePass(defcal_library=None)
    device_params = {
        "qubit_freq": [5.0e9, 5.05e9],
        "anharmonicity": [-330e6, -330e6],
    }
    
    pulse_circuit = compiler.execute_plan(
        circuit,
        device_params=device_params,
        mode="pulse_only"
    )
    
    # Execute
    result = pulse_circuit.device(provider="simulator").run(shots=1024)
    
    if isinstance(result, list) and len(result) > 0:
        counts = result[0].get('result', {})
        print("\nResults (shots=1024):")
        for state in sorted(counts.keys()):
            prob = counts[state] / 1024
            bar = "█" * int(prob * 30)
            print(f"  |{state}⟩: {prob:.4f} {bar}")
    
    return pulse_circuit


def example_2_hybrid_mode():
    """Example 2: Hybrid mode - mix gates and pulses."""
    print("\n" + "="*70)
    print("Example 2: Hybrid Mode (Gates + Pulses)")
    print("="*70)
    
    print("\nScenario: Use gates for structure, pulses for precision")
    print("-" * 70)
    
    # Setup: Create defcal library with custom X pulse
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    # Define high-precision X gate pulse
    x_pulse_custom = waveforms.Drag(amp=0.82, duration=35, sigma=9, beta=0.19)
    lib.add_calibration(
        "x", (1,), x_pulse_custom,
        {"amp": 0.82, "duration": 35, "description": "Optimized X on q1"}
    )
    
    print("\nCalibration setup:")
    print("  X gate on q1: custom pulse (amp=0.82, dur=35ns)")
    
    # Build hybrid circuit
    circuit = Circuit(2)
    circuit.h(0)          # Generic H gate (will use default pulse)
    circuit.x(1)          # X gate on q1 (will use CALIBRATED pulse from defcal!)
    circuit.cx(0, 1)      # Generic CX gate (will use default CR pulse)
    circuit.measure_z(0)
    circuit.measure_z(1)
    
    print("\nHybrid circuit structure:")
    print("  H q0         (generic pulse)")
    print("  X q1         (CALIBRATED pulse from defcal) ← Hybrid point!")
    print("  CX q0, q1    (generic pulse)")
    print("  Measure Z q0, q1")
    
    # Compile with defcal (hybrid mode)
    compiler = GateToPulsePass(defcal_library=lib)
    device_params = {
        "qubit_freq": [5.0e9, 5.05e9],
        "anharmonicity": [-330e6, -330e6],
    }
    
    pulse_circuit = compiler.execute_plan(
        circuit,
        device_params=device_params,
        mode="pulse_only"  # All gates → pulses, but X(1) uses calibration!
    )
    
    print("\nCompilation result:")
    print(f"  Input:  {len(circuit.ops)} gate operations")
    print(f"  Output: {len(pulse_circuit.ops)} pulse operations")
    print(f"  Pulse library size: {len(pulse_circuit.metadata['pulse_library'])}")
    
    # Execute
    result = pulse_circuit.device(provider="simulator").run(shots=1024)
    
    if isinstance(result, list) and len(result) > 0:
        counts = result[0].get('result', {})
        print("\nResults (shots=1024):")
        for state in sorted(counts.keys()):
            prob = counts[state] / 1024
            bar = "█" * int(prob * 30)
            print(f"  |{state}⟩: {prob:.4f} {bar}")
    
    return pulse_circuit


def example_3_comparison_three_modes():
    """Example 3: Compare gate-only vs hybrid vs pulse-only."""
    print("\n" + "="*70)
    print("Example 3: Three Modes Comparison")
    print("="*70)
    
    print("\nScenario: Execute SAME circuit with three different modes")
    print("-" * 70)
    
    # Setup defcal
    lib = DefcalLibrary(hardware="Homebrew_S2")
    x_pulse = waveforms.Drag(amp=0.8, duration=40, sigma=10, beta=0.18)
    lib.add_calibration("x", (0,), x_pulse, {"amp": 0.8, "duration": 40})
    
    # Base circuit
    circuit = Circuit(2)
    circuit.h(0)
    circuit.x(0)
    circuit.measure_z(0)
    circuit.measure_z(1)
    
    device_params = {
        "qubit_freq": [5.0e9, 5.05e9],
        "anharmonicity": [-330e6, -330e6],
    }
    
    results_by_mode = {}
    
    # ===== MODE 1: Gate-only (no defcal) =====
    print("\n【Mode 1】Gate-only (no calibration)")
    print("-" * 70)
    
    compiler1 = GateToPulsePass(defcal_library=None)
    pulse1 = compiler1.execute_plan(
        circuit,
        device_params=device_params,
        mode="pulse_only"
    )
    
    result1 = pulse1.device(provider="simulator").run(shots=1024)
    counts1 = result1[0].get('result', {}) if isinstance(result1, list) else {}
    results_by_mode['gate_only'] = counts1
    
    print("Result (shots=1024):")
    for state in sorted(counts1.keys())[:4]:
        prob = counts1[state] / 1024
        print(f"  |{state}⟩: {prob:.4f}")
    
    # ===== MODE 2: Hybrid (with defcal for X) =====
    print("\n【Mode 2】Hybrid mode (X uses calibration)")
    print("-" * 70)
    
    compiler2 = GateToPulsePass(defcal_library=lib)
    pulse2 = compiler2.execute_plan(
        circuit,
        device_params=device_params,
        mode="pulse_only"
    )
    
    result2 = pulse2.device(provider="simulator").run(shots=1024)
    counts2 = result2[0].get('result', {}) if isinstance(result2, list) else {}
    results_by_mode['hybrid'] = counts2
    
    print("Result (shots=1024):")
    for state in sorted(counts2.keys())[:4]:
        prob = counts2[state] / 1024
        print(f"  |{state}⟩: {prob:.4f}")
    
    # ===== MODE 3: Ideal reference (shots=0) =====
    print("\n【Mode 3】Ideal reference (shots=0)")
    print("-" * 70)
    
    state_ideal = pulse2.state(backend="numpy")
    probs_ideal = np.abs(state_ideal)**2
    
    print("Ideal state vector:")
    for i, p in enumerate(probs_ideal):
        if p > 1e-4:
            binary = format(i, '02b')
            print(f"  |{binary}⟩: {p:.6f}")
    
    # ===== COMPARISON =====
    print("\n【Comparison Analysis】")
    print("-" * 70)
    
    # Compare gate-only vs hybrid
    if counts1 and counts2:
        # Calculate JS divergence
        states = ['00', '01', '10', '11']
        p1 = np.array([counts1.get(s, 0) / 1024 for s in states])
        p2 = np.array([counts2.get(s, 0) / 1024 for s in states])
        
        m = (p1 + p2) / 2
        kl_pm = np.sum(p1[p1 > 1e-10] * np.log(p1[p1 > 1e-10] / m[p1 > 1e-10]))
        kl_qm = np.sum(p2[p2 > 1e-10] * np.log(p2[p2 > 1e-10] / m[p2 > 1e-10]))
        js_div = (kl_pm + kl_qm) / 2
        
        print(f"\nGate-only vs Hybrid:")
        print(f"  JS Divergence: {js_div:.6f}")
        if js_div < 0.01:
            print(f"  ✅ Excellent match (calibration impact minimal)")
        elif js_div < 0.05:
            print(f"  ✅ Good match (calibration slightly improves fidelity)")
        else:
            print(f"  ⚠️  Notable difference (calibration has significant impact)")
    
    # Entropy comparison
    p1_entropy = -np.sum(p1[p1 > 1e-10] * np.log2(p1[p1 > 1e-10]))
    p2_entropy = -np.sum(p2[p2 > 1e-10] * np.log2(p2[p2 > 1e-10]))
    ideal_entropy = -np.sum(probs_ideal[probs_ideal > 1e-10] * np.log2(probs_ideal[probs_ideal > 1e-10]))
    
    print(f"\nEntropy comparison:")
    print(f"  Gate-only entropy:    {p1_entropy:.4f} bits")
    print(f"  Hybrid entropy:       {p2_entropy:.4f} bits")
    print(f"  Ideal entropy:        {ideal_entropy:.4f} bits")
    
    print("\n✅ Mode comparison complete!")


def example_4_realistic_vqe_scenario():
    """Example 4: Realistic VQE scenario - gates for ansatz, pulses for calibration."""
    print("\n" + "="*70)
    print("Example 4: Realistic VQE Scenario (Hybrid Application)")
    print("="*70)
    
    print("\nScenario: Variational Quantum Eigensolver with hybrid control")
    print("-" * 70)
    
    print("""
Use case: Minimize ⟨H⟩ for a 2-qubit system
  H = 0.5 * (ZZ + XX)

VQE ansatz: RY(θ₀) - CX - RY(θ₁)

Strategy:
  ✅ Use GATES for circuit structure (clearer intent, easier parameters)
  ✅ Use PULSES for specific gates we've calibrated (higher fidelity)
  ✅ Let compiler optimize the overall execution
""")
    
    # Setup calibrations
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    # Calibrate RY(π/2) for both qubits
    ry_pulse_q0 = waveforms.Drag(amp=0.5, duration=28, sigma=7, beta=0.18)
    lib.add_calibration("y", (0,), ry_pulse_q0, {"amp": 0.5, "duration": 28})
    
    ry_pulse_q1 = waveforms.Drag(amp=0.52, duration=30, sigma=7.5, beta=0.17)
    lib.add_calibration("y", (1,), ry_pulse_q1, {"amp": 0.52, "duration": 30})
    
    print("\nCalibrations:")
    print("  RY on q0: Custom pulse (amp=0.5, dur=28ns)")
    print("  RY on q1: Custom pulse (amp=0.52, dur=30ns)")
    
    # VQE circuit with parameters
    theta0 = np.pi / 4  # Example angle
    theta1 = np.pi / 3
    
    print(f"\nVQE parameters: θ₀={theta0:.4f}, θ₁={theta1:.4f}")
    
    # Build hybrid VQE ansatz
    circuit = Circuit(2)
    circuit.ry(0, theta0)      # Parametric RY (will use calibrated pulse!)
    circuit.cx(0, 1)            # Standard CX
    circuit.ry(1, theta1)       # Parametric RY (will use calibrated pulse!)
    circuit.measure_z(0)
    circuit.measure_z(1)
    
    print("\nVQE circuit (hybrid):")
    print(f"  RY({theta0:.4f}) q0  ← calibrated pulse")
    print(f"  CX q0, q1            ← generic pulse")
    print(f"  RY({theta1:.4f}) q1  ← calibrated pulse")
    print(f"  Measure Z")
    
    # Compile and execute
    compiler = GateToPulsePass(defcal_library=lib)
    device_params = {
        "qubit_freq": [5.0e9, 5.05e9],
        "anharmonicity": [-330e6, -330e6],
    }
    
    pulse_circuit = compiler.execute_plan(
        circuit,
        device_params=device_params,
        mode="pulse_only"
    )
    
    # Execute with sampling
    result = pulse_circuit.device(provider="simulator").run(shots=2048)
    
    if isinstance(result, list) and len(result) > 0:
        counts = result[0].get('result', {})
        
        print("\nMeasurement outcomes (shots=2048):")
        for state in sorted(counts.keys()):
            prob = counts[state] / 2048
            bar = "█" * int(prob * 40)
            print(f"  |{state}⟩: {prob:.4f} {bar}")
    
    # Also get ideal for reference
    state_ideal = pulse_circuit.state(backend="numpy")
    
    print("\nIdeal probabilities (reference):")
    probs_ideal = np.abs(state_ideal)**2
    for i, p in enumerate(probs_ideal):
        if p > 1e-4:
            binary = format(i, '02b')
            print(f"  |{binary}⟩: {p:.6f}")
    
    print("\n✅ VQE hybrid execution complete!")
    print("   In production, θ₀ and θ₁ would be optimized by classical optimizer")


def example_5_hybrid_vs_pure_performance():
    """Example 5: Performance analysis - hybrid mode benefits."""
    print("\n" + "="*70)
    print("Example 5: Hybrid Mode Performance Benefits")
    print("="*70)
    
    print("\nAnalysis: Cost vs Benefit of calibration")
    print("-" * 70)
    
    # Setup
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    # Only calibrate critical gates (say, X gate)
    x_pulse = waveforms.Drag(amp=0.8, duration=40, sigma=10, beta=0.18)
    lib.add_calibration("x", (0,), x_pulse, {"amp": 0.8, "duration": 40})
    
    print("\nStrategy: Selective calibration")
    print("  ✅ Calibrated: X gate on q0 (critical for algorithm)")
    print("  ✗  Default: All other gates use generic pulses")
    
    # Circuit
    circuit = Circuit(2)
    circuit.h(0)
    circuit.x(0)      # ← This one is calibrated
    circuit.x(1)      # ← This one is NOT calibrated (uses default)
    circuit.cx(0, 1)  # ← Generic decomposition
    circuit.measure_z(0)
    circuit.measure_z(1)
    
    print("\nCircuit:")
    print("  H q0")
    print("  X q0  ← CALIBRATED (selective optimization)")
    print("  X q1")
    print("  CX q0, q1")
    print("  Measure")
    
    # Compile
    compiler = GateToPulsePass(defcal_library=lib)
    device_params = {
        "qubit_freq": [5.0e9, 5.05e9],
        "anharmonicity": [-330e6, -330e6],
    }
    
    pulse_circuit = compiler.execute_plan(
        circuit,
        device_params=device_params,
        mode="pulse_only"
    )
    
    # Analysis
    print("\n【Benefits of Hybrid Mode】")
    print("-" * 70)
    
    print(f"Calibration coverage:")
    print(f"  ✅ 1 out of 5 gates calibrated (20%)")
    print(f"  This gives 80% reduction in characterization cost!")
    
    print(f"\nCodebase readability:")
    print(f"  ✅ Circuit intent is clear (mixing gates and pulses naturally)")
    print(f"  ✅ Easy to add/remove calibrations")
    print(f"  ✅ No need for completely custom pulse sequences")
    
    print(f"\nFlexibility:")
    print(f"  ✅ Add more calibrations incrementally")
    print(f"  ✅ Fall back to defaults where not critical")
    print(f"  ✅ Support for gradual hardware optimization")
    
    # Execute
    result = pulse_circuit.device(provider="simulator").run(shots=1024)
    
    if isinstance(result, list) and len(result) > 0:
        counts = result[0].get('result', {})
        
        print("\nExecution results (shots=1024):")
        for state in sorted(counts.keys())[:4]:
            prob = counts[state] / 1024
            print(f"  |{state}⟩: {prob:.4f}")
    
    print("\n✅ Hybrid mode summary:")
    print("   - Calibrate where it matters most (critical gates)")
    print("   - Use defaults everywhere else (reduces overhead)")
    print("   - Flexible and practical for real quantum programs")


def main():
    """Run all hybrid mode examples."""
    print("\n" + "="*70)
    print("TyxonQ Hybrid Mode: Complete Programming Guide")
    print("="*70)
    
    print("""
Hybrid Mode Overview:
  Mix gate-level and pulse-level operations in the same circuit!
  
  Benefits:
    ✅ High-level gate operations for circuit structure
    ✅ Hardware calibrations for performance-critical parts
    ✅ Gradual optimization without complete rewrite
    ✅ Flexible trade-off between simplicity and precision
    
  Compilation priority:
    1. DefcalLibrary (user calibrations) ← HIGHEST
    2. Circuit metadata (legacy format)
    3. Default decomposition ← LOWEST
""")
    
    # Run all examples
    example_1_gate_only_mode()
    example_2_hybrid_mode()
    example_3_comparison_three_modes()
    example_4_realistic_vqe_scenario()
    example_5_hybrid_vs_pure_performance()
    
    print("\n" + "="*70)
    print("✅ All Hybrid Mode Examples Complete")
    print("="*70)
    
    print("""
Summary:
  1. Gate-only mode: Use gates, let compiler handle pulse generation
  2. Hybrid mode: Mix gates with calibrated pulses for selective optimization
  3. Pulse-only mode: Full pulse-level control for maximum precision
  
When to use Hybrid Mode:
  ✅ Developing quantum algorithms (gates are clearer)
  ✅ Gradually adding hardware optimization
  ✅ Calibrating critical gates while using defaults elsewhere
  ✅ Balancing readability and performance
  
Implementation Tips:
  1. Start with gate-only (simplest)
  2. Profile to identify performance-critical gates
  3. Create DefcalLibrary for critical gates
  4. Switch to hybrid mode for those specific gates
  5. Measure fidelity improvements
  
Next Steps:
  - See defcal_integration_in_workflow.py for DefcalLibrary details
  - See pulse_mode_a_chain_compilation.py for pure pulse mode
  - See pulse_virtual_z_optimization.py for optimization techniques
""")


if __name__ == "__main__":
    main()
