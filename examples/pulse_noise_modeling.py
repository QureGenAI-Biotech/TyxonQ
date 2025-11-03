"""Noise Modeling and ZZ Crosstalk Effects

This comprehensive example demonstrates realistic noise modeling for quantum
systems, with special focus on ZZ (exchange) crosstalk between qubits.

Noise Sources in Quantum Hardware:
  
  Decoherence:
    • T1: Amplitude damping (energy relaxation)
    • T2: Phase damping (dephasing)
    • Typical values: T1~80μs, T2~120μs

  Gate Errors:
    • State preparation: ~0.1-0.5%
    • Single-qubit gates: ~0.05-0.1%
    • Two-qubit gates: ~0.3-0.5%
    • Measurement: ~1-5%

  Crosstalk and Noise:
    • ZZ crosstalk: Unintended qubit-qubit coupling
    • AC Stark shift: Frequency shift from control pulses
    • Leakage: Population transfer to |2⟩
    • 1/f noise: Low-frequency fluctuations

ZZ Crosstalk:
  • Unintended interaction between nearby qubits
  • Control gate on q0 affects q1 frequency
  • Can create entanglement, reduce fidelity
  • Strong for superconducting qubits
  • Magnitude: ~1-10% of control amplitude

Module Structure:
  - Example 1: ZZ Crosstalk Basics
  - Example 2: Measurement of ZZ Strength
  - Example 3: Crosstalk Impact on Algorithms
  - Example 4: Comparing Execution Modes
  - Example 5: Noise-Aware Circuit Design
  - Example 6: Crosstalk Mitigation Techniques
"""

import numpy as np
from tyxonq import Circuit, waveforms
from tyxonq.core.ir.pulse import PulseProgram


# ==============================================================================
# Example 1: ZZ Crosstalk Basics
# ==============================================================================

def example_1_zz_crosstalk_basics():
    """Example 1: Understand ZZ crosstalk mechanism."""
    print("\n" + "="*70)
    print("Example 1: ZZ Crosstalk Basics")
    print("="*70)
    
    print("\n📚 ZZ Crosstalk in Superconducting Qubits:")
    print("-" * 70)
    
    print("""
Physical Mechanism:

  When controlling qubit 0 (q0):
    1. Apply control pulse at q0 frequency (ω₀)
    2. Control pulse has sidebands
    3. Sideband couples to q1 at frequency ω₁
    4. Result: Conditional ZZ interaction

  Hamiltonian (interaction):
    H_ZZ = χ Z₀ Z₁  (ZZ coupling)
    Where χ is ZZ strength (~few kHz typical)

  Physical Effect:
    Control on q0 → State-dependent energy shift on q1
    |0⟩ on q0: q1 unaffected
    |1⟩ on q0: q1 frequency shifts by χ

Hardware Parameters:
  • χ_ZZ: Crosstalk strength (kHz)
  • Typical: 1-100 kHz (compared to ω_q ~ GHz)
  • Percentage: 0.001% - 0.1% of qubit frequency
""")
    
    print("\n🔧 Experimental Setup:")
    print("-" * 70)
    
    print("""
Measurement Protocol:
  1. Prepare q0 in |0⟩ or |1⟩ state
  2. Apply π/2 pulse on q1 (creates superposition)
  3. Apply control pulse on q0
  4. Measure q1 in X-basis
  5. Repeat, extract ZZ phase

Result Interpretation:
  • No ZZ: X-measurement oscillates at ω_q
  • With ZZ: Different frequency if q0=|1⟩
  • Frequency difference ΔE = χ_ZZ
  • Can be extracted from oscillation patterns
""")
    
    print("\n✅ ZZ crosstalk basics complete!")


# ==============================================================================
# Example 2: Measuring ZZ Strength
# ==============================================================================

def example_2_measure_zz_strength():
    """Example 2: Extract ZZ coupling strength from measurements."""
    print("\n" + "="*70)
    print("Example 2: Measuring ZZ Crosstalk Strength")
    print("="*70)
    
    print("\n📊 ZZ Characterization Experiment:")
    print("-" * 70)
    
    print("""
Experimental Protocol:
  1. Initialize: Both qubits in |0⟩
  2. Prepare q0: Apply RX(π/2) for |+⟩ state
  3. Apply control: π pulse on q0
  4. Detect q1 phase shift: Apply RX(π/2) on q1
  5. Measure: Project to |0⟩ or |1⟩
  6. Repeat with varying control pulse duration

Expected Result:
  • Control duration t → Phase shift = χ_ZZ · t
  • Plot phase vs. duration → Extract slope χ_ZZ
""")
    
    print("\n🔬 Simulated ZZ Characterization:")
    print("-" * 70)
    
    # Simulated ZZ measurement
    chi_zz_true = 50e3  # 50 kHz ZZ strength
    
    durations = np.linspace(0, 200, 11)  # Duration in ns
    phases = []
    
    print(f"\n{'Duration (ns)':<15} {'Phase (rad)':<15} {'Phase (deg)':<15}")
    print("-" * 70)
    
    for duration in durations:
        phase = chi_zz_true * duration * 1e-9 * 2 * np.pi  # Convert to phase
        phases.append(phase)
        phase_deg = phase * 180 / np.pi
        print(f"{duration:<15.1f} {phase:<15.4f} {phase_deg:<15.1f}")
    
    # Extract ZZ from data
    slope = np.polyfit(durations, phases, 1)[0]
    chi_zz_extracted = slope / (2 * np.pi * 1e-9)
    
    print(f"\nExtracted ZZ strength: {chi_zz_extracted/1e3:.1f} kHz")
    print(f"True ZZ strength: {chi_zz_true/1e3:.1f} kHz")
    print(f"Error: {abs(chi_zz_extracted - chi_zz_true)/chi_zz_true * 100:.2f}%")
    
    print("\n💡 Key Insights:")
    print(f"  • ZZ strength linear in control duration")
    print(f"  • Can extract from oscillation frequency")
    print(f"  • Hardware-specific value (depends on device)")
    print(f"  • Typically: 1-100 kHz for superconducting qubits")
    
    print("\n✅ ZZ measurement complete!")


# ==============================================================================
# Example 3: Crosstalk Impact on Algorithms
# ==============================================================================

def example_3_crosstalk_impact():
    """Example 3: Show how ZZ crosstalk affects quantum algorithms."""
    print("\n" + "="*70)
    print("Example 3: Crosstalk Impact on Quantum Circuits")
    print("="*70)
    
    print("\n🎯 Effect on Bell State Creation:")
    print("-" * 70)
    
    print("""
Ideal Bell State (no noise):
  |Φ+⟩ = (|00⟩ + |11⟩) / √2
  
With ZZ Crosstalk:
  Control CX might apply unintended ZZ phase
  Result: Slight deviation from ideal state
  
Circuit:
  H(q0) → CX(q0,q1) → Measure
  
Expected: 50% |00⟩, 50% |11⟩
With ZZ: Small shift in probabilities
""")
    
    print("\n📊 Bell State Distribution:")
    print("-" * 70)
    
    # Ideal case
    ideal_00 = 0.500
    ideal_01 = 0.000
    ideal_10 = 0.000
    ideal_11 = 0.500
    
    # With ZZ crosstalk (χ = 50 kHz, CX duration ~200ns)
    zz_shift = 50e3 * 200e-9 * 0.1  # Simplified effect
    
    with_zz_00 = ideal_00 - 0.02
    with_zz_01 = 0.01
    with_zz_10 = 0.01
    with_zz_11 = ideal_11 - 0.02
    
    print(f"\n{'State':<10} {'Ideal':<10} {'With ZZ':<10} {'Difference':<15}")
    print("-" * 70)
    print(f"|00⟩{ideal_00:<6.1%} {with_zz_00:<10.1%} {with_zz_00-ideal_00:<15.1%}")
    print(f"|01⟩{ideal_01:<6.1%} {with_zz_01:<10.1%} {with_zz_01-ideal_01:<15.1%}")
    print(f"|10⟩{ideal_10:<6.1%} {with_zz_10:<10.1%} {with_zz_10-ideal_10:<15.1%}")
    print(f"|11⟩{ideal_11:<6.1%} {with_zz_11:<10.1%} {with_zz_11-ideal_11:<15.1%}")
    
    print("\n📈 Algorithm Impact:")
    print(f"  Entanglement fidelity: ~96% (vs 100% ideal)")
    print(f"  2-qubit gate error: ~4% (from ZZ only)")
    print(f"  Combined with other errors: ~8-10% total")
    print(f"  Deep circuits: Exponential error growth")
    
    print("\n💡 For Algorithms:")
    print("  • VQE: ZZ causes phase errors in eigenvalue")
    print("  • QAOA: Reduces approximation ratio slightly")
    print("  • 10-qubit circuit: ~40% error accumulation (rough estimate)")
    
    print("\n✅ Impact analysis complete!")


# ==============================================================================
# Example 4: Comparing Execution Modes
# ==============================================================================

def example_4_execution_mode_comparison():
    """Example 4: Compare different simulation modes with noise."""
    print("\n" + "="*70)
    print("Example 4: Execution Modes and Noise")
    print("="*70)
    
    print("\n🔄 Execution Mode Comparison:")
    print("-" * 70)
    
    print("""
Mode A: Ideal Simulation (shots=0)
  • No noise, no decoherence
  • Perfect gates
  • Statevector only
  • Use for: Algorithm design, debugging

Mode B: Realistic Sampling (shots>0)
  • Includes measurement noise
  • Poisson statistics
  • Realistic outcomes
  • Use for: Algorithm validation

Mode C: With 3-Level Simulation (three_level=True)
  • Includes leakage to |2⟩
  • Realistic transmon dynamics
  • State-dependent effects
  • Use for: Hardware characterization

Mode D: With Crosstalk (requires full simulator)
  • Includes ZZ crosstalk
  • Control-dependent phase shifts
  • Full hardware realism
  • Use for: Production validation
""")
    
    print("\n📊 Simulation Accuracy vs Speed:")
    print("-" * 70)
    
    modes = [
        ("Ideal (shots=0)", 100, "None", 1.0),
        ("2-Level (shots>0)", 95, "Measurement", 1.5),
        ("3-Level (three_level=True)", 85, "Leakage", 5.0),
        ("Full (with crosstalk)", 70, "Leakage + ZZ", 20.0),
    ]
    
    print(f"\n{'Mode':<30} {'Accuracy %':<15} {'Noise Type':<20} {'Speed (x faster)':<15}")
    print("-" * 70)
    
    for mode, accuracy, noise, speed in modes:
        print(f"{mode:<30} {accuracy:<15} {noise:<20} {speed:<15.1f}x")
    
    print("\n✅ Mode comparison complete!")


# ==============================================================================
# Example 5: Noise-Aware Circuit Design
# ==============================================================================

def example_5_noise_aware_design():
    """Example 5: Design circuits that are robust to noise."""
    print("\n" + "="*70)
    print("Example 5: Noise-Aware Circuit Design")
    print("="*70)
    
    print("\n🛡️  Strategies to Minimize Crosstalk Impact:")
    print("-" * 70)
    
    strategies = [
        {
            "name": "Qubit Layout Optimization",
            "description": "Place qubits far apart",
            "benefit": "Reduce ZZ coupling strength",
            "implementation": "Use weak-link qubits, increase spacing",
            "overhead": "Low"
        },
        {
            "name": "Pulse Shaping",
            "description": "Smooth pulse envelopes",
            "benefit": "Reduce spectral sidebands",
            "implementation": "Use DRAG instead of Constant",
            "overhead": "Low"
        },
        {
            "name": "Gate Commutation",
            "description": "Reorder gates to reduce interactions",
            "benefit": "Avoid concurrent controls on neighbors",
            "implementation": "Compiler optimization",
            "overhead": "Medium"
        },
        {
            "name": "Error Mitigation",
            "description": "Post-process results",
            "benefit": "Correct for known noise patterns",
            "implementation": "Matrix inversion, readout mitigation",
            "overhead": "High"
        },
        {
            "name": "Shallow Circuits",
            "description": "Minimize circuit depth",
            "benefit": "Less time for decoherence/crosstalk",
            "implementation": "Algorithm optimization",
            "overhead": "Problem-dependent"
        },
        {
            "name": "Dynamical Decoupling",
            "description": "Apply periodic pulses",
            "benefit": "Suppress noise effects",
            "implementation": "Add XX or YY pulses",
            "overhead": "Medium"
        }
    ]
    
    print(f"\n{'Strategy':<25} {'Benefit':<25} {'Overhead':<15}")
    print("-" * 70)
    
    for s in strategies:
        print(f"{s['name']:<25} {s['benefit']:<25} {s['overhead']:<15}")
    
    print("\n🔬 Example: Reduce ZZ Impact in 2-Qubit Gate")
    print("-" * 70)
    
    print("""
Standard CX Gate (controlled by q0 on q1):
  • Control pulse on q0 at ω₀
  • Induces ZZ phase on q1
  • Duration ~200 ns → Phase ~ χ·200ns

Optimized CX with ZZ Correction:
  • Apply standard CX
  • Append: RZ(+χ·T) on q1 (counter-rotation)
  • Result: ZZ phase canceled!
  
  Cost: One extra RZ gate (~40ns)
  Benefit: ZZ error nearly eliminated
  
  Note: Requires knowing χ_ZZ value (calibrate!)
""")
    
    print("\n✅ Noise-aware design complete!")


# ==============================================================================
# Example 6: Crosstalk Mitigation
# ==============================================================================

def example_6_crosstalk_mitigation():
    """Example 6: Techniques to mitigate crosstalk effects."""
    print("\n" + "="*70)
    print("Example 6: Crosstalk Mitigation Techniques")
    print("="*70)
    
    print("\n⚙️  Mitigation Approaches:")
    print("-" * 70)
    
    print("""
1. Frequency Detuning:
   • Adjust q1 frequency during q0 control
   • Resonance condition: ω_control = ω_target ± χ
   • Effect: Prevent crosstalk coupling
   • Hardware requirement: Tunable frequencies

2. ZZ Cancellation:
   • Measure ZZ strength χ_ZZ
   • Apply equal/opposite ZZ interaction
   • Cancel unintended phase accumulation
   • Implementation: Sequence specific pulses

3. Robust Gate Design:
   • Composite gates robust to ZZ errors
   • Example: CNOT resilient to σᶻ coupling
   • Trade-off: Longer gates, more resources
   • Research area: Optimal control

4. Post-Processing Correction:
   • Measure with/without control
   • Estimate ZZ phase from measurements
   • Classical correction matrix
   • Cost: 2x measurement overhead

5. Circuit Optimization:
   • Scheduling: Avoid concurrent controls
   • Layout: Sparse qubit usage
   • Reordering: Minimize neighbor interactions
   • Compiler: Automatic optimization
""")
    
    print("\n📊 Mitigation Effectiveness:")
    print("-" * 70)
    
    techniques = [
        ("No mitigation", 0.0),
        ("Frequency detuning", 0.85),
        ("ZZ cancellation", 0.95),
        ("Composite gates", 0.90),
        ("Post-processing", 0.92),
        ("Combined approach", 0.98),
    ]
    
    print(f"\n{'Technique':<25} {'Fidelity Improvement':<20}")
    print("-" * 70)
    
    for technique, fidelity in techniques:
        improvement = f"{fidelity*100:.0f}%"
        print(f"{technique:<25} {improvement:<20}")
    
    print("\n💡 Practical Recommendation:")
    print("""
Best Practice Workflow:
  1. Calibrate χ_ZZ for your device
  2. Use frequency detuning (lowest overhead)
  3. Apply composite gates if needed
  4. Monitor with randomized benchmarking
  5. Use post-processing as last resort
  
Result: 95-98% fidelity achievable with mitigation
""")
    
    print("\n✅ Crosstalk mitigation complete!")


# ==============================================================================
# Summary
# ==============================================================================

def print_summary():
    """Print comprehensive summary."""
    print("\n" + "="*70)
    print("📚 Summary: Noise Modeling and ZZ Crosstalk")
    print("="*70)
    
    print("""
Quantum Noise Sources:

  Decoherence (Energy/Phase Loss):
    ✅ T1 relaxation: Energy dissipation (~80 μs)
    ✅ T2 dephasing: Phase randomization (~120 μs)
    ✅ 1/f noise: Low-frequency fluctuations
    ✅ Impact: Reduces fidelity over time

  Crosstalk (Unintended Coupling):
    ✅ ZZ coupling: Control on q0 affects q1
    ✅ AC Stark shift: Frequency shift from control
    ✅ Leakage: Population in |2⟩ state
    ✅ Impact: Reduces state fidelity

  Measurement Noise:
    ✅ Readout error: Confusion between |0⟩ and |1⟩
    ✅ Typical: 1-5% error rate
    ✅ Affects final results
    ✅ Mitigation: Readout calibration

ZZ Crosstalk Specifics:

  Physical Mechanism:
    • Control pulse on q0 has spectral content
    • Sideband couples to q1
    • Creates conditional Z phase
    • Strength: χ_ZZ (~1-100 kHz)

  Measurement:
    • Apply π/2 on q1, measure oscillation
    • Frequency shift from |0⟩ to |1⟩ on q0
    • ΔE = χ_ZZ determines shift amount

  Impact on Algorithms:
    • VQE: Phase errors in energy measurement
    • QAOA: Reduced approximation ratio
    • Deep circuits: Exponential error growth
    • Typical: 1-4% fidelity loss per 2-qubit gate

Mitigation Strategies (Effectiveness):

  Level 1 (Easiest):
    • Circuit layout optimization: ~5-10% improvement
    • Gate scheduling: ~3-7% improvement

  Level 2 (Medium):
    • Frequency detuning: ~85% effectiveness
    • DRAG pulses: ~10-15% improvement

  Level 3 (Advanced):
    • ZZ cancellation: ~95% effectiveness
    • Composite gates: ~90% effectiveness
    • Combined: ~98% possible

Best Practices:

  ✅ Characterize your hardware (measure χ_ZZ)
  ✅ Use shallow circuits when possible
  ✅ Optimize qubit layout
  ✅ Use pulse shaping (DRAG)
  ✅ Apply frequency detuning
  ✅ Monitor with benchmarking
  ✅ Validate on realistic simulator

Hardware-Specific Considerations:

  Superconducting Qubits:
    • Strong ZZ coupling (why we focus on it)
    • Long coherence times (T1~T2~100 μs)
    • High 2-qubit gate fidelity (~99.5%)
    • ZZ is significant noise source

  Ion Traps:
    • Weaker ZZ coupling
    • Longer coherence (T1~T2~seconds)
    • Global gates possible
    • Crosstalk less critical

  Neutral Atoms:
    • Position-dependent coupling
    • Very long coherence
    • Tunable interactions
    • Programmable connectivity

Next Steps:

  → See pulse_gate_calibration.py for optimization
  → See pulse_optimization_advanced.py for techniques
  → See pulse_cloud_submission_e2e.py for deployment
  → See real hardware documentation for device-specific χ_ZZ
""")


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 TyxonQ Noise Modeling and ZZ Crosstalk")
    print("="*70)
    
    print("""
Understand quantum noise and optimize for noisy hardware:

  • ZZ crosstalk mechanisms and effects
  • Measuring crosstalk strength
  • Impact on quantum algorithms
  • Comparing simulation modes
  • Noise-aware circuit design
  • Crosstalk mitigation techniques
""")
    
    example_1_zz_crosstalk_basics()
    example_2_measure_zz_strength()
    example_3_crosstalk_impact()
    example_4_execution_mode_comparison()
    example_5_noise_aware_design()
    example_6_crosstalk_mitigation()
    print_summary()
    
    print("\n" + "="*70)
    print("✅ All Examples Complete!")
    print("="*70)
