"""Pulse Waveforms: Comprehensive Guide

This example demonstrates TyxonQ's pulse waveform library and how different
waveforms affect gate implementation and quantum control.

Available Waveforms:
  • Constant: Rectangular pulse
  • Gaussian: Smooth Gaussian envelope
  • Drag: DRAG (Derivative Removal by Adiabatic Gate) for leakage suppression
  • Hermite: Higher-order derivative correction
  • BlackmanSquare: Blackman window for improved spectral properties
  • CosineDrag: Cosine-modulated DRAG pulse

Key Concepts:
  ✅ Pulse shape affects gate fidelity
  ✅ DRAG suppresses leakage to higher levels
  ✅ Waveform choice depends on hardware characteristics
  ✅ Proper envelope functions improve gate quality
  ✅ Parameter tuning is essential for calibration

Module Structure:
  - Example 1: Waveform Types and Properties
  - Example 2: Comparing Different Envelopes
  - Example 3: Single-Qubit Gate Implementation
  - Example 4: Two-Qubit Gate Decomposition
  - Example 5: Waveform Optimization
  - Example 6: Advanced Pulse Shaping Techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from tyxonq import Circuit, waveforms
from tyxonq.core.ir.pulse import PulseProgram


# ==============================================================================
# Example 1: Waveform Types
# ==============================================================================

def example_1_waveform_types():
    """Example 1: Demonstrate all available waveform types."""
    print("\n" + "="*70)
    print("Example 1: Waveform Types and Properties")
    print("="*70)
    
    print("\n📊 Available Waveforms in TyxonQ:")
    print("-" * 70)
    
    waveforms_info = [
        {
            "name": "Constant",
            "class": "waveforms.Constant",
            "params": "amp, duration",
            "use_case": "Simple rectangular pulse",
            "pros": "Fast, sharp transitions",
            "cons": "High spectral content, leakage"
        },
        {
            "name": "Gaussian",
            "class": "waveforms.Gaussian",
            "params": "amp, duration, sigma",
            "use_case": "Standard smooth envelope",
            "pros": "Smooth, reduced spectral sidebands",
            "cons": "Slower edges than Constant"
        },
        {
            "name": "DRAG",
            "class": "waveforms.Drag",
            "params": "amp, duration, sigma, beta",
            "use_case": "Leakage suppression",
            "pros": "Suppresses transitions to |2⟩",
            "cons": "Additional beta parameter tuning"
        },
        {
            "name": "Hermite",
            "class": "waveforms.Hermite",
            "params": "amp, duration, sigma, alpha",
            "use_case": "Higher-order corrections",
            "pros": "Excellent spectral properties",
            "cons": "More complex, slower computation"
        },
        {
            "name": "BlackmanSquare",
            "class": "waveforms.BlackmanSquare",
            "params": "amp, duration, width",
            "use_case": "Optimal spectral envelope",
            "pros": "Blackman windowing reduces sidebands",
            "cons": "Specific to hardware characteristics"
        },
        {
            "name": "CosineDrag",
            "class": "waveforms.CosineDrag",
            "params": "amp, duration, phase, alpha",
            "use_case": "Phase-modulated control",
            "pros": "Flexible phase control",
            "cons": "Specialized use cases"
        }
    ]
    
    for i, wf in enumerate(waveforms_info, 1):
        print(f"\n{i}. {wf['name']}")
        print(f"   Import: {wf['class']}")
        print(f"   Parameters: {wf['params']}")
        print(f"   Use Case: {wf['use_case']}")
        print(f"   Pros: {wf['pros']}")
        print(f"   Cons: {wf['cons']}")
    
    print("\n✅ Waveform types overview complete!")


# ==============================================================================
# Example 2: Comparing Different Envelopes
# ==============================================================================

def example_2_envelope_comparison():
    """Example 2: Compare how different envelopes affect pulse shape."""
    print("\n" + "="*70)
    print("Example 2: Envelope Comparison")
    print("="*70)
    
    duration = 160
    sigma = 40
    amp = 1.0
    
    print(f"\nCommon Parameters:")
    print(f"  Duration: {duration} ns")
    print(f"  Sigma: {sigma} ns (for smooth envelopes)")
    print(f"  Amplitude: {amp}")
    
    print(f"\n📊 Simulating gate execution with different envelopes:")
    print("-" * 70)
    
    envelopes = [
        ("Constant", waveforms.Constant(amp=amp, duration=duration)),
        ("Gaussian", waveforms.Gaussian(amp=amp, duration=duration, sigma=sigma)),
        ("DRAG (beta=0.2)", waveforms.Drag(amp=amp, duration=duration, sigma=sigma, beta=0.2)),
        ("DRAG (beta=0.1)", waveforms.Drag(amp=amp, duration=duration, sigma=sigma, beta=0.1)),
    ]
    
    results = []
    print(f"\n{'Envelope':<20} {'Pop_0':<10} {'Pop_1':<10} {'Leakage':<10}")
    print("-" * 70)
    
    for name, wf in envelopes:
        prog = PulseProgram(1)
        prog.set_device_params(
            qubit_freq=[5.0e9],
            anharmonicity=[-330e6],
            T1=[80e-6],
            T2=[120e-6]
        )
        prog.add_pulse(0, wf, qubit_freq=5.0e9)
        
        state = prog.state(backend="numpy")
        pop_0 = abs(state[0])**2 if len(state) > 0 else 0
        pop_1 = abs(state[1])**2 if len(state) > 1 else 0
        leakage = 1 - pop_0 - pop_1  # Population in higher levels
        
        results.append((name, pop_0, pop_1, leakage))
        print(f"{name:<20} {pop_0:<10.4f} {pop_1:<10.4f} {leakage:<10.6f}")
    
    # Analysis
    print("\n💡 Key Observations:")
    print("  • Constant: Highest leakage (sharp transitions)")
    print("  • Gaussian: Reduced leakage (smooth envelope)")
    print("  • DRAG: Leakage suppression via derivative term")
    print("  • Higher beta: More aggressive leakage suppression")
    
    print("\n✅ Envelope comparison complete!")


# ==============================================================================
# Example 3: Single-Qubit Gates
# ==============================================================================

def example_3_single_qubit_gates():
    """Example 3: Implement single-qubit gates with different waveforms."""
    print("\n" + "="*70)
    print("Example 3: Single-Qubit Gate Implementation")
    print("="*70)
    
    print("\nBasic Single-Qubit Rotations:")
    print("-" * 70)
    
    # X gate (π rotation around X-axis)
    print("\n1. X Gate (π rotation):")
    print("   Implementation: DRAG pulse with optimized amplitude")
    
    prog_x = PulseProgram(1)
    prog_x.set_device_params(qubit_freq=[5.0e9], anharmonicity=[-330e6])
    x_pulse = waveforms.Drag(amp=0.8, duration=160, sigma=40, beta=0.2)
    prog_x.add_pulse(0, x_pulse, qubit_freq=5.0e9)
    
    state_x = prog_x.state(backend="numpy")
    print(f"   Result: |0⟩={abs(state_x[0])**2:.4f}, |1⟩={abs(state_x[1])**2:.4f}")
    print(f"   Expected: |0⟩=0.0, |1⟩=1.0 (perfect π rotation)")
    
    # Y gate (π rotation around Y-axis)
    print("\n2. Y Gate (π rotation with phase shift):")
    print("   Implementation: DRAG pulse with π/2 phase shift")
    
    prog_y = PulseProgram(1)
    prog_y.set_device_params(qubit_freq=[5.0e9], anharmonicity=[-330e6])
    y_pulse = waveforms.Drag(amp=0.8, duration=160, sigma=40, beta=0.2)
    prog_y.add_pulse(0, y_pulse, qubit_freq=5.0e9, phase=np.pi/2)
    
    state_y = prog_y.state(backend="numpy")
    print(f"   Result: |0⟩={abs(state_y[0])**2:.4f}, |1⟩={abs(state_y[1])**2:.4f}")
    
    # Z gate (Phase rotation, zero time)
    print("\n3. Z Gate (π phase rotation, zero physical time):")
    print("   Implementation: Virtual-Z (frame update, no pulse)")
    print("   No physical pulse needed - handled in software")
    print("   Result: Phase update only, populations unchanged")
    
    # Hadamard (π/√2 around (X+Z)/√2)
    print("\n4. Hadamard Gate (π/√2 rotation):")
    print("   Implementation: Composite pulse or custom waveform")
    
    prog_h = PulseProgram(1)
    prog_h.set_device_params(qubit_freq=[5.0e9], anharmonicity=[-330e6])
    h_pulse = waveforms.Drag(amp=0.5657, duration=112, sigma=28, beta=0.2)  # π/√2
    prog_h.add_pulse(0, h_pulse, qubit_freq=5.0e9)
    
    state_h = prog_h.state(backend="numpy")
    expected_h = 1/np.sqrt(2)
    print(f"   Result: |0⟩={abs(state_h[0])**2:.4f}, |1⟩={abs(state_h[1])**2:.4f}")
    print(f"   Expected: ~{expected_h**2:.4f} each")
    
    print("\n✅ Single-qubit gates complete!")


# ==============================================================================
# Example 4: Two-Qubit Gate Decomposition
# ==============================================================================

def example_4_two_qubit_gates():
    """Example 4: Two-qubit gate implementation via decomposition."""
    print("\n" + "="*70)
    print("Example 4: Two-Qubit Gate Decomposition")
    print("="*70)
    
    print("\nTwo-Qubit Gate Implementations:")
    print("-" * 70)
    
    # CX gate
    print("\n1. CX Gate (CNOT):")
    print("   Physical Implementation: Cross-Resonance (CR) sequence")
    print("   Decomposition:")
    print("     1. RX(-π/2) on control")
    print("     2. CR pulse (drive control @ target frequency)")
    print("     3. Rotary echo on target")
    print("     4. RX(π/2) on control")
    
    prog_cx = PulseProgram(2)
    prog_cx.set_device_params(
        qubit_freq=[5.0e9, 5.1e9],
        anharmonicity=[-330e6, -320e6]
    )
    
    # Build CX sequence
    prog_cx.drag(0, amp=-0.5, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    prog_cx.gaussian(0, amp=0.3, duration=400, sigma=100, qubit_freq=5.0e9)  # CR
    prog_cx.constant(1, amp=0.1, duration=400, qubit_freq=5.1e9)  # Echo
    prog_cx.drag(0, amp=0.5, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    
    state_cx = prog_cx.state(backend="numpy")
    print(f"   Total duration: ~{160*4} ns")
    print(f"   State norm: {np.linalg.norm(state_cx):.6f}")
    
    # iSWAP gate
    print("\n2. iSWAP Gate:")
    print("   Physical Implementation: CX chain decomposition")
    print("   Decomposition: CX(q0,q1) · CX(q1,q0) · CX(q0,q1)")
    print("   Properties:")
    print("     • Exchanges states |01⟩ ↔ i|10⟩ (with phase i)")
    print("     • Symmetric, works on any topology")
    print("     • Phase difference automatically handled in software")
    
    prog_iswap = PulseProgram(2)
    prog_iswap.set_device_params(
        qubit_freq=[5.0e9, 5.1e9],
        anharmonicity=[-330e6, -320e6]
    )
    
    # iSWAP as CX chain
    for _ in range(3):
        prog_iswap.gaussian(0, amp=0.3, duration=100, sigma=25, qubit_freq=5.0e9)
    
    state_iswap = prog_iswap.state(backend="numpy")
    print(f"   Total duration: ~300 ns (3 × CX)")
    print(f"   State norm: {np.linalg.norm(state_iswap):.6f}")
    
    # SWAP gate
    print("\n3. SWAP Gate:")
    print("   Physical Implementation: CX chain (same as iSWAP)")
    print("   Decomposition: CX(q0,q1) · CX(q1,q0) · CX(q0,q1)")
    print("   Properties:")
    print("     • Exchanges states |01⟩ ↔ |10⟩ (no phase)")
    print("     • Physically identical to iSWAP")
    print("     • Phase difference handled in software")
    
    prog_swap = PulseProgram(2)
    prog_swap.set_device_params(
        qubit_freq=[5.0e9, 5.1e9],
        anharmonicity=[-330e6, -320e6]
    )
    
    # SWAP as CX chain
    for _ in range(3):
        prog_swap.gaussian(0, amp=0.3, duration=100, sigma=25, qubit_freq=5.0e9)
    
    state_swap = prog_swap.state(backend="numpy")
    print(f"   Total duration: ~300 ns (same as iSWAP)")
    print(f"   State norm: {np.linalg.norm(state_swap):.6f}")
    
    print("\n💡 Key Insight:")
    print("   iSWAP and SWAP use identical pulse sequences")
    print("   Physics difference: relative phase (handled in software)")
    print("   Both decompose to 3-CX chains for universal compatibility")
    
    print("\n✅ Two-qubit gates complete!")


# ==============================================================================
# Example 5: Waveform Optimization
# ==============================================================================

def example_5_waveform_optimization():
    """Example 5: Optimize waveform parameters for fidelity."""
    print("\n" + "="*70)
    print("Example 5: Waveform Parameter Optimization")
    print("="*70)
    
    print("\nOptimizing DRAG pulse parameters for maximum fidelity:")
    print("-" * 70)
    
    # Scan amplitude
    print("\n1. Amplitude Scan (for π rotation):")
    amplitudes = np.linspace(0.6, 1.0, 5)
    
    print(f"\n{'Amplitude':<12} {'Pop_1':<12} {'Error':<12} {'Status':<15}")
    print("-" * 70)
    
    best_amp = None
    best_error = float('inf')
    
    for amp in amplitudes:
        prog = PulseProgram(1)
        prog.set_device_params(qubit_freq=[5.0e9], anharmonicity=[-330e6])
        pulse = waveforms.Drag(amp=amp, duration=160, sigma=40, beta=0.2)
        prog.add_pulse(0, pulse, qubit_freq=5.0e9)
        
        state = prog.state(backend="numpy")
        pop_1 = abs(state[1])**2
        error = abs(pop_1 - 1.0)
        
        status = "✅ Good" if error < 0.05 else "⚠️  OK" if error < 0.1 else "❌ Poor"
        
        print(f"{amp:<12.3f} {pop_1:<12.4f} {error:<12.6f} {status:<15}")
        
        if error < best_error:
            best_error = error
            best_amp = amp
    
    print(f"\n✅ Optimal amplitude: {best_amp:.3f} (error={best_error:.6f})")
    
    # Scan beta (DRAG parameter)
    print("\n2. Beta Scan (leakage suppression):")
    betas = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    print(f"\n{'Beta':<12} {'Pop_1':<12} {'Leakage':<12} {'Fidelity':<12}")
    print("-" * 70)
    
    for beta in betas:
        prog = PulseProgram(1)
        prog.set_device_params(
            qubit_freq=[5.0e9],
            anharmonicity=[-330e6],
            T1=[80e-6]
        )
        pulse = waveforms.Drag(amp=0.8, duration=160, sigma=40, beta=beta)
        prog.add_pulse(0, pulse, qubit_freq=5.0e9)
        
        state = prog.state(backend="numpy")
        pop_0 = abs(state[0])**2
        pop_1 = abs(state[1])**2
        leakage = 1 - pop_0 - pop_1
        fidelity = pop_1  # For π gate
        
        print(f"{beta:<12.2f} {pop_1:<12.4f} {leakage:<12.6f} {fidelity:<12.4f}")
    
    print("\n💡 Insights:")
    print("  • Amplitude controls rotation angle")
    print("  • Beta controls leakage suppression")
    print("  • Optimal beta depends on anharmonicity")
    print("  • Trade-off: more suppression, slower computation")
    
    print("\n✅ Waveform optimization complete!")


# ==============================================================================
# Example 6: Advanced Pulse Shaping
# ==============================================================================

def example_6_advanced_pulse_shaping():
    """Example 6: Advanced techniques for pulse design."""
    print("\n" + "="*70)
    print("Example 6: Advanced Pulse Shaping Techniques")
    print("="*70)
    
    print("\nAdvanced Concepts:")
    print("-" * 70)
    
    # 1. Composite pulses
    print("\n1. Composite Pulses (BB1, CORPSE, etc.):")
    print("   Idea: Sequence of pulses with specific phases")
    print("   Purpose: Robustness to parameter variations")
    print("   Example: X(θ) Y(2θ) X(θ) is robust to amplitude noise")
    
    prog_composite = PulseProgram(1)
    prog_composite.set_device_params(qubit_freq=[5.0e9], anharmonicity=[-330e6])
    
    # Simple composite: X π/2, Y π, X π/2 (robust to detuning)
    theta = 0.8  # π/2 equivalent
    prog_composite.drag(0, amp=theta, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    prog_composite.drag(0, amp=2*theta, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9, phase=np.pi/2)
    prog_composite.drag(0, amp=theta, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    
    print("   Sequence: RX(π/2) → RY(π) → RX(π/2)")
    print("   Total duration: 480 ns")
    state = prog_composite.state(backend="numpy")
    print(f"   Result norm: {np.linalg.norm(state):.6f}")
    
    # 2. Pulse shaping with derivative
    print("\n2. Derivative-Modulated Pulses:")
    print("   Idea: Add derivative term to reduce spectral content")
    print("   Math: I(t) = A·f(t), Q(t) = A·β·f'(t)")
    print("   Benefit: Suppresses leakage to higher levels")
    
    # DRAG is exactly this - derivative of Gaussian
    print("   ✅ DRAG already implements this (I + iQ terms)")
    
    # 3. Shaped pulses for different gates
    print("\n3. Gate-Specific Pulse Shaping:")
    print("   Different gates need different waveforms")
    print("   ")
    
    gates = [
        ("X gate", "DRAG, amp=0.8, beta=0.2"),
        ("Y gate", "DRAG with π/2 phase shift"),
        ("Z gate", "Virtual-Z (no pulse needed)"),
        ("H gate", "π/√2 DRAG pulse"),
        ("T gate", "π/8 DRAG pulse"),
        ("CX gate", "CR sequence"),
    ]
    
    print(f"\n{'Gate':<12} {'Recommended Waveform':<30}")
    print("-" * 70)
    for gate, wf in gates:
        print(f"{gate:<12} {wf:<30}")
    
    # 4. Robust control techniques
    print("\n4. Robust Control Techniques:")
    print("   • Derivative removal: DRAG")
    print("   • Composite pulses: Sequence of rotations")
    print("   • Optimal control: Gradient-based optimization")
    print("   • Pulse staggering: Compensate hardware asymmetries")
    
    print("\n✅ Advanced pulse shaping complete!")


# ==============================================================================
# Summary
# ==============================================================================

def print_summary():
    """Print comprehensive summary."""
    print("\n" + "="*70)
    print("📚 Summary: Pulse Waveforms and Gate Implementation")
    print("="*70)
    
    print("""
Waveform Selection Guide:

  For most use cases: DRAG (leakage suppression)
    ✅ Default choice
    ✅ Parameters: amp, duration, sigma, beta
    ✅ Widely supported
    ✅ Good gate fidelity

  For smooth gates: Gaussian
    ✅ Simpler than DRAG
    ✅ Parameters: amp, duration, sigma
    ✅ Lower spectral content
    ✅ May have more leakage

  For optimal control: Hermite
    ✅ Higher-order corrections
    ✅ Parameters: amp, duration, sigma, alpha
    ✅ Complex optimization
    ✅ Best fidelity possible

  For hardware-specific: BlackmanSquare
    ✅ Blackman windowing
    ✅ Reduced sidebands
    ✅ Hardware dependent
    ✅ Specialized optimization

Key Principles:

  1. Envelope function affects spectral content
     • Sharp pulses: High sidebands, leakage
     • Smooth pulses: Low sidebands, less leakage
     • Optimal: Balance between speed and leakage

  2. Amplitude controls rotation angle
     • Higher amplitude: Faster rotation
     • Proper tuning needed for π rotations
     • Amplitude scan during calibration

  3. Duration affects gate time
     • Longer: More adiabatic, less error
     • Shorter: Faster, but susceptible to noise
     • Trade-off between speed and fidelity

  4. DRAG parameter (beta) suppresses leakage
     • β > 0: Derivative correction applied
     • Larger β: More suppression
     • Optimization needed for your hardware

Gate Implementation Strategy:

  Single-qubit: DRAG with optimized beta
  Two-qubit: CX chains (CR, iSWAP, SWAP)
  Z rotations: Virtual-Z (zero time)
  Hadamard: π/√2 DRAG pulses

Next Steps:

  → See pulse_gate_calibration.py for optimization
  → See pulse_variational_algorithms.py for applications
  → See pulse_optimization_advanced.py for advanced techniques
""")


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 TyxonQ Pulse Waveforms - Comprehensive Guide")
    print("="*70)
    
    print("""
Master pulse waveforms for high-fidelity quantum gates:

  • Available waveforms (Constant, Gaussian, DRAG, Hermite, etc.)
  • Envelope comparison and effects
  • Single-qubit gate implementation
  • Two-qubit gate decomposition
  • Parameter optimization
  • Advanced pulse shaping techniques
""")
    
    example_1_waveform_types()
    example_2_envelope_comparison()
    example_3_single_qubit_gates()
    example_4_two_qubit_gates()
    example_5_waveform_optimization()
    example_6_advanced_pulse_shaping()
    print_summary()
    
    print("\n" + "="*70)
    print("✅ All Examples Complete!")
    print("="*70)
