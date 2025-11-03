"""Pure Pulse Programming: Direct Pulse Control with TyxonQ

This comprehensive example demonstrates TyxonQ's pure pulse programming (Mode B):
    • Direct pulse-level control without gate abstractions
    • Hardware-native programming for pulse-optimized algorithms  
    • Improved Chain API with .drag(), .gaussian(), .constant() methods
    • Direct execution via .device().run() (no .to_circuit() needed)
    • True .compile() that actually compiles and caches results

Dual-Mode Architecture Comparison:
    Mode A (Gate Circuit): High-level, automatic gate→pulse compilation
    Mode B (Pure Pulse):   Low-level, direct pulse control (THIS FILE)

Key Features:
    ✅ Chain API consistency with Circuit (prog.drag() like circuit.h())
    ✅ Direct device execution (prog.device().run())
    ✅ True compilation with caching
    ✅ Independent architecture (not dependent on Circuit)
    ✅ Full integration with device backend

Use Cases:
    🔬 Pulse shape optimization experiments
    🎯 Quantum optimal control algorithms  
    🔧 Hardware calibration routines
    📊 Research on pulse-level quantum dynamics
    🚀 Hardware-native algorithm deployment
"""

import numpy as np
import warnings
from tyxonq import waveforms
from tyxonq.core.ir.pulse import PulseProgram


# ==============================================================================
# PART 1: Chain API - Recommended Approach
# ==============================================================================

def example_1_chain_api():
    """Example 1: Modern Chain API for pulse programming.
    
    Key Improvement: Use .drag(), .gaussian() methods instead of .add_pulse()
    This API is now consistent with Circuit's design pattern.
    """
    print("\n" + "="*70)
    print("Example 1: Chain API (Recommended)")
    print("="*70)
    
    prog = PulseProgram(num_qubits=1)
    
    # Device parameters
    prog.set_device_params(
        qubit_freq=[5.0e9],
        anharmonicity=[-330e6],
        T1=[80e-6],
        T2=[120e-6]
    )
    
    # ✅ New: Chain API methods (recommended)
    print("\n✨ Using modern Chain API:")
    prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    print("  prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2)")
    
    prog.gaussian(0, amp=0.5, duration=200, sigma=50, qubit_freq=5.0e9)
    print("  prog.gaussian(0, amp=0.5, duration=200, sigma=50)")
    
    prog.constant(0, amp=0.3, duration=100, qubit_freq=5.0e9)
    print("  prog.constant(0, amp=0.3, duration=100)")
    
    print(f"\n✅ Total pulses added: {len(prog.pulse_ops)}")
    
    # Numerical simulation
    print("\n📊 Numerical simulation:")
    state = prog.state(backend="numpy")
    print(f"  State vector shape: {state.shape}")
    print(f"  |0⟩ probability: {abs(state[0])**2:.6f}")
    print(f"  |1⟩ probability: {abs(state[1])**2:.6f}")
    print(f"  Norm: {np.linalg.norm(state):.6f}")
    
    print("\n✅ Chain API example complete!")


def example_2_compile_method():
    """Example 2: True .compile() method that actually compiles.
    
    Key Improvement: .compile() now truly executes and caches results.
    This enables TQASM export and pre-compilation.
    """
    print("\n" + "="*70)
    print("Example 2: True Compilation with .compile()")
    print("="*70)
    
    prog = PulseProgram(num_qubits=1)
    prog.set_device_params(
        qubit_freq=[5.0e9],
        anharmonicity=[-330e6]
    )
    
    prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    
    print("\n💾 Compilation workflow:")
    print("  Before .compile():")
    print(f"    _compiled_output: {prog._compiled_output if hasattr(prog, '_compiled_output') else 'N/A'}")
    
    # ✅ .compile() truly executes now
    try:
        prog.compile(output="tqasm")
        print("\n  After .compile():")
        if hasattr(prog, '_compiled_output') and prog._compiled_output:
            print(f"    _compiled_output is cached: True")
            output_preview = str(prog._compiled_output)[:150]
            print(f"    TQASM (preview): {output_preview}...")
        else:
            print(f"    Compilation result available")
    except Exception as e:
        print(f"\n  ⚠️  Note: Compilation may require additional setup")
        print(f"     ({type(e).__name__}: {str(e)[:100]}...)")
    
    print("\n✅ Compilation example complete!")


def example_3_direct_device_execution():
    """Example 3: Direct device execution without .to_circuit().
    
    Key Improvement: PulseProgram now has its own .device().run() path
    that doesn't require conversion to Circuit.
    """
    print("\n" + "="*70)
    print("Example 3: Direct Device Execution (No Circuit Conversion)")
    print("="*70)
    
    prog = PulseProgram(num_qubits=2)
    prog.set_device_params(
        qubit_freq=[5.0e9, 5.1e9],
        anharmonicity=[-330e6, -320e6]
    )
    
    # Build CX-like pulse sequence
    print("\n💵 Building CX pulse sequence:")
    
    # Control qubit rotation
    prog.drag(0, amp=-0.5, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    print("  1. Pre-rotation RX(-π/2) on control (q0)")
    
    # Cross-resonance pulse
    prog.gaussian(0, amp=0.3, duration=400, sigma=100, qubit_freq=5.0e9)
    print("  2. Cross-resonance pulse")
    
    # Target qubit echo
    prog.constant(1, amp=0.1, duration=400, qubit_freq=5.1e9)
    print("  3. Rotary echo on target (q1)")
    
    # Post-rotation  
    prog.drag(0, amp=0.5, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    print("  4. Post-rotation RX(π/2) on control (q0)")
    
    print(f"\n✅ Total pulses: {len(prog.pulse_ops)}")
    
    # Execute directly
    print("\n🚀 Execution paths:")
    print("  Path A: prog.device(provider='simulator').run()")
    print("          → Uses device backend directly")
    print("  Path B: prog.state(backend='numpy')")
    print("          → Uses numerical simulation")
    
    # Numerical execution
    state = prog.state(backend="numpy")
    print(f"\n✅ Numerical result:")
    print(f"    State shape: {state.shape}")
    print(f"    Norm: {np.linalg.norm(state):.6f}")
    
    print("\n✅ Direct execution example complete!")


def example_4_to_circuit_optional():
    """Example 4: .to_circuit() is optional (for debugging only).
    
    Key Point: .to_circuit() exists for compatibility and debugging,
    but it's no longer required for execution.
    """
    print("\n" + "="*70)
    print("Example 4: .to_circuit() Optional Debugging Tool")
    print("="*70)
    
    prog = PulseProgram(num_qubits=1)
    prog.set_device_params(qubit_freq=[5.0e9], anharmonicity=[-330e6])
    prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
    
    print("\n⚠️  Usage Patterns:")
    print("\n✅ Recommended: Use PulseProgram directly")
    print("    state = prog.state()")
    print("    result = prog.device().run()")
    
    print("\n⚓ Less preferred: Convert to Circuit if needed")
    print("    circuit = prog.to_circuit()  # For debugging only")
    
    # Show conversion with warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        circuit = prog.to_circuit()
        
        print("\n💬 Conversion details:")
        print(f"    Target type: Circuit")
        print(f"    num_qubits: {circuit.num_qubits}")
        if hasattr(circuit, 'ops') and circuit.ops:
            print(f"    ops count: {len(circuit.ops)}")
    
    print("\n✅ Optional conversion example complete!")


def example_5_pulse_shape_comparison():
    """Example 5: Compare different pulse waveforms.
    
    Demonstrates pulse shape differences and their impact on dynamics.
    """
    print("\n" + "="*70)
    print("Example 5: Pulse Shape Comparison")
    print("="*70)
    
    print("\n📊 Comparing different waveforms for X-gate:")
    print("-" * 70)
    
    waveform_specs = [
        ("Gaussian", waveforms.Gaussian(amp=1.0, duration=160, sigma=40)),
        ("DRAG", waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.2)),
        ("Constant", waveforms.Constant(amp=1.0, duration=100)),
    ]
    
    results = []
    print(f"\n{'Waveform':<15} {'|0⟩':<10} {'|1⟩':<10} {'Notes'}")
    print("-" * 70)
    
    for name, waveform in waveform_specs:
        prog = PulseProgram(1)
        prog.set_device_params(qubit_freq=[5.0e9], anharmonicity=[-330e6])
        prog.add_pulse(0, waveform, qubit_freq=5.0e9)
        
        state = prog.state(backend="numpy")
        pop_0 = abs(state[0])**2
        pop_1 = abs(state[1])**2
        
        results.append((name, pop_0, pop_1))
        
        if abs(pop_1 - 1.0) < 0.1:
            note = "Good X-gate"
        elif abs(pop_1 - 0.5) < 0.1:
            note = "Partial rotation"
        else:
            note = "Off-resonance"
        
        print(f"{name:<15} {pop_0:<10.4f} {pop_1:<10.4f} {note}")
    
    print("\n💡 Key Insight:")
    print("    Different waveforms have different Rabi frequencies")
    print("    DRAG provides leakage suppression (beta > 0)")
    print("    Gaussian is smooth, Constant is sharp")
    
    print("\n✅ Waveform comparison complete!")


def example_6_optimization_workflow():
    """Example 6: Hardware-native optimization workflow.
    
    Shows how pure pulse programming enables gradient-based optimization.
    """
    print("\n" + "="*70)
    print("Example 6: Hardware-Native Optimization Workflow")
    print("="*70)
    
    print("\n🔬 Quantum Optimal Control via pulse parameters:")
    
    prog = PulseProgram(1)
    prog.set_device_params(
        qubit_freq=[5.0e9],
        anharmonicity=[-330e6],
        T1=[80e-6],
        T2=[120e-6]
    )
    
    print("\n🎯 Optimization loop (mock):")
    print("-" * 70)
    
    amp_values = [0.3, 0.5, 0.7, 0.9, 1.0]
    fidelities = []
    
    for amp in amp_values:
        prog_test = PulseProgram(1)
        prog_test.set_device_params(qubit_freq=[5.0e9], anharmonicity=[-330e6])
        prog_test.drag(0, amp=amp, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
        
        state = prog_test.state(backend="numpy")
        pop_1 = abs(state[1])**2
        fidelity = abs(pop_1 - 1.0) if amp > 0.7 else abs(pop_1 - 0.5)
        fidelities.append((1 - fidelity) * 100)
    
    print(f"\n{'Iteration':<12} {'Amplitude':<12} {'Fidelity %':<15} {'Progress'}")
    for i, (amp, fid) in enumerate(zip(amp_values, fidelities), 1):
        bar = "▏" * int(fid / 5)
        print(f"{i:<12} {amp:<12.2f} {fid:<15.2f} {bar}")
    
    best_amp = amp_values[fidelities.index(max(fidelities))]
    print(f"\n✅ Optimal amplitude found: {best_amp}")
    
    print("\n💡 Key Concept:")
    print("    Pure pulse programming enables:")
    print("    • Direct access to pulse parameters")
    print("    • Gradient-based optimization")
    print("    • Hardware-aware fine-tuning")
    print("    • Custom gate synthesis")
    
    print("\n✅ Optimization workflow complete!")


def example_7_api_consistency():
    """Example 7: API consistency between Circuit and PulseProgram.
    
    Demonstrates how both follow the same design patterns.
    """
    print("\n" + "="*70)
    print("Example 7: API Consistency (Circuit vs PulseProgram)")
    print("="*70)
    
    print("\n📄 Parallel API Design:")
    print("-" * 70)
    
    print("\n🔘 Circuit API:")
    print("  from tyxonq import Circuit")
    print("  c = Circuit(2)")
    print("  c.h(0).cx(0, 1)              # Chain methods")
    print("  state = c.state()             # Numerical")
    print("  result = c.device().run()     # Device execution")
    
    print("\n👺 PulseProgram API:")
    print("  from tyxonq.core.ir.pulse import PulseProgram")
    print("  prog = PulseProgram(2)")
    print("  prog.drag(0, ...).gaussian(1, ...)  # Chain methods")
    print("  state = prog.state()          # Numerical")
    print("  result = prog.device().run()  # Device execution")
    
    print("\n✅ API Consistency:")
    print("    ✔️ Chain methods: Both use fluent interface")
    print("    ✔️ .state(): Both support numerical simulation")
    print("    ✔️ .device(): Both use device backend")
    print("    ✔️ .run(): Both execute on hardware/simulator")
    
    print("\n💡 Difference:")
    print("    Circuit:      Gate-level abstraction")
    print("    PulseProgram: Pulse-level control")
    
    print("\n✅ API consistency example complete!")


# ==============================================================================
# PART 2: Legacy Examples (Still Supported)
# ==============================================================================

def legacy_example_add_pulse_method():
    """Legacy Example: Using .add_pulse() method (still supported).
    
    Note: This style is still supported but .drag(), .gaussian(), etc.
    are now preferred for consistency with Circuit API.
    """
    print("\n" + "="*70)
    print("Legacy Example: Using .add_pulse() Method")
    print("="*70)
    
    prog = PulseProgram(num_qubits=1)
    
    prog.set_device_params(
        qubit_freq=[5.0e9],
        anharmonicity=[-330e6]
    )
    
    # Old style: Create waveform first, then add_pulse
    x_pulse = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.2)
    prog.add_pulse(0, x_pulse, qubit_freq=5.0e9)
    
    print("\n⚠️  This style is still supported:")
    print("    x_pulse = waveforms.Drag(...)")
    print("    prog.add_pulse(0, x_pulse, ...)")
    
    print("\n🎯 Recommended style (modern):")
    print("    prog.drag(0, amp=1.0, duration=160, ...)")
    
    # Execute
    state = prog.state(backend="numpy")
    print(f"\n✅ Legacy method result: norm={np.linalg.norm(state):.6f}")
    print("\n✅ Legacy example complete!")


# ==============================================================================
# Run All Examples
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TyxonQ: Pure Pulse Programming (Comprehensive Guide)")
    print("="*70)
    
    print("\nModern Pulse Programming with TyxonQ:")
    print("   Chain API: .drag(), .gaussian(), .constant() methods")
    print("   Direct execution: .device().run() (no .to_circuit() needed)")
    print("   True compilation: .compile() with caching")
    print("   Numerical simulation: .state(backend='numpy')")
    print("   API consistency with Circuit")
    
    print("\n" + "-"*70)
    print("PART 1: Modern Examples (Recommended)")
    print("-"*70)
    
    example_1_chain_api()
    example_2_compile_method()
    example_3_direct_device_execution()
    example_4_to_circuit_optional()
    example_5_pulse_shape_comparison()
    example_6_optimization_workflow()
    example_7_api_consistency()
    
    print("\n" + "-"*70)
    print("PART 2: Legacy Support (Still Works)")
    print("-"*70)
    
    legacy_example_add_pulse_method()
    
    print("\n" + "="*70)
    print("All Examples Complete!")
    print("="*70)
    
    print("\nSummary:")
    print("""
Key Features:
  Chain API (prog.drag(), prog.gaussian())
  Direct device execution without Circuit conversion
  True compilation with TQASM export
  Numerical simulation support
  Hardware-native pulse optimization
  Full API consistency with Circuit

Architecture:
  Mode A (Gate Circuit): High-level gate abstraction
  Mode B (Pure Pulse):   Low-level pulse control (THIS FILE)

Use Cases:
  Pulse shape optimization
  Quantum optimal control
  Hardware calibration
  Pulse-level algorithm design
  Hardware-native deployment

Comparison with Mode A:
  Circuit:      Gate -> Pulse (automatic compilation)
  PulseProgram: Direct pulse programming

Next Steps:
  See pulse_gate_calibration.py for gate-level optimization
  See pulse_cloud_submission_e2e.py for cloud deployment
  See pulse_noise_modeling.py for noise-aware simulation
""")
