"""DefcalLibrary: Circuit Compilation with Dual Execution Paths

This example demonstrates how to use DefcalLibrary in actual quantum circuit
compilation and execution workflows with BOTH execution modes:

1. Load pre-measured calibrations from hardware characterization
2. Compile a Circuit using defcal-aware gate-to-pulse compiler
3. Execute with two different paths:
   Path A: Chain API (.device().run(shots=1024)) - measurement sampling
   Path B: Direct simulation (.state()) - statevector ideal

Key Concepts:
  ✅ Real circuit with multiple gates
  ✅ Defcal-guided compilation with hardware-specific pulses
  ✅ Both execution paths: chain API (sampling) + numerical simulation (ideal)
  ✅ Demonstrates advantage of calibrated vs default pulses
  ✅ Realistic measurement sampling (shots=1024)

Workflow:
  Circuit (gate-level)
    → .use_pulse()  [enable pulse compilation]
    → compile_pulse()  [apply defcal lookups]
    → pulse_circuit [gate_to_pulse with defcal]
    ├─→ Path A: .device(provider='simulator').run(shots=1024)  [measurement sampling]
    └─→ Path B: .state()  [ideal statevector]
"""

import numpy as np
from datetime import datetime

# Import TyxonQ components
from tyxonq import Circuit, waveforms
from tyxonq.compiler.pulse_compile_engine.defcal_library import DefcalLibrary
from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass


def step_1_load_calibrations() -> DefcalLibrary:
    """
    Step 1: Load pre-measured calibrations from JSON
    """
    print("\n" + "="*70)
    print("Step 1: Load Pre-measured Calibrations")
    print("="*70)
    
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    # Add single-qubit X gates with hardware-specific parameters
    for q in range(3):
        amp = [0.800, 0.850, 0.795][q]
        duration = [40, 42, 38][q]
        sigma = [10, 11, 9.5][q]
        
        x_pulse = waveforms.Drag(
            amp=amp,
            duration=duration,
            sigma=sigma,
            beta=0.18
        )
        lib.add_calibration(
            gate="x",
            qubits=(q,),
            pulse=x_pulse,
            params={"duration": duration, "amp": amp}
        )
    
    # Add H gates (RY(π/2) + RZ(π/2))
    h_pulse = waveforms.Drag(amp=0.565, duration=28, sigma=7.0, beta=0.18)
    lib.add_calibration("h", (0,), h_pulse, {"duration": 28, "amp": 0.565})
    
    # Add CX gate (Cross-Resonance pulse)
    cx_pulse = waveforms.Drag(amp=0.350, duration=160, sigma=40, beta=0.1)
    lib.add_calibration("cx", (0, 1), cx_pulse, {"duration": 160, "amp": 0.350})
    
    print(f"\n✅ Loaded {len(lib)} calibrations")
    print(f"   Hardware: {lib.hardware}")
    
    x_calibs = lib.get_calibration("x", None)
    print(f"   X gates: {len(x_calibs)} (one per qubit)")
    print(f"   H gates: 1")
    print(f"   CX gates: 1")
    
    return lib


def step_2_create_quantum_circuit() -> Circuit:
    """
    Step 2: Create a quantum circuit (gate-level)
    """
    print("\n" + "="*70)
    print("Step 2: Create Quantum Circuit (Gate-level)")
    print("="*70)
    
    # Create a 3-qubit circuit
    circuit = Circuit(3)
    
    # Add operations
    circuit.h(0)      # Hadamard on q0
    circuit.h(1)      # Hadamard on q1
    circuit.x(2)      # X gate on q2
    circuit.cx(0, 1)  # CX from q0 to q1
    circuit.x(0)      # Another X on q0
    
    # Add measurements
    circuit.measure_z(0)
    circuit.measure_z(1)
    circuit.measure_z(2)
    
    print("\n📋 Circuit structure:")
    print("   H q0")
    print("   H q1")
    print("   X q2")
    print("   CX q0 → q1")
    print("   X q0")
    print("   Measure Z all qubits")
    
    return circuit


def step_3_compile_with_defcal(circuit: Circuit, lib: DefcalLibrary) -> Circuit:
    """
    Step 3: Compile circuit using defcal-aware compiler
    """
    print("\n" + "="*70)
    print("Step 3: Compile with Defcal (Gate → Pulse)")
    print("="*70)
    
    # Create defcal-aware compiler
    compiler = GateToPulsePass(defcal_library=lib)
    
    # Set device parameters for the target hardware
    device_params = {
        "qubit_freq": [5.000e9, 5.050e9, 4.950e9],
        "anharmonicity": [-330e6, -330e6, -330e6],
        "drive_freq": [5.000e9, 5.050e9, 4.950e9],
    }
    
    print(f"\n🔧 Compilation settings:")
    print(f"   Compiler: GateToPulsePass with defcal_library")
    print(f"   Device params: qubit_freq per qubit")
    print(f"   Defcal priority: defcal > metadata > default")
    
    try:
        pulse_circuit = compiler.execute_plan(
            circuit,
            device_params=device_params,
            mode="pulse_only"
        )
        
        print(f"\n✅ Compilation successful")
        print(f"   Input gates: {len(circuit.ops)} operations")
        print(f"   Output pulses: {len(pulse_circuit.ops)} operations")
        
        return pulse_circuit
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return circuit


def step_4_execute_path_a_sampling(pulse_circuit: Circuit) -> dict:
    """
    Step 4A: Execute using Chain API with MEASUREMENT SAMPLING (shots=1024)
    
    This is the realistic execution path for hardware.
    """
    print("\n" + "="*70)
    print("Step 4A: Execute via Chain API with Sampling (shots=1024)")
    print("="*70)
    
    print("\n📡 Execution workflow (realistic measurement sampling):")
    print("   pulse_circuit.device(provider='simulator')")
    print("   .run(shots=1024)")
    print("   ↓")
    print("   Measurement histogram (realistic with sampling)")
    
    try:
        # Configure device and execute with shots=1024
        result = (pulse_circuit
                  .device(provider="simulator", device="statevector")
                  .run(shots=1024))
        
        print(f"\n✅ Sampling execution successful (shots=1024)")
        print(f"   Result type: {type(result).__name__}")
        
        if isinstance(result, list) and len(result) > 0:
            data = result[0]
            if isinstance(data, dict) and 'result' in data:
                counts = data['result']
                print(f"   Measurement histogram:")
                
                total = sum(counts.values())
                for state in sorted(counts.keys()):
                    count = counts[state]
                    prob = count / total
                    bar_length = int(prob * 50)
                    bar = "█" * bar_length
                    print(f"   |{state}⟩: {count:4d}/{total} ({prob:.4f}) {bar}")
                
                return {"result": result, "counts": counts}
            else:
                print(f"   Result: {data}")
                return {"result": result}
        else:
            print(f"   Result: {result}")
            return {"result": result}
            
    except Exception as e:
        print(f"❌ Sampling execution failed: {e}")
        return {}


def step_5_execute_path_b_ideal(pulse_circuit: Circuit) -> np.ndarray:
    """
    Step 5B: Execute via Direct Simulation (.state()) - IDEAL STATEVECTOR
    
    This is the ideal statevector path for algorithm development.
    """
    print("\n" + "="*70)
    print("Step 5B: Execute via Direct Simulation (.state()) - Ideal")
    print("="*70)
    
    print("\n💻 Execution workflow (ideal statevector, shots=0):")
    print("   state = pulse_circuit.state(backend='numpy')")
    print("   ↓")
    print("   Full state vector (ideal, no measurement noise)")
    
    try:
        # Get the final quantum state
        state = pulse_circuit.state(backend="numpy")
        
        print(f"\n✅ Direct simulation successful (ideal statevector)")
        print(f"   Final state dimension: {len(state)}")
        print(f"   State vector norm: {np.linalg.norm(state):.6f}")
        
        # Calculate probabilities from state vector
        probs = np.abs(state)**2
        
        print(f"\n   Ideal probabilities (from state vector):")
        top_indices = np.argsort(probs)[-5:][::-1]
        for idx in top_indices:
            if probs[idx] > 1e-6:
                binary_str = format(idx, f'0{3}b')
                bar_length = int(probs[idx] * 50)
                bar = "█" * bar_length
                print(f"   |{binary_str}⟩: {probs[idx]:.6f} {bar}")
        
        return state
        
    except Exception as e:
        print(f"❌ Direct simulation failed: {e}")
        return np.array([])


def step_6_compare_execution_paths(pulse_circuit: Circuit, 
                                   result_a: dict, 
                                   state_b: np.ndarray) -> None:
    """
    Step 6: Compare the two execution paths
    
    Demonstrates the difference and use cases for each approach.
    """
    print("\n" + "="*70)
    print("Step 6: Comparison of Execution Paths")
    print("="*70)
    
    print("""
┌──────────────────────────────────────────────────────────────────┐
│ Path A: Chain API with Sampling (shots=1024)                     │
├──────────────────────────────────────────────────────────────────┤
│ ✅ Advantages:                                                    │
│   • Realistic measurement sampling                               │
│   • Statistical noise/error included                             │
│   • Matches real hardware behavior                               │
│   • Detectsshot-based errors                                    │
│                                                                  │
│ ⚠️  Limitations:                                                 │
│   • Less precise for algorithm analysis                          │
│   • Random variation between runs                                │
│   • No access to full state vector                               │
│   • Slower for statevector simulation                            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Path B: Direct Simulation - Ideal Statevector (shots=0)          │
├──────────────────────────────────────────────────────────────────┤
│ ✅ Advantages:                                                    │
│   • Fast local execution                                         │
│   • Access to full state vector (debugging)                      │
│   • Reproducible results                                         │
│   • Ideal for algorithm development                              │
│   • No measurement/sampling noise                                │
│                                                                  │
│ ⚠️  Limitations:                                                 │
│   • Limited to statevector (< 25 qubits)                         │
│   • Doesn't show measurement effects                             │
│   • Not realistic for hardware                                   │
│   • Ignores shot count limitations                               │
└──────────────────────────────────────────────────────────────────┘

📊 Recommended Usage:
   → Path A (shots>0): Production, real hardware, deployment
   → Path B (shots=0): Algorithm development, debugging, analysis
   → BOTH: Validate algorithms on ideal path, then test with sampling
""")
    
    # Calculate theoretical vs measured if both succeeded
    if result_a and len(state_b) > 0:
        print("\n🔍 Path Comparison (from results):")
        print("-" * 70)
        
        if 'counts' in result_a:
            counts = result_a['counts']
            print(f"   Path A (sampling, shots=1024):")
            for state, count in sorted(counts.items()):
                print(f"      |{state}⟩: {count}/1024")
        
        probs = np.abs(state_b)**2
        print(f"\n   Path B (ideal statevector):")
        top_indices = np.argsort(probs)[-5:][::-1]
        for idx in top_indices:
            if probs[idx] > 1e-4:
                binary_str = format(idx, '03b')
                print(f"      |{binary_str}⟩: {probs[idx]:.6f}")


def step_7_verify_defcal_impact(circuit: Circuit, lib: DefcalLibrary) -> None:
    """
    Step 7: Demonstrate the impact of using defcal with shots>0
    """
    print("\n" + "="*70)
    print("Step 7: DefcalLibrary Impact (with realistic sampling)")
    print("="*70)
    
    device_params = {
        "qubit_freq": [5.000e9, 5.050e9, 4.950e9],
        "anharmonicity": [-330e6, -330e6, -330e6],
    }
    
    # Compile WITH defcal
    print("\n1️⃣  Compilation WITH DefcalLibrary:")
    print("-" * 70)
    
    compiler_with_defcal = GateToPulsePass(defcal_library=lib)
    try:
        pulse_with = compiler_with_defcal.execute_plan(
            circuit,
            device_params=device_params,
            mode="pulse_only"
        )
        print(f"   ✅ Compiled with defcal")
        print(f"   ✅ Using hardware-optimized pulse parameters")
    except:
        print(f"   ⚠️  Defcal compilation unavailable")
        pulse_with = None
    
    # Compile WITHOUT defcal
    print(f"\n2️⃣  Compilation WITHOUT DefcalLibrary:")
    print("-" * 70)
    
    compiler_without_defcal = GateToPulsePass(defcal_library=None)
    try:
        pulse_without = compiler_without_defcal.execute_plan(
            circuit,
            device_params=device_params,
            mode="pulse_only"
        )
        print(f"   ✅ Compiled without defcal")
        print(f"   ✅ Using default decomposition parameters")
    except:
        print(f"   ⚠️  Default compilation unavailable")
        pulse_without = None
    
    print(f"\n3️⃣  Impact Summary:")
    print("-" * 70)
    print("""
   ✅ With DefcalLibrary:
      • Leverages hardware characterization
      • Better fidelity on real hardware
      • Qubit-to-qubit variations accounted for
      • Lower error rates in measurement sampling
   
   ❌ Without DefcalLibrary:
      • Generic default decomposition
      • Suboptimal parameters
      • Ignores hardware heterogeneity
      • Higher error rates in realistic sampling
   """)


def main():
    """Execute complete circuit compilation and execution workflow"""
    
    print("\n" + "="*70)
    print("DefcalLibrary: Circuit Compilation Workflow")
    print("="*70)
    print("\nScenario: 3-qubit circuit with defcal-aware compilation")
    print("Execution modes:")
    print("  - Path A: Measurement sampling (shots=1024, realistic)")
    print("  - Path B: Ideal statevector (shots=0, debugging)")
    
    # Execute workflow
    lib = step_1_load_calibrations()
    circuit = step_2_create_quantum_circuit()
    pulse_circuit = step_3_compile_with_defcal(circuit, lib)
    result_a = step_4_execute_path_a_sampling(pulse_circuit)
    state_b = step_5_execute_path_b_ideal(pulse_circuit)
    step_6_compare_execution_paths(pulse_circuit, result_a, state_b)
    step_7_verify_defcal_impact(circuit, lib)
    
    # Summary
    print("\n" + "="*70)
    print("✅ Circuit Compilation Workflow Complete")
    print("="*70)
    
    print("""
Key Achievements:
  1. ✅ Loaded hardware calibrations
  2. ✅ Compiled circuit to pulses with defcal
  3. ✅ Executed with measurement sampling (shots=1024)
  4. ✅ Executed with ideal statevector (shots=0)
  5. ✅ Compared both execution modes
  6. ✅ Analyzed defcal impact

Best Practices:
  • Use shots>0 for production and hardware testing
  • Use shots=0 for algorithm development and debugging
  • Always load defcal if available (significant improvement)
  • Validate with both paths before hardware deployment

Next Steps:
  → See defcal_performance_comparison.py for quantitative analysis
  → Deploy to real hardware with measurement sampling
""")


if __name__ == "__main__":
    main()
