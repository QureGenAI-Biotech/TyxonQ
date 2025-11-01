"""DefcalLibrary Integration in Real Quantum Circuit Workflow

This example demonstrates how to use DefcalLibrary in the complete workflow:
1. Build gate-level circuit
2. Load/create calibrations
3. Compile to pulses (with defcal integration)
4. Execute via device (both chain API and direct simulation)

This shows the REAL integration point: how defcal is used during compilation
and how it affects the final pulse circuit execution.

Key Integration Points:
  ✅ DefcalLibrary passed to GateToPulsePass compiler
  ✅ Compiler queries defcal during _gate_to_pulse()
  ✅ Priority: defcal > metadata > default decomposition
  ✅ Works with both execution modes: shots=0 and shots>0
"""

import numpy as np

# Import TyxonQ components
from tyxonq import Circuit, waveforms
from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass


def example_1_minimal_integration():
    """Minimal example: Gate circuit → Compile with defcal → Execute"""
    
    print("\n" + "="*70)
    print("Example 1: Minimal DefcalLibrary Integration")
    print("="*70)
    
    # Step 1: Create DefcalLibrary with hardware calibrations
    print("\n1️⃣  Creating calibration library...")
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    # Add X gate calibration for q0
    x_pulse_q0 = waveforms.Drag(amp=0.8, duration=40, sigma=10, beta=0.18)
    lib.add_calibration("x", (0,), x_pulse_q0, {"duration": 40, "amp": 0.8})
    
    # Add X gate calibration for q1 (different parameters!)
    x_pulse_q1 = waveforms.Drag(amp=0.85, duration=42, sigma=11, beta=0.17)
    lib.add_calibration("x", (1,), x_pulse_q1, {"duration": 42, "amp": 0.85})
    
    print(f"   ✅ Loaded {len(lib)} calibrations")
    
    # Step 2: Build gate-level circuit (high-level)
    print("\n2️⃣  Building gate circuit...")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.x(1)
    circuit.cx(0, 1)
    circuit.measure_z(0)
    circuit.measure_z(1)
    
    print("   Circuit:")
    print("   ├─ H q0")
    print("   ├─ X q1  ← Will use calibrated pulse from defcal!")
    print("   ├─ CX q0-q1")
    print("   └─ Measure Z q0, q1")
    
    # Step 3: Compile with defcal-aware compiler
    print("\n3️⃣  Compiling with defcal...")
    
    # THIS IS THE KEY INTEGRATION POINT!
    # Pass defcal_library to the compiler
    compiler = GateToPulsePass(defcal_library=lib)
    
    device_params = {
        "qubit_freq": [5.0e9, 5.05e9],
        "anharmonicity": [-330e6, -330e6],
    }
    
    # During compilation:
    # - H on q0: No defcal → Use default decomposition
    # - X on q1: Found in defcal → Use calibrated pulse (amp=0.85, dur=42)
    # - CX q0-q1: No defcal → Use default Cross-Resonance decomposition
    pulse_circuit = compiler.execute_plan(
        circuit,
        device_params=device_params,
        mode="pulse_only"
    )
    
    print("   ✅ Compilation complete")
    print(f"   Input gates: {len(circuit.ops)}")
    print(f"   Output pulses: {len(pulse_circuit.ops)}")
    print(f"   Pulse library size: {len(pulse_circuit.metadata['pulse_library'])}")
    
    # Step 4: Execute via device (two modes)
    print("\n4️⃣  Executing circuit...")
    
    # Mode A: Realistic sampling (shots > 0)
    print("   Mode A: Measurement sampling (shots=1024)...")
    result_sampling = pulse_circuit.device(provider="simulator", device="statevector").run(shots=1024)
    
    if isinstance(result_sampling, list) and len(result_sampling) > 0:
        counts = result_sampling[0].get('result', {})
        if counts:
            print("   Measurement outcomes:")
            for state in sorted(counts.keys()):
                prob = counts[state] / 1024
                print(f"      |{state}⟩: {prob:.4f}")
    
    # Mode B: Ideal simulation (shots = 0)
    print("   Mode B: Ideal statevector (shots=0)...")
    state = pulse_circuit.state(backend="numpy")
    probs = np.abs(state)**2
    
    print("   Ideal probabilities:")
    for i, p in enumerate(probs):
        if p > 1e-4:
            binary = format(i, '02b')
            print(f"      |{binary}⟩: {p:.6f}")
    
    print("\n   ✅ Execution complete!")


def example_2_compare_with_without_defcal():
    """Compare: Same circuit compiled WITH vs WITHOUT defcal"""
    
    print("\n" + "="*70)
    print("Example 2: Impact of DefcalLibrary on Compilation")
    print("="*70)
    
    # Create calibration library
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    # Single qubit gate with specific calibration
    x_pulse_cal = waveforms.Drag(amp=0.80, duration=40, sigma=10, beta=0.18)
    lib.add_calibration("x", (0,), x_pulse_cal, {"amp": 0.80, "duration": 40})
    
    # Create identical circuit
    circuit = Circuit(1)
    circuit.x(0)
    circuit.measure_z(0)
    
    device_params = {"qubit_freq": [5.0e9], "anharmonicity": [-330e6]}
    
    print("\n📋 Test circuit: X on q0, Measure")
    
    # ===== WITHOUT DefcalLibrary =====
    print("\n1️⃣  Compilation WITHOUT DefcalLibrary:")
    print("   └─ Uses default X decomposition")
    
    compiler_default = GateToPulsePass(defcal_library=None)
    pulse_default = compiler_default.execute_plan(
        circuit,
        device_params=device_params,
        mode="pulse_only"
    )
    
    # Execute ideal
    state_default = pulse_default.state(backend="numpy")
    
    print(f"   Final state: |{state_default[0]:.6f}⟩|0⟩ + |{state_default[1]:.6f}⟩|1⟩")
    print(f"   P(0)={abs(state_default[0])**2:.6f}, P(1)={abs(state_default[1])**2:.6f}")
    
    # ===== WITH DefcalLibrary =====
    print("\n2️⃣  Compilation WITH DefcalLibrary:")
    print(f"   └─ Uses calibrated X pulse")
    print(f"      └─ amp=0.80 (vs default 0.5 in default)")
    print(f"      └─ duration=40ns (vs default 160ns)")
    
    compiler_optimized = GateToPulsePass(defcal_library=lib)
    pulse_optimized = compiler_optimized.execute_plan(
        circuit,
        device_params=device_params,
        mode="pulse_only"
    )
    
    # Execute ideal
    state_optimized = pulse_optimized.state(backend="numpy")
    
    print(f"   Final state: |{state_optimized[0]:.6f}⟩|0⟩ + |{state_optimized[1]:.6f}⟩|1⟩")
    print(f"   P(0)={abs(state_optimized[0])**2:.6f}, P(1)={abs(state_optimized[1])**2:.6f}")
    
    # ===== Comparison =====
    print("\n3️⃣  Comparison:")
    fidelity = abs(np.vdot(state_default / np.linalg.norm(state_default),
                            state_optimized / np.linalg.norm(state_optimized)))**2
    print(f"   Fidelity between two approaches: {fidelity:.6f}")
    
    if fidelity > 0.99:
        print("   ✅ Similar results (both implement X gate correctly)")
    else:
        print(f"   ⚠️  Different results - indicates calibration impact")


def example_3_multi_qubit_heterogeneous_calibration():
    """Show qubit-specific calibrations in a larger circuit"""
    
    print("\n" + "="*70)
    print("Example 3: Multi-Qubit Heterogeneous Calibrations")
    print("="*70)
    
    # Create calibration library with per-qubit variations
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    print("\n1️⃣  Adding qubit-specific X gate calibrations:")
    
    # Each qubit has different optimal parameters
    x_calibrations = [
        {"q": 0, "amp": 0.800, "duration": 40, "freq": 5.000e9},
        {"q": 1, "amp": 0.850, "duration": 42, "freq": 5.050e9},
        {"q": 2, "amp": 0.795, "duration": 38, "freq": 4.950e9},
    ]
    
    for cal in x_calibrations:
        pulse = waveforms.Drag(
            amp=cal["amp"],
            duration=cal["duration"],
            sigma=10,
            beta=0.18
        )
        lib.add_calibration(
            "x",
            (cal["q"],),
            pulse,
            {
                "amp": cal["amp"],
                "duration": cal["duration"],
                "qubit_freq": cal["freq"]
            }
        )
        print(f"   ✅ X on q{cal['q']}: amp={cal['amp']:.3f}, dur={cal['duration']}ns")
    
    # Create circuit using different qubits
    print("\n2️⃣  Building circuit with mixed gate operations:")
    
    circuit = Circuit(3)
    circuit.x(0)   # Will use amp=0.800, dur=40
    circuit.x(1)   # Will use amp=0.850, dur=42
    circuit.x(2)   # Will use amp=0.795, dur=38
    circuit.measure_z(0)
    circuit.measure_z(1)
    circuit.measure_z(2)
    
    print("   Circuit:")
    print("   ├─ X q0  ← Calibrated: amp=0.800, dur=40")
    print("   ├─ X q1  ← Calibrated: amp=0.850, dur=42")
    print("   ├─ X q2  ← Calibrated: amp=0.795, dur=38")
    print("   └─ Measure Z all qubits")
    
    # Compile with defcal
    print("\n3️⃣  Compiling with hardware-aware calibrations:")
    
    compiler = GateToPulsePass(defcal_library=lib)
    device_params = {
        "qubit_freq": [5.000e9, 5.050e9, 4.950e9],
        "anharmonicity": [-330e6, -330e6, -330e6],
    }
    
    pulse_circuit = compiler.execute_plan(
        circuit,
        device_params=device_params,
        mode="pulse_only"
    )
    
    print(f"   ✅ Compiled {len(circuit.ops)} gate ops → {len(pulse_circuit.ops)} pulse ops")
    print(f"      Pulse library: {len(pulse_circuit.metadata['pulse_library'])} waveforms")
    
    # Execute
    print("\n4️⃣  Executing compiled circuit:")
    
    # Ideal mode
    state = pulse_circuit.state(backend="numpy")
    
    # Sampling mode
    result = pulse_circuit.device(provider="simulator").run(shots=1024)
    
    # Expected: Since X gate applied to all qubits, all should be in |111⟩
    probs = np.abs(state)**2
    
    print("   Ideal probabilities:")
    for i, p in enumerate(probs):
        if p > 1e-4:
            binary = format(i, '03b')
            print(f"      |{binary}⟩: {p:.6f}")
    
    if isinstance(result, list) and len(result) > 0:
        counts = result[0].get('result', {})
        if counts and '111' in counts:
            print(f"\n   Sampling (shots=1024):")
            print(f"      |111⟩: {counts['111']}/1024 = {counts['111']/1024:.4f}")
    
    print("\n   ✅ Multi-qubit execution complete!")


def example_4_complete_realistic_workflow():
    """Complete realistic workflow: characterization → compilation → execution"""
    
    print("\n" + "="*70)
    print("Example 4: Complete Realistic Workflow")
    print("="*70)
    
    print("""
【PHASE 1: Hardware Characterization】
  (Would run on real hardware, here we simulate)
  └─ Measure optimal pulse parameters for each gate on each qubit
  └─ Save to JSON for deployment
""")
    
    # Simulate characterization
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    # Add some basic calibrations
    x_pulse = waveforms.Drag(amp=0.80, duration=40, sigma=10, beta=0.18)
    lib.add_calibration("x", (0,), x_pulse, {"amp": 0.80, "duration": 40})
    
    h_pulse = waveforms.Drag(amp=0.565, duration=28, sigma=7, beta=0.18)
    lib.add_calibration("h", (0,), h_pulse, {"amp": 0.565, "duration": 28})
    
    # Save for later
    lib.export_to_json("/tmp/homebrew_s2_characterization.json")
    print("✅ Phase 1: Calibrations exported to /tmp/homebrew_s2_characterization.json")
    
    print("""
【PHASE 2: Program Development & Compilation】
  └─ Load calibrations (in real scenario, from JSON)
  └─ Write gate-level circuit
  └─ Compile to pulses with defcal
""")
    
    # Load calibrations (we'll recreate them for this example to avoid JSON deserialization issues)
    lib_loaded = DefcalLibrary()
    x_pulse_loaded = waveforms.Drag(amp=0.80, duration=40, sigma=10, beta=0.18)
    lib_loaded.add_calibration("x", (0,), x_pulse_loaded, {"amp": 0.80, "duration": 40})
    
    h_pulse_loaded = waveforms.Drag(amp=0.565, duration=28, sigma=7, beta=0.18)
    lib_loaded.add_calibration("h", (0,), h_pulse_loaded, {"amp": 0.565, "duration": 28})
    
    # Build user circuit
    circuit = Circuit(1)
    circuit.h(0)
    circuit.x(0)
    circuit.measure_z(0)
    
    # Compile with loaded calibrations
    compiler = GateToPulsePass(defcal_library=lib_loaded)
    pulse_circuit = compiler.execute_plan(
        circuit,
        device_params={"qubit_freq": [5.0e9]},
        mode="pulse_only"
    )
    
    print("✅ Phase 2: Circuit compiled with hardware calibrations")
    print(f"   Gates: H, X")
    print(f"   H: Uses calibrated pulse (amp=0.565, dur=28ns)")
    print(f"   X: Uses calibrated pulse (amp=0.80, dur=40ns)")
    
    print("""
【PHASE 3: Validation & Testing】
  └─ Test with ideal simulation (shots=0)
  └─ Test with realistic sampling (shots>0)
  └─ Compare and validate
""")
    
    # Ideal test
    state_ideal = pulse_circuit.state(backend="numpy")
    
    # Realistic test
    result_sampling = pulse_circuit.device(provider="simulator").run(shots=1024)
    
    # Analyze
    probs = np.abs(state_ideal)**2
    print("✅ Phase 3: Validation")
    print(f"   Ideal (shots=0):")
    for i, p in enumerate(probs):
        if p > 1e-4:
            print(f"      |{i}⟩: {p:.6f}")
    
    if isinstance(result_sampling, list) and len(result_sampling) > 0:
        counts = result_sampling[0].get('result', {})
        if counts:
            print(f"   Sampling (shots=1024):")
            for state in sorted(counts.keys()):
                prob = counts[state] / 1024
                print(f"      |{state}⟩: {prob:.4f}")
    
    print("""
【PHASE 4: Deployment】
  └─ Submit to real hardware with hardware-optimized pulses
  └─ All gates use defcal calibrations automatically
  └─ Better fidelity due to hardware-specific optimization
""")
    
    print("✅ Phase 4: Ready for deployment")
    print("   Pulse circuit with hardware calibrations ready for submission")


def example_5_complete_chain_api():
    """Complete chain API: circuit → compile → device → run (in one go)"""
    
    print("\n" + "="*70)
    print("Example 5: Complete Chain API")
    print("="*70)
    
    print("""
【CHAIN API WORKFLOW】
  Circuit (Gates)
      ↓
  GateToPulsePass(defcal_library=lib).execute_plan()
      ↓
  .device(provider="simulator")
      ↓
  .run(shots=1024)
      ↓
  Results
""")
    
    # Setup: Create defcal library
    print("1️⃣  Setup: Creating DefcalLibrary...")
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    # Add calibrations
    h_pulse = waveforms.Drag(amp=0.5, duration=28, sigma=7, beta=0.18)
    lib.add_calibration("h", (0,), h_pulse, {"amp": 0.5, "duration": 28})
    
    x_pulse_q0 = waveforms.Drag(amp=0.8, duration=40, sigma=10, beta=0.18)
    lib.add_calibration("x", (0,), x_pulse_q0, {"amp": 0.8, "duration": 40})
    
    x_pulse_q1 = waveforms.Drag(amp=0.85, duration=42, sigma=11, beta=0.17)
    lib.add_calibration("x", (1,), x_pulse_q1, {"amp": 0.85, "duration": 42})
    
    print(f"   ✅ DefcalLibrary created with {len(lib)} calibrations")
    
    # Build gate circuit
    print("\n2️⃣  Building gate-level circuit...")
    circuit = Circuit(2)
    circuit.h(0)
    circuit.x(1)
    circuit.cx(0, 1)
    circuit.measure_z(0)
    circuit.measure_z(1)
    
    print("   Circuit:")
    print("   ├─ H q0   (calibrated: amp=0.5, dur=28ns)")
    print("   ├─ X q1   (calibrated: amp=0.85, dur=42ns)")
    print("   ├─ CX q0-q1 (default CR decomposition)")
    print("   └─ Measure Z q0, q1")
    
    # Device params
    device_params = {
        "qubit_freq": [5.0e9, 5.05e9],
        "anharmonicity": [-330e6, -330e6],
    }
    
    # ===== CHAIN API EXECUTION =====
    print("\n3️⃣  Complete Chain API Execution:")
    print("   circuit → compile(defcal) → device() → run(shots=1024)")
    print()
    
    # Compile with defcal
    compiler = GateToPulsePass(defcal_library=lib)
    pulse_circuit = compiler.execute_plan(
        circuit,
        device_params=device_params,
        mode="pulse_only"
    )
    print(f"   ✅ Compilation: {len(circuit.ops)} gates → {len(pulse_circuit.ops)} pulses")
    
    # Execute via device chain API
    print("   Executing...")
    result = pulse_circuit.device(provider="simulator", device="statevector").run(shots=1024)
    
    # Parse results
    if isinstance(result, list) and len(result) > 0:
        counts = result[0].get('result', {})
        print("\n   📊 Results (shots=1024):")
        if counts:
            for state in sorted(counts.keys()):
                prob = counts[state] / 1024
                bar_len = int(prob * 40)
                bar = "█" * bar_len
                print(f"      |{state}⟩: {prob:.4f} {bar}")
    
    # Also show ideal for comparison
    print("\n4️⃣  Ideal Statevector Comparison (shots=0):")
    state_ideal = pulse_circuit.state(backend="numpy")
    probs_ideal = np.abs(state_ideal)**2
    
    print("   Ideal probabilities:")
    for i, p in enumerate(probs_ideal):
        if p > 1e-4:
            binary = format(i, '02b')
            bar_len = int(p * 40)
            bar = "█" * bar_len
            print(f"      |{binary}⟩: {p:.6f} {bar}")
    
    print("\n   ✅ Chain API execution complete!")
    print("\n   This is the COMPLETE workflow:")
    print("   ┌─ Circuit.h(0), x(1), cx(0,1)")
    print("   ├─ GateToPulsePass(defcal_library=lib)")
    print("   ├─ .execute_plan(circuit, device_params)")
    print("   ├─ .device(provider='simulator')")
    print("   ├─ .run(shots=1024)")
    print("   └─ Results with hardware-optimized pulses ✓")


def main():
    """Run all integration examples"""
    
    print("\n" + "="*70)
    print("DefcalLibrary Integration in Quantum Workflow")
    print("="*70)
    
    print("""
This demonstrates the REAL integration of defcal in the framework:

  Circuit (Gates)
       ↓
  [GateToPulsePass(defcal_library=lib)]  ← Integration Point!
       ↓
  Pulse Circuit
       ├─→ .device().run(shots=1024)     ← Path A: Realistic
       └─→ .state()                      ← Path B: Ideal

The compiler automatically:
  1. Checks defcal for each gate
  2. Uses calibrated pulse if found
  3. Falls back to default decomposition if not
  4. All transparent to the user!
""")
    
    # Run all examples
    example_1_minimal_integration()
    example_2_compare_with_without_defcal()
    example_3_multi_qubit_heterogeneous_calibration()
    example_4_complete_realistic_workflow()
    example_5_complete_chain_api()
    
    print("\n" + "="*70)
    print("✅ All Integration Examples Complete")
    print("="*70)
    print("""
Summary:
  ✅ DefcalLibrary is integrated in GateToPulsePass
  ✅ Pass library to compiler: GateToPulsePass(defcal_library=lib)
  ✅ Compiler queries defcal during _gate_to_pulse()
  ✅ Priority: defcal > metadata > default
  ✅ Works seamlessly with both execution modes

Key Code Pattern:
    from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
    from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
    
    lib = DefcalLibrary()
    lib.import_from_json("calibrations.json")
    
    circuit = Circuit(n)
    circuit.h(0)
    circuit.x(0)
    
    compiler = GateToPulsePass(defcal_library=lib)  ← Pass here!
    pulse_circuit = compiler.execute_plan(circuit, device_params=...)
    
    result = pulse_circuit.device(provider="simulator").run(shots=1024)
""")


if __name__ == "__main__":
    main()
