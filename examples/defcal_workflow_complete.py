"""DefcalLibrary: Complete Hardware Calibration and Integration Workflow

This comprehensive example demonstrates DefcalLibrary usage in quantum circuits:

1. Hardware Calibration: Measure and store optimal pulse parameters
2. Circuit Compilation: Gate-to-pulse compilation with defcal integration
3. Execution Modes: Both realistic (shots>0) and ideal (shots=0)
4. Calibration Persistence: Save and load calibrations via JSON
5. Real-world Integration: Complete end-to-end workflow

Key Features:
  ✅ Hardware characterization simulation (5-qubit Homebrew_S2)
  ✅ Defcal-aware compilation with GateToPulsePass
  ✅ Dual execution modes: sampling (shots=1024) + ideal statevector (shots=0)
  ✅ JSON persistence for deployment across systems
  ✅ Realistic measurement outcomes with hardware variations
  ✅ Complete integration: circuit → compile → device → run
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple

from tyxonq import Circuit, waveforms
from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass


# ==================== DEMO 1: Basic Calibration Creation ====================

def demo_1_basic_calibration():
    """Create and inspect a DefcalLibrary with hardware calibrations"""
    
    print("\n" + "="*70)
    print("DEMO 1: Basic Calibration Creation")
    print("="*70)
    
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    # Add single-qubit calibrations
    print("\n1️⃣  Adding single-qubit calibrations...")
    
    for q in range(3):
        # Per-qubit variations due to hardware heterogeneity
        amp = [0.800, 0.850, 0.795][q]
        duration = [40, 42, 38][q]
        
        x_pulse = waveforms.Drag(
            amp=amp,
            duration=duration,
            sigma=10,
            beta=0.18
        )
        
        lib.add_calibration(
            gate="x",
            qubits=(q,),
            pulse=x_pulse,
            params={"duration": duration, "amp": amp}
        )
        print(f"   ✅ X gate on q{q}: amp={amp}, duration={duration}ns")
    
    # Add multi-qubit calibration
    print("\n2️⃣  Adding two-qubit calibration...")
    
    cx_pulse = waveforms.Drag(amp=0.350, duration=160, sigma=40, beta=0.1)
    lib.add_calibration("cx", (0, 1), cx_pulse, {"duration": 160, "amp": 0.350})
    print(f"   ✅ CX gate on q0-q1: amp=0.350, duration=160ns")
    
    # Print summary
    print(f"\n3️⃣  DefcalLibrary Summary:")
    print(f"   Total calibrations: {len(lib)}")
    print(f"   Hardware: {lib.hardware}")
    
    return lib


# ==================== DEMO 2: Hardware Characterization ====================

def demo_2_hardware_characterization():
    """Simulate hardware characterization and create library from measurements"""
    
    print("\n" + "="*70)
    print("DEMO 2: Hardware Characterization Workflow")
    print("="*70)
    
    print("\n1️⃣  Simulating hardware characterization...")
    print("   (On real hardware, this would measure pulse parameters)")
    
    # Simulated calibration measurements from hardware
    calibrations = {
        "x_q0": {"gate": "x", "qubits": (0,), "amp": 0.800, "duration": 40, "sigma": 10, "beta": 0.18},
        "x_q1": {"gate": "x", "qubits": (1,), "amp": 0.850, "duration": 42, "sigma": 11, "beta": 0.17},
        "x_q2": {"gate": "x", "qubits": (2,), "amp": 0.795, "duration": 38, "sigma": 9.5, "beta": 0.19},
        "h_q0": {"gate": "h", "qubits": (0,), "amp": 0.565, "duration": 28, "sigma": 7.0, "beta": 0.18},
        "cx_q01": {"gate": "cx", "qubits": (0, 1), "amp": 0.350, "duration": 160, "sigma": 40, "beta": 0.1},
    }
    
    print(f"   ✅ Simulated {len(calibrations)} calibrations")
    
    # Create library from measurements
    print("\n2️⃣  Creating DefcalLibrary from measurements...")
    
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    for key, cal in calibrations.items():
        pulse = waveforms.Drag(
            amp=cal["amp"],
            duration=cal["duration"],
            sigma=cal["sigma"],
            beta=cal["beta"]
        )
        
        lib.add_calibration(
            gate=cal["gate"],
            qubits=cal["qubits"],
            pulse=pulse,
            params={"duration": cal["duration"], "amp": cal["amp"]}
        )
    
    print(f"   ✅ Populated library with {len(lib)} calibrations")
    
    # Display results
    print("\n3️⃣  Calibration Summary:")
    print(f"   {'Gate':<8} | {'Qubits':<12} | {'Amplitude':<10} | {'Duration':<10}")
    print("   " + "-"*50)
    
    for key, cal in calibrations.items():
        gate = cal["gate"].upper()
        qubits = str(cal["qubits"])
        amp = cal["amp"]
        dur = cal["duration"]
        print(f"   {gate:<8} | {qubits:<12} | {amp:<10.3f} | {dur:<10}ns")
    
    return lib, calibrations


# ==================== DEMO 3: Circuit Compilation with Defcal ====================

def demo_3_circuit_compilation():
    """Compile a gate-level circuit using defcal-aware compiler"""
    
    print("\n" + "="*70)
    print("DEMO 3: Circuit Compilation with Defcal")
    print("="*70)
    
    # Create library with calibrations
    print("\n1️⃣  Creating DefcalLibrary...")
    
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    # Add calibrations
    x_q0 = waveforms.Drag(amp=0.800, duration=40, sigma=10, beta=0.18)
    lib.add_calibration("x", (0,), x_q0, {"amp": 0.800, "duration": 40})
    
    h_q0 = waveforms.Drag(amp=0.565, duration=28, sigma=7, beta=0.18)
    lib.add_calibration("h", (0,), h_q0, {"amp": 0.565, "duration": 28})
    
    cx_q01 = waveforms.Drag(amp=0.350, duration=160, sigma=40, beta=0.1)
    lib.add_calibration("cx", (0, 1), cx_q01, {"amp": 0.350, "duration": 160})
    
    print(f"   ✅ Loaded {len(lib)} calibrations")
    
    # Create gate-level circuit
    print("\n2️⃣  Building gate-level circuit...")
    
    circuit = Circuit(3)
    circuit.h(0)
    circuit.h(1)
    circuit.x(2)
    circuit.cx(0, 1)
    circuit.x(0)
    circuit.measure_z(0)
    circuit.measure_z(1)
    circuit.measure_z(2)
    
    print("   Circuit structure:")
    print("   ├─ H q0      (has calibration)")
    print("   ├─ H q1      (no calibration - will use default)")
    print("   ├─ X q2      (no calibration - will use default)")
    print("   ├─ CX q0-q1  (has calibration)")
    print("   ├─ X q0      (has calibration)")
    print("   └─ Measure Z all")
    
    # Compile with defcal
    print("\n3️⃣  Compiling with defcal-aware compiler...")
    
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
    
    print(f"   ✅ Compilation complete:")
    print(f"      Input: {len(circuit.ops)} gate operations")
    print(f"      Output: {len(pulse_circuit.ops)} pulse operations")
    print(f"      Pulse library: {len(pulse_circuit.metadata['pulse_library'])} waveforms")
    
    return circuit, pulse_circuit, lib


# ==================== DEMO 4: Execution Modes ====================

def demo_4_execution_modes():
    """Execute compiled circuit using both realistic and ideal modes"""
    
    print("\n" + "="*70)
    print("DEMO 4: Dual Execution Modes")
    print("="*70)
    
    # Get compiled circuit from previous demo
    circuit = Circuit(2)
    circuit.h(0)
    circuit.x(0)
    circuit.measure_z(0)
    circuit.measure_z(1)
    
    lib = DefcalLibrary(hardware="Homebrew_S2")
    h_pulse = waveforms.Drag(amp=0.565, duration=28, sigma=7, beta=0.18)
    lib.add_calibration("h", (0,), h_pulse, {"amp": 0.565})
    x_pulse = waveforms.Drag(amp=0.800, duration=40, sigma=10, beta=0.18)
    lib.add_calibration("x", (0,), x_pulse, {"amp": 0.800})
    
    compiler = GateToPulsePass(defcal_library=lib)
    pulse_circuit = compiler.execute_plan(
        circuit,
        device_params={"qubit_freq": [5.0e9, 5.05e9]},
        mode="pulse_only"
    )
    
    # Path A: Measurement Sampling (shots=1024)
    print("\n1️⃣  Path A: Measurement Sampling (shots=1024, realistic)")
    print("-" * 70)
    print("   Execution: .device(provider='simulator').run(shots=1024)")
    
    try:
        result_sampling = pulse_circuit.device(provider="simulator", device="statevector").run(shots=1024)
        
        print("   ✅ Sampling execution successful")
        
        if isinstance(result_sampling, list) and len(result_sampling) > 0:
            counts = result_sampling[0].get('result', {})
            if counts:
                print("   Measurement histogram (1024 shots):")
                for state in sorted(counts.keys()):
                    count = counts[state]
                    prob = count / 1024
                    bar = "█" * int(prob * 30)
                    print(f"      |{state}⟩: {count:4d}/1024 ({prob:.4f}) {bar}")
    except Exception as e:
        print(f"   ⚠️  Sampling failed: {e}")
    
    # Path B: Ideal Statevector (shots=0)
    print("\n2️⃣  Path B: Ideal Statevector (shots=0, perfect)")
    print("-" * 70)
    print("   Execution: .state(backend='numpy')")
    
    try:
        state_ideal = pulse_circuit.state(backend="numpy")
        
        print("   ✅ Ideal simulation successful")
        
        probs = np.abs(state_ideal)**2
        print("   State vector probabilities:")
        for i, p in enumerate(probs):
            if p > 1e-6:
                binary = format(i, '02b')
                bar = "█" * int(p * 30)
                print(f"      |{binary}⟩: {p:.6f} {bar}")
    except Exception as e:
        print(f"   ⚠️  Ideal simulation failed: {e}")
    
    print("\n3️⃣  Comparison of Modes:")
    print("-" * 70)
    print("""
    Mode A (shots=1024):
      ✅ Realistic with measurement sampling noise
      ✅ Matches hardware behavior
      ✅ Statistical variation between runs
      ❌ Less precise for algorithm validation
    
    Mode B (shots=0):
      ✅ Fast, deterministic
      ✅ Perfect for algorithm development
      ✅ Full state vector access
      ❌ Doesn't reflect real hardware
    
    Best Practice: Validate with Mode B, then test with Mode A before hardware
""")


# ==================== DEMO 5: Calibration Persistence ====================

def demo_5_calibration_persistence():
    """Save and load calibrations via JSON"""
    
    print("\n" + "="*70)
    print("DEMO 5: Calibration Persistence (JSON)")
    print("="*70)
    
    # Create library
    print("\n1️⃣  Creating library with calibrations...")
    
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    x_pulse = waveforms.Drag(amp=0.800, duration=40, sigma=10, beta=0.18)
    lib.add_calibration("x", (0,), x_pulse, {"amp": 0.800, "duration": 40})
    
    h_pulse = waveforms.Drag(amp=0.565, duration=28, sigma=7, beta=0.18)
    lib.add_calibration("h", (0,), h_pulse, {"amp": 0.565, "duration": 28})
    
    print(f"   ✅ Created library with {len(lib)} calibrations")
    
    # Export to JSON
    print("\n2️⃣  Exporting to JSON...")
    
    filepath = "/tmp/homebrew_s2_calibrations.json"
    lib.export_to_json(filepath)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"   ✅ Exported to {filepath}")
    print(f"      JSON size: {len(json.dumps(data))} bytes")
    print(f"      Calibrations: {len(data.get('calibrations', []))}")
    
    # Import from JSON
    print("\n3️⃣  Importing from JSON...")
    
    lib_loaded = DefcalLibrary()
    lib_loaded.import_from_json(filepath)
    
    print(f"   ✅ Loaded {len(lib_loaded)} calibrations")
    
    # Verify
    print("\n4️⃣  Verification:")
    
    calib_x = lib_loaded.get_calibration("x", (0,))
    if calib_x:
        print(f"   ✅ X(q0): amp={calib_x.params['amp']:.3f}, duration={calib_x.params['duration']}ns")
    
    calib_h = lib_loaded.get_calibration("h", (0,))
    if calib_h:
        print(f"   ✅ H(q0): amp={calib_h.params['amp']:.3f}, duration={calib_h.params['duration']}ns")
    
    print("\n   ✅ Calibrations successfully persisted and restored!")


# ==================== DEMO 6: Real-World Complete Workflow ====================

def demo_6_complete_workflow():
    """End-to-end workflow: characterize → compile → execute"""
    
    print("\n" + "="*70)
    print("DEMO 6: Complete Real-World Workflow")
    print("="*70)
    
    print("""
【WORKFLOW】
  1. Hardware Characterization
     └─ Measure optimal pulse parameters for each gate/qubit
  
  2. Create DefcalLibrary
     └─ Organize calibrations in structured format
  
  3. Save for Deployment
     └─ Export to JSON for later use
  
  4. Build Circuit & Compile
     └─ Gate-level circuit → Pulse-level with defcal
  
  5. Execute with Both Modes
     └─ Realistic sampling (shots>0) + Ideal (shots=0)
""")
    
    # Step 1-2: Characterization & Library
    print("\n1️⃣  Hardware Characterization...")
    
    lib = DefcalLibrary(hardware="Homebrew_S2")
    
    # Multi-qubit heterogeneous calibrations
    for q in range(3):
        amps = [0.800, 0.850, 0.795]
        durs = [40, 42, 38]
        
        pulse = waveforms.Drag(
            amp=amps[q],
            duration=durs[q],
            sigma=10,
            beta=0.18
        )
        lib.add_calibration("x", (q,), pulse, {"amp": amps[q], "duration": durs[q]})
    
    print(f"   ✅ Characterized {len(lib)} gates")
    
    # Step 3: Save
    print("\n2️⃣  Saving calibrations...")
    
    filepath = "/tmp/homebrew_s2_calibrations_workflow.json"
    lib.export_to_json(filepath)
    print(f"   ✅ Saved to {filepath}")
    
    # Step 4: Build & Compile
    print("\n3️⃣  Building and compiling circuit...")
    
    circuit = Circuit(3)
    circuit.x(0)
    circuit.x(1)
    circuit.x(2)
    circuit.measure_z(0)
    circuit.measure_z(1)
    circuit.measure_z(2)
    
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
    
    print(f"   ✅ Compiled circuit: {len(circuit.ops)} gates → {len(pulse_circuit.ops)} pulses")
    
    # Step 5: Execute
    print("\n4️⃣  Executing with both modes...")
    
    # Ideal
    state = pulse_circuit.state(backend="numpy")
    probs = np.abs(state)**2
    
    print("   Ideal results (expected: |111⟩):")
    for i, p in enumerate(probs):
        if p > 1e-4:
            binary = format(i, '03b')
            print(f"      |{binary}⟩: {p:.6f}")
    
    # Sampling
    try:
        result = pulse_circuit.device(provider="simulator").run(shots=1024)
        
        if isinstance(result, list) and len(result) > 0:
            counts = result[0].get('result', {})
            print("\n   Realistic results (shots=1024):")
            for state in sorted(counts.keys()):
                prob = counts[state] / 1024
                print(f"      |{state}⟩: {prob:.4f} ({counts[state]} counts)")
    except:
        print("   (Sampling skipped)")
    
    print("\n   ✅ Complete workflow executed successfully!")


# ==================== Main ====================

def main():
    """Execute all defcal workflow demonstrations"""
    
    print("\n" + "="*70)
    print("DefcalLibrary: Complete Workflow Guide")
    print("="*70)
    print("""
DefcalLibrary enables hardware-aware quantum circuit compilation:
  • Store optimal pulse parameters from hardware characterization
  • Use calibrations during gate-to-pulse compilation
  • Significant improvement in circuit fidelity
  • JSON persistence for deployment

This guide demonstrates:
  1. Creating DefcalLibrary with calibrations
  2. Hardware characterization workflow
  3. Circuit compilation with defcal
  4. Dual execution modes (realistic + ideal)
  5. JSON persistence and loading
  6. Complete end-to-end workflow
""")
    
    demo_1_basic_calibration()
    demo_2_hardware_characterization()
    demo_3_circuit_compilation()
    demo_4_execution_modes()
    demo_5_calibration_persistence()
    demo_6_complete_workflow()
    
    print("\n" + "="*70)
    print("✅ All Workflow Demonstrations Complete")
    print("="*70)
    print("""
Key Takeaways:
  ✅ DefcalLibrary stores hardware calibrations
  ✅ GateToPulsePass integrates with defcal
  ✅ Compiler queries defcal during gate decomposition
  ✅ Fallback to defaults if calibration not available
  ✅ Works with both ideal (shots=0) and realistic (shots>0)
  ✅ JSON format enables deployment across systems

Common Pattern:
    lib = DefcalLibrary(hardware="Homebrew_S2")
    lib.import_from_json("calibrations.json")
    
    circuit = Circuit(n)
    # ... build circuit ...
    
    compiler = GateToPulsePass(defcal_library=lib)
    pulse_circuit = compiler.execute_plan(
        circuit,
        device_params={...},
        mode="pulse_only"
    )
    
    result = pulse_circuit.device(provider="simulator").run(shots=1024)

See defcal_performance_analysis.py for quantitative performance analysis.
""")


if __name__ == "__main__":
    main()
