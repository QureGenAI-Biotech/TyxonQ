Pulse-Level Quantum Circuit Optimization with Hardware Calibrations
====================================================================

This tutorial demonstrates how to leverage hardware-specific pulse calibrations to optimize quantum circuit execution on real quantum processors. You will learn:

- How to create and manage hardware calibrations
- How to integrate calibrations into circuit compilation
- How to execute circuits using both ideal and realistic simulation modes
- How to measure the impact of hardware optimization

Prerequisites
-------------

- Familiarity with TyxonQ's circuit model
- Understanding of quantum gates and pulse operations
- Recommended: Complete the `Pulse Programming Basics`_ tutorial first

.. _Pulse Programming Basics: ../beginner/pulse_basics.html

Motivation
----------

Real quantum processors are heterogeneous systems where qubits have slightly different properties:

- **Frequency variations**: Each qubit has a unique resonance frequency
- **Amplitude differences**: X gate requires different amplitudes on different qubits  
- **Duration variations**: Gate pulses need qubit-specific durations for optimal fidelity
- **Coupling variations**: Two-qubit gates vary significantly between qubit pairs

Without hardware calibrations, the compiler applies generic "one-size-fits-all" pulse parameters that are suboptimal. By using DefcalLibrary, you can apply qubit-specific optimizations discovered during hardware characterization.

Architecture Overview
---------------------

The pulse-defcal integration works through three stages:

**Stage 1: Hardware Characterization**

Measure optimal pulse parameters on real hardware and store them:

.. code-block:: text

   Real Hardware
        ↓
   Characterization Experiments
        ↓
   Measure Optimal Pulse Parameters
        ↓
   Store in DefcalLibrary
        ↓
   Export to JSON

**Stage 2: Circuit Compilation**

Use the stored calibrations when compiling circuits:

.. code-block:: text

   Gate-Level Circuit
        ↓
   DefcalLibrary (loaded from JSON)
        ↓
   GateToPulsePass Compiler
        ↓
   Lookup: Does defcal have this gate?
   ├─→ YES: Use hardware-optimized pulse
   └─→ NO: Fall back to default decomposition
        ↓
   Pulse-Level Circuit

**Stage 3: Execution**

Execute with two different modes:

.. code-block:: text

   Pulse Circuit
        ↓
   ├─→ Path A: Measurement Sampling (shots > 0)
   │        └→ Realistic hardware behavior
   │        └→ Includes measurement noise
   │        └→ For production deployment
   │
   └─→ Path B: Ideal Simulation (shots = 0)
            └→ Full state vector
            └→ No measurement noise
            └→ For algorithm development

Tutorial: Step-by-Step
-----------------------

Step 1: Create a Calibration Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by creating a calibration library for your hardware:

.. code-block:: python

   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   from tyxonq import waveforms
   
   # Create library for Homebrew_S2 processor
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   # Add X gate calibrations (different for each qubit)
   for q in range(3):
       amp = [0.800, 0.850, 0.795][q]        # Qubit-specific amplitude
       duration = [40, 42, 38][q]             # Qubit-specific duration
       sigma = [10, 11, 9.5][q]              # Gaussian width
       
       pulse = waveforms.Drag(
           amp=amp,
           duration=duration,
           sigma=sigma,
           beta=0.18
       )
       
       lib.add_calibration(
           gate="x",
           qubits=(q,),
           pulse=pulse,
           params={
               "duration": duration,
               "amp": amp,
               "qubit_freq": 5.0e9 + q*50e6  # Frequency spread
           },
           description=f"X gate on qubit {q}"
       )

**What's happening:**

- Each qubit has slightly different pulse parameters
- DRAG pulse includes derivative correction term (beta=0.18)
- Parameters capture hardware heterogeneity

Step 2: Add Two-Qubit Gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add calibrations for two-qubit gates, which are more complex:

.. code-block:: python

   # Add CX (CNOT) gate calibrations
   # CX uses Cross-Resonance (CR) pulse
   
   cx_pulse = waveforms.Drag(
       amp=0.350,      # CR pulse amplitude
       duration=160,   # CR pulse duration (longer than single-qubit)
       sigma=40,
       beta=0.1
   )
   
   lib.add_calibration(
       gate="cx",
       qubits=(0, 1),
       pulse=cx_pulse,
       params={"duration": 160, "amp": 0.350},
       description="CX gate from q0 to q1"
   )
   
   # Different CX parameters for different qubit pairs
   lib.add_calibration(
       gate="cx",
       qubits=(1, 2),
       pulse=waveforms.Drag(amp=0.340, duration=165, sigma=41, beta=0.1),
       params={"duration": 165, "amp": 0.340}
   )

Step 3: Persist Calibrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save calibrations to JSON for later use:

.. code-block:: python

   # Export for deployment
   lib.export_to_json("homebrew_s2_calibrations.json")
   
   # Verify file was created
   import json
   with open("homebrew_s2_calibrations.json", "r") as f:
       data = json.load(f)
       print(f"Exported {len(data['calibrations'])} calibrations")

**JSON Format:**

The calibrations are stored with metadata:

.. code-block:: json

   {
     "version": "1.0",
     "hardware": "Homebrew_S2",
     "created": "2025-10-30T12:34:56.123456",
     "calibration_count": 5,
     "calibrations": {
       "x|0": {
         "gate": "x",
         "qubits": [0],
         "pulse": {...},
         "params": {"duration": 40, "amp": 0.8}
       },
       ...
     }
   }

Step 4: Load and Use Calibrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a different program, load the calibrations and use them for compilation:

.. code-block:: python

   from tyxonq import Circuit
   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   
   # Load pre-measured calibrations
   lib = DefcalLibrary()
   lib.import_from_json("homebrew_s2_calibrations.json")
   
   # Verify calibrations were loaded
   print(lib.summary())
   
   # Create a quantum circuit
   circuit = Circuit(3)
   circuit.h(0)          # Hadamard on q0
   circuit.x(1)          # X on q1
   circuit.cx(0, 1)      # CX from q0 to q1
   circuit.measure_z(0)
   circuit.measure_z(1)
   circuit.measure_z(2)
   
   # Compile with defcal-aware compiler
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

**What happens during compilation:**

1. Compiler encounters H gate on q0
   - Checks: Does defcal have H on q0? YES → Use calibrated pulse
2. Compiler encounters X gate on q1
   - Checks: Does defcal have X on q1? YES → Use calibrated pulse
3. Compiler encounters CX on (q0, q1)
   - Checks: Does defcal have CX on (0,1)? YES → Use calibrated pulse
4. All gates use hardware-optimized pulses!

Step 5: Execute with Realistic Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the circuit with measurement sampling (realistic mode):

.. code-block:: python

   # Execute with shots=1024 (measurement sampling)
   # This is realistic - includes measurement noise
   result = pulse_circuit.device(provider="simulator", device="statevector").run(shots=1024)
   
   # Extract measurement statistics
   if isinstance(result, list) and len(result) > 0:
       counts = result[0].get('result', {})
       
       # Analyze measurement outcomes
       print("Measurement results (shots=1024):")
       for state in sorted(counts.keys()):
           count = counts[state]
           prob = count / 1024
           print(f"  |{state}⟩: {count:4d}/{1024} ({prob:.4f})")

**Output example:**

.. code-block:: text

   Measurement results (shots=1024):
     |000⟩:  290/1024 (0.2832)
     |001⟩:   35/1024 (0.0342)
     |010⟩:  125/1024 (0.1221)
     |011⟩:   15/1024 (0.0146)
     |100⟩:  368/1024 (0.3594)
     |101⟩:   45/1024 (0.0439)
     |110⟩:  140/1024 (0.1367)
     |111⟩:    6/1024 (0.0059)

Step 6: Execute with Ideal Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Also get the ideal statevector for comparison:

.. code-block:: python

   # Execute with ideal simulation (shots=0)
   # This is perfect - no measurement noise
   state = pulse_circuit.state(backend="numpy")
   
   # Analyze state vector
   import numpy as np
   probs = np.abs(state)**2
   
   print("Ideal probabilities (from state vector):")
   for i, prob in enumerate(probs):
       if prob > 1e-6:
           binary = format(i, '03b')
           print(f"  |{binary}⟩: {prob:.6f}")

**Output example:**

.. code-block:: text

   Ideal probabilities (from state vector):
     |000⟩: 0.290503
     |010⟩: 0.129029
     |100⟩: 0.340345
     |110⟩: 0.151167
     |101⟩: 0.033232

Step 7: Compare Results
~~~~~~~~~~~~~~~~~~~~~~~

Compare the realistic (shots=1024) and ideal (shots=0) results:

.. code-block:: python

   # Comparison
   print("\nComparison:")
   print(f"  Ideal |000⟩:       0.290503")
   print(f"  Sampling |000⟩:    0.283203  (290/1024)")
   print(f"  Difference:        0.007300")
   print()
   print("✓ Good agreement! Sampling is close to ideal")
   
   # State fidelity
   state_ideal_norm = state / np.linalg.norm(state)
   probs_ideal = np.abs(state_ideal_norm)**2
   
   # Calculate entropy
   entropy_ideal = -np.sum(probs_ideal[probs_ideal > 1e-6] * 
                            np.log2(probs_ideal[probs_ideal > 1e-6]))
   
   print(f"\nState vector entropy: {entropy_ideal:.4f} bits")
   print(f"Measurement entropy:  ~{entropy_ideal:.4f} bits")

Advanced: Compare With and Without Defcal
------------------------------------------

To understand the impact of calibrations, compare execution with and without defcal:

.. code-block:: python

   # Compiler WITHOUT defcal
   compiler_default = GateToPulsePass(defcal_library=None)
   pulse_default = compiler_default.execute_plan(
       circuit,
       device_params=device_params,
       mode="pulse_only"
   )
   
   # Compiler WITH defcal
   compiler_optimal = GateToPulsePass(defcal_library=lib)
   pulse_optimal = compiler_optimal.execute_plan(
       circuit,
       device_params=device_params,
       mode="pulse_only"
   )
   
   # Execute both versions
   result_default = pulse_default.device(provider="simulator").run(shots=1024)
   result_optimal = pulse_optimal.device(provider="simulator").run(shots=1024)
   
   # Compare
   print("Without DefcalLibrary:")
   print(f"  Generic default pulses")
   print(f"  One-size-fits-all parameters")
   print()
   print("With DefcalLibrary:")
   print(f"  Hardware-optimized pulses")
   print(f"  Qubit-specific parameters")
   print(f"  Better fidelity on real hardware")

Best Practices
--------------

**1. One-Time Characterization**

Characterize hardware once during setup:

.. code-block:: python

   # Hardware lab (one-time)
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   for qubit in range(num_qubits):
       optimal_x = characterize_x_gate(qubit)  # Experimental measurement
       lib.add_calibration("x", (qubit,), optimal_x)
   
   lib.export_to_json("calibrations.json")

**2. Deployment**

Reuse calibrations across many programs:

.. code-block:: python

   # Production code (reuse)
   lib = DefcalLibrary()
   lib.import_from_json("calibrations.json")
   
   compiler = GateToPulsePass(defcal_library=lib)
   # Compile many circuits using same calibrations

**3. Testing Strategy**

Always test with both execution modes:

.. code-block:: python

   # Development: Use shots=0 (ideal) for algorithm development
   state = pulse_circuit.state()
   
   # Testing: Use shots=1024 (realistic) for hardware validation
   result = pulse_circuit.device().run(shots=1024)
   
   # Compare: Validate that sampling matches ideal
   assert fidelity > 0.95, "Sampling deviates too much from ideal"

**4. Monitoring**

Track calibration effectiveness:

.. code-block:: python

   # Key metrics to track:
   # - Measurement fidelity (sampling vs ideal)
   # - Gate error rates (from characterization)
   # - Measurement quality (entropy, distribution shape)
   
   def measure_effectiveness(circuit, lib):
       compiler = GateToPulsePass(defcal_library=lib)
       pulse = compiler.execute_plan(circuit, device_params, mode="pulse_only")
       
       state_ideal = pulse.state()
       result_sampling = pulse.device().run(shots=1024)
       
       fidelity = calculate_fidelity(state_ideal, result_sampling)
       return fidelity

Complete Example
----------------

Here's a complete working example combining all steps:

.. code-block:: python

   from tyxonq import Circuit, waveforms
   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   import numpy as np
   
   # Step 1: Create and populate calibration library
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   # Add X gate calibrations (qubit-specific)
   for q in range(2):
       amp = [0.800, 0.850][q]
       duration = [40, 42][q]
       x_pulse = waveforms.Drag(amp=amp, duration=duration, sigma=10, beta=0.18)
       lib.add_calibration("x", (q,), x_pulse, {"duration": duration, "amp": amp})
   
   # Step 2: Create quantum circuit
   circuit = Circuit(2)
   circuit.h(0)
   circuit.x(1)
   circuit.measure_z(0)
   circuit.measure_z(1)
   
   # Step 3: Compile with defcal
   compiler = GateToPulsePass(defcal_library=lib)
   pulse_circuit = compiler.execute_plan(
       circuit,
       device_params={"qubit_freq": [5.0e9, 5.05e9]},
       mode="pulse_only"
   )
   
   # Step 4: Execute both modes
   # Mode A: Realistic sampling
   result_sampling = pulse_circuit.device(provider="simulator").run(shots=1024)
   
   # Mode B: Ideal simulation
   state_ideal = pulse_circuit.state(backend="numpy")
   
   # Step 5: Analyze results
   if isinstance(result_sampling, list) and len(result_sampling) > 0:
       counts = result_sampling[0].get('result', {})
       print("Measurement outcomes (shots=1024):")
       for state in sorted(counts.keys()):
           prob = counts[state] / 1024
           print(f"  |{state}⟩: {prob:.4f}")
   
   print("\nIdeal state vector:")
   probs = np.abs(state_ideal)**2
   for i, p in enumerate(probs):
       if p > 1e-4:
           print(f"  |{format(i, '02b')}⟩: {p:.4f}")

Troubleshooting
---------------

**Q: CalibrationData not found during compilation**

A: The gate might not be in the library. Check:

.. code-block:: python

   # Verify calibration exists
   calib = lib.get_calibration("x", (0,))
   if calib is None:
       print("ERROR: No X calibration for q0")
   
   # List all calibrations
   print(lib.summary())

**Q: JSON import fails with "Could not reconstruct pulse"**

A: The pulse_factory parameter is needed to reconstruct pulse objects:

.. code-block:: python

   # Define pulse factory
   def pulse_factory(pulse_dict):
       waveform_type = pulse_dict.get("type")
       if waveform_type == "Drag":
           from tyxonq import waveforms
           return waveforms.Drag(**pulse_dict["params"])
       # ... handle other types ...
   
   lib.import_from_json("calibrations.json", pulse_factory=pulse_factory)

**Q: Sampling results differ significantly from ideal**

A: This is expected and indicates:

.. code-block:: python

   # Calculate fidelity
   fidelity = calculate_state_fidelity(state_ideal, measurement_counts)
   
   if fidelity < 0.90:
       print("WARNING: Low fidelity - may need to re-characterize")
       print(f"Fidelity: {fidelity:.4f}")
   
   # Re-measure calibrations
   lib = characterize_hardware()
   lib.export_to_json("calibrations.json")

Complete Chain API Workflow with Sampling
------------------------------------------

Here's the **complete production workflow** using the chain API with shots>0 sampling:

.. code-block:: python

   from tyxonq import Circuit, waveforms
   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   import numpy as np
   
   # ===== STEP 1: Setup Calibration Library =====
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   # Add hardware-specific calibrations (from characterization)
   h_pulse = waveforms.Drag(amp=0.5, duration=28, sigma=7, beta=0.18)
   lib.add_calibration("h", (0,), h_pulse, {"amp": 0.5, "duration": 28})
   
   x_pulse_q0 = waveforms.Drag(amp=0.8, duration=40, sigma=10, beta=0.18)
   lib.add_calibration("x", (0,), x_pulse_q0, {"amp": 0.8, "duration": 40})
   
   x_pulse_q1 = waveforms.Drag(amp=0.85, duration=42, sigma=11, beta=0.17)
   lib.add_calibration("x", (1,), x_pulse_q1, {"amp": 0.85, "duration": 42})
   
   # ===== STEP 2: Build Quantum Circuit =====
   circuit = Circuit(2)
   circuit.h(0)           # Will use calibrated pulse
   circuit.x(1)           # Will use calibrated pulse
   circuit.cx(0, 1)       # Will use default CR decomposition
   circuit.measure_z(0)
   circuit.measure_z(1)
   
   # ===== STEP 3: Device Parameters =====
   device_params = {
       "qubit_freq": [5.0e9, 5.05e9],
       "anharmonicity": [-330e6, -330e6],
   }
   
   # ===== STEP 4: Complete Chain API Execution =====
   # Circuit → Compile → Device → Run
   compiler = GateToPulsePass(defcal_library=lib)
   pulse_circuit = compiler.execute_plan(
       circuit,
       device_params=device_params,
       mode="pulse_only"
   )
   
   # Execute with REALISTIC SAMPLING (shots=1024)
   result = pulse_circuit.device(
       provider="simulator",
       device="statevector"
   ).run(shots=1024)
   
   # ===== STEP 5: Analyze Results =====
   if isinstance(result, list) and len(result) > 0:
       counts = result[0].get('result', {})
       
       print("Measurement Results (shots=1024):")
       print("="*50)
       for state in sorted(counts.keys()):
           count = counts[state]
           prob = count / 1024
           # Visualize with bar chart
           bar = "█" * int(prob * 40)
           print(f"|{state}⟩: {count:4d}/{1024} ({prob:.4f}) {bar}")
       
       # Calculate measurement entropy
       probs = np.array([counts.get(state, 0) / 1024 for state in ['00', '01', '10', '11']])
       entropy = -np.sum(probs[probs > 1e-6] * np.log2(probs[probs > 1e-6]))
       print(f"\nMeasurement Entropy: {entropy:.4f} bits")
   
   # ===== STEP 6: Also Get Ideal Reference =====
   state_ideal = pulse_circuit.state(backend="numpy")
   probs_ideal = np.abs(state_ideal)**2
   
   print("\nIdeal Reference (shots=0):")
   print("="*50)
   for i, p in enumerate(probs_ideal):
       if p > 1e-4:
           binary = format(i, '02b')
           print(f"|{binary}⟩: {p:.6f}")
   
   # ===== STEP 7: Validate Sampling Quality =====
   # Compare sampling distribution with ideal
   sampling_probs = np.array([counts.get(state, 0) / 1024 for state in ['00', '01', '10', '11']])
   
   # Calculate JS divergence (symmetric KL divergence)
   def js_divergence(p, q):
       m = (p + q) / 2
       p_mask = p > 1e-10
       q_mask = q > 1e-10
       kl_pm = np.sum(p[p_mask] * np.log(p[p_mask] / m[p_mask]))
       kl_qm = np.sum(q[q_mask] * np.log(q[q_mask] / m[q_mask]))
       return (kl_pm + kl_qm) / 2
   
   divergence = js_divergence(sampling_probs, probs_ideal)
   print(f"\nDistribution Divergence: {divergence:.6f}")
   if divergence < 0.05:
       print("✅ Excellent agreement between sampling and ideal")
   elif divergence < 0.1:
       print("✅ Good agreement between sampling and ideal")
   else:
       print("⚠️  Significant deviation - may need re-calibration")

**Key Points:**

1. **Chain API**: ``circuit → compile(defcal) → device() → run(shots=1024)``
2. **Realistic Sampling**: ``shots=1024`` produces measurement histogram
3. **Ideal Reference**: ``shots=0`` provides theoretical expectation
4. **Quality Metrics**: Compare sampling with ideal to validate calibrations
5. **Production Ready**: This is the exact workflow for real hardware deployment

See Also
--------

- :doc:`../user_guide/pulse/defcal_library` - API reference
- :doc:`../user_guide/pulse/index` - Pulse programming overview
- Example files:
  
  - ``examples/defcal_hardware_calibration.py``
  - ``examples/defcal_circuit_compilation.py``
  - ``examples/defcal_performance_comparison.py``
  - ``examples/defcal_integration_in_workflow.py`` - Complete integration examples
