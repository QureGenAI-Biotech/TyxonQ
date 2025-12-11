Hybrid Mode Integration: Practical Guide to Mixed Gate-Pulse Programming
=========================================================================

This tutorial teaches you how to effectively use TyxonQ's hybrid mode to build quantum circuits that leverage both high-level gate abstractions and low-level pulse optimizations.

Prerequisites
-------------

- Familiarity with quantum circuits and gates
- Understanding of pulse basics (see `Pulse Programming Basics`_)
- Knowledge of DefcalLibrary (see `DefcalLibrary API`_)

.. _Pulse Programming Basics: ../beginner/pulse_basics.html
.. _DefcalLibrary API: ../user_guide/pulse/defcal_library.html

Motivation: Why Hybrid Mode?
----------------------------

Consider these scenarios:

**Scenario 1: Algorithm Development**

You're developing a VQE algorithm. Gates are perfect for expressing the ansatz clearly:

.. code-block:: python

   circuit.ry(0, theta)
   circuit.ry(1, phi)
   circuit.cx(0, 1)

But you want higher fidelity. Hybrid mode lets you calibrate critical gates without rewriting the entire circuit.

**Scenario 2: Gradual Hardware Optimization**

You have a working gate-level circuit. You profiled it and found that CX gates are the bottleneck. Instead of converting everything to pulses, you can:

1. Keep most gates as-is (clear, simple)
2. Calibrate just CX gates (optimized, precise)
3. Let the compiler handle the rest (automated)

**Scenario 3: Team Collaboration**

- Algorithm team writes gate-level circuits
- Hardware team provides calibrations via DefcalLibrary
- Both teams work independently
- Compiler integrates everything automatically

Architecture of Hybrid Compilation
----------------------------------

When you compile a hybrid circuit, here's what happens:

.. code-block:: text

   1. User Input:
      Circuit (gates) + DefcalLibrary (calibrations)
         ↓
   2. Compiler (GateToPulsePass):
      For each gate operation:
        ├─ Check: Is there a calibration in DefcalLibrary?
        │  └─→ YES: Use calibrated pulse
        │  └─→ NO: Use default decomposition
        ↓
   3. Output:
      Pulse Circuit (all gates converted to pulses)
         ↓
   4. Execution:
      .device().run() → Realistic sampling or .state() → Ideal vector

The key insight: **Calibrations are queried at compile time, not execution time.**

Tutorial: Building Your First Hybrid Circuit
---------------------------------------------

Step 1: Hardware Characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, characterize your hardware and measure optimal pulses:

.. code-block:: python

   # This step typically runs once on real hardware
   from tyxonq import waveforms
   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   
   # Create calibration library
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   # After hardware characterization, you'd have measured pulses
   # For this example, we'll use typical values
   
   # Measure X gate on q0 (example result: needs high amplitude)
   x_pulse_q0 = waveforms.Drag(
       amp=0.82,       # Measured from characterization
       duration=38,    # Optimized duration
       sigma=9.5,      # Measured sigma
       beta=0.19       # Measured beta
   )
   lib.add_calibration(
       "x", (0,),
       x_pulse_q0,
       params={"amp": 0.82, "duration": 38}
   )
   
   # Export for deployment
   lib.export_to_json("homebrew_s2_calibrations.json")
   
   print("✅ Calibrations saved for deployment")

**Output:**

File `homebrew_s2_calibrations.json` now contains your hardware-specific calibrations.

Step 2: Load Calibrations
~~~~~~~~~~~~~~~~~~~~~~~~~~

In your quantum algorithm code, load the pre-measured calibrations:

.. code-block:: python

   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   
   # Load calibrations measured in Step 1
   lib = DefcalLibrary()
   lib.import_from_json("homebrew_s2_calibrations.json")
   
   # Verify calibrations loaded
   print(f"Loaded {len(lib)} calibrations")
   print(lib.summary())

Step 3: Build Gate-Level Circuit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write your quantum algorithm using high-level gates:

.. code-block:: python

   from tyxonq import Circuit
   
   # Build circuit with gates (clean, clear intent)
   circuit = Circuit(2)
   
   # Ansatz structure (VQE example)
   theta = 0.5  # Example parameter
   
   circuit.ry(0, theta)      # Parameterized rotation
   circuit.cx(0, 1)          # Entangling gate
   circuit.ry(1, -theta)     # Inverse rotation
   
   circuit.measure_z(0)
   circuit.measure_z(1)
   
   print("\nCircuit (gate-level):")
   print("  RY(θ) q0")
   print("  CX q0, q1")
   print("  RY(-θ) q1")
   print("  Measure Z")

Step 4: Compile with Hybrid Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compile the circuit with DefcalLibrary. The compiler automatically applies calibrations:

.. code-block:: python

   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   
   # Create compiler with defcal library
   compiler = GateToPulsePass(defcal_library=lib)
   
   # Define device parameters
   device_params = {
       "qubit_freq": [5.0e9, 5.05e9],       # Qubit frequencies
       "anharmonicity": [-330e6, -330e6],   # Anharmonicity
   }
   
   # Compile circuit to pulses
   # The compiler automatically:
   # ✅ Queries defcal for each gate
   # ✅ Uses calibrated pulse where available
   # ✅ Uses default decomposition otherwise
   pulse_circuit = compiler.execute_plan(
       circuit,
       device_params=device_params,
       mode="pulse_only"
   )
   
   print(f"\n✅ Compilation successful:")
   print(f"   Input gates: {len(circuit.ops)}")
   print(f"   Output pulses: {len(pulse_circuit.ops)}")
   print(f"   Pulse library: {len(pulse_circuit.metadata['pulse_library'])} waveforms")

Step 5: Execute and Validate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute with both realistic and ideal modes:

.. code-block:: python

   import numpy as np
   
   # ===== Mode A: Realistic Sampling (production) =====
   print("\n【Mode A】Realistic sampling (shots=1024)")
   print("-" * 50)
   
   result = pulse_circuit.device(provider="simulator", device="statevector").run(shots=1024)
   
   if isinstance(result, list) and len(result) > 0:
       counts = result[0].get('result', {})
       
       print("Measurement outcomes:")
       for state in sorted(counts.keys()):
           prob = counts[state] / 1024
           bar = "█" * int(prob * 40)
           print(f"  |{state}⟩: {prob:.4f} {bar}")
       
       # Calculate statistics
       probs = np.array([counts.get(f"{i:02b}", 0) / 1024 for i in range(4)])
       entropy = -np.sum(probs[probs > 1e-10] * np.log2(probs[probs > 1e-10]))
       print(f"\nMeasurement entropy: {entropy:.4f} bits")
   
   # ===== Mode B: Ideal Reference (validation) =====
   print("\n【Mode B】Ideal statevector (shots=0)")
   print("-" * 50)
   
   state = pulse_circuit.state(backend="numpy")
   probs = np.abs(state)**2
   
   print("Ideal probabilities:")
   for i, p in enumerate(probs):
       if p > 1e-4:
           binary = format(i, '02b')
           print(f"  |{binary}⟩: {p:.6f}")
   
   # ===== Validation =====
   print("\n【Validation】")
   print("-" * 50)
   
   # Calculate fidelity
   if counts:
       sampling_probs = np.array([counts.get(f"{i:02b}", 0) / 1024 for i in range(4)])
       
       # JS divergence
       m = (sampling_probs + probs) / 2
       mask1 = sampling_probs > 1e-10
       mask2 = probs > 1e-10
       
       kl1 = np.sum(sampling_probs[mask1] * np.log(sampling_probs[mask1] / m[mask1]))
       kl2 = np.sum(probs[mask2] * np.log(probs[mask2] / m[mask2]))
       js_div = (kl1 + kl2) / 2
       
       print(f"JS Divergence: {js_div:.6f}")
       if js_div < 0.01:
           print("✅ Excellent: Sampling matches ideal very closely")
       elif js_div < 0.05:
           print("✅ Good: Sampling closely follows ideal distribution")
       else:
           print("⚠️  Notable deviation: May need recalibration")

Advanced: Comparing Hybrid vs Gate-Only
---------------------------------------

To understand the benefit of hybrid mode, compare execution with and without calibrations:

.. code-block:: python

   # ===== WITHOUT calibrations (baseline) =====
   compiler_baseline = GateToPulsePass(defcal_library=None)
   pulse_baseline = compiler_baseline.execute_plan(
       circuit,
       device_params=device_params,
       mode="pulse_only"
   )
   
   # ===== WITH calibrations (hybrid) =====
   compiler_hybrid = GateToPulsePass(defcal_library=lib)
   pulse_hybrid = compiler_hybrid.execute_plan(
       circuit,
       device_params=device_params,
       mode="pulse_only"
   )
   
   # Execute both
   result_baseline = pulse_baseline.device(provider="simulator").run(shots=2048)
   result_hybrid = pulse_hybrid.device(provider="simulator").run(shots=2048)
   
   # Analyze
   counts_baseline = result_baseline[0].get('result', {})
   counts_hybrid = result_hybrid[0].get('result', {})
   
   print("Comparison:")
   print("-" * 50)
   
   for state in ['00', '01', '10', '11']:
       prob_baseline = counts_baseline.get(state, 0) / 2048
       prob_hybrid = counts_hybrid.get(state, 0) / 2048
       
       improvement = (prob_hybrid - prob_baseline) * 100
       arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
       
       print(f"|{state}⟩ Baseline: {prob_baseline:.4f}  Hybrid: {prob_hybrid:.4f}  {arrow} {abs(improvement):.1f}%")
   
   # Fidelity improvement
   print("\n" + "="*50)
   print("Fidelity Analysis:")
   print("="*50)
   
   state_ideal = pulse_hybrid.state(backend="numpy")
   
   fidelity_baseline = calculate_fidelity(state_ideal, counts_baseline)
   fidelity_hybrid = calculate_fidelity(state_ideal, counts_hybrid)
   
   print(f"Fidelity (baseline): {fidelity_baseline:.4f}")
   print(f"Fidelity (hybrid):   {fidelity_hybrid:.4f}")
   print(f"Improvement:         {(fidelity_hybrid - fidelity_baseline)*100:.2f}%")

Realistic Application: VQE with Hybrid Mode
-------------------------------------------

Here's how to apply hybrid mode in a real VQE optimization loop:

.. code-block:: python

   from tyxonq import Circuit, waveforms
   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   import numpy as np
   
   # ===== Setup =====
   # Load calibrations
   lib = DefcalLibrary()
   lib.import_from_json("vqe_calibrations.json")
   
   device_params = {
       "qubit_freq": [5.0e9, 5.05e9, 4.95e9],
       "anharmonicity": [-330e6, -330e6, -330e6],
   }
   
   # VQE parameters
   num_qubits = 3
   num_layers = 2
   
   def build_ansatz(params):
       """Build hybrid VQE ansatz."""
       circuit = Circuit(num_qubits)
       
       # Parameterized rotations (single-qubit)
       for layer in range(num_layers):
           for q in range(num_qubits):
               theta = params[layer * num_qubits + q]
               circuit.ry(q, theta)  # ← Will use calibrated pulse if available!
           
           # Entanglement (two-qubit)
           for q in range(num_qubits - 1):
               circuit.cx(q, q + 1)  # ← Will use default CR
       
       # Measurement
       for q in range(num_qubits):
           circuit.measure_z(q)
       
       return circuit
   
   def evaluate_ansatz(params):
       """Evaluate VQE ansatz with hybrid compilation."""
       circuit = build_ansatz(params)
       
       # Compile with hybrid mode
       compiler = GateToPulsePass(defcal_library=lib)
       pulse_circuit = compiler.execute_plan(
           circuit,
           device_params=device_params,
           mode="pulse_only"
       )
       
       # Execute
       result = pulse_circuit.device(provider="simulator").run(shots=2048)
       counts = result[0].get('result', {})
       
       # Compute expectation value (example: ZZ expectation)
       expect_zz = 0
       for bitstring, count in counts.items():
           z0, z1 = int(bitstring[0]), int(bitstring[1])
           expect_zz += (1 if z0 == z1 else -1) * count
       
       expect_zz /= 2048
       
       return expect_zz
   
   # ===== Optimization Loop =====
   from scipy.optimize import minimize
   
   initial_params = np.random.randn(num_layers * num_qubits)
   
   # Optimize with hybrid-compiled circuits
   result = minimize(
       evaluate_ansatz,
       initial_params,
       method='COBYLA',
       options={'maxiter': 100}
   )
   
   print(f"✅ VQE optimization complete:")
   print(f"   Optimal energy: {result.fun:.6f}")
   print(f"   Optimal parameters: {result.x}")

Best Practices Summary
----------------------

1. **Profile First**
   
   .. code-block:: python
   
      # Measure which gates are bottlenecks
      # Only calibrate critical ones
      profiled_gates = profile_circuit_execution(circuit)
      critical_gates = get_top_k_expensive(profiled_gates, k=3)
      
      for gate_name, qubits in critical_gates:
          lib.add_calibration(gate_name, qubits, measured_pulse, params)

2. **Validate Improvements**
   
   .. code-block:: python
   
      # Always measure fidelity improvement
      fidelity_baseline = run_and_measure(circuit, lib=None)
      fidelity_hybrid = run_and_measure(circuit, lib=lib)
      
      if fidelity_hybrid <= fidelity_baseline:
          revert_calibrations()  # Not worth it

3. **Iterate Gradually**
   
   .. code-block:: python
   
      # Start with gate-only
      # Add one calibration at a time
      # Measure improvement
      # Continue if beneficial
      
      for gate in critical_gates:
          lib.add_calibration(gate, ...)
          improvement = measure_improvement()
          if improvement < threshold:
              lib.remove_calibration(gate)

4. **Production Deployment**
   
   .. code-block:: python
   
      # Always include validation
      lib = load_calibrations("production.json")
      assert lib.validate()
      
      circuit = build_algorithm()
      compiler = GateToPulsePass(defcal_library=lib)
      pulse_circuit = compiler.execute_plan(circuit, device_params)
      
      # Deploy to hardware
      submit_to_cloud(pulse_circuit)

See Also
--------

- :doc:`../user_guide/pulse/hybrid_mode` - Hybrid mode user guide
- :doc:`../user_guide/pulse/defcal_library` - DefcalLibrary reference
- :doc:`../user_guide/pulse/index` - Complete pulse guide

Example Files
~~~~~~~~~~~~~

- ``examples/pulse_hybrid_mode_complete.py`` - 5 complete examples
- ``examples/defcal_integration_in_workflow.py`` - DefcalLibrary workflow
- ``examples/pulse_mode_a_chain_compilation.py`` - Chain compilation
