Hybrid Mode: Mixing Gates and Pulses
====================================

Overview
--------

Hybrid mode allows you to mix gate-level and pulse-level operations in the same quantum circuit. This provides maximum flexibility by letting you:

- Use **gates** (H, X, CX, etc.) for high-level circuit structure and clarity
- Use **pulses** for hardware-optimized, calibrated operations
- Compile both seamlessly together
- Gradually optimize your circuits incrementally

This is particularly useful when you want to:

- Start with gate-level circuits for simplicity
- Add hardware calibrations only for performance-critical operations
- Balance code readability with execution precision
- Migrate existing gate-based code without complete rewrite

Key Concepts
------------

Three Execution Modes
~~~~~~~~~~~~~~~~~~~~~

**1. Gate-Only Mode** (Baseline)

All operations are expressed as gates. The compiler automatically decomposes them to pulses:

.. code-block:: python

   circuit = Circuit(2)
   circuit.h(0)       # Generic Hadamard
   circuit.cx(0, 1)   # Generic CNOT
   circuit.measure_z(0)

Advantages:
   - Simple and clear intent
   - Generic pulse parameters
   - No need for calibrations

Disadvantages:
   - Uses default pulse parameters
   - May not optimal for specific hardware
   - Cannot apply calibrations

**2. Hybrid Mode** (Mixed Abstraction) ⭐

Mix gates with hardware-calibrated operations:

.. code-block:: python

   lib = DefcalLibrary()
   lib.add_calibration("x", (0,), calibrated_pulse, params)
   
   circuit = Circuit(2)
   circuit.h(0)              # Generic gate
   circuit.x(0)              # Calibrated gate (from defcal)
   circuit.cx(0, 1)          # Generic gate
   circuit.measure_z(0)

Advantages:
   - Clear circuit structure (gates are self-documenting)
   - Hardware optimization where needed
   - Selective calibration reduces characterization cost
   - Flexible: easy to add/remove calibrations

Disadvantages:
   - Requires DefcalLibrary setup
   - Mixed abstraction level

**3. Pulse-Only Mode** (Full Control)

All operations expressed as pulses for maximum precision:

.. code-block:: python

   prog = PulseProgram(2)
   prog.add_pulse(0, drag_pulse_1, qubit_freq=5.0e9)
   prog.add_pulse(0, drag_pulse_2, qubit_freq=5.0e9)
   prog.add_pulse((0, 1), cr_pulse, qubit_freq=5.0e9, drive_freq=5.05e9)

Advantages:
   - Maximum hardware control
   - Precise pulse parameter tuning
   - Optimal gate fidelity

Disadvantages:
   - Complex to write
   - Requires detailed hardware knowledge
   - Difficult to maintain

**Comparison Table**

.. table::

   ========== =========== ======== ========
   Criterion  Gate-Only   Hybrid   Pulse   
   ========== =========== ======== ========
   Simplicity High        Medium   Low
   Clarity    Excellent   Good     Poor
   Precision  Generic     Optimized Optimal
   Overhead   Zero        Selective Full
   ========== =========== ======== ========

Compilation Priority
~~~~~~~~~~~~~~~~~~~~

When compiling a hybrid circuit, the compiler uses this priority:

1. **DefcalLibrary** (user-provided calibrations) ← HIGHEST
2. **Circuit metadata** (legacy calibration format)
3. **Default decomposition** (physics-based gates) ← LOWEST

This ensures hardware-specific calibrations override defaults:

.. code-block:: python

   # Compiler looks for calibration in this order:
   
   # 1. Check DefcalLibrary (highest priority)
   calib = defcal_library.get_calibration("x", (0,))
   if calib:
       use calib.pulse  # ✅ Use calibrated pulse
   
   # 2. Check circuit metadata
   elif "x" in circuit.metadata["pulse_calibrations"]:
       use calibration_from_metadata
   
   # 3. Use default decomposition
   else:
       use default_x_gate_decomposition()

Using Hybrid Mode
-----------------

Step-by-Step Guide
~~~~~~~~~~~~~~~~~~

**Step 1: Create DefcalLibrary**

.. code-block:: python

   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   from tyxonq import waveforms
   
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   # Add calibrations for performance-critical gates
   x_pulse = waveforms.Drag(amp=0.8, duration=40, sigma=10, beta=0.18)
   lib.add_calibration("x", (0,), x_pulse, {"amp": 0.8, "duration": 40})

**Step 2: Build Circuit with Gates**

.. code-block:: python

   from tyxonq import Circuit
   
   circuit = Circuit(2)
   circuit.h(0)              # High-level gate
   circuit.x(0)              # Will use calibrated pulse if available
   circuit.cx(0, 1)          # Will use default if no calibration
   circuit.measure_z(0)

**Step 3: Compile with DefcalLibrary**

.. code-block:: python

   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   
   compiler = GateToPulsePass(defcal_library=lib)
   
   device_params = {
       "qubit_freq": [5.0e9, 5.05e9],
       "anharmonicity": [-330e6, -330e6],
   }
   
   pulse_circuit = compiler.execute_plan(
       circuit,
       device_params=device_params,
       mode="pulse_only"  # Converts all gates to pulses
   )

**Step 4: Execute**

.. code-block:: python

   # Mode A: Realistic sampling (production)
   result = pulse_circuit.device(provider="simulator").run(shots=1024)
   
   # Mode B: Ideal reference (validation)
   state = pulse_circuit.state(backend="numpy")

Complete Example
~~~~~~~~~~~~~~~~

Here's a complete example of hybrid mode in action:

.. code-block:: python

   from tyxonq import Circuit, waveforms
   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   import numpy as np
   
   # ===== STEP 1: Setup Calibrations =====
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   # Calibrate critical gates only
   x_pulse_q0 = waveforms.Drag(amp=0.8, duration=40, sigma=10, beta=0.18)
   lib.add_calibration("x", (0,), x_pulse_q0, {"amp": 0.8, "duration": 40})
   
   x_pulse_q1 = waveforms.Drag(amp=0.85, duration=42, sigma=11, beta=0.17)
   lib.add_calibration("x", (1,), x_pulse_q1, {"amp": 0.85, "duration": 42})
   
   # ===== STEP 2: Build Hybrid Circuit =====
   circuit = Circuit(2)
   circuit.h(0)              # Generic H (uses default)
   circuit.x(0)              # X on q0 (uses calibration!)
   circuit.x(1)              # X on q1 (uses calibration!)
   circuit.cx(0, 1)          # CX (uses default)
   circuit.measure_z(0)
   circuit.measure_z(1)
   
   # ===== STEP 3: Compile with Hybrid =====
   compiler = GateToPulsePass(defcal_library=lib)
   
   pulse_circuit = compiler.execute_plan(
       circuit,
       device_params={
           "qubit_freq": [5.0e9, 5.05e9],
           "anharmonicity": [-330e6, -330e6],
       },
       mode="pulse_only"
   )
   
   # ===== STEP 4: Execute =====
   # Realistic sampling
   result = pulse_circuit.device(provider="simulator").run(shots=1024)
   
   counts = result[0].get('result', {})
   for state, count in sorted(counts.items()):
       prob = count / 1024
       print(f"|{state}⟩: {prob:.4f}")
   
   # Ideal reference
   state = pulse_circuit.state(backend="numpy")
   print(f"\nIdeal state vector: {state}")

Best Practices
--------------

**1. Selective Calibration Strategy**

Don't calibrate everything. Focus on gates that matter:

.. code-block:: python

   lib = DefcalLibrary()
   
   # ✅ Calibrate performance-critical gates
   lib.add_calibration("x", (0,), optimized_x_pulse, params)  # Often used
   lib.add_calibration("cx", (0, 1), optimized_cx_pulse, params)  # Bottleneck
   
   # ✗ Skip gates that aren't bottlenecks
   # Don't calibrate: rare gates, gates with good default performance

**Benefits:**
   - 80% reduction in characterization cost
   - Faster development cycle
   - Easier maintenance

**2. Gradual Migration**

Start simple, add optimization incrementally:

.. code-block:: python

   # Phase 1: Pure gate circuit (baseline)
   circuit = Circuit(2)
   circuit.h(0)
   circuit.cx(0, 1)
   
   # Phase 2: Add calibrations for H
   lib.add_calibration("h", (0,), calibrated_h, params)
   
   # Phase 3: Add calibrations for CX
   lib.add_calibration("cx", (0, 1), calibrated_cx, params)
   
   # Compiler automatically benefits from improvements!

**3. Performance Validation**

Always compare hybrid vs gate-only:

.. code-block:: python

   # Without calibrations (baseline)
   compiler_baseline = GateToPulsePass(defcal_library=None)
   pulse_baseline = compiler_baseline.execute_plan(circuit, device_params)
   
   # With calibrations (hybrid)
   compiler_hybrid = GateToPulsePass(defcal_library=lib)
   pulse_hybrid = compiler_hybrid.execute_plan(circuit, device_params)
   
   # Execute both and compare
   result_baseline = pulse_baseline.device().run(shots=1024)
   result_hybrid = pulse_hybrid.device().run(shots=1024)
   
   # Analyze improvement in fidelity
   fidelity_gain = calculate_fidelity(result_baseline, result_hybrid)
   if fidelity_gain > threshold:
       deploy_calibrations()

**4. Production Checklist**

Before deploying hybrid circuits:

.. code-block:: python

   lib = load_calibrations("production_calibrations.json")
   
   # ✅ Validate all calibrations
   assert lib.validate(), "Invalid calibrations!"
   
   # ✅ Test with both execution modes
   result_sampling = circuit.device().run(shots=1024)
   state_ideal = circuit.state()
   
   # ✅ Verify fidelity improvement
   fidelity = measure_fidelity(state_ideal, result_sampling)
   assert fidelity > min_fidelity, f"Low fidelity: {fidelity}"
   
   # ✅ Ready for deployment
   submit_to_hardware()

Common Patterns
---------------

**Pattern 1: VQE with Hybrid Ansatz**

.. code-block:: python

   lib = DefcalLibrary()
   
   # Calibrate parameterized rotations (used many times)
   for q in range(num_qubits):
       lib.add_calibration("y", (q,), optimized_ry_pulse, params)
   
   # Build VQE ansatz with gates
   circuit = Circuit(num_qubits)
   for q in range(num_qubits):
       circuit.ry(q, theta[q])  # Uses calibrated pulse!
   
   for q in range(num_qubits - 1):
       circuit.cx(q, q+1)  # Uses default CX
   
   # Compile with hybrid
   compiler = GateToPulsePass(defcal_library=lib)
   pulse_circuit = compiler.execute_plan(circuit, device_params)

**Pattern 2: Calibration-Only for Bottleneck**

.. code-block:: python

   lib = DefcalLibrary()
   
   # Only calibrate the two-qubit gate (bottleneck)
   lib.add_calibration("cx", (0, 1), optimized_cr_pulse, params)
   
   # All other gates use defaults
   circuit = Circuit(3)
   circuit.h(0)
   circuit.h(1)
   circuit.h(2)
   circuit.cx(0, 1)      # ← Calibrated (critical!)
   circuit.ry(2, theta)
   circuit.measure_z(2)
   
   # Compile and execute
   compiler = GateToPulsePass(defcal_library=lib)
   pulse_circuit = compiler.execute_plan(circuit, device_params)

**Pattern 3: Hybrid with DefcalLibrary JSON**

.. code-block:: python

   # Load pre-measured calibrations
   lib = DefcalLibrary()
   lib.import_from_json("hardware_calibrations.json")
   
   # Use in circuit
   circuit = build_user_circuit()
   
   # Compile automatically applies calibrations
   compiler = GateToPulsePass(defcal_library=lib)
   pulse_circuit = compiler.execute_plan(circuit, device_params)
   
   # Deploy to real hardware
   submit_to_cloud(pulse_circuit)

Troubleshooting
---------------

**Problem: Calibration not applied**

If your calibrated gate still uses default pulse:

.. code-block:: python

   # Check 1: Is the gate name correct?
   lib.add_calibration("x", (0,), pulse, params)  # lowercase "x"
   # ✗ Wrong: uppercase "X"
   # ✗ Wrong: different gate name like "rx"
   
   # Check 2: Are the qubits correct?
   lib.add_calibration("x", (0,), pulse, params)
   circuit.x(0)  # ✓ Correct: matches qubit 0
   # ✗ Wrong: circuit.x(1) doesn't match
   
   # Check 3: Validate calibration exists
   assert lib.has_calibration("x", (0,)), "Calibration not found!"
   
   # Check 4: Verify compilation uses defcal
   compiler = GateToPulsePass(defcal_library=lib)
   assert compiler.defcal_library is not None

**Problem: Performance degradation with hybrid**

If hybrid mode performs worse than expected:

.. code-block:: python

   # Verify calibrations are correct
   state_ideal = lib.get_calibration("x", (0,)).pulse.state()
   # Compare with experimental measurement
   
   # Try with fewer calibrations
   lib_minimal = DefcalLibrary()
   lib_minimal.add_calibration("x", (0,), pulse, params)
   # Test with only the most critical gate
   
   # Measure fidelity improvement
   fidelity_gain = compare_execution(with_defcal, without_defcal)
   if fidelity_gain < threshold:
       revert_calibrations()

See Also
--------

- :doc:`defcal_library` - Detailed DefcalLibrary API reference
- :doc:`index` - Complete pulse programming overview
- :doc:`../../tutorials/advanced/pulse_defcal_integration` - Integration tutorial
- :doc:`../../tutorials/advanced/pulse_hybrid_mode_integration` - Hybrid mode tutorial

Example Files
~~~~~~~~~~~~~

For complete working examples, see:

- ``examples/pulse_hybrid_mode_complete.py`` - 5 comprehensive examples
- ``examples/defcal_integration_in_workflow.py`` - DefcalLibrary integration
- ``examples/pulse_mode_a_chain_compilation.py`` - Chain compilation modes
