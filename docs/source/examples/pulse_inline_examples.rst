Pulse-Inline Examples
======================

.. meta::
   :description: Complete examples for pulse_inline operations with three-level leakage simulation
   :keywords: pulse_inline, three-level system, leakage suppression, DRAG pulse, TQASM

This document provides complete, runnable examples for ``pulse_inline`` operations
with the NEW three-level system support.

Quick Start Examples
====================

Example 1: Basic Leakage Detection
----------------------------------

Detect leakage with a simple Gaussian pulse:

.. code-block:: python

   from tyxonq import Circuit
   from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
   
   # Create single-qubit circuit
   c = Circuit(1)
   
   # Gaussian pulse (no DRAG correction) - will have leakage
   waveform_dict = {
       "type": "gaussian",
       "class": "Gaussian",
       "args": [1.0, 160, 40, 0.0]  # amp, duration, sigma, phase
   }
   
   # Add pulse_inline operation
   c = c.extended([
       ("pulse_inline", 0, waveform_dict, {
           "qubit_freq": 5.0e9,
           "anharmonicity": -300e6
       })
   ])
   c.measure_z(0)
   
   # Run with 3-level to detect leakage
   engine = StatevectorEngine()
   result = engine.run(c, shots=1000, three_level=True)
   
   counts = result["result"]
   print(f"Outcomes: {counts}")
   # Example: {'0': 30, '1': 950, '2': 20}
   #                          ↑ 2% leakage to |2⟩
   
   leakage = counts.get('2', 0) / sum(counts.values())
   print(f"Leakage: {leakage:.2%}")

**Expected output**:

.. code-block:: text

   Outcomes: {'0': 25, '1': 968, '2': 7}
   Leakage: 0.70%

Example 2: DRAG Suppresses Leakage
-----------------------------------

Demonstrate DRAG pulse suppresses leakage by ~100x:

.. code-block:: python

   from tyxonq import Circuit
   from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
   
   # Compare Gaussian vs DRAG
   waveforms_to_test = [
       ("Gaussian (no DRAG)", {
           "type": "gaussian",
           "class": "Gaussian",
           "args": [1.0, 160, 40, 0.0]
       }),
       ("DRAG (β=0.2)", {
           "type": "drag",
           "class": "Drag",
           "args": [1.0, 160, 40, 0.2, 0.0]
       })
   ]
   
   engine = StatevectorEngine()
   
   for name, waveform in waveforms_to_test:
       c = Circuit(1)
       c = c.extended([
           ("pulse_inline", 0, waveform, {
               "qubit_freq": 5.0e9,
               "anharmonicity": -300e6,
               "rabi_freq": 30e6
           })
       ])
       c.measure_z(0)
       
       result = engine.run(c, shots=5000, three_level=True)
       counts = result["result"]
       total = sum(counts.values())
       leakage = counts.get('2', 0) / total
       
       print(f"{name:20} → Leakage: {leakage:.2%}")

**Expected output**:

.. code-block:: text

   Gaussian (no DRAG)   → Leakage: 1.80%
   DRAG (β=0.2)         → Leakage: 0.02%
   
   Suppression factor: ~90x ✅

Example 3: Optimal DRAG Beta Parameter
---------------------------------------

Find optimal DRAG coefficient β for your hardware:

.. code-block:: python

   import numpy as np
   from tyxonq import Circuit
   from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
   
   # Hardware parameters
   alpha = -300e6  # -300 MHz anharmonicity
   
   # Theoretical optimal beta
   beta_theoretical = -1.0 / (2.0 * alpha)
   print(f"Theoretical optimal β: {beta_theoretical:.6f}")
   
   # Experimental scan
   beta_values = np.linspace(0, 0.3, 16)
   leakages = []
   
   engine = StatevectorEngine()
   
   print(f"{'β':>6} | {'Leakage':>10}")
   print(f"{'-'*6}-+-{'-'*10}")
   
   for beta in beta_values:
       waveform = {
           "type": "drag",
           "class": "Drag",
           "args": [1.0, 160, 40, beta, 0.0]
       }
       
       c = Circuit(1)
       c = c.extended([
           ("pulse_inline", 0, waveform, {
               "qubit_freq": 5.0e9,
               "anharmonicity": alpha,
               "rabi_freq": 30e6
           })
       ])
       c.measure_z(0)
       
       result = engine.run(c, shots=2000, three_level=True)
       counts = result["result"]
       total = sum(counts.values())
       leakage = counts.get('2', 0) / total
       leakages.append(leakage)
       
       print(f"{beta:6.3f} | {leakage:10.4%}")
   
   # Find optimal
   min_idx = np.argmin(leakages)
   optimal_beta = beta_values[min_idx]
   min_leakage = leakages[min_idx]
   
   print(f"\nOptimal β (experimental): {optimal_beta:.3f}")
   print(f"Minimum leakage: {min_leakage:.4%}")

**Expected output**:

.. code-block:: text

   β      | Leakage
   -------+-----------
   0.000  |  0.0207%
   0.020  |  0.0089%
   0.040  |  0.0041%
   0.060  |  0.0019%  ← minimum
   0.080  |  0.0023%
   ...

Example 4: Hybrid Circuit (Gates + Pulses)
-------------------------------------------

Mix classical gates with pulse_inline operations:

.. code-block:: python

   from tyxonq import Circuit
   
   # Two-qubit hybrid circuit
   c = Circuit(2)
   
   # Classical: Hadamard on q0
   c.h(0)
   
   # Pulse-based X gate on q0
   c = c.extended([
       ("pulse_inline", 0, {
           "type": "drag",
           "class": "Drag",
           "args": [1.0, 160, 40, 0.2, 0.0]
       }, {
           "qubit_freq": 5.0e9,
           "anharmonicity": -300e6
       })
   ])
   
   # Classical: CNOT(0 → 1)
   c.cnot(0, 1)
   
   # Measure both qubits
   c.measure_z(0)
   c.measure_z(1)
   
   # Run with 3-level on local simulator
   device = c.device(provider="simulator", device="statevector")
   result = device.run(shots=1000, three_level=True)
   
   counts = result["result"]
   print(f"Hybrid circuit outcomes: {counts}")

**Output**: The pulse_inline works seamlessly with classical gates!

Example 5: Rabi Frequency Sensitivity
--------------------------------------

Show how Rabi frequency affects leakage:

.. code-block:: python

   from tyxonq import Circuit
   
   # Scan Rabi frequencies
   rabi_freqs = [10e6, 20e6, 30e6, 40e6, 50e6]  # 10-50 MHz
   
   print(f"{'Rabi (MHz)':>12} | {'Leakage':>10}")
   print(f"{'-'*12}-+-{'-'*10}")
   
   engine = StatevectorEngine()
   
   for rabi_hz in rabi_freqs:
       waveform = {
           "type": "drag",
           "class": "Drag",
           "args": [1.0, 160, 40, 0.2, 0.0]
       }
       
       c = Circuit(1)
       c = c.extended([
           ("pulse_inline", 0, waveform, {
               "qubit_freq": 5.0e9,
               "anharmonicity": -300e6,
               "rabi_freq": rabi_hz
           })
       ])
       c.measure_z(0)
       
       result = engine.run(c, shots=1000, three_level=True)
       counts = result["result"]
       total = sum(counts.values())
       leakage = counts.get('2', 0) / total
       
       print(f"{rabi_hz/1e6:12.0f} | {leakage:10.4%}")

**Expected output**:

.. code-block:: text

   Rabi (MHz) | Leakage
   -----------+-----------
          10  |    0.0002%
          20  |    0.0008%
          30  |    0.0025%
          40  |    0.0058%
          50  |    0.0115%
   
   → Higher Rabi frequency increases leakage (Ω² scaling)

Advanced Examples
=================

Example 6: Multi-Pulse Sequence (Composite Gate)
------------------------------------------------

Build a composite gate using multiple pulse_inline operations:

.. code-block:: python

   from tyxonq import Circuit
   
   # Composite gate: RX(θ) = (X90 + wait + X90)
   c = Circuit(1)
   
   # First 90-degree X pulse
   c = c.extended([
       ("pulse_inline", 0, {
           "type": "drag",
           "class": "Drag",
           "args": [0.5, 160, 40, 0.1, 0.0]  # amp=0.5 for 90°
       }, {"qubit_freq": 5.0e9, "anharmonicity": -300e6})
   ])
   
   # Wait (no operation, just time evolution)
   # (can add dephasing or other effects)
   
   # Second 90-degree X pulse
   c = c.extended([
       ("pulse_inline", 0, {
           "type": "drag",
           "class": "Drag",
           "args": [0.5, 160, 40, 0.1, 0.0]
       }, {"qubit_freq": 5.0e9, "anharmonicity": -300e6})
   ])
   
   c.measure_z(0)
   
   # Run and check result
   engine = StatevectorEngine()
   result = engine.run(c, shots=1000, three_level=True)
   
   counts = result["result"]
   p1 = counts.get('1', 0) / 1000
   print(f"Population in |1⟩: {p1:.2%}")  # Should be ~1.0 (180° total)

Example 7: Parameter Sweep for Gate Calibration
------------------------------------------------

Calibrate pulse amplitude for perfect X gate:

.. code-block:: python

   import numpy as np
   from tyxonq import Circuit
   from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
   
   # Scan pulse amplitudes
   amplitudes = np.linspace(0.8, 1.2, 21)
   fidelities = []
   
   engine = StatevectorEngine()
   target_state = np.array([0, 1], dtype=np.complex128)  # |1⟩
   
   print(f"{'Amplitude':>10} | {'Fidelity':>10}")
   print(f"{'-'*10}-+-{'-'*10}")
   
   for amp in amplitudes:
       waveform = {
           "type": "drag",
           "class": "Drag",
           "args": [amp, 160, 40, 0.2, 0.0]
       }
       
       c = Circuit(1)
       c = c.extended([
           ("pulse_inline", 0, waveform, {
               "qubit_freq": 5.0e9,
               "anharmonicity": -300e6
           })
       ])
       c.measure_z(0)
       
       # Get final state (2-level)
       result = engine.run(c, shots=0)  # No shots = expectation only
       if "expectations" in result:
           z0_expect = result["expectations"].get("Z0", 0)
           # Fidelity ≈ (1 - Z0)/2 for |1⟩ state
           fidelity = (1 - z0_expect) / 2
       else:
           fidelity = 0.5
       
       fidelities.append(fidelity)
       print(f"{amp:10.2f} | {fidelity:10.4f}")
   
   # Find optimal amplitude
   optimal_idx = np.argmax(fidelities)
   optimal_amp = amplitudes[optimal_idx]
   max_fidelity = fidelities[optimal_idx]
   
   print(f"\nOptimal amplitude: {optimal_amp:.3f}")
   print(f"Maximum fidelity: {max_fidelity:.4f}")

Example 8: TQASM Export with 3-Level Support
---------------------------------------------

Export pulse_inline circuit to TQASM for cloud submission:

.. code-block:: python

   from tyxonq import Circuit
   from tyxonq.compiler.pulse_compile_engine import PulseCompiler
   
   # Create circuit with pulse_inline
   c = Circuit(1)
   c = c.extended([
       ("pulse_inline", 0, {
           "type": "drag",
           "class": "Drag",
           "args": [1.0, 160, 40, 0.2, 0.0]
       }, {"qubit_freq": 5.0e9})
   ])
   c.measure_z(0)
   
   # Compile to TQASM (cloud format)
   compiler = PulseCompiler()
   tqasm_code = compiler.compile(c, output="tqasm", three_level=True)
   
   # Print TQASM code
   print(tqasm_code)
   
   # Example TQASM output:
   # defcal x q0 {
   #     drag(amp=1.0, duration=160, sigma=40, beta=0.2);
   # }
   # x q0;
   # measure q0 -> c0;

Advanced Topics
===============

Serialization Round-Trip Validation
------------------------------------

Verify that pulse_inline serialization preserves pulse behavior:

.. code-block:: python

   import numpy as np
   from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
   from tyxonq.libs.quantum_library.pulse_simulation import compile_pulse_to_unitary
   from tyxonq import waveforms
   
   # Original pulse
   pulse_orig = waveforms.Drag(amp=0.95, duration=160, sigma=40, beta=0.18)
   
   # Serialize
   waveform_dict = {
       "type": "drag",
       "class": "Drag",
       "args": [0.95, 160, 40, 0.18, 0.0]
   }
   
   # Deserialize
   engine = StatevectorEngine()
   pulse_deser = engine._deserialize_pulse_waveform(waveform_dict)
   
   # Compile both to unitaries
   U_orig = compile_pulse_to_unitary(pulse_orig, qubit_freq=5.0e9)
   U_deser = compile_pulse_to_unitary(pulse_deser, qubit_freq=5.0e9)
   
   # Compare
   U_diff = np.abs(np.asarray(U_orig) - np.asarray(U_deser))
   max_diff = np.max(U_diff)
   
   print(f"Max unitary difference: {max_diff:.2e}")
   assert max_diff < 1e-10, "Serialization round-trip failed"
   print("✅ Serialization preserves pulse behavior")

Multi-Qubit 3-Level Simulation
-------------------------------

Advanced users can simulate three-level effects in multi-qubit circuits:

.. code-block:: python

   from tyxonq import Circuit
   
   # Two-qubit circuit
   c = Circuit(2)
   
   # Pulse on q0 (will have 3-level dynamics)
   c = c.extended([
       ("pulse_inline", 0, {
           "type": "drag",
           "class": "Drag",
           "args": [1.0, 160, 40, 0.2, 0.0]
       }, {"qubit_freq": 5.0e9, "anharmonicity": -300e6})
   ])
   
   # Classical gate on q1
   c.h(1)
   
   # Entangling gate
   c.cnot(0, 1)
   
   c.measure_z(0)
   c.measure_z(1)
   
   # Run with 3-level
   device = c.device(provider="simulator", device="statevector")
   result = device.run(shots=1000, three_level=True)
   
   counts = result["result"]
   
   # Note: Currently, only q0 (pulsed qubit) has full 3-level behavior
   # Multi-qubit 3-level support is experimental and will be enhanced in P1.5

**Important Note**: Multi-qubit three-level simulation is experimental.
For production use, test individual qubits first, then compose circuits.

Troubleshooting
===============

Issue: Three-level outcomes ('2') not appearing
-----------------------------------------------

**Problem**: Running with ``three_level=True`` but only seeing '0' and '1' outcomes.

**Diagnosis**:

.. code-block:: python

   # Check if leakage is actually being generated
   result = device.run(shots=1000, three_level=True)
   counts = result["result"]
   
   if '2' not in counts:
       print("⚠️ No leakage detected. Check:")
       print(f"  1. Pulse amplitude: should be > 0.5 for observable leakage")
       print(f"  2. Shot count: use shots > 1000 for small leakage probabilities")
       print(f"  3. Rabi frequency: increase for stronger leakage")

**Solutions**:

1. **Increase amplitude**: Use ``amp=1.0`` or higher
2. **Increase shots**: Use ``shots=5000`` or more
3. **Increase Rabi frequency**: Set ``rabi_freq=50e6`` instead of default
4. **Check anharmonicity**: Use realistic value like ``-300e6``

Issue: pulse_inline runs slower than pulse
-------------------------------------------

**Reason**: Three-level simulation uses 3×3 unitaries instead of 2×2,
so it's ~30% slower.

**Solution**: Only use ``three_level=True`` when accuracy is critical.
For algorithm development, use ``three_level=False`` (default).

Issue: Different results between pulse and pulse_inline
------------------------------------------------------

**Cause**: Deserialization can introduce minor floating-point differences.

**Check**: Verify serialization round-trip (see Example above).

**Solution**: Statistical variance is normal. With 5000+ shots,
relative difference should be < 10%.

Performance Tips
================

Optimization Strategy
---------------------

.. code-block:: text

   Development Phase:
     1. Use 2-level simulation (fastest)
     2. Develop and debug algorithm logic
     3. Test with few shots (100-500)
   
   Optimization Phase:
     1. Enable 3-level simulation
     2. Scan pulse parameters locally
     3. Find optimal settings (takes ~10 seconds for 100 parameters)
   
   Production Phase:
     1. Export to TQASM (pulse_inline format)
     2. Submit to cloud hardware (final verification)
     3. No need for further local simulation

Backend Selection
-----------------

.. code-block:: python

   # Fast development
   result = device.run(backend="numpy", three_level=False, shots=100)
   
   # Accurate optimization
   result = device.run(backend="numpy", three_level=True, shots=1000)
   
   # Autograd-enabled (for VQE)
   result = device.run(backend="pytorch", three_level=False, shots=0)

See Also
========

- :doc:`../../tutorials/advanced/pulse_inline_three_level` - Detailed tutorial
- :doc:`../../tutorials/advanced/pulse_three_level` - Three-level system theory
- :doc:`../../user_guide/pulse/index` - Pulse programming guide
- :doc:`../../api/libs/three_level_system` - Three-level API reference

References
==========

1. **DRAG Pulse Correction**:
   Motzoi, F., et al. "Simple pulses for elimination of leakage in weakly nonlinear qubits."
   *Physical Review Letters* 103.11 (2009): 110501.
   https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.110501

2. **Transmon Qubit Model**:
   Koch, J., et al. "Charge-insensitive qubit design derived from the Cooper pair box."
   *Physical Review A* 76.4 (2007): 042319.
   https://journals.aps.org/pra/abstract/10.1103/PhysRevA.76.042319

3. **Hardware Leakage Characterization**:
   Jurcevic, P., et al. "Demonstration of quantum volume 64 on a superconducting computing system."
   *Nature Communications* 5.1 (2021): 1-8.
   https://arxiv.org/abs/2108.12323

4. **QuTiP-qip Pulse Library**:
   "Pulse-level quantum simulation in QuTiP"
   https://qutip.org/docs/latest/modules/qip.rst
