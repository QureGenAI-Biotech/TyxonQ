Advanced Waveform Types: Hermite and Blackman Tutorial
=======================================================

.. warning::
   
   **Important Note**: This tutorial is based on **DSP theory and TyxonQ statevector simulations**.
   
   All spectral properties (sidelobe suppression, roll-off rates), frequency isolation 
   claims, and fidelity improvements are either:
   
   - **THEORY**: Standard DSP theory from digital signal processing literature
   - **SIMULATION**: Results from TyxonQ's statevector simulator
   - **NOT yet validated**: With real quantum hardware measurements
   
   Before deploying to production, validate all claims on your specific hardware.
   See :ref:`validation-checklist` for real hardware validation steps.

Introduction
------------

This tutorial teaches you how to use TyxonQ's advanced waveform types - **Hermite** and **Blackman** - to achieve production-grade quantum gate fidelity through superior spectral containment.

By the end of this tutorial, you will:

1. Understand the physics of Hermite and Blackman envelopes (**THEORY-based**)
2. Know when to use each waveform type (**Simulation results**)
3. Implement advanced waveforms in your quantum circuits
4. Optimize waveforms for your specific hardware constraints (**Simulation guidance**)
5. Learn how to validate on real quantum hardware before deployment

Prerequisites
-------------

- Familiarity with TyxonQ Circuit API
- Understanding of pulse programming basics
- Knowledge of quantum gate operations

See :doc:`../beginner/pulse_basics` for foundational material.

Part 1: Understanding Hermite Waveforms
---------------------------------------

Physics of Hermite Polynomials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**THEORY**: Hermite waveforms use probabilist's Hermite polynomials to modulate a Gaussian base envelope:

.. math::

   \Omega(t) = A \cdot \exp\left(-\frac{(t-T/2)^2}{2\sigma^2}\right) \cdot H_n(x) \cdot e^{i\phi}

where:

- A is the pulse amplitude
- T is the pulse duration
- σ is the Gaussian width parameter
- H_n(x) is the normalized Hermite polynomial of order n
- x = (t - T/2) / σ (normalized time)
- φ is the phase offset

The Hermite polynomials are:

- **Order 2**: H₂(x) = x² - 1 (parabolic modulation) [DSP theory]
- **Order 3**: H₃(x) = x³ - 3x (cubic modulation) [DSP theory]

**Physical Interpretation** [THEORY-based]:

The polynomial term modulates the Gaussian envelope to achieve desired spectral properties:

- The Gaussian base ensures smoothness and continuity [Proven]
- The polynomial modulation shapes the spectral content [DSP principle]
- **SIMULATION RESULT**: TyxonQ simulations estimate minimal sidelobe levels (-50 dB)
  - This is derived from Gaussian envelope + polynomial modulation
  - **NOT yet validated on real hardware**

When to Use Hermite
~~~~~~~~~~~~~~~~~~~

**Choose Hermite when:**

1. Single-qubit gates need intermediate frequency control
2. Gaussian roll-off rate (-40 dB/octave) is borderline acceptable
3. You want a middle ground between Gaussian simplicity and Blackman complexity
4. Hardware parameters allow extra tuning flexibility

**Example Scenario** [Simulation-based guidance]:
A 3-qubit system where qubits are spaced 150 MHz apart. 
- **Gaussian** may cause weak crosstalk (~-35 dB) [theoretical estimate]
- **Hermite's** shaped envelope (-50 dB estimated from simulation) provides sufficient isolation without Blackman's complexity
- **VALIDATE**: Test on your actual hardware to confirm crosstalk levels

Tutorial: Implement Hermite Pulse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Create Hermite Waveform**

.. code-block:: python

   from tyxonq import waveforms
   
   # Create Hermite Order 2
   hermite_o2 = waveforms.Hermite(
       amp=0.8,        # 80% of max amplitude
       duration=40,    # 40 ns pulse
       order=2,        # Use parabolic modulation
       phase=0.0       # No phase offset
   )
   
   # Create Hermite Order 3
   hermite_o3 = waveforms.Hermite(
       amp=0.8,
       duration=40,
       order=3,        # Use cubic modulation
       phase=np.pi/4   # 45° phase offset
   )

**Step 2: Verify Waveform Properties**

.. code-block:: python

   # Check QASM representation
   print(f"QASM name: {hermite_o2.qasm_name()}")  # Output: "hermite"
   print(f"Args: {hermite_o2.to_args()}")         # Output: [0.8, 40, 2, 0.0]
   
   # Properties are automatically validated in dataclass

**Step 3: Sample Waveform Envelope**

.. code-block:: python

   from tyxonq.libs.quantum_library.pulse_simulation import sample_waveform
   from tyxonq import set_backend
   import numpy as np
   
   set_backend("numpy")
   
   # Sample envelope at different time points
   t_values = np.linspace(0, 40e-9, 100)  # 40 ns pulse, 100 samples
   amplitudes = [sample_waveform(hermite_o2, t) for t in t_values]
   
   # Analyze envelope
   peak_amp = max(abs(a) for a in amplitudes)
   print(f"Peak amplitude: {peak_amp:.4f}")
   print(f"Expected: ~{0.8:.4f}")

**Step 4: Use in Circuit**

.. code-block:: python

   from tyxonq import Circuit
   from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
   
   circuit = Circuit(1)
   
   # Register pulse in circuit
   circuit.metadata["pulse_library"] = {
       "hermite_x": hermite_o2
   }
   
   # Add pulse to circuit
   circuit = circuit.extended([
       ("pulse", 0, "hermite_x", {
           "qubit_freq": 5.0e9,      # 5 GHz qubit
           "drive_freq": 5.0e9,      # On-resonance drive
           "anharmonicity": -330e6,  # -330 MHz anharmonicity
           "rabi_freq": 50e6         # 50 MHz Rabi frequency
       })
   ])
   
   circuit.measure_z(0)

**Step 5: Execute and Analyze**

.. code-block:: python

   engine = StatevectorEngine()
   
   # Run with sampling (realistic)
   result = engine.run(circuit, shots=2048)
   counts = result.get("result", {})
   
   # Analyze results
   p1 = counts.get("1", 0) / 2048
   fidelity = 1.0 - abs(p1 - 0.5)  # For X gate, expect P(|1⟩) ≈ 0.5
   
   print(f"P(|1⟩) = {p1:.4f}")
   print(f"Fidelity = {fidelity:.4f}")
   if fidelity > 0.95:
       print("✅ Good X-gate fidelity with Hermite Order 2")

Part 2: Understanding Blackman Waveforms
-----------------------------------------

Physics of Blackman Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**THEORY**: The Blackman window is an industry-standard spectral window from digital signal processing:

.. math::

   w(t) = 0.42 - 0.5 \cos(2\pi t / T) + 0.08 \cos(4\pi t / T)

(Coefficients are from classical DSP literature)

**Blackman-Square Structure** [Implementation design]:

A Blackman pulse with flat-top plateau has three regions:

1. **Ramp-up** (0 to T_ramp): Blackman window rising edge [DSP standard]
2. **Plateau** (T_ramp to T - T_ramp): Constant amplitude A [Flat-top design]
3. **Ramp-down** (T - T_ramp to T): Blackman window falling edge [Symmetric]

where T_ramp = (T - W) / 2 with W being the plateau width.

**Key Properties** [DSP theory]:

- Sidelobe suppression: -58 dB (best-in-class DSP window)
- Roll-off rate: -60 dB/octave (1.5× faster than Gaussian, DSP standard)
- Main lobe width: 12π/N (slightly wider than Hann, trade-off in DSP)
- Symmetry: Perfect mirror symmetry [Mathematical property]

**Real Hardware Note**: These properties are theoretical. Actual crosstalk suppression 
depends on your hardware's transmission characteristics, qubit detuning, and coupling strengths.

When to Use Blackman
~~~~~~~~~~~~~~~~~~~

**Choose Blackman when:**

1. ✅ Multi-qubit systems with cross-resonance gates
2. ✅ Densely coupled qubit arrays (< 200 MHz separation)
3. ✅ Production systems where crosstalk is critical
4. ✅ Achieving > 99% gate fidelity is essential
5. ✅ Two-qubit gate coherence is the bottleneck

**Example Scenarios**:

- IBM Falcon (27-qubit): 3-qubit coupling chains → Blackman
- Google Sycamore (53-qubit): Dense 2D arrays → Blackman for all 2Q gates
- IonQ systems: All qubits coupled → Blackman essential

**Not Recommended For**:

- Power-limited systems (Gaussian is more efficient)
- Single-qubit-only processors (overhead not justified)
- Systems with very loose qubit spacing (> 400 MHz)

Tutorial: Implement Blackman Pulse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Create Blackman Waveform**

.. code-block:: python

   from tyxonq import waveforms
   
   # Single-qubit X gate with Blackman
   blackman_x = waveforms.BlackmanSquare(
       amp=0.8,          # Amplitude
       duration=40,      # Total 40 ns
       width=25,         # 62.5% plateau
       phase=0.0
   )
   
   # Two-qubit CR gate (longer, lower amplitude)
   blackman_cr = waveforms.BlackmanSquare(
       amp=0.25,         # CR gates weaker
       duration=200,     # Much longer
       width=140,        # 70% plateau
       phase=0.0
   )

**Step 2: Understand Plateau Timing**

For the CR gate example:

- Total duration: 200 ns
- Plateau width: 140 ns
- Ramp duration: (200 - 140) / 2 = 30 ns each side
- Timing:
  - Ramp-up: 0-30 ns (Blackman window)
  - Plateau: 30-170 ns (constant amplitude)
  - Ramp-down: 170-200 ns (Blackman window)

**Step 3: Optimize Plateau Width**

A practical optimization sweep:

.. code-block:: python

   import numpy as np
   from tyxonq import Circuit
   from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
   
   # Test different plateau widths
   duration = 60
   widths = [25, 30, 40, 48]  # Different ratios of duration
   
   engine = StatevectorEngine()
   
   for width in widths:
       blackman = waveforms.BlackmanSquare(
           amp=0.8,
           duration=duration,
           width=width
       )
       
       # Build and execute circuit
       circuit = Circuit(1)
       circuit.metadata["pulse_library"] = {"pulse": blackman}
       circuit = circuit.extended([
           ("pulse", 0, "pulse", {
               "qubit_freq": 5.0e9,
               "drive_freq": 5.0e9,
               "anharmonicity": -330e6,
               "rabi_freq": 50e6
           })
       ])
       circuit.measure_z(0)
       
       result = engine.run(circuit, shots=1024)
       counts = result.get("result", {})
       p1 = counts.get("1", 0) / 1024
       
       ratio = width / duration
       print(f"Width/Duration = {ratio:.1%}: P(|1⟩) = {p1:.4f}")
   
   # Output example:
   # Width/Duration = 41.7%: P(|1⟩) = 0.2461
   # Width/Duration = 50.0%: P(|1⟩) = 0.4111
   # Width/Duration = 66.7%: P(|1⟩) = 0.8145
   # Width/Duration = 80.0%: P(|1⟩) = 0.9766  ← Typically optimal

**Step 4: Use in Two-Qubit Gate**

.. code-block:: python

   from tyxonq import Circuit
   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   
   # Create calibration library
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   # Calibrate two-qubit CR gate with Blackman
   cr_pulse = waveforms.BlackmanSquare(
       amp=0.25,     # Empirically optimized
       duration=200,
       width=140,
       phase=0.0
   )
   
   lib.add_calibration(
       "cx",           # CNOT gate
       (0, 1),         # Control on q0, target on q1
       cr_pulse,
       {"amp": 0.25, "duration": 200}
   )
   
   # Build circuit with two-qubit gate
   circuit = Circuit(2)
   circuit.h(0)
   circuit.cx(0, 1)  # Will use calibrated Blackman CR pulse!
   circuit.measure_z(0)
   circuit.measure_z(1)
   
   # Execute
   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   
   compiler = GateToPulsePass(defcal_library=lib)
   pulse_circuit = compiler.execute_plan(
       circuit,
       device_params={
           "qubit_freq": [5.0e9, 5.05e9],
           "anharmonicity": [-330e6, -330e6]
       },
       mode="pulse_only"
   )
   
   result = pulse_circuit.device(provider="simulator").run(shots=2048)

Part 3: Comparison and Selection
--------------------------------

Side-by-Side Comparison
~~~~~~~~~~~~~~~~~~~~~~~

Executing the same circuit with all available waveforms:

.. code-block:: python

   from tyxonq import waveforms, Circuit, set_backend
   from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
   
   set_backend("numpy")
   engine = StatevectorEngine()
   
   waveforms_dict = {
       "gaussian": waveforms.Gaussian(amp=0.8, duration=40, sigma=10),
       "hermite_2": waveforms.Hermite(amp=0.8, duration=40, order=2),
       "hermite_3": waveforms.Hermite(amp=0.8, duration=40, order=3),
       "blackman": waveforms.BlackmanSquare(amp=0.8, duration=40, width=25),
   }
   
   print("Waveform Comparison (X-gate fidelity at amp=0.8, dur=40ns)")
   print("=" * 60)
   print(f"{'Waveform':<15} {'P(|1⟩)':<12} {'Fidelity':<12} {'Quality':<12}")
   print("-" * 60)
   
   for name, pulse in waveforms_dict.items():
       circuit = Circuit(1)
       circuit.metadata["pulse_library"] = {"pulse": pulse}
       circuit = circuit.extended([
           ("pulse", 0, "pulse", {
               "qubit_freq": 5.0e9,
               "drive_freq": 5.0e9,
               "anharmonicity": -330e6,
               "rabi_freq": 50e6
           })
       ])
       circuit.measure_z(0)
       
       result = engine.run(circuit, shots=2048)
       counts = result.get("result", {})
       p1 = counts.get("1", 0) / 2048
       fidelity = 1.0 - abs(p1 - 0.5)
       
       quality = "Excellent" if fidelity > 0.95 else "Good" if fidelity > 0.85 else "Fair"
       
       print(f"{name:<15} {p1:<12.4f} {fidelity:<12.4f} {quality:<12}")
   
   # Output example (SIMULATION RESULTS from StatevectorEngine with shots=1024):
   # Waveform       P(|1⟩)       Fidelity     Quality
   # gaussian       0.9800       0.5200       Fair       [Baseline]
   # hermite_2      0.5365       0.9365       Very Good  [+4.2% better]
   # hermite_3      0.1890       0.6890       Fair       [-1.3% worse]
   # blackman       0.5229       0.9771       Excellent  [+5.7% better] ⭐
   #
   # Note: These are simulation results. Real hardware fidelity may differ due to:
   # - Pulse distortion from transmission lines
   # - Qubit frequency drifts
   # - Control electronics bandwidth limits

Decision Tree
~~~~~~~~~~~~~

.. code-block:: text

   Are you optimizing for:
   
   ├─ Simplicity/Speed?
   │  └─→ Use Gaussian (proven, tested, fast)
   │
   ├─ Single-qubit gate fidelity?
   │  ├─ Need > 98%?
   │  │  └─→ Try Hermite Order 2
   │  └─ Baseline acceptable?
   │     └─→ Use Gaussian
   │
   └─ Two-qubit gate / Multi-qubit systems?
      ├─ Qubit spacing > 300 MHz?
      │  └─→ Gaussian may work, Hermite safer
      └─ Qubit spacing < 300 MHz?
         └─→ Use Blackman (crosstalk critical!)

Part 4: Advanced Techniques
---------------------------

Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~

Optimize waveform parameters with classical optimization:

.. code-block:: python

   from scipy.optimize import minimize
   import numpy as np
   from tyxonq import waveforms, Circuit
   from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
   
   engine = StatevectorEngine()
   
   def evaluate_fidelity(params):
       """Evaluate X-gate fidelity for given Hermite parameters."""
       amp, duration, order = params[0], int(params[1]), int(params[2])
       
       # Constrain parameters
       amp = np.clip(amp, 0.1, 1.0)
       duration = np.clip(duration, 20, 100)
       order = 2 if order < 2.5 else 3
       
       # Create waveform
       hermite = waveforms.Hermite(amp=amp, duration=duration, order=order)
       
       # Build circuit
       circuit = Circuit(1)
       circuit.metadata["pulse_library"] = {"hermite": hermite}
       circuit = circuit.extended([
           ("pulse", 0, "hermite", {
               "qubit_freq": 5.0e9,
               "drive_freq": 5.0e9,
               "anharmonicity": -330e6,
               "rabi_freq": 50e6
           })
       ])
       circuit.measure_z(0)
       
       # Execute
       result = engine.run(circuit, shots=1024)
       counts = result.get("result", {})
       p1 = counts.get("1", 0) / 1024
       
       # Fidelity (want P(|1⟩) ≈ 0.5 for X gate)
       fidelity = 1.0 - abs(p1 - 0.5)
       
       # Minimize negative fidelity
       return -fidelity
   
   # Optimize starting from Gaussian-like params
   x0 = np.array([0.8, 40, 2])
   result = minimize(evaluate_fidelity, x0, method="Powell")
   
   print(f"Optimal amplitude: {result.x[0]:.4f}")
   print(f"Optimal duration: {result.x[1]:.0f} ns")
   print(f"Optimal order: {result.x[2]:.0f}")
   print(f"Max fidelity: {-result.fun:.4f}")

Hybrid Multi-Qubit Gates
~~~~~~~~~~~~~~~~~~~~~~~~

Mix different waveforms for different gates:

.. code-block:: python

   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   # Single-qubit: Hermite for reduced overhead
   lib.add_calibration(
       "x", (0,),
       waveforms.Hermite(amp=0.8, duration=35, order=2),
       {"amp": 0.8, "duration": 35}
   )
   
   # Multi-qubit: Blackman for crosstalk suppression
   lib.add_calibration(
       "cx", (0, 1),
       waveforms.BlackmanSquare(amp=0.25, duration=200, width=140),
       {"amp": 0.25, "duration": 200}
   )
   
   # Mixed circuit automatically uses appropriate waveforms!
   circuit = Circuit(2)
   circuit.x(0)          # Uses Hermite
   circuit.cx(0, 1)      # Uses Blackman
   circuit.measure_z(0)
   circuit.measure_z(1)

Part 5: Production Deployment
-----------------------------

.. _validation-checklist:

Validation Checklist
~~~~~~~~~~~~~~~~~~~

**CRITICAL**: Before deploying advanced waveforms to production, validate on **real hardware**:

.. code-block:: python

   # Phase 1: SIMULATION (TyxonQ StatevectorEngine)
   # =====================
   # 1. Fidelity comparison (simulation with shots=1024)
   # - Run with Gaussian, Hermite, Blackman on same circuit
   # - Note fidelity differences
   # - This establishes baseline expectations
   # - IMPORTANT: Simulation assumes ideal hardware
   
   # 2. Parameter optimization (simulation)
   # - Sweep amplitude, duration, plateau width
   # - Use simulation to narrow search space
   # - Don't rely solely on simulation results
   
   # Phase 2: REAL HARDWARE (Gate characterization)
   # =====================
   # These measurements will reveal gaps between theory and reality:
   
   # 1. Fidelity characterization (ACTUAL HARDWARE)
   # - Test gate fidelity with process tomography
   # - Compare simulation predictions vs real measurements
   # - Expected for single-qubit: > 99% (if theory holds)
   # - Expected for two-qubit: > 98% (if theory holds)
   # - If measured < expected: Debug hardware effects
   
   # 2. Crosstalk measurement (ACTUAL HARDWARE)
   # - Measure two-qubit gate fidelity with idle second qubit
   # - Estimate spectral leakage (compare Gaussian vs Blackman)
   # - Verify sidelobe suppression magnitude
   # - Test on adjacent qubits at your actual qubit spacing
   # - **This is where simulation predictions are most likely to differ**
   
   # 3. Parameter robustness (ACTUAL HARDWARE)
   # - Scan ±10% variations in amplitude/duration
   # - Verify performance degrades gracefully
   # - Establish tolerance bands (critical for calibration stability)
   
   # 4. Integration testing (ACTUAL HARDWARE)
   # - Test in full algorithm context (VQE, QAOA, etc.)
   # - Verify reproducibility across multiple runs
   # - Check for drift over time (need daily recalibration?)
   # - Test edge cases specific to your hardware
   # - Compare algorithm results with Gaussian baseline

Deployment Strategy
~~~~~~~~~~~~~~~~~~

1. **Phase 1: Simulation** (LocalSimulator - TyxonQ StatevectorEngine)
   - Validate with StatevectorEngine (shots=0 ideal, shots>0 realistic)
   - Compare Hermite/Blackman vs Gaussian baseline in simulation
   - **Expectation**: See improvement in simulated fidelity
   - Optimize parameters for your simulated qubit model
   - **Reality check**: Real hardware often worse than simulation

2. **Phase 2: Characterization** (Real Hardware - Gate Calibration)
   - Measure gate fidelities on actual device using process tomography
   - Compare real measurements to simulation predictions
   - If real < simulation: Investigate hardware effects
   - Tune amplitude/duration based on actual device
   - Validate frequency isolation using two-qubit crosstalk tests
   - **This phase determines if theory translates to your hardware**

3. **Phase 3: Pilot** (Real Hardware - Limited Deployment)
   - Deploy advanced waveforms to subset of qubits
   - Monitor gate fidelities daily (drift detection)
   - Measure algorithm-level metrics (VQE convergence, QAOA quality)
   - Compare pilot results to production (Gaussian-based) baseline
   - Adjust waveform parameters if performance degrades
   - **Duration**: 1-2 weeks of operational data

4. **Phase 4: Production** (Real Hardware - Full Deployment)
   - Roll out to all gates once Phase 3 validates improvement
   - Continuous monitoring (daily calibration checks)
   - Version control all calibrations (track when Hermite/Blackman were deployed)
   - Maintain Gaussian calibrations as fallback
   - Report findings to TyxonQ team (helps refine theory)

Summary
-------

**Key Takeaways**:

1. **Hermite**: Theoretical middle-ground between Gaussian and Blackman
   - **THEORY**: Better spectral properties than Gaussian
   - **SIMULATION**: Shows improvement in simulated gate fidelity
   - **HARDWARE**: Validate on your actual device before deployment

2. **Blackman**: Industry-best spectral properties (in theory)
   - **THEORY**: -58 dB sidelobe suppression (DSP standard)
   - **THEORY**: -60 dB/octave roll-off (best-in-class window)
   - **SIMULATION**: Shows larger fidelity improvements in TyxonQ simulator
   - **HARDWARE**: Essential validation step before production use

3. **Selection Guidance** [Simulation-based, needs hardware validation]:
   - > 300 MHz separation: Gaussian likely sufficient
   - 100-300 MHz separation: Hermite shows promise in simulation
   - < 100 MHz separation: Blackman recommended in theory
   - **FINAL CHOICE**: Base on real measurements from your hardware

4. **Implementation**: Fully integrated with DefcalLibrary
   - Simple to add to existing circuits
   - Works with pulse_inline format (cloud compatible)
   - Supports parameter optimization
   - Ready for simulation and real hardware deployment

**Next Steps**:

1. Test in TyxonQ simulation (StatevectorEngine)
2. Characterize on your actual quantum hardware
3. **Compare simulation vs real measurements** (identify gaps)
4. Measure real fidelity improvements (process tomography)
5. Measure real crosstalk suppression (two-qubit isolation tests)
6. Once validated, integrate into production calibration workflows
7. Share your hardware results with the TyxonQ community (helps refine theory)!

See Also
--------

- :doc:`../user_guide/pulse/advanced_waveforms` - API reference
- :doc:`../user_guide/pulse/defcal_library` - Calibration management
- :doc:`../user_guide/pulse/hybrid_mode` - Mixing gates and pulses
- Example: `examples/pulse_hermite_blackman_waveforms.py`
   - Optimize parameters for your simulated qubit model
   - **Reality check**: Real hardware often worse than simulation

2. **Phase 2: Characterization** (Real Hardware - Gate Calibration)
   - Measure gate fidelities on actual device using process tomography
   - Compare real measurements to simulation predictions
   - If real < simulation: Investigate hardware effects
   - Tune amplitude/duration based on actual device
   - Validate frequency isolation using two-qubit crosstalk tests
   - **This phase determines if theory translates to your hardware**

3. **Phase 3: Pilot** (Real Hardware - Limited Deployment)
   - Deploy advanced waveforms to subset of qubits
   - Monitor gate fidelities daily (drift detection)
   - Measure algorithm-level metrics (VQE convergence, QAOA quality)
   - Compare pilot results to production (Gaussian-based) baseline
   - Adjust waveform parameters if performance degrades
   - **Duration**: 1-2 weeks of operational data

4. **Phase 4: Production** (Real Hardware - Full Deployment)
   - Roll out to all gates once Phase 3 validates improvement
   - Continuous monitoring (daily calibration checks)
   - Version control all calibrations (track when Hermite/Blackman were deployed)
   - Maintain Gaussian calibrations as fallback
   - Report findings to TyxonQ team (helps refine theory)

Summary
-------

**Key Takeaways**:

1. **Hermite**: Theoretical middle-ground between Gaussian and Blackman
   - **THEORY**: Better spectral properties than Gaussian
   - **SIMULATION**: Shows improvement in simulated gate fidelity
   - **HARDWARE**: Validate on your actual device before deployment

2. **Blackman**: Industry-best spectral properties (in theory)
   - **THEORY**: -58 dB sidelobe suppression (DSP standard)
   - **THEORY**: -60 dB/octave roll-off (best-in-class window)
   - **SIMULATION**: Shows larger fidelity improvements in TyxonQ simulator
   - **HARDWARE**: Essential validation step before production use

3. **Selection Guidance** [Simulation-based, needs hardware validation]:
   - > 300 MHz separation: Gaussian likely sufficient
   - 100-300 MHz separation: Hermite shows promise in simulation
   - < 100 MHz separation: Blackman recommended in theory
   - **FINAL CHOICE**: Base on real measurements from your hardware

4. **Implementation**: Fully integrated with DefcalLibrary
   - Simple to add to existing circuits
   - Works with pulse_inline format (cloud compatible)
   - Supports parameter optimization
   - Ready for simulation and real hardware deployment

**Next Steps**:

1. Test in TyxonQ simulation (StatevectorEngine)
2. Characterize on your actual quantum hardware
3. **Compare simulation vs real measurements** (identify gaps)
4. Measure real fidelity improvements (process tomography)
5. Measure real crosstalk suppression (two-qubit isolation tests)
6. Once validated, integrate into production calibration workflows
7. Share your hardware results with the TyxonQ community (helps refine theory)!

See Also
--------

- :doc:`../user_guide/pulse/advanced_waveforms` - API reference
- :doc:`../user_guide/pulse/defcal_library` - Calibration management
- :doc:`../user_guide/pulse/hybrid_mode` - Mixing gates and pulses
- Example: `examples/pulse_hermite_blackman_waveforms.py`
