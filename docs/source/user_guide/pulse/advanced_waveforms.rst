Advanced Waveforms: Hermite and Blackman
=========================================

Overview
--------

TyxonQ provides two advanced waveform types for high-fidelity quantum gate implementation:

- **Hermite** - Polynomial envelope with minimal spectral leakage
- **BlackmanSquare** - Industry-standard Blackman window with flat-top plateau

These waveforms extend TyxonQ's waveform library from 8 basic types to 10 complete types, enabling production-grade quantum control with superior frequency containment properties.

Key Features
~~~~~~~~~~~~

**Hermite Waveforms**

Hermite polynomial envelopes provide smooth, shaped pulses using Hermite polynomials to modulate a Gaussian base:

- Order 2: H₂(x) = x² - 1 (parabolic modulation)
- Order 3: H₃(x) = x³ - 3x (cubic modulation)
- Properties: Minimal spectral leakage, smooth envelope, PyTorch-compatible

**BlackmanSquare Waveforms**

Blackman window with flat-top plateau provides industry-best frequency containment:

- Structure: Smooth ramp-up → flat plateau → smooth ramp-down
- Properties: -58 dB sidelobe suppression, -60 dB/octave roll-off rate
- Application: Multi-qubit gates where crosstalk is critical

Quick Start
-----------

Creating Hermite Waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tyxonq import waveforms, Circuit
   from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
   
   # Create Hermite pulse (order 2)
   hermite = waveforms.Hermite(
       amp=0.8,        # Amplitude
       duration=40,    # Duration in nanoseconds
       order=2,        # Polynomial order (2 or 3)
       phase=0.0       # Phase offset in radians
   )
   
   # Use in circuit
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
   engine = StatevectorEngine()
   result = engine.run(circuit, shots=1024)

Creating BlackmanSquare Waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from tyxonq import waveforms, Circuit
   from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
   
   # Create Blackman pulse
   blackman = waveforms.BlackmanSquare(
       amp=0.8,            # Amplitude
       duration=60,        # Total duration (ns)
       width=30,           # Plateau width (ns)
       phase=0.0           # Phase offset (radians)
   )
   
   # Use in circuit
   circuit = Circuit(1)
   circuit.metadata["pulse_library"] = {"blackman": blackman}
   circuit = circuit.extended([
       ("pulse", 0, "blackman", {
           "qubit_freq": 5.0e9,
           "drive_freq": 5.0e9,
           "anharmonicity": -330e6,
           "rabi_freq": 50e6
       })
   ])
   circuit.measure_z(0)
   
   # Execute
   engine = StatevectorEngine()
   result = engine.run(circuit, shots=1024)

API Reference
-------------

Hermite Class
~~~~~~~~~~~~~

.. code-block:: python

   class Hermite:
       """Hermite polynomial envelope waveform.
       
       Uses probabilist's Hermite polynomials to create a smooth envelope
       with minimal frequency spectral leakage.
       """
       
       amp: float or Parameter
           Pulse amplitude (0 to 1 typical)
       
       duration: int
           Pulse duration in nanoseconds
       
       order: int (default: 2)
           Hermite polynomial order (2 or 3)
           - 2: H₂(x) = x² - 1 (lighter modulation)
           - 3: H₃(x) = x³ - 3x (stronger modulation)
       
       phase: float (default: 0.0)
           Phase offset in radians for envelope rotation
       
       def qasm_name() -> str:
           Return QASM name "hermite"
       
       def to_args() -> List:
           Return [amp, duration, order, phase]

BlackmanSquare Class
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class BlackmanSquare:
       """Blackman window with flat-top envelope.
       
       Combines Blackman window (excellent frequency properties)
       with a flat-top plateau for high-precision quantum gates.
       """
       
       amp: float or Parameter
           Pulse amplitude (0 to 1 typical)
       
       duration: int
           Total pulse duration in nanoseconds
       
       width: float or Parameter
           Flat-top plateau width in nanoseconds
           Must be < duration
           Typical: 0.5-0.8 × duration
       
       phase: float (default: 0.0)
           Phase offset in radians for envelope rotation
       
       def qasm_name() -> str:
           Return QASM name "blackman_square"
       
       def to_args() -> List:
           Return [amp, duration, width, phase]

Frequency Domain Properties
----------------------------

Comparison of all waveform types:

.. table::

   ================  =========  ===========  ================  ====================
   Waveform          Main Lobe  Sidelobe     Roll-off Rate     Best For
   ================  =========  ===========  ================  ====================
   Gaussian          6π/N†      -40 dB†      -40 dB/octave†    Single-qubit (simple)
   Hermite           ~8π/N†     -50 dB*†     -40 dB/octave†    Mixed scenarios
   Flattop           7π/N†      -44 dB†      -50 dB/octave†    High-precision 1Q
   **Blackman**      12π/N†     **-58 dB**†  **-60 dB/octave**† **Multi-qubit** ⭐
   ================  =========  ===========  ================  ====================

| † **THEORY (DSP Standard)**: Main Lobe, Sidelobe, and Roll-off for Gaussian, Hermite, 
|   Flattop, and Blackman are standard values from digital signal processing theory. 
|   These represent theoretical frequency domain properties of the window functions 
|   and are NOT validated with real hardware measurements (no real device data yet).
| \* **SIMULATION**: Hermite sidelobe level (-50 dB) is estimated from TyxonQ statevector 
|   simulations based on Gaussian envelope + polynomial modulation. Actual hardware 
|   performance may differ.

Selecting the Right Waveform
-----------------------------

**Use Gaussian for:**
- Simple single-qubit gates
- When pulse length is constrained
- Baseline/reference implementations
- Limited computational resources

**Use Hermite for:**
- Intermediate scenarios needing better frequency properties
- Mixed single/multi-qubit systems
- When Gaussian doesn't quite meet performance targets
- Exploring spectral shaping effects

**Use Blackman for:**
- ✅ Multi-qubit CR (Cross-Resonance) gates
- ✅ Densely coupled qubit arrays
- ✅ Crosstalk-critical applications
- ✅ Production quantum systems

**Use Flattop for:**
- High-precision single-qubit when power allows
- Long-duration constant-amplitude regions

Performance Characteristics
---------------------------

Spectral Leakage
~~~~~~~~~~~~~~~~

The key advantage of Hermite and Blackman is reduced spectral leakage:

- **Gaussian**: -40 dB/octave roll-off rate † THEORY
  - Side-lobes decay relatively slowly (DSP standard)
  - Risk: Coupling to neighboring qubits at 100-200 MHz separation
  - **NOT validated with real hardware**

- **Hermite**: -40 dB/octave roll-off (similar to Gaussian) † THEORY
  - But with shaped modulation reducing peak sidelobe (DSP principle)
  - Intermediate option
  - **Performance validation pending real hardware measurements**

- **Blackman** ⭐: -60 dB/octave roll-off rate † THEORY
  - Side-lobes decay 1.5× faster than Gaussian (DSP standard)
  - Peak sidelobe: -58 dB (best in class for digital windows)
  - Excellent for multi-qubit isolation (theoretical advantage)
  - **IMPORTANT**: Actual hardware crosstalk suppression requires real device testing

Envelope Smoothness
~~~~~~~~~~~~~~~~~~~

All waveforms have smooth, continuous envelopes:

- No sharp discontinuities that cause high-frequency artifacts
- Continuous derivatives for low-amplitude leakage
- Compatible with hardware implementations

Time Domain Properties
~~~~~~~~~~~~~~~~~~~~~~

.. table::

   Property                 Gaussian       Hermite        Blackman
   ======================  ==============  ============   ==============
   Peak amplitude            Full           ~Full          ~Full
   Rise time                 Smooth†      Smooth†       Designed†
   Plateau                   None           None           Yes ✓†
   Fall time                 Smooth†      Smooth†       Designed†
   Overshoot                 None           Minimal        None
   Symmetry                  Yes†         Yes†          Yes†

| † **THEORY/DESIGN**: Time-domain properties derived from waveform envelope equations.
|   Gaussian/Hermite smoothness and Blackman plateau design are theoretical based on 
|   mathematical properties. **NOT yet validated with oscilloscope measurements on 
|   real quantum hardware**.
|   
|   Actual rise/fall times and overshoot depend on:
|   - Hardware bandwidth limitations
|   - AWG (Arbitrary Waveform Generator) capabilities
|   - Transmission line effects
|   - Real qubit response dynamics

Advanced Topics
---------------

Combining with DefcalLibrary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use advanced waveforms in hardware calibrations:

.. code-block:: python

   from tyxonq import waveforms
   from tyxonq.compiler.pulse_compile_engine import DefcalLibrary
   
   lib = DefcalLibrary(hardware="Homebrew_S2")
   
   # Calibrate CX gate with Blackman CR pulse
   cr_pulse = waveforms.BlackmanSquare(
       amp=0.25,     # CR pulses typically weaker
       duration=200, # Longer than single-qubit
       width=140,    # 70% plateau
       phase=0.0
   )
   
   lib.add_calibration(
       "cx", (0, 1), cr_pulse,
       {"amp": 0.25, "duration": 200}
   )

Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~

Hermite and Blackman support parameter optimization through DefcalLibrary:

.. code-block:: python

   # Use as optimization targets
   hermite = waveforms.Hermite(
       amp=0.8,      # Can be optimized
       duration=40,
       order=2,
       phase=0.0     # Can be optimized
   )
   
   # Or in pulse_inline format for cloud execution
   waveform_dict = {
       "type": "hermite",
       "class": "Hermite",
       "args": [0.8, 40, 2, 0.0]  # Serializable for TQASM export
   }

Common Use Cases
----------------

Single-Qubit High-Fidelity Gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Target: >99% fidelity X gate on q0
   hermite_x = waveforms.Hermite(
       amp=0.82,
       duration=35,
       order=2
   )

Two-Qubit CR Gate
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Target: Minimize crosstalk to q1 (200 MHz away)
   cr_pulse = waveforms.BlackmanSquare(
       amp=0.3,       # CR pulses are weaker than single-qubit
       duration=200,  # Longer duration for adiabatic evolution
       width=140,     # 70% plateau
       phase=0.0
   )
   # Blackman's -58 dB sidelobe suppression at 200 MHz isolation

Multi-Qubit Array
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For densely packed qubits (all gates use Blackman)
   for gate_type in ["x", "y", "cx"]:
       pulse = waveforms.BlackmanSquare(
           amp=gate_params[gate_type]["amp"],
           duration=gate_params[gate_type]["duration"],
           width=gate_params[gate_type]["width"]
       )
       lib.add_calibration(gate_type, (...), pulse, {...})

Troubleshooting
---------------

.. note::
   
   **IMPORTANT**: Current properties are based on **THEORY and SIMULATION only**.
   No real hardware data has been collected yet. All claims about frequency suppression,
   crosstalk reduction, and fidelity improvements should be validated on actual 
   quantum hardware before production deployment.

Problem: Hermite Order 3 Has Lower Fidelity Than Order 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Order 3 cubic modulation may be too strong for your pulse parameters.

**Solution**:
- Start with Order 2 (recommended for most cases)
- Only use Order 3 if you have specific frequency shaping requirements
- Validate with gate fidelity characterization **on real hardware**
- Report results to TyxonQ team for empirical validation

Problem: Blackman Width Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Plateau width affects effective amplitude and gate time.

**Solution**:
- Empirical sweep: Test width = 0.5, 0.6, 0.7, 0.8 × duration
- Monitor gate fidelity vs width in **simulation first** using StatevectorEngine
- Then validate on **real hardware** with process tomography
- Typical theoretical optimal: 0.6-0.7 × duration (may differ on real device)

Problem: Expected Crosstalk Reduction Not Observed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Theoretical spectral advantages may not translate to real hardware due to:
- AWG bandwidth limitations
- Transmission line filtering
- Qubit response non-idealities
- Coupling strength variations

**Solution**:
- Measure actual crosstalk on your hardware using two-qubit experiments
- Compare Gaussian vs Blackman in same conditions
- If improvement < expected, investigate hardware-specific limitations
- Report findings for model refinement

Problem: Incompatibility with Older Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: New waveform types not recognized in legacy code paths.

**Solution**:
- Upgrade to TyxonQ v0.2.1+
- Check StatevectorEngine.\_deserialize_pulse_waveform() supports new types
- For TQASM export, ensure compiler is updated

See Also
--------

- :doc:`index` - Complete pulse programming overview
- :doc:`defcal_library` - Hardware calibration management
- :doc:`hybrid_mode` - Mixing gates and pulses
- :doc:`../../tutorials/advanced/pulse_advanced_waveforms` - Tutorial
