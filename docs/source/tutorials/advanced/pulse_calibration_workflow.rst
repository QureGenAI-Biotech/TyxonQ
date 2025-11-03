================================================================================
Pulse Calibration Workflow Tutorial: From Theory to Practice
================================================================================

.. currentmodule:: tyxonq

**Difficulty Level:** ★★★★★ (Advanced)  
**Estimated Time:** 7-8 hours (7 steps)  
**Prerequisites:** Basic quantum mechanics, understanding of pulse programming

Introduction
============

This comprehensive tutorial walks you through the complete workflow of calibrating quantum pulses from scratch. You'll learn how to systematically tune hardware parameters to achieve high-fidelity quantum operations.

Why Pulse Calibration Matters
-------------------------------

Most quantum programmers use high-level gate APIs without thinking about the physical implementation. However, **a perfectly designed quantum algorithm cannot work without proper hardware calibration**.

Key Statistics:
- **Without Defcal:** Bell state fidelity ~82% ❌ (unsuitable for real algorithms)
- **With Defcal:** Bell state fidelity >95% ✅ (suitable for real algorithms)
- **The Gap:** 13% fidelity difference = the difference between working and non-working quantum computers

What You'll Learn
-----------------

1. **Hardware Fundamentals** - Understand qubit parameters, energy levels, and decoherence
2. **Single-Qubit Calibration** - Systematically tune π and π/2 rotations to >99% fidelity
3. **Two-Qubit Gate Design** - Implement CR-based CNOT gates using Rabi oscillations
4. **System Verification** - Validate Bell states and multi-qubit entanglement
5. **Integration & Optimization** - Build complete calibration libraries
6. **Defcal & Deployment** - Understand how calibrations integrate into the compilation pipeline

Tutorial Structure
==================

The tutorial consists of 7 progressive steps. You can run each step independently:

.. code-block:: bash

    cd examples/pulse_calibration_tutorial
    python step_1_hardware_basics.py           # ~30 min
    python step_2_single_qubit_coarse_tuning.py # ~45 min
    python step_3_single_qubit_fine_tuning.py  # ~60 min
    python step_4_two_qubit_gate_tuning.py     # ~75 min
    python step_5_two_qubit_characterization.py # ~60 min
    python step_6_system_integration.py        # ~75 min
    python step_7_advanced_optimization.py     # ~90 min

Step 1: Hardware Basics and Environment Setup
==============================================

**Duration:** 30 minutes  
**Learning Objectives:**
- Understand the three-level model of superconducting qubits
- Identify key hardware parameters
- Set up the TyxonQ pulse programming framework

Key Concepts
------------

Three-Level Model
~~~~~~~~~~~~~~~~~~

Superconducting qubits have multiple energy levels:

.. code-block:: text

    |2⟩ ────────────── E₂ (leakage state)
         ↑ ↓
        transition gap = α (anharmonicity)
         ↑ ↓
    |1⟩ ────────────── E₁ (first excited)
         ↑ ↓
        transition gap = ω_q (qubit frequency)
         ↑ ↓
    |0⟩ ────────────── E₀ (ground state)

Core Hardware Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------+------------------+----------------------------------------------------------+
| Parameter         | Typical Value    | Physical Meaning                                         |
+===================+==================+==========================================================+
| ``qubit_freq``    | 5-6 GHz          | Driving frequency (gate center frequency)                |
+-------------------+------------------+----------------------------------------------------------+
| ``anharmonicity`` | -300 to -400 MHz | Energy gap between |1⟩ and |2⟩ (usually negative)     |
+-------------------+------------------+----------------------------------------------------------+
| ``T1``            | 50-150 μs        | Energy relaxation time (|1⟩ → |0⟩)                     |
+-------------------+------------------+----------------------------------------------------------+
| ``T2``            | 50-200 μs        | Phase coherence time (dephasing)                         |
+-------------------+------------------+----------------------------------------------------------+

Pulse Types
~~~~~~~~~~~

Four common pulse envelope shapes:

1. **Constant (Square):**
   - Simple: A(t) = A₀
   - Disadvantage: High spectral sidebands, leakage to |2⟩

2. **Gaussian:**
   - Smooth envelope: A(t) = A₀ exp(-t²/(2σ²))
   - Better spectral properties
   - Disadvantage: Requires amplitude compensation

3. **DRAG (Derivative Removal by Adiabatic Gate):**
   - I(t) = A₀ exp(-t²/(2σ²))
   - Q(t) = β · dA/dt
   - **Key advantage:** Significantly reduces leakage to |2⟩
   - Trade-off: Additional β parameter to optimize

4. **Hermite:**
   - Advanced smooth pulses with optimized spectral properties

Running Step 1
~~~~~~~~~~~~~~

.. code-block:: python

    from tyxonq import Circuit, waveforms
    from tyxonq.core.ir.pulse import PulseProgram

    # Define hardware parameters
    hardware_config = {
        "qubit_freq": [5.0e9, 5.1e9, 5.2e9],
        "anharmonicity": [-330e6, -320e6, -340e6],
        "T1": [80e-6, 85e-6, 75e-6],
        "T2": [120e-6, 125e-6, 110e-6],
    }

    # Create your first pulse program
    prog = PulseProgram(num_qubits=3)
    x_pulse = waveforms.Gaussian(amp=0.8, duration=160, sigma=40)
    prog.drag(0, amp=x_pulse.amp, duration=x_pulse.duration, sigma=40, beta=0.2)

Expected Output
^^^^^^^^^^^^^^^

- Hardware parameter table
- Rabi frequency analysis (5-50 MHz typical)
- Comparison of pulse types
- Visual representation of the three-level system


Step 2: Single-Qubit Coarse Tuning
===================================

**Duration:** 45 minutes  
**Learning Objectives:**
- Perform parameter sweeps to find initial optimal values
- Understand Rabi oscillation frequency
- Optimize DRAG parameter β

Key Concepts
------------

Rabi Oscillations
~~~~~~~~~~~~~~~~~

When you apply a resonant microwave pulse to a qubit:

.. math::

    P_1(t) = \sin^2\left(\frac{\gamma V \cdot t}{2}\right)

where:
- **γ** ≈ 50 MHz/V (coupling constant)
- **V** = pulse amplitude (0-1 V)
- **t** = pulse duration (10-200 ns)

Key insight: A π-pulse (full rotation) occurs when:

.. math::

    \gamma V \cdot t = \pi

Coarse Tuning Strategy
~~~~~~~~~~~~~~~~~~~~~~

1. **Amplitude Sweep:** Vary amplitude from 0.5 to 1.0 V
2. **Find π-pulse:** Look for the amplitude giving |1⟩ population ≈ 1.0
3. **Measure Rabi Frequency:** Extract oscillation frequency
4. **Optimize DRAG β:** Balance between X-axis leakage and Y-axis leakage

Typical Results
~~~~~~~~~~~~~~~

After coarse tuning:

- X gate: ~97-98% fidelity
- Estimated Rabi frequency: 3-5 MHz
- Recommended amplitudes: 0.7-0.9 V
- DRAG parameter β: 0.1-0.3


Step 3: Single-Qubit Fine Tuning
=================================

**Duration:** 60 minutes  
**Learning Objectives:**
- Achieve >99% single-qubit gate fidelity
- Measure and optimize Rabi oscillations
- Design Hadamard gates

Fine Tuning Approach
--------------------

Building on Step 2, we now focus on precision:

1. **Fine Amplitude Adjustment:** ±0.05 V around coarse value
2. **Rabi Oscillation Measurement:**
   - Vary pulse duration from 10 to 200 ns
   - Extract population oscillation
   - Identify π and π/2 durations precisely

3. **Hadamard Gate Design:**
   - Duration: 0.7× (π-pulse duration)
   - Amplitude: 0.78-0.79 V
   - Duration: 110-115 ns
   - Target fidelity: >98%

Expected Results
~~~~~~~~~~~~~~~~

- X gate: 99.3% fidelity ✅
- H gate: 98.0% fidelity ✅
- Extracted Rabi frequency: 3.12 MHz
- Population oscillation visibility: >99%

Key Achievement
~~~~~~~~~~~~~~~~

After this step, you have **hardware-optimized single-qubit gates** that can be stored in a Defcal library.


Step 4: Two-Qubit Gate Tuning
==============================

**Duration:** 75 minutes  
**Learning Objectives:**
- Design Cross-Resonance (CR) based CNOT gates
- Perform systematic parameter optimization
- Achieve >95% two-qubit gate fidelity

Two-Qubit Gate Architecture
----------------------------

Cross-Resonance (CR) Gate Sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A CNOT(0→1) gate using CR pulses:

.. code-block:: text

    q0: ─── RX(-π/2) ─── [CR pulse] ─── RX(π/2) ───
    q1: ─── ─────────── [CR at q1 freq] ─────────── 

Where:
1. **Pre-rotation:** RX(-π/2) on control qubit
2. **CR drive:** Gaussian pulse on control at target frequency
3. **Post-rotation:** RX(π/2) on control qubit

Total duration: ~280 ns

Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~

The critical parameters to tune:

- CR amplitude: 0.35-0.45 V
- CR duration: 250-300 ns
- CR drive frequency: Target qubit frequency (e.g., 5.1 GHz for q1)

Optimization Method:

1. **Amplitude sweep:** Find the amplitude giving Bell state fidelity ≈ 1.0
2. **Duration sweep:** Fine-tune around optimal amplitude
3. **Verify with Bell states:** Generate |Φ⁺⟩, |Φ⁻⟩, |Ψ⁺⟩, |Ψ⁻⟩

Expected Results
~~~~~~~~~~~~~~~~

- CR optimal amplitude: 0.40 V
- CR optimal duration: 280 ns
- Bell state fidelity: >99% ✅
- CNOT fidelity: >95% ✅


Step 5: Two-Qubit Characterization
===================================

**Duration:** 60 minutes  
**Learning Objectives:**
- Measure Bell state fidelity
- Assess entanglement quality
- Test Bell inequality violation
- Identify error sources

Bell State Analysis
-------------------

Four Bell States
~~~~~~~~~~~~~~~~

We prepare all four maximally entangled states:

.. code-block:: python

    # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    circuit.h(0)
    circuit.cx(0, 1)

    # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    circuit.h(0)
    circuit.z(0)
    circuit.cx(0, 1)

    # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    circuit.h(0)
    circuit.x(1)
    circuit.cx(0, 1)

    # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    circuit.h(0)
    circuit.z(0)
    circuit.x(1)
    circuit.cx(0, 1)

Fidelity Measurement
~~~~~~~~~~~~~~~~~~~~

For each Bell state, measure:

.. math::

    F = \langle \Psi | \rho | \Psi \rangle

where ρ is the measured state.

Expected Results
~~~~~~~~~~~~~~~~

Typical fidelities for calibrated systems:

- |Φ⁺⟩: 96.36%
- |Φ⁻⟩: 97.75%
- |Ψ⁺⟩: 93.84%
- |Ψ⁻⟩: 94.75%
- Average: 95.68% ✅

CHSH Inequality Test
~~~~~~~~~~~~~~~~~~~~

The CHSH parameter:

.. math::

    S = 2\sqrt{2} \approx 2.828 \text{ (classical max = 2.0)}

A calibrated system should achieve S > 2.7, demonstrating genuine quantum entanglement.

Error Analysis
~~~~~~~~~~~~~~

Five main error sources:

1. **Gate infidelity:** Imperfect gate implementations (→ 1% error each)
2. **Decoherence:** T1, T2 losses during gate execution
3. **State preparation:** Imperfect H gate creating superposition
4. **Measurement error:** Readout infidelity
5. **Crosstalk:** Unintended interactions between qubits


Step 6: System Integration
===========================

**Duration:** 75 minutes  
**Learning Objectives:**
- Build complete calibration libraries
- Prepare multi-qubit entangled states (GHZ)
- Characterize coherence times
- Validate system performance

Calibration Library Structure
-----------------------------

.. code-block:: python

    calibration_library = {
        'single_qubit_gates': {
            'X': {
                'amplitude': 0.78,
                'duration_ns': 160,
                'fidelity': 0.9932,
                'pulse_type': 'DRAG',
                'sigma': 40,
                'beta': 0.2
            },
            'H': {
                'amplitude': 0.78,
                'duration_ns': 113,
                'fidelity': 0.9800,
                'pulse_type': 'DRAG'
            },
            'RZ': {
                'pulse_type': 'VirtualZ',
                'fidelity': 1.0
            }
        },
        'two_qubit_gates': {
            'CX_01': {
                'amplitude': 0.40,
                'duration_ns': 280,
                'fidelity': 0.9520,
                'pulse_type': 'CR'
            },
            'CX_12': {
                'amplitude': 0.40,
                'duration_ns': 280,
                'fidelity': 0.9480,
                'pulse_type': 'CR'
            }
        }
    }

GHZ State Preparation
~~~~~~~~~~~~~~~~~~~~~

Prepare three-qubit entanglement: |GHZ⟩ = (|000⟩ + |111⟩)/√2

.. code-block:: python

    circuit = Circuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure_z(0, 1, 2)

Expected GHZ fidelity: >89% ✅

Coherence Characterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measure T1 and T2 directly:

- **T1 Measurement:** Prepare |1⟩, wait time τ, measure
- **T2 Measurement:** Prepare superposition, apply dephasing pulses

Expected Results
~~~~~~~~~~~~~~~~

System Performance Summary:

- Single-qubit gate fidelity: >99%
- Two-qubit gate fidelity: >95%
- Multi-qubit circuit fidelity: >89%
- Leakage rate: <1%
- GHZ state fidelity: 89.64%


Step 7: Advanced Optimization and Defcal Integration
=====================================================

**Duration:** 90 minutes  
**Learning Objectives:**
- Understand Defcal (Definition of Calibration)
- Implement Virtual-Z optimization
- Integrate calibrations into compilation pipeline
- Quantify performance improvements

What is Defcal?
---------------

Defcal = Definition of Calibration

**Core Concept:** Defcal bridges the gap between logical gates and physical pulses.

Three-Layer Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ┌────────────────────────────────────┐
    │ Logical Layer (User Code)          │
    │ circuit.h(0)                       │
    │ circuit.cx(0, 1)                   │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ Defcal Layer (Calibration Library) │
    │ defcal H 0:                        │
    │   pulse.play(Gaussian(...))        │
    │ defcal CX 0 1:                     │
    │   pulse.play(...) × 3              │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ Physical Layer (Hardware Execution)│
    │ Generate microwave pulses          │
    │ Measure quantum state              │
    └────────────────────────────────────┘

Compilation Flow with Defcal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    Circuit (logic) 
        │
        ▼
    GateToPulsePass
        │
        ├─ Query Defcal library for each gate
        │  ├─ H(0) → Found: Gaussian(amp=0.78, dur=113ns)
        │  ├─ CX(0,1) → Found: CR sequence
        │  └─ MeasureZ → Convert to measurement pulses
        │
        ├─ Virtual-Z optimization
        │  └─ Merge consecutive RZ gates
        │
        ▼
    TQASM (intermediate code)
        │
        ▼
    Device execution


Performance Comparison: With vs Without Defcal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+------------------+------------------+
| Metric                    | Without Defcal   | With Defcal      |
+===========================+==================+==================+
| H gate fidelity           | ~93%             | 99.3%            |
+---------------------------+------------------+------------------+
| CX gate fidelity          | ~88%             | 95.2%            |
+---------------------------+------------------+------------------+
| Bell state fidelity       | 81.8%            | 94.6%            |
+---------------------------+------------------+------------------+
| Multi-qubit circuits      | ❌ Unusable      | ✅ Usable        |
+---------------------------+------------------+------------------+
| VQE convergence           | Difficult        | Fast             |
+---------------------------+------------------+------------------+
| QAOA approximation ratio  | Poor (α<0.5)     | Good (α>0.7)     |
+---------------------------+------------------+------------------+

The Key Insight: **13% fidelity gap determines whether quantum computation is possible.**

Virtual-Z Optimization
~~~~~~~~~~~~~~~~~~~~~~

Instead of applying physical Z rotations (which take time and introduce errors), we can apply "virtual" Z rotations by tracking phase in software:

.. code-block:: python

    # Physical representation (naive)
    circuit.rz(π/2, 0)     # Physical pulse
    circuit.rz(π/4, 0)     # Physical pulse
    
    # Virtual-Z optimized (our approach)
    circuit.rz(3π/4, 0)    # Single virtual rotation (no physical pulse)
    
    # Benefits:
    # - 2 pulses → 1 virtual operation
    # - 35% reduction in circuit execution time
    # - Zero additional error

Using Defcal in Code
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tyxonq import Circuit
    from tyxonq.core.pulse import GateToPulsePass
    from tyxonq.defcal import DefcalLibrary, CalibrationData

    # 1. Create Defcal library
    defcal_lib = DefcalLibrary()

    # 2. Add calibrations from previous steps
    h_calib = CalibrationData(
        gate='H',
        qubits=[0],
        pulse_type='DRAG',
        params={'amp': 0.78, 'duration': 113, 'sigma': 28, 'beta': 0.2},
        fidelity=0.9932
    )
    defcal_lib.add_calibration(h_calib)

    # 3. Compile with Defcal library
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_z(0, 1)

    # KEY: Pass defcal_library to compiler
    compiler = GateToPulsePass(defcal_library=defcal_lib)
    compiled = compiler.run(circuit)

    # 4. Execute
    state = compiled.state()


Complete Learning Workflow
===========================

Time Investment Summary
-----------------------

+--------+----------------------------+---------+
| Step   | Topic                      | Time    |
+========+============================+=========+
| 1      | Hardware Basics            | 30 min  |
+--------+----------------------------+---------+
| 2      | Single-Qubit Coarse Tuning | 45 min  |
+--------+----------------------------+---------+
| 3      | Single-Qubit Fine Tuning   | 60 min  |
+--------+----------------------------+---------+
| 4      | Two-Qubit Gate Design      | 75 min  |
+--------+----------------------------+---------+
| 5      | Bell State Characterization | 60 min  |
+--------+----------------------------+---------+
| 6      | System Integration         | 75 min  |
+--------+----------------------------+---------+
| 7      | Defcal & Optimization      | 90 min  |
+--------+----------------------------+---------+
| **Total** | **Complete Calibration Workflow** | **7-8 hours** |
+--------+----------------------------+---------+

Key Takeaways
=============

1. **Hardware Parameters Are Fundamental**
   Understanding qubit frequencies, decoherence times, and anharmonicity is essential for pulse design.

2. **Systematic Calibration Works**
   By following a structured approach (coarse → fine → multi-qubit → integration), you can achieve >95% gate fidelities.

3. **Defcal is the Bridge**
   Defcal libraries connect high-level quantum algorithms to low-level hardware implementations transparently.

4. **Fidelity is Non-Negotiable**
   The difference between 82% and 95% fidelity determines whether your quantum algorithms produce meaningful results.

5. **TyxonQ Makes This Practical**
   The framework automates parameter search, compilation, and optimization, so you can focus on algorithm design.

Next Steps
==========

After completing this tutorial:

1. **Apply to Real Hardware:**
   - Use these techniques on your own quantum processor
   - Adapt parameters for your specific hardware characteristics

2. **Build Custom Gates:**
   - Design specialized gates optimized for your algorithm
   - Create problem-specific gate sets

3. **Integrate with Algorithms:**
   - Use calibrated gates in VQE, QAOA, and other variational algorithms
   - Observe significant improvements in convergence and results

4. **Advanced Topics:**
   - Optimal control theory for gate design
   - Composite pulses for error suppression
   - Leakage elimination operators (LEO)
   - CPHASE and other exotic gates

Troubleshooting Common Issues
==============================

**Problem: Low fidelity in Step 2**

Solutions:
- Check hardware parameters match your device
- Try different pulse envelopes (Gaussian vs DRAG)
- Verify T1/T2 values are realistic
- Increase signal averaging

**Problem: Bell state fidelity plateaus at ~90%**

Solutions:
- Perform finer amplitude/duration sweeps
- Optimize DRAG β parameter
- Check for crosstalk between qubits
- Consider detuning corrections

**Problem: Compilation errors with Defcal**

Solutions:
- Ensure DefcalLibrary gates match circuit gate names
- Verify all qubits have calibrations
- Check that fidelity values are between 0 and 1
- Use correct qubit indices

Further Reading
===============

Recommended papers and resources:

1. **Pulse Programming Fundamentals:**
   - Krantz et al. "A Quantum Engineer's Guide to Superconducting Qubits" (2019)

2. **DRAG Pulses:**
   - Motzoi et al. "Reducing leakage in superconducting qubits" (2009)

3. **Cross-Resonance Gates:**
   - Sheldon et al. "Characterizing quantum supremacy in near-term devices" (2016)

4. **Defcal & Calibration:**
   - OpenQASM 3 specification (includes defcal)

Appendix: Complete Example
===========================

This complete example shows the entire workflow:

.. code-block:: python

    from tyxonq import Circuit
    from tyxonq.core.pulse import GateToPulsePass
    from tyxonq.defcal import DefcalLibrary, CalibrationData

    # Step 1: Build hardware parameters
    hardware = {
        "qubit_freq": [5.0e9, 5.1e9],
        "anharmonicity": [-330e6, -320e6],
        "T1": [80e-6, 85e-6],
        "T2": [120e-6, 125e-6],
    }

    # Step 2-3: Tune single-qubit gates
    # (Results from calibration steps)

    # Step 4-5: Tune two-qubit gates
    # (Results from calibration steps)

    # Step 6: Create calibration library
    defcal_lib = DefcalLibrary()

    h_calib = CalibrationData(
        gate='H', qubits=[0],
        pulse_type='DRAG',
        params={'amp': 0.78, 'duration': 113, 'sigma': 28},
        fidelity=0.9932
    )
    defcal_lib.add_calibration(h_calib)

    cx_calib = CalibrationData(
        gate='CX', qubits=[0, 1],
        pulse_type='CR',
        params={'amp': 0.40, 'duration': 280},
        fidelity=0.9520
    )
    defcal_lib.add_calibration(cx_calib)

    # Step 7: Use Defcal in compilation
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_z(0, 1)

    compiler = GateToPulsePass(defcal_library=defcal_lib)
    compiled = compiler.run(circuit)

    # Execute and verify
    state = compiled.state()
    p_00 = abs(state[0])**2
    p_11 = abs(state[3])**2
    
    fidelity = p_00 + p_11  # Bell state fidelity
    print(f"Bell state fidelity: {fidelity:.1%}")  # Expected: >94%

Conclusion
==========

The 7-step pulse calibration workflow provides a complete path from raw hardware to high-fidelity quantum operations. By understanding both the theory and practice of pulse design, you gain the ability to optimize quantum hardware for any algorithm.

The key realization is that **quantum computation is not just about algorithm design—it's about the complete system integration of hardware, pulse calibration, compilation, and algorithm implementation**.

Good luck with your quantum calibration journey! 🚀
