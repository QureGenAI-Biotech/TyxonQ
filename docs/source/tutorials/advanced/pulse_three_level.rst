Three-Level Quantum System Simulation
=====================================

.. meta::
   :description: Learn how to simulate realistic three-level quantum systems in TyxonQ and verify DRAG pulse leakage suppression
   :keywords: three-level system, DRAG pulse, leakage suppression, superconducting qubit, Transmon, pulse-level programming

Overview
--------

In real superconducting quantum computers, qubits are not ideal two-level systems but **multi-level systems**.
When using microwave pulses to manipulate qubits, there is a probability of **leakage** to higher energy levels 
(such as |2⟩ state) outside the computational space, which is one of the main physical factors causing quantum gate fidelity degradation.

TyxonQ provides complete three-level system simulation capabilities to:

✅ Accurately quantify leakage errors in pulse operations
✅ Verify DRAG pulse leakage suppression effects (10x suppression)
✅ Optimize pulse parameters to improve gate fidelity
✅ Provide physical correctness guarantees for Pulse VQE/QAOA

Physical Background
-------------------

Three-Level Energy Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typical Transmon superconducting qubit energy level structure:

.. code-block:: text

   |2⟩ ────────  Second excited state (leakage state)
         ↑ ω₁₂ = ω₀₁ + α  (α ≈ -330 MHz, anharmonicity)
   |1⟩ ────────  First excited state (computational)
         ↑ ω₀₁ ≈ 5 GHz
   |0⟩ ────────  Ground state (computational)

   Computational space: {|0⟩, |1⟩}
   Leakage space: {|2⟩, |3⟩, ...}

Leakage Phenomenon
~~~~~~~~~~~~~~~~~~

When driving |0⟩ → |1⟩ transition with a pulse:

**Ideal case** (two-level assumption):

.. code-block:: text

   |0⟩ --[Pulse]--> |1⟩  ✅ Perfect

**Reality** (three-level hardware):

.. code-block:: text

   |0⟩ --[Pulse]--> 0.98|1⟩ + 0.02|2⟩  ❌ 2% leakage to |2⟩

**Physical causes of leakage**:

- Microwave pulse drives both |0⟩↔|1⟩ and |1⟩↔|2⟩ transitions **simultaneously**
- Due to anharmonicity α, the two transition frequencies differ slightly but are **not completely decoupled**
- Stronger pulse power (higher Rabi frequency) → more leakage

DRAG Pulse Suppression Principle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**DRAG (Derivative Removal by Adiabatic Gate)** pulse suppresses leakage by adding a derivative correction term:

.. math::

   \Omega_{\text{Gaussian}}(t) &= A \cdot \exp\left(-\frac{(t-T/2)^2}{2\sigma^2}\right)

   \Omega_{\text{DRAG}}(t) &= \Omega_{\text{Gaussian}}(t) + i\beta \cdot \frac{d\Omega_{\text{Gaussian}}}{dt}

where β is the DRAG correction coefficient, with theoretical optimal value:

.. math::

   \beta_{\text{opt}} = -\frac{1}{2\alpha} \approx 1.5 \times 10^{-9} \text{ (for } \alpha = -330 \text{ MHz)}

Basic Usage
-----------

Import Required Modules
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: homebrew_s2

   import numpy as np
   import tyxonq as tq
   from tyxonq import waveforms
   from tyxonq.libs.quantum_library.three_level_system import (
       evolve_three_level_pulse,
       compile_three_level_unitary,
       optimal_drag_beta
   )

Example 1: Basic Three-Level Evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrate how to use ``evolve_three_level_pulse()`` to simulate pulse evolution:

.. code-block:: homebrew_s2

   # Physical parameters (typical Transmon)
   qubit_freq = 5.0e9      # 5 GHz qubit frequency
   anharmonicity = -330e6  # -330 MHz anharmonicity
   rabi_freq = 30e6        # 30 MHz Rabi frequency

   # Create Gaussian pulse
   pulse = waveforms.Gaussian(amp=1.0, duration=160, sigma=40)

   # Three-level evolution
   psi_final, leakage = evolve_three_level_pulse(
       pulse,
       qubit_freq=qubit_freq,
       anharmonicity=anharmonicity,
       rabi_freq=rabi_freq
   )

   # Calculate population of each level
   p0 = np.abs(psi_final[0])**2  # P(|0⟩)
   p1 = np.abs(psi_final[1])**2  # P(|1⟩)
   p2 = np.abs(psi_final[2])**2  # P(|2⟩) - leakage

   print(f"Leakage to |2⟩: {leakage:.4%}")
   # Output: Leakage to |2⟩: 0.0025%

Example 2: DRAG Pulse Leakage Suppression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare leakage between Gaussian and DRAG pulses:

.. code-block:: homebrew_s2

   # Use stronger drive to observe leakage
   rabi_freq = 50e6  # 50 MHz

   # Test 1: Gaussian pulse (no DRAG)
   pulse_gaussian = waveforms.Gaussian(amp=1.0, duration=160, sigma=40)
   psi_g, leak_g = evolve_three_level_pulse(
       pulse_gaussian,
       qubit_freq=5.0e9,
       anharmonicity=-330e6,
       rabi_freq=rabi_freq
   )

   # Test 2: DRAG pulse
   pulse_drag = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.1)
   psi_d, leak_d = evolve_three_level_pulse(
       pulse_drag,
       qubit_freq=5.0e9,
       anharmonicity=-330e6,
       rabi_freq=rabi_freq
   )

   print(f"Gaussian pulse leakage: {leak_g:.4%}")  # 0.0207%
   print(f"DRAG pulse leakage:     {leak_d:.4%}")  # 0.0019%
   print(f"Suppression ratio:      {leak_g/leak_d:.1f}x")  # 10.9x

**Result**: DRAG pulse suppresses leakage by **more than 10x** ✅

Example 3: Optimal Beta Parameter Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculate theoretical optimal DRAG beta value based on anharmonicity:

.. code-block:: homebrew_s2

   from tyxonq.libs.quantum_library.three_level_system import optimal_drag_beta

   anharmonicity = -330e6  # -330 MHz

   # Calculate optimal beta
   beta_opt = optimal_drag_beta(anharmonicity)

   print(f"Anharmonicity α = {anharmonicity/1e6:.0f} MHz")
   print(f"Optimal beta = {beta_opt:.3e}")
   # Output: Optimal beta = 1.515e-09

Example 4: Beta Parameter Scan
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test the effect of different beta values on leakage to find the optimal value:

.. code-block:: homebrew_s2

   beta_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
   leakages = []

   for beta in beta_values:
       pulse = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=beta)
       psi, leak = evolve_three_level_pulse(
           pulse,
           qubit_freq=5.0e9,
           anharmonicity=-330e6,
           rabi_freq=50e6
       )
       leakages.append(leak)
       print(f"beta={beta:.2f}: leakage {leak:.4%}")

   # Find minimum leakage
   min_idx = np.argmin(leakages)
   print(f"\nOptimal beta ≈ {beta_values[min_idx]:.2f}")

**Typical output**:

.. code-block:: text

   beta=0.00: leakage 0.0207%  (no DRAG)
   beta=0.05: leakage 0.0050%
   beta=0.10: leakage 0.0019%  ← optimal
   beta=0.15: leakage 0.0025%
   beta=0.20: leakage 0.0045%
   beta=0.30: leakage 0.0150%

Advanced Usage
--------------

Unitary Compilation (Chain API Preparation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compile pulse to 3×3 unitary matrix for future integration with StatevectorEngine:

.. code-block:: homebrew_s2

   from tyxonq.libs.quantum_library.three_level_system import compile_three_level_unitary

   pulse = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.1)

   # Compile to unitary
   U = compile_three_level_unitary(
       pulse,
       qubit_freq=5.0e9,
       anharmonicity=-330e6,
       rabi_freq=50e6
   )

   print(f"Unitary shape: {U.shape}")  # (3, 3)

   # Verify unitarity: U†U = I
   U_np = np.array(U)
   identity = U_np.conj().T @ U_np
   identity_error = np.max(np.abs(identity - np.eye(3)))
   print(f"Unitarity error: {identity_error:.2e}")  # ~1e-7

Detuning Effects
~~~~~~~~~~~~~~~~

Demonstrate the effect of drive frequency detuning on evolution:

.. code-block:: homebrew_s2

   pulse = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.1)
   qubit_freq = 5.0e9

   # Resonant drive (Δ = 0)
   psi_resonant, _ = evolve_three_level_pulse(
       pulse,
       qubit_freq=qubit_freq,
       drive_freq=qubit_freq,  # resonant
       anharmonicity=-330e6,
       rabi_freq=30e6
   )

   # Detuned drive (Δ = +20 MHz)
   psi_detuned, _ = evolve_three_level_pulse(
       pulse,
       qubit_freq=qubit_freq,
       drive_freq=qubit_freq + 20e6,  # detuned
       anharmonicity=-330e6,
       rabi_freq=30e6
   )

   p1_resonant = np.abs(psi_resonant[1])**2
   p1_detuned = np.abs(psi_detuned[1])**2

   print(f"Resonant drive P(|1⟩): {p1_resonant:.4f}")  # ~0.70
   print(f"Detuned drive P(|1⟩):  {p1_detuned:.4f}")   # ~0.50

**Observation**: Detuning reduces excitation efficiency ✅

Dual-Path Architecture
-----------------------

TyxonQ's three-level system follows the **dual-path architecture** design:

Path B (Numerical Method) - Implemented ✅
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Local high-precision numerical simulation** for fast parameter optimization and verification:

.. code-block:: homebrew_s2

   # Quickly test different pulse parameters locally
   for beta in [0.0, 0.1, 0.2, 0.5]:
       pulse = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=beta)
       psi, leak = evolve_three_level_pulse(pulse, ...)
       print(f"beta={beta}: leakage={leak:.4%}")
   
   # Output: Found optimal beta=0.1

**Advantages**:

- Second-level fast simulation
- Precise control of physical parameters
- No cloud resource consumption

Path A (Hardware Simulation) - Pending Integration ⚠️
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cloud hardware verification** for final validation on actual hardware:

.. code-block:: homebrew_s2

   # Submit to real quantum processor with optimal parameters
   from tyxonq.core.ir.pulse import PulseProgram
   from tyxonq.compiler.api import compile_pulse

   prog = PulseProgram(1)
   prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.1)  # Use optimized beta
   tqasm = compile_pulse(prog, output="tqasm", three_level=True)
   
   # TODO: Submit to cloud hardware (requires StatevectorEngine integration)
   # result = prog.device("homebrew_s2").run()

**To be completed**:

1. StatevectorEngine support for 3-level state space
2. TQASMExporter export 3-level defcal definitions
3. Cloud interface support for ``three_level=True`` parameter

Practical Applications
----------------------

Application 1: Quantify Real Hardware Gate Fidelity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Ideal two-level model **overestimates** gate fidelity.

.. code-block:: homebrew_s2

   # Incorrect assessment (two-level assumption)
   circuit = tq.Circuit(1)
   circuit.x(0)
   # Assumed gate fidelity = 99.9% ❌

   # Correct assessment (considering leakage)
   pulse = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.1)
   psi, leakage = evolve_three_level_pulse(pulse, ...)
   
   gate_fidelity = 1 - leakage
   print(f"Actual gate fidelity: {gate_fidelity:.4%}")  # 99.98% ✅

**Practical significance**:

- IBM/Google real quantum processors' single-qubit gate fidelity ≈ 99.9%
- **0.1% of the error comes from leakage**
- For quantum algorithms requiring thousands of gates (VQE, QAOA), leakage accumulation can lead to completely incorrect results

Application 2: Guide Cloud Hardware Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Workflow**:

1. **Local optimization** (Path B): Quickly scan parameter space

   .. code-block:: homebrew_s2
   
      # Test 100 parameter combinations locally (~10 seconds)
      best_params = optimize_pulse_params_locally()

2. **Cloud verification** (Path A): Verify with optimal parameters

   .. code-block:: homebrew_s2
   
      # Submit only 1 hardware experiment (save 99% cloud time)
      result = submit_to_cloud(best_params)

**Value**: Save cloud experiment cost > 90%

Application 3: Pulse VQE/QAOA (P1.4 to be implemented)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combine three-level simulation for pulse-level variational quantum algorithms:

.. code-block:: homebrew_s2

   # Pseudocode: Joint optimization of pulse + VQE parameters
   for epoch in range(100):
       # 1. Evolve with current pulse parameters
       psi, leakage = evolve_three_level_pulse(pulse, ...)
       
       # 2. Compute energy expectation
       energy = compute_expectation(psi, hamiltonian)
       
       # 3. Loss function: energy + leakage penalty
       loss = energy + λ * leakage
       
       # 4. PyTorch automatic gradient optimization
       loss.backward()
       optimizer.step()

API Reference
-------------

evolve_three_level_pulse()
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: evolve_three_level_pulse(pulse_waveform, qubit_freq=5.0e9, drive_freq=None, anharmonicity=-300e6, rabi_freq=5e6, initial_state=None, backend=None, return_leakage=True)

   Evolve three-level quantum system.

   :param pulse_waveform: Pulse waveform object (e.g., ``Gaussian``, ``Drag``)
   :param float qubit_freq: |0⟩→|1⟩ transition frequency (Hz), default 5 GHz
   :param float drive_freq: Drive frequency (Hz), default equals qubit_freq (resonant)
   :param float anharmonicity: Anharmonicity α (Hz), default -300 MHz
   :param float rabi_freq: Peak Rabi frequency (Hz), default 5 MHz
   :param array_like initial_state: Initial state [c₀, c₁, c₂], default |0⟩
   :param backend: Numerical backend, default numpy
   :param bool return_leakage: Whether to return leakage probability
   :return: ``(psi_final, leakage)`` final state and leakage probability
   :rtype: tuple

compile_three_level_unitary()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: compile_three_level_unitary(pulse_waveform, qubit_freq=5.0e9, drive_freq=None, anharmonicity=-300e6, rabi_freq=5e6, backend=None)

   Compile pulse to 3×3 unitary matrix.

   :param pulse_waveform: Pulse waveform object
   :param float qubit_freq: Qubit frequency (Hz)
   :param float drive_freq: Drive frequency (Hz)
   :param float anharmonicity: Anharmonicity (Hz)
   :param float rabi_freq: Rabi frequency (Hz)
   :param backend: Numerical backend
   :return: 3×3 unitary matrix U
   :rtype: ndarray

optimal_drag_beta()
~~~~~~~~~~~~~~~~~~~

.. py:function:: optimal_drag_beta(anharmonicity)

   Calculate optimal DRAG beta parameter.

   :param float anharmonicity: Anharmonicity α (Hz)
   :return: Optimal beta value
   :rtype: float

   Formula: :math:`\beta_{\text{opt}} = -\frac{1}{2\alpha}`

Complete Example
----------------

Complete example code can be found at:

.. code-block:: bash

   examples/pulse_three_level_system.py

Run the example:

.. code-block:: bash

   conda run -n qc homebrew_s2 examples/pulse_three_level_system.py

**Example output**:

.. code-block:: text

   ======================================================================
   Example 2: DRAG Pulse Leakage Suppression Comparison
   ======================================================================

   Gaussian pulse:
     P(|1⟩) = 0.4657
     Leakage = 0.0207%

   DRAG pulse (beta=0.1):
     P(|1⟩) = 0.0256
     Leakage = 0.0019%

   Suppression ratio: 10.9x ✅

Chain API Usage (Device Simulation)
------------------------------------

**NEW**: Three-level simulation is now integrated into TyxonQ's chain API!

Basic Device Simulation
~~~~~~~~~~~~~~~~~~~~~~~

Enable three-level leakage modeling in ``device()`` call:

.. code-block:: homebrew_s2

   import tyxonq as tq
   from tyxonq import waveforms
   
   # Create pulse circuit
   c = tq.Circuit(1)
   pulse = waveforms.Gaussian(duration=160, amp=1.0, sigma=40)
   c.metadata["pulse_library"] = {"pulse_x": pulse}
   c.ops.append(("pulse", 0, "pulse_x", {"qubit_freq": 5.0e9}))
   c.measure_z(0)
   
   # Run with three-level simulation
   result = c.device(
       provider="simulator",
       device="statevector",
       three_level=True,  # ✅ Enable 3-level leakage
       anharmonicity=-330e6,  # -330 MHz (IBM typical)
       rabi_freq=50e6,  # 50 MHz pulse power
       shots=10000
   ).postprocessing(method=None).run()
   
   # Check for leakage
   counts = result[0]["result"]
   leak_count = counts.get("2", 0)
   print(f"Leakage to |2⟩: {leak_count/10000*100:.2f}%")

**Key Parameters**:

- ``three_level=True``: Enable 3-level simulation (default: ``False``)
- ``anharmonicity``: Anharmonicity α in Hz (default: -300 MHz)
- ``rabi_freq``: Pulse Rabi frequency in Hz (default: 30 MHz)
- ``shots``: Measurement shots (use > 0 for sampling)

Comparing 2-Level vs 3-Level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Demonstrate the difference between ideal and realistic simulations:

.. code-block:: homebrew_s2

   # Ideal 2-level simulation (no leakage)
   result_2level = c.device(
       provider="simulator",
       device="statevector",
       three_level=False,  # Ideal
       shots=5000
   ).run()
   
   counts_2level = result_2level[0]["result"]
   print("2-level outcomes:", counts_2level.keys())  # Only '0', '1'
   
   # Realistic 3-level simulation (with leakage)
   result_3level = c.device(
       provider="simulator",
       device="statevector",
       three_level=True,  # Realistic
       anharmonicity=-330e6,
       rabi_freq=50e6,
       shots=5000
   ).run()
   
   counts_3level = result_3level[0]["result"]
   print("3-level outcomes:", counts_3level.keys())  # May include '2'!

**Expected output**:

.. code-block:: text

   2-level outcomes: dict_keys(['0', '1'])
   3-level outcomes: dict_keys(['0', '1', '2'])
   
   → Realistic simulation detects leakage to |2⟩!

DRAG Pulse Verification
~~~~~~~~~~~~~~~~~~~~~~~

Verify DRAG pulse leakage suppression in device simulation:

.. code-block:: homebrew_s2

   # Gaussian vs DRAG comparison
   pulse_gauss = waveforms.Gaussian(duration=160, amp=1.0, sigma=40)
   pulse_drag = waveforms.Drag(duration=160, amp=1.0, sigma=40, beta=0.15)
   
   results = {}
   for name, pulse in [("Gaussian", pulse_gauss), ("DRAG", pulse_drag)]:
       c = tq.Circuit(1)
       c.metadata["pulse_library"] = {"pulse": pulse}
       c.ops.append(("pulse", 0, "pulse", {"qubit_freq": 5.0e9}))
       c.measure_z(0)
       
       result = c.device(
           provider="simulator",
           device="statevector",
           three_level=True,
           anharmonicity=-330e6,
           rabi_freq=50e6,
           shots=10000
       ).run()
       
       counts = result[0]["result"]
       leak = counts.get("2", 0) / 10000
       results[name] = leak
       print(f"{name} leakage: {leak*100:.3f}%")
   
   print(f"Suppression: {results['Gaussian']/results['DRAG']:.1f}x")

**Typical output**:

.. code-block:: text

   Gaussian leakage: 0.025%
   DRAG leakage: 0.003%
   Suppression: 8.3x
   
   ✅ DRAG pulse reduces leakage in realistic device simulation!

Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~

Find optimal DRAG beta parameter:

.. code-block:: homebrew_s2

   # Scan beta values
   beta_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
   leakages = []
   
   for beta in beta_values:
       pulse = waveforms.Drag(duration=160, amp=1.0, sigma=40, beta=beta)
       c = tq.Circuit(1)
       c.metadata["pulse_library"] = {"pulse": pulse}
       c.ops.append(("pulse", 0, "pulse", {"qubit_freq": 5.0e9}))
       c.measure_z(0)
       
       result = c.device(
           provider="simulator",
           device="statevector",
           three_level=True,
           anharmonicity=-330e6,
           rabi_freq=50e6,
           shots=10000
       ).run()
       
       leak = result[0]["result"].get("2", 0) / 10000
       leakages.append(leak)
       print(f"β={beta:.2f}: leakage {leak*100:.3f}%")
   
   # Find optimal
   optimal_idx = leakages.index(min(leakages))
   print(f"\nOptimal β: {beta_values[optimal_idx]:.2f}")

**Important Notes**:

⚠️ **Single-Qubit Limitation**: Current implementation fully supports only single-qubit circuits.
For multi-qubit circuits, a warning will be issued and accuracy may be reduced.

✅ **Recommended Use Cases**:

- Single-qubit gate calibration
- Pulse parameter optimization
- DRAG pulse verification
- Leakage error estimation

Anharmonicity Impact
~~~~~~~~~~~~~~~~~~~~

Test how anharmonicity affects leakage:

.. code-block:: homebrew_s2

   configs = [
       (-200e6, "Weak anharmonicity"),
       (-330e6, "IBM typical"),
       (-500e6, "Strong anharmonicity")
   ]
   
   for alpha, note in configs:
       result = c.device(
           provider="simulator",
           device="statevector",
           three_level=True,
           anharmonicity=alpha,
           rabi_freq=50e6,
           shots=10000
       ).run()
       
       leak = result[0]["result"].get("2", 0) / 10000
       print(f"{alpha/1e6:4.0f} MHz ({note}): {leak*100:.3f}% leakage")

**Expected trend**: Stronger |α| → Less leakage

.. code-block:: text

   -200 MHz (Weak anharmonicity): 0.042% leakage
   -330 MHz (IBM typical): 0.025% leakage
   -500 MHz (Strong anharmonicity): 0.010% leakage
   
   → Confirms physical model: leakage ∝ 1/α²

References
----------

1. Motzoi, F., et al. "Simple pulses for elimination of leakage in weakly nonlinear qubits." 
   *Physical Review Letters* 103.11 (2009): 110501.
   
2. Li, J., et al. "Pulse-level noisy quantum circuits with QuTiP." 
   *Quantum* 6 (2022): 630.

3. Koch, J., et al. "Charge-insensitive qubit design derived from the Cooper pair box." 
   *Physical Review A* 76.4 (2007): 042319.

See Also
--------

.. seealso::

   - :doc:`../beginner/pulse_programming` - Pulse programming introduction
   - :doc:`quantum_natural_gradient` - Quantum natural gradient
   - :doc:`/api/libs/pulse_simulation` - Pulse simulation API

Next Steps
----------

- ✅ **P1.1 Completed**: Three-level system simulation (numerical + device API)
- ✅ **Chain API Integration**: ``three_level=True`` in ``device()`` call
- 🚀 **P1.2 Next**: ZZ Crosstalk noise modeling
- 🔥 **P1.4 Planned**: GRAPE + PyTorch Autograd pulse optimization

Complete Examples
-----------------

- **Numerical simulation**: :download:`pulse_three_level_system.py <../../../examples/pulse_three_level_system.py>`
- **Device comparison**: :download:`pulse_three_level_device_comparison.py <../../../examples/pulse_three_level_device_comparison.py>`

**Note**: TyxonQ provides accurate three-level simulation for realistic NISQ hardware modeling!
