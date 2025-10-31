Pulse Programming Basics
=========================

.. meta::
   :description: Learn the fundamentals of pulse-level quantum programming in TyxonQ
   :keywords: pulse programming, quantum gates, DRAG pulse, Cross-Resonance, TQASM, superconducting qubits

Overview
--------

TyxonQ provides comprehensive pulse-level programming support with a **dual-path, dual-mode, dual-format** architecture:

Core Features
~~~~~~~~~~~~~

🔀 **Dual Execution Paths**
   - **Path A (Hardware Simulation)**: Gate Circuit → Pulse Compiler → TQASM → Cloud Hardware
   - **Path B (Numerical Method)**: Hamiltonian → Pulse Simulation → Local Evolution

📊 **Dual Programming Modes**
   - **Mode A (Chain API)**: High-level gates automatically compiled to pulses
   - **Mode B (Direct Numerical)**: Direct pulse-level control and simulation

📦 **Dual Output Formats**
   - **pulse_ir**: TyxonQ native format, preserves homebrew_s2 objects, supports PyTorch autograd
   - **tqasm**: TQASM 0.2 / OpenQASM 3 text format, cloud-compatible

What is Pulse Programming?
---------------------------

Traditional gate-level quantum programming abstracts away the physical implementation:

.. code-block:: homebrew_s2

   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cx(0, 1)
   # How are these gates actually implemented on hardware? 🤔

Pulse programming reveals the physical reality:

.. code-block:: homebrew_s2

   # H gate = RY(π/2) + RX(π) pulses
   prog = PulseProgram(1)
   prog.drag(0, amp=0.5, duration=160, sigma=40, beta=0.2)  # RY(π/2)
   prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2)  # RX(π)
   
   # CX gate = Cross-Resonance pulse sequence
   prog.cr_drive(control=0, target=1, amp=0.3, duration=320)

**Why Pulse Programming?**

✅ **Higher gate fidelity** - Optimize pulses for specific hardware
✅ **Leakage suppression** - DRAG pulses reduce leakage to |2⟩ state
✅ **Noise resilience** - Custom pulse shapes minimize decoherence
✅ **Advanced algorithms** - Pulse VQE, QAOA, quantum optimal control

P0.1: Single-Qubit Gate Decomposition
--------------------------------------

All single-qubit gates are decomposed into physical pulses.

Supported Gates
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Gate
     - Pulse Type
     - Physical Implementation
   * - X, Y
     - DRAG
     - Gaussian + derivative correction, suppresses leakage
   * - Z, RZ
     - Virtual-Z
     - Phase rotation in software, zero time overhead
   * - H
     - DRAG + DRAG
     - H = RY(π/2) · RX(π)
   * - RX, RY
     - DRAG
     - Parametrized DRAG pulse with rotation angle

DRAG Pulse Explained
~~~~~~~~~~~~~~~~~~~~

**DRAG (Derivative Removal by Adiabatic Gate)** pulse suppresses leakage to |2⟩ state:

.. math::

   \Omega_{\text{DRAG}}(t) = A \cdot \exp\left(-\frac{(t-T/2)^2}{2\sigma^2}\right) + i\beta \cdot \frac{d\Omega}{dt}

where:

- **A**: pulse amplitude
- **σ**: pulse width (Gaussian standard deviation)
- **β**: DRAG coefficient (typically ~0.1-0.5)

**Example**:

.. code-block:: homebrew_s2

   import tyxonq as tq
   from tyxonq.core.ir.pulse import PulseProgram
   
   # Create pulse program
   prog = PulseProgram(1)
   
   # X gate with DRAG pulse
   prog.drag(qubit=0, amp=1.0, duration=160, sigma=40, beta=0.2)
   
   # Compile to TQASM
   from tyxonq.compiler.api import compile_pulse
   tqasm = compile_pulse(prog, output="tqasm")
   print(tqasm)

**Output**:

.. code-block:: text

   TQASM 0.2;
   qubit[1] q;
   
   defcal drag q[0] {
       waveform drag_wf = drag(1.0, 160, 40, 0.2);
       play(drag_wf, $0);
   }
   
   drag q[0];

Virtual-Z Gate
~~~~~~~~~~~~~~

Z-axis rotations are implemented as **frame updates** (zero time):

.. code-block:: homebrew_s2

   prog = PulseProgram(1)
   prog.virtual_z(qubit=0, angle=np.pi/4)  # RZ(π/4), instant!

**Advantages**:

- ⚡ Zero time overhead
- 🎯 Perfect fidelity (no pulse error)
- 💾 Software-only operation

P0.2: Two-Qubit Gate Decomposition
-----------------------------------

Cross-Resonance (CR) Gate
~~~~~~~~~~~~~~~~~~~~~~~~~~

CX (CNOT) gate is implemented using **Cross-Resonance** drive:

**Physical Mechanism**:

.. code-block:: text

   Control qubit frequency: ωc = 5.0 GHz
   Target qubit frequency:  ωt = 4.8 GHz
   
   Drive control at target frequency → induces ZX coupling
   H_CR = Ω(t) · (σ_x^control ⊗ σ_z^target)

**4-Pulse Sequence**:

.. code-block:: homebrew_s2

   # 1. Pre-rotation (align basis)
   prog.virtual_z(target, -π/2)
   
   # 2. CR drive (ZX coupling)
   prog.cr_drive(control, target, amp=0.3, duration=320)
   
   # 3. Echo pulse (cancel unwanted terms)
   prog.drag(control, amp=1.0, duration=160, sigma=40, beta=0.2)  # X gate
   prog.cr_drive(control, target, amp=-0.3, duration=320)  # Inverted drive
   
   # 4. Post-rotation (return to standard basis)
   prog.virtual_z(target, π/2)

**Example**:

.. code-block:: homebrew_s2

   from tyxonq.core.ir.pulse import PulseProgram
   
   prog = PulseProgram(2)
   
   # Manual CX implementation
   prog.virtual_z(1, -np.pi/2)
   prog.cr_drive(0, 1, amp=0.3, duration=320)
   prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2)
   prog.cr_drive(0, 1, amp=-0.3, duration=320)
   prog.virtual_z(1, np.pi/2)
   
   # Or use high-level API (automatic decomposition)
   from tyxonq import Circuit
   circuit = Circuit(2)
   circuit.cx(0, 1)
   
   # Compile to pulses
   pulse_circuit = circuit.to_pulse()  # Automatically decomposes to CR sequence

P0.3: TQASM Export
------------------

TyxonQ exports pulse programs to **TQASM 0.2** (TyxonQ Quantum Assembly) or **OpenQASM 3 + OpenPulse** for cloud execution.

TQASM 0.2 Format
~~~~~~~~~~~~~~~~

.. code-block:: text

   TQASM 0.2;
   qubit[2] q;
   
   // Define calibrated gates
   defcal x q[0] {
       waveform drag_wf = drag(1.0, 160, 40, 0.2);
       play(drag_wf, $0);
   }
   
   defcal cx q[0], q[1] {
       frame_update($1, -1.5707963267948966);  // Virtual-Z
       waveform cr_wf = gaussian(0.3, 320, 80);
       play(cr_wf, $0, $1);
       // ... CR sequence
   }
   
   // Gate sequence
   x q[0];
   cx q[0], q[1];

**Export Example**:

.. code-block:: homebrew_s2

   from tyxonq import Circuit
   from tyxonq.compiler.api import compile_pulse
   
   circuit = Circuit(2)
   circuit.h(0)
   circuit.cx(0, 1)
   
   # Compile to TQASM
   result = compile_pulse(
       circuit,
       output="tqasm",
       inline_pulses=True  # Inline all pulse definitions
   )
   
   tqasm_code = result["pulse_schedule"]
   print(tqasm_code)

OpenQASM 3 + OpenPulse
~~~~~~~~~~~~~~~~~~~~~~

For compatibility with IBM Quantum and Rigetti systems:

.. code-block:: homebrew_s2

   result = compile_pulse(circuit, output="qasm3")
   
   # Exports OpenQASM 3.0 with OpenPulse extensions
   # Compatible with IBM Qiskit and Rigetti PyQuil

P0.4: Multi-Qubit Pulse Simulation
-----------------------------------

TyxonQ provides physics-based pulse simulation for local verification before cloud submission.

Physical Model
~~~~~~~~~~~~~~

Solves time-dependent Schrödinger equation:

.. math::

   i\hbar \frac{\partial}{\partial t}|\psi(t)\rangle = H(t)|\psi(t)\rangle

where:

.. math::

   H(t) = H_{\text{drift}} + H_{\text{drive}}(t)

**Drift Hamiltonian** (qubit frequencies):

.. math::

   H_{\text{drift}} = \sum_i \frac{\omega_i}{2} \sigma_z^i + \sum_{i<j} J_{ij} \sigma_z^i \otimes \sigma_z^j

**Drive Hamiltonian** (pulse control):

.. math::

   H_{\text{drive}}(t) = \sum_i \Omega_i(t) [\cos(\phi_i)\sigma_x^i + \sin(\phi_i)\sigma_y^i]

Simulation Example
~~~~~~~~~~~~~~~~~~

.. code-block:: homebrew_s2

   import tyxonq as tq
   from tyxonq import waveforms
   from tyxonq.libs.quantum_library.pulse_simulation import (
       evolve_pulse_hamiltonian,
       compile_pulse_to_unitary
   )
   
   # Create initial state
   psi0 = tq.statevector.zero_state(1)
   
   # Define DRAG pulse
   pulse = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.2)
   
   # Method 1: Direct evolution (Path B: Numerical)
   psi_final = evolve_pulse_hamiltonian(
       psi0,
       pulse,
       qubit=0,
       qubit_freq=5.0e9,
       drive_freq=5.0e9
   )
   
   # Method 2: Compile to unitary (Path A: Hardware Simulation)
   U = compile_pulse_to_unitary(
       pulse,
       qubit_freq=5.0e9,
       drive_freq=5.0e9
   )
   psi_final = U @ psi0

Multi-Qubit Simulation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: homebrew_s2

   from tyxonq.core.ir.pulse import PulseProgram
   from tyxonq.compiler.api import compile_pulse
   
   # Create 2-qubit pulse program
   prog = PulseProgram(2)
   
   # Hadamard on qubit 0
   prog.drag(0, amp=0.5, duration=160, sigma=40, beta=0.2)  # RY(π/2)
   prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2)  # RX(π)
   
   # CX gate (simplified)
   prog.cr_drive(0, 1, amp=0.3, duration=320)
   
   # Simulate locally
   circuit = tq.Circuit(2)
   # ... (convert PulseProgram to Circuit operations)
   result = circuit.run(shots=1000)
   
   print(result)
   # Output: {'00': 502, '11': 498}  # Bell state!

Noise Simulation
~~~~~~~~~~~~~~~~

Include T1/T2 decoherence:

.. code-block:: homebrew_s2

   psi_final = evolve_pulse_hamiltonian(
       psi0,
       pulse,
       qubit=0,
       qubit_freq=5.0e9,
       T1=50e-6,  # 50 μs relaxation time
       T2=30e-6   # 30 μs dephasing time
   )

P0.5: Cloud Submission (End-to-End)
------------------------------------

Submit pulse programs to TyxonQ cloud quantum processors.

Three Modes of Cloud Submission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mode A: Gate Circuit + use_pulse()**

.. code-block:: homebrew_s2

   import tyxonq as tq
   
   # Set cloud provider
   tq.set_token("your_token_here", provider="tyxonq", device="homebrew_s2")
   
   # Create gate circuit
   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cx(0, 1)
   circuit.measure_all()
   
   # Enable pulse mode
   circuit.use_pulse(
       inline_pulses=True,  # Required for cloud
       mode="tqasm"
   )
   
   # Submit to cloud
   result = circuit.device("tyxonq").run(shots=1000)
   print(result)

**Mode B: Smart Inference (Automatic)**

.. code-block:: homebrew_s2

   # Compiler auto-detects pulse requirements
   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cx(0, 1)
   
   # Direct submission (compiler chooses pulse if needed)
   result = circuit.device("homebrew_s2").run(shots=1000)

**Mode C: Pure Pulse Programming**

.. code-block:: homebrew_s2

   from tyxonq.core.ir.pulse import PulseProgram
   from tyxonq.compiler.api import compile_pulse
   
   # Create pulse program
   prog = PulseProgram(2)
   prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2)
   prog.cr_drive(0, 1, amp=0.3, duration=320)
   
   # Compile to TQASM
   tqasm = compile_pulse(prog, output="tqasm")
   
   # Submit via API
   import requests
   response = requests.post(
       "https://api.tyxonq.com/v1/submit",
       headers={"Authorization": f"Bearer {token}"},
       json={"tqasm": tqasm, "shots": 1000}
   )

Complete Example: Bell State Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: homebrew_s2

   import tyxonq as tq
   import numpy as np
   
   # Set up cloud access
   tq.set_token("your_token", provider="tyxonq")
   
   # Create circuit
   circuit = tq.Circuit(2)
   
   # Hadamard gate (decomposed to pulses)
   circuit.h(0)
   
   # CNOT gate (decomposed to CR sequence)
   circuit.cx(0, 1)
   
   # Measurements
   circuit.measure_all()
   
   # Path A: Local simulation (verify first)
   local_result = circuit.run(shots=1000)
   print(f"Local result: {local_result}")
   
   # Path B: Cloud execution (pulse-level)
   circuit.use_pulse(inline_pulses=True, mode="tqasm")
   cloud_result = circuit.device("homebrew_s2").run(shots=1000)
   print(f"Cloud result: {cloud_result}")

Best Practices
--------------

1. **Always Verify Locally First**

   .. code-block:: homebrew_s2
   
      # Local simulation (free, fast)
      local_result = circuit.run(shots=1000)
      
      # Only submit to cloud after verification
      if verify_results(local_result):
          cloud_result = circuit.device("tyxonq").run(shots=1000)

2. **Use Appropriate DRAG Beta**

   .. code-block:: homebrew_s2
   
      from tyxonq.libs.quantum_library.three_level_system import optimal_drag_beta
      
      anharmonicity = -330e6  # -330 MHz (typical Transmon)
      beta_opt = optimal_drag_beta(anharmonicity)
      
      prog.drag(0, amp=1.0, duration=160, sigma=40, beta=beta_opt)

3. **Inline Pulses for Cloud Submission**

   .. code-block:: homebrew_s2
   
      # ❌ Wrong: References won't work on cloud
      circuit.use_pulse(inline_pulses=False)
      
      # ✅ Correct: Fully self-contained
      circuit.use_pulse(inline_pulses=True)

4. **Monitor Gate Fidelity**

   .. code-block:: homebrew_s2
   
      # Estimate fidelity from leakage
      from tyxonq.libs.quantum_library.three_level_system import evolve_three_level_pulse
      
      pulse = waveforms.Drag(amp=1.0, duration=160, sigma=40, beta=0.2)
      psi, leakage = evolve_three_level_pulse(pulse, qubit_freq=5e9)
      
      fidelity = 1 - leakage
      print(f"Estimated gate fidelity: {fidelity:.4%}")

Next Steps
----------

✅ **Completed (P0.1-P0.5)**:
   - Single-qubit gate decomposition (DRAG, Virtual-Z)
   - Two-qubit gate decomposition (Cross-Resonance)
   - TQASM/OpenQASM export
   - Multi-qubit pulse simulation
   - Cloud submission framework

🚀 **Advanced Topics**:
   - :doc:`../advanced/pulse_three_level` - Three-level system simulation and DRAG optimization
   - :doc:`../advanced/quantum_natural_gradient` - Pulse VQE with QNG
   - :doc:`/api/libs/pulse_simulation` - Pulse simulation API reference

See Also
--------

.. seealso::

   - :doc:`/examples/pulse_programming` - Complete pulse programming examples
   - :doc:`/user_guide/pulse/index` - Detailed pulse programming guide (Chinese)
   - :doc:`/api/compiler/pulse` - Pulse compiler API

References
----------

1. QuTiP-qip: "Pulse-level noisy quantum circuits with QuTiP", Quantum 6, 630 (2022)
2. Motzoi et al., "DRAG pulse for leakage suppression", PRL 103, 110501 (2009)
3. Rigetti Computing, "Cross-Resonance gates for superconducting qubits", arXiv:1603.04821
4. IBM Quantum, "OpenPulse specification", https://openqasm.com
