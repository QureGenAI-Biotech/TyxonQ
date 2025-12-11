=================================================
iSWAP and SWAP Gate Pulse-Level Decomposition
=================================================

.. meta::
   :description: Complete guide to iSWAP and SWAP gate decomposition at pulse level with CX chain implementation
   :keywords: iSWAP, SWAP, two-qubit gates, pulse decomposition, CX chain, cross-resonance

Overview
--------

**NEW FEATURE**: TyxonQ now provides native support for iSWAP and SWAP gates with
automatic pulse-level decomposition.

This document covers:

✅ iSWAP and SWAP gate definitions and physical properties
✅ Pulse-level decomposition using CX chain
✅ Why both gates decompose identically
✅ Practical examples and use cases
✅ Three-level leakage simulation with iSWAP/SWAP
✅ Comparison with other two-qubit gates

Key Features
~~~~~~~~~~~~

- **Native Support**: Both gates available as first-class Circuit methods
- **Automatic Decomposition**: Gate → CX chain → Pulse sequence (transparent)
- **Hardware-Ready**: Compiles to cross-resonance pulses for superconducting qubits
- **Leakage-Aware**: Compatible with ``three_level=True`` simulation
- **Equivalent**: Both gates decompose to identical pulse sequences
- **Universal**: Works on all qubit topologies

Gate Definitions
----------------

iSWAP Gate
~~~~~~~~~~

**Mathematical Definition:**

The iSWAP gate exchanges quantum states and applies a relative phase:

.. math::

   \text{iSWAP} = \exp\left(-i\frac{\pi}{4} \sigma_x \otimes \sigma_x\right)

**Matrix Representation:**

.. code-block:: text

   [[1,  0,  0,  0 ],
    [0,  0, 1j,  0 ],
    [0, 1j,  0,  0 ],
    [0,  0,  0,  1 ]]

**State Transformations:**

.. code-block:: text

   iSWAP|00⟩ = |00⟩     (unchanged)
   iSWAP|01⟩ = i|10⟩    (swapped + phase i)
   iSWAP|10⟩ = i|01⟩    (swapped + phase i)
   iSWAP|11⟩ = |11⟩     (unchanged)

**Physical Interpretation:**

- Implements XX coupling (Heisenberg model)
- Energy-preserving interaction
- Common in Rydberg atom and trapped ion platforms
- Native on Rigetti and IonQ quantum processors

**Key Property**: The relative phase factor is crucial for algorithms like:

- Fermi-Hubbard model simulation
- Quantum chemistry (UCCSD ansatz)
- Variational quantum eigensolvers (VQE)

SWAP Gate
~~~~~~~~~

**Mathematical Definition:**

The SWAP gate exchanges quantum states without phase:

.. code-block:: text

   SWAP = [[1, 0, 0, 0],
           [0, 0, 1, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 1]]

**State Transformations:**

.. code-block:: text

   SWAP|00⟩ = |00⟩
   SWAP|01⟩ = |10⟩    (swapped, no phase)
   SWAP|10⟩ = |01⟩    (swapped, no phase)
   SWAP|11⟩ = |11⟩

**Properties:**

- SWAP² = I (applying twice = identity)
- SWAP is Hermitian (SWAP† = SWAP)
- Commutes with single-qubit gates on different qubits
- Useful for qubit routing and layout optimization

**Key Property**: Pure state exchange without phase modification.

Pulse-Level Decomposition
--------------------------

CX Chain Decomposition
~~~~~~~~~~~~~~~~~~~~~~

Both iSWAP and SWAP are decomposed using the same **3-CX chain**:

.. math::

   \text{iSWAP}(q_0, q_1) = \text{CX}(q_0,q_1) \cdot \text{CX}(q_1,q_0) \cdot \text{CX}(q_0,q_1)

   \text{SWAP}(q_0, q_1) = \text{CX}(q_0,q_1) \cdot \text{CX}(q_1,q_0) \cdot \text{CX}(q_0,q_1)

**Why the same decomposition?**

The physical difference (relative phase) is **absorbed into software** during execution:

1. **Gate level**: Distinct unitary matrices
2. **Pulse level**: Identical CX chain (phase handled elsewhere)
3. **Simulation**: Statevector simulator applies correct phase in software

**Reference:**

Shende & Markov, "Minimal universal two-qubit controlled-NOT-based circuits",
*Physical Review A* **72**, 062305 (2005) [arXiv:quant-ph/0308033]

Implementation in TyxonQ
~~~~~~~~~~~~~~~~~~~~~~~~

The decomposition is implemented in ``gate_to_pulse.py``:

.. code-block:: python

   def _decompose_iswap_gate(self, op, device_params, circuit):
       """Decompose iSWAP to CX chain."""
       q0, q1 = op[1], op[2]
       pulse_ops = []
       
       # Three CX gates
       cx_ops_1 = self._decompose_cx_gate(("cx", q0, q1), device_params, circuit)
       pulse_ops.extend(cx_ops_1)
       
       cx_ops_2 = self._decompose_cx_gate(("cx", q1, q0), device_params, circuit)
       pulse_ops.extend(cx_ops_2)
       
       cx_ops_3 = self._decompose_cx_gate(("cx", q0, q1), device_params, circuit)
       pulse_ops.extend(cx_ops_3)
       
       return pulse_ops

**Pulse Sequence** (per CX gate):

Each CX decomposes to **4 pulse types**:

.. code-block:: text

   CX(q0, q1) pulse sequence:
   ├─ Pre-RX pulse on q0 (DRAG Gaussian)
   ├─ CR (Cross-Resonance) on q0
   ├─ Echo (DRAG Gaussian) on q1 (calibration)
   └─ Post-RX pulse on q0 (DRAG Gaussian)

**Total pulse count**: 3 CX × 4 pulses/CX = 12 pulses (after optimization)

Usage Examples
--------------

Basic iSWAP Usage
~~~~~~~~~~~~~~~~~

Create and run a simple iSWAP circuit:

.. code-block:: python

   import tyxonq as tq
   
   # Create 2-qubit circuit
   c = tq.Circuit(2)
   c.h(0)           # Hadamard on q0 (superposition)
   c.iswap(0, 1)    # iSWAP gate
   c.measure_z(0).measure_z(1)
   
   # Run with 1000 shots
   result = c.device(
       provider="simulator",
       device="statevector"
   ).run(shots=1000)
   
   print(f"Measurement outcomes: {result}")

Basic SWAP Usage
~~~~~~~~~~~~~~~~

Create and run a SWAP circuit:

.. code-block:: python

   import tyxonq as tq
   
   # Prepare |10⟩ state (q0=1, q1=0)
   c = tq.Circuit(2)
   c.x(0)          # X gate on q0 → |1⟩
   c.swap(0, 1)    # After SWAP: q0=0, q1=1 (|01⟩)
   c.measure_z(0).measure_z(1)
   
   # Run
   result = c.device(provider="simulator", device="statevector").run(shots=1000)
   
   # Expected: Measurements should be {'01': 1000}

Multi-Qubit Circuit with iSWAP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mix iSWAP with other gates:

.. code-block:: python

   import tyxonq as tq
   
   c = tq.Circuit(3)
   c.h(0)                 # Prepare superposition on q0
   c.iswap(0, 1)         # Exchange q0 and q1
   c.cx(1, 2)            # Standard CNOT between q1 and q2
   c.iswap(1, 2)         # Exchange q1 and q2
   c.measure_z(0).measure_z(1).measure_z(2)
   
   result = c.device(provider="simulator", device="statevector").run(shots=500)

Fermi-Hubbard Model Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use iSWAP for physics simulation (physics-native interaction):

.. code-block:: python

   import tyxonq as tq
   import numpy as np
   
   # Simulate 1D Fermi-Hubbard chain (4 qubits)
   c = tq.Circuit(4)
   
   # Initial state preparation
   for i in range(4):
       c.h(i)
   
   # Hubbard hopping terms (iSWAP is natural for this)
   # H_hop = -t \sum_i (c†_i c_{i+1} + h.c.)
   for step in range(3):
       for i in range(3):
           c.iswap(i, i+1)  # Hopping along chain
   
   # Measurement
   for i in range(4):
       c.measure_z(i)
   
   result = c.device(provider="simulator", device="statevector").run(shots=1000)

Three-Level Leakage Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulate with realistic three-level transmon qubits:

.. code-block:: python

   import tyxonq as tq
   
   c = tq.Circuit(2)
   c.h(0)
   c.iswap(0, 1)
   c.measure_z(0).measure_z(1)
   
   # 2-level ideal simulation
   result_2level = c.device(
       provider="simulator",
       device="statevector",
       three_level=False  # Ideal qubits
   ).run(shots=1000)
   
   # 3-level realistic simulation (includes leakage)
   result_3level = c.device(
       provider="simulator",
       device="statevector",
       three_level=True,   # Enable leakage modeling
       rabi_freq=30e6,     # Rabi frequency for leakage calculation
   ).run(shots=1000)
   
   # Compare outcomes
   print(f"2-level: {result_2level}")
   print(f"3-level: {result_3level}")
   print(f"Leakage impact: {abs(result_3level - result_2level)}")

Export to TQASM (Cloud Submission)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export iSWAP/SWAP circuits for cloud submission:

.. code-block:: python

   import tyxonq as tq
   from tyxonq.compiler import compile_api
   
   # Create circuit with iSWAP
   c = tq.Circuit(2)
   c.h(0)
   c.iswap(0, 1)
   c.measure_z(0).measure_z(1)
   
   # Compile to TQASM (for cloud submission)
   result = compile_api(
       c,
       output="tqasm",
       options={
           "mode": "pulse_only",
           "device_params": {
               "qubit_freq": [5.0e9, 5.1e9],
               "anharmonicity": [-330e6, -320e6]
           }
       }
   )
   
   tqasm_code = result["circuit"]
   print("TQASM code ready for cloud submission:")
   print(tqasm_code)
   
   # Save to file
   with open("iswap_circuit.qasm", "w") as f:
       f.write(tqasm_code)

Comparison with Other Two-Qubit Gates
--------------------------------------

Gate Characteristics
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Two-Qubit Gate Comparison
   :header-rows: 1
   :widths: 15 15 15 15 15

   * - Property
     - iSWAP
     - SWAP
     - CX
     - RXX(π/2)
   * - State exchange
     - ✓ (with phase)
     - ✓ (no phase)
     - Partial
     - Partial
   * - Relative phase
     - π/2 phase
     - None
     - Variable
     - Fixed
   * - Native hardware
     - Rigetti, IonQ
     - Few platforms
     - Most platforms
     - Variable
   * - Good for routing
     - ✗
     - ✓
     - ✗
     - ✗
   * - Good for physics
     - ✓
     - ✗
     - ✓
     - ✓
   * - Pulse efficiency
     - 3 CX (decomposed)
     - 3 CX (decomposed)
     - 1 CR pulse
     - Variable

When to Use Each Gate
~~~~~~~~~~~~~~~~~~~~~

**Use iSWAP when:**

- Simulating condensed matter physics (Hubbard, Heisenberg models)
- Using quantum chemistry ansatz (UCCSD with XX interactions)
- Working on platforms with native iSWAP (Rigetti, IonQ)
- Need energy-preserving interactions
- Algorithm specifically requires the phase factor

**Use SWAP when:**

- Optimizing qubit routing
- Implementing layout-aware compilation
- Need pure state exchange without phase
- Rearranging qubit logical order
- Circuit requires permutation

**Use CX when:**

- Maximum universality needed
- Decomposing arbitrary unitaries
- Working on all superconducting platforms
- Need minimal gate count

Testing and Verification
------------------------

Unit Tests
~~~~~~~~~~

TyxonQ provides comprehensive test coverage:

.. code-block:: python

   # Tests in: tests_core_module/test_iswap_swap_pulse_decomposition.py
   
   pytest tests_core_module/test_iswap_swap_pulse_decomposition.py -v
   
   # Test categories:
   # 1. test_iswap_decomposition_to_cx_chain
   # 2. test_swap_decomposition_to_cx_chain
   # 3. test_iswap_swap_equivalence_in_pulse_form
   # 4. test_iswap_with_multiple_qubits
   # 5. test_swap_with_measurement
   # 6. test_iswap_decomposition_structure
   # 7. test_swap_gate_equivalence

Integration Tests
~~~~~~~~~~~~~~~~~

Test with three-level simulation:

.. code-block:: python

   # Tests in: tests_core_module/test_pulse_inline_tqasm_three_level.py
   
   pytest tests_core_module/test_pulse_inline_tqasm_three_level.py -v
   
   # Verifies:
   # - TQASM export (no three_level in syntax)
   # - Device chain parameter passing
   # - three_level in device().run()

Common Pitfalls and Solutions
------------------------------

Pitfall 1: Forgetting Phase Factor in iSWAP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Treating iSWAP like SWAP (ignoring phase)

**Solution**:

.. code-block:: python

   # ✗ WRONG (ignoring phase)
   # iswap = [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]  ← This is SWAP!
   
   # ✓ CORRECT (phase factor included)
   # iswap = [[1,0,0,0], [0,0,1j,0], [0,1j,0,0], [0,0,0,1]]  ← iSWAP
   
   # Use built-in gate (no manual matrix needed)
   c.iswap(0, 1)  # Correct implementation

Pitfall 2: Pulse Decomposition Complexity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Expecting fast single-pulse execution

**Solution**:

.. code-block:: python

   # Remember: iSWAP/SWAP = 3 CX gates = ~12 pulses
   # This is slower than native single-pulse CX
   
   # For performance-critical apps:
   # - Use CX where possible
   # - Reserve iSWAP for physics-required algorithms
   
   import time
   
   c = tq.Circuit(2)
   c.iswap(0, 1)
   
   start = time.time()
   result = c.device(provider="simulator").run(shots=1000)
   elapsed = time.time() - start
   print(f"Execution time: {elapsed:.3f}s (includes CX chain compilation)")

Pitfall 3: Leakage in Three-Level Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Unexpected measurement of |2⟩ state in three_level=True

**Solution**:

.. code-block:: python

   # The |2⟩ state is physical leakage to excited level
   # It's not an error—it's expected behavior in real hardware
   
   result = c.device(
       provider="simulator",
       device="statevector",
       three_level=True
   ).run(shots=1000)
   
   # Result may include measurements of qubit 2 (leakage):
   # {'00': 950, '10': 30, '20': 10, ...}  ← '20' and '2x' are leakage
   
   # Mitigation strategies:
   # 1. Use DRAG pulses (beta parameter ≠ 0)
   # 2. Reduce pulse amplitude
   # 3. Increase pulse duration
   # 4. Use calibration-optimized waveforms

Performance Considerations
---------------------------

Execution Time
~~~~~~~~~~~~~~

.. code-block:: text

   Gate-level: iSWAP(q0,q1)           → 1 operation
   Decomposed:  CX(0,1) + CX(1,0) + CX(0,1)  → 3 operations
   Pulse-level: 3 × 4 pulses           → ~12 pulses (after optimization)
   
   Expected slowdown: 3-5x vs single native gate
   But: Identical to universal CX construction

Memory Usage
~~~~~~~~~~~~

Pulse-level circuits use significantly more memory than gate-level:

.. code-block:: python

   import sys
   
   # Gate-level circuit (small)
   gate_circuit = tq.Circuit(10)
   for i in range(9):
       gate_circuit.iswap(i, i+1)
   print(f"Gate circuit size: {sys.getsizeof(gate_circuit)} bytes")
   
   # Pulse-level circuit (larger, due to waveform dicts)
   from tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse import GateToPulsePass
   pass_inst = GateToPulsePass()
   pulse_circuit = pass_inst.execute_plan(gate_circuit, mode="pulse_only")
   print(f"Pulse circuit size: {sys.getsizeof(pulse_circuit)} bytes")

Related Documentation
---------------------

- :doc:`/api/core/circuit_methods`: iswap() and swap() API reference
- :doc:`/user_guide/pulse/index`: Pulse programming comprehensive guide
- :doc:`/tutorials/advanced/pulse_three_level`: Three-level simulation tutorial
- :doc:`/tutorials/advanced/pulse_inline_three_level`: pulse_inline with 3-level
- :doc:`/tutorials/advanced/pulse_zz_crosstalk`: ZZ crosstalk modeling

API Reference
-------------

Circuit Methods
~~~~~~~~~~~~~~~

.. automethod:: tyxonq.core.ir.circuit.Circuit.iswap
.. automethod:: tyxonq.core.ir.circuit.Circuit.swap

Compiler Functions
~~~~~~~~~~~~~~~~~~

.. autoclass:: tyxonq.compiler.pulse_compile_engine.native.gate_to_pulse.GateToPulsePass
   :members: _decompose_iswap_gate, _decompose_swap_gate

References
----------

1. Shende & Markov, "Minimal universal two-qubit controlled-NOT-based circuits",
   *Physical Review A* **72**, 062305 (2005)
   https://doi.org/10.1103/PhysRevA.72.062305 [arXiv:quant-ph/0308033]

2. Rigetti, "A Practical Quantum Instruction Set Architecture" (2017)
   https://arxiv.org/abs/1903.02492

3. QuTiP-qip Processor Model, *Quantum* **6**, 630 (2022)
   https://quantum-journal.org/papers/q-2022-04-24-710/

4. Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010)
   Cambridge University Press, Chapter 4.5

5. Krantz et al., "A Quantum Engineer's Guide to Superconducting Qubits" (2019)
   *Reviews of Modern Physics* **90**, 015001
   https://doi.org/10.1103/RevModPhys.90.015001

.. note::

   This documentation corresponds to TyxonQ v0.2.0+

   Last updated: 2025-10-30

   Author: TyxonQ Development Team
