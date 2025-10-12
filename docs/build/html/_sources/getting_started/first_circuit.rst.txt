==========================
Your First Quantum Circuit
==========================

Welcome to the world of quantum circuits with TyxonQ! This guide will take you from zero to creating, manipulating, and executing quantum circuits.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

Quantum circuits are the fundamental building blocks of quantum computing. In TyxonQ, circuits are represented by the :class:`Circuit` class, which provides an intuitive API for constructing and executing quantum algorithms.

**Key Concepts**

- **Qubits**: The basic unit of quantum information
- **Gates**: Operations performed on qubits
- **Measurement**: Extracting classical information from quantum states
- **Circuit**: An ordered sequence of gates and measurements

Creating Your First Circuit
============================

Basic Circuit Construction
---------------------------

Let's start with the simplest example - creating a Bell state:

.. code-block:: python

   import tyxonq as tq
   
   # Create a circuit with 2 qubits
   circuit = tq.Circuit(2)
   
   # Add Hadamard gate to qubit 0
   circuit.h(0)
   
   # Add CNOT gate with control=0, target=1
   circuit.cnot(0, 1)
   
   print(f"Circuit has {circuit.num_qubits} qubits")
   print(f"Number of gates: {len(circuit.ops)}")

**Output**::

   Circuit has 2 qubits
   Number of gates: 2

Chained API Construction
-------------------------

TyxonQ supports method chaining for cleaner code:

.. code-block:: python

   # Build the same Bell state circuit with chaining
   bell_circuit = tq.Circuit(2).h(0).cnot(0, 1)
   
   # More complex example: GHZ state
   ghz_circuit = (
       tq.Circuit(3)
       .h(0)
       .cnot(0, 1)
       .cnot(1, 2)
   )

Common Quantum Gates
====================

Single-Qubit Gates
------------------

.. list-table:: Supported Single-Qubit Gates in TyxonQ
   :header-rows: 1
   :widths: 15 25 60

   * - Gate Name
     - Method Call
     - Description
   * - Pauli-X
     - ``circuit.x(qubit)``
     - Bit-flip gate (quantum NOT)
   * - Pauli-Y
     - ``circuit.y(qubit)``
     - Y-axis rotation
   * - Pauli-Z
     - ``circuit.z(qubit)``
     - Phase-flip gate
   * - Hadamard
     - ``circuit.h(qubit)``
     - Superposition gate
   * - S gate
     - ``circuit.s(qubit)``
     - Phase gate (π/2 rotation)
   * - T gate
     - ``circuit.t(qubit)``
     - π/4 phase gate

**Example: Single-Qubit Gate Operations**

.. code-block:: python

   # Create a single-qubit circuit and apply gates
   single_qubit = tq.Circuit(1)
   
   # Apply Hadamard gate to create superposition
   single_qubit.h(0)
   
   # Apply Pauli-X gate
   single_qubit.x(0)
   
   # View circuit summary
   print(single_qubit.gate_summary())

Parameterized Rotation Gates
-----------------------------

Parameterized gates allow you to specify rotation angles:

.. code-block:: python

   import numpy as np
   
   # Parameterized rotation gates
   param_circuit = tq.Circuit(2)
   
   # RX gate: rotate around X-axis by π/4
   param_circuit.rx(0, np.pi/4)
   
   # RY gate: rotate around Y-axis by π/2
   param_circuit.ry(1, np.pi/2)
   
   # RZ gate: rotate around Z-axis by arbitrary angle
   theta = 0.5
   param_circuit.rz(0, theta)

Two-Qubit Gates
---------------

.. list-table:: Two-Qubit Gates
   :header-rows: 1
   :widths: 15 25 60

   * - Gate Name
     - Method Call
     - Description
   * - CNOT
     - ``circuit.cnot(control, target)``
     - Controlled-NOT gate
   * - CZ
     - ``circuit.cz(control, target)``
     - Controlled-Z gate
   * - SWAP
     - ``circuit.swap(qubit1, qubit2)``
     - Swap gate
   * - CX
     - ``circuit.cx(control, target)``
     - Alias for CNOT

**Example: Creating Entangled States**

.. code-block:: python

   # Create maximally entangled Bell state
   entangled = (
       tq.Circuit(2)
       .h(0)        # Create superposition
       .cnot(0, 1)  # Create entanglement
   )
   
   # Create 3-qubit GHZ state
   ghz_state = (
       tq.Circuit(3)
       .h(0)
       .cnot(0, 1)
       .cnot(1, 2)
   )

Measurement Operations
======================

Adding Measurements
-------------------

Measurement is how we extract classical information from quantum circuits:

.. code-block:: python

   # Create circuit and add measurements
   measured_circuit = (
       tq.Circuit(2)
       .h(0)
       .cnot(0, 1)
       .measure_z(0)  # Measure qubit 0
       .measure_z(1)  # Measure qubit 1
   )
   
   # Or measure all qubits
   all_measured = tq.Circuit(2).h(0).cnot(0, 1)
   
   # Add measurements to all qubits
   for i in range(2):
       all_measured.measure_z(i)

Executing Circuits
==================

Basic Execution
---------------

Use TyxonQ's chain API to execute circuits:

.. code-block:: python

   # Execute Bell state circuit
   circuit = tq.Circuit(2).h(0).cnot(0, 1)
   
   # Add measurements
   for i in range(2):
       circuit.measure_z(i)
   
   # Execute circuit (using default settings)
   result = circuit.run()
   
   print("Measurement result:", result)

Specifying Execution Options
-----------------------------

You can customize execution options:

.. code-block:: python

   # Specify shots and device type
   result = (
       circuit
       .device(provider="simulator", device="statevector", shots=1024)
       .run()
   )
   
   # View counts result
   if isinstance(result, list) and result:
       counts = result[0].get("result", {})
       print("Counts:", counts)
       # Expected output: {'00': 512, '11': 512}

Different Device Types
----------------------

.. code-block:: python

   # Statevector simulator (exact simulation)
   sv_result = (
       circuit
       .device(provider="simulator", device="statevector")
       .run()
   )
   
   # Density matrix simulator (supports noise)
   dm_result = (
       circuit
       .device(provider="simulator", device="density_matrix")
       .run()
   )
   
   # Matrix product state simulator (for large circuits)
   mps_result = (
       circuit
       .device(provider="simulator", device="mps")
       .run()
   )

Circuit Properties and Methods
===============================

Querying Circuit Information
-----------------------------

.. code-block:: python

   circuit = tq.Circuit(3).h(0).cnot(0, 1).rx(2, 0.5)
   
   # Basic attributes
   print(f"Number of qubits: {circuit.num_qubits}")
   print(f"Number of operations: {len(circuit.ops)}")
   
   # Gate operation summary
   print(f"Gate summary: {circuit.gate_summary()}")
   
   # View instruction list
   print(f"Instructions: {circuit.instructions}")

Metadata Management
-------------------

Add descriptive information to circuits:

.. code-block:: python

   # Add metadata
   circuit_with_meta = (
       tq.Circuit(2)
       .h(0).cnot(0, 1)
       .with_metadata(description="Bell state preparation", author="Alice")
   )
   
   print("Metadata:", circuit_with_meta.metadata)

Advanced Examples
=================

Quantum Fourier Transform
--------------------------

Implement Quantum Fourier Transform (QFT):

.. code-block:: python

   def qft_circuit(n_qubits):
       """Build n-qubit Quantum Fourier Transform circuit"""
       circuit = tq.Circuit(n_qubits)
       
       for j in range(n_qubits):
           circuit.h(j)
           for k in range(j + 1, n_qubits):
               angle = np.pi / (2 ** (k - j))
               # Controlled phase gate (simplified to RZ for demonstration)
               circuit.rz(k, angle / 2)
       
       return circuit
   
   # Create 3-qubit QFT circuit
   qft_3 = qft_circuit(3)
   print(f"QFT circuit gate count: {len(qft_3.ops)}")

Variational Quantum Circuit
----------------------------

Create a parameterized variational circuit:

.. code-block:: python

   def variational_circuit(n_qubits, layers, parameters):
       """Build variational quantum circuit"""
       circuit = tq.Circuit(n_qubits)
       param_idx = 0
       
       for layer in range(layers):
           # Single-qubit rotation layer
           for i in range(n_qubits):
               circuit.ry(i, parameters[param_idx])
               param_idx += 1
           
           # Entangling layer
           for i in range(n_qubits - 1):
               circuit.cnot(i, i + 1)
       
       return circuit
   
   # Create variational circuit
   n_qubits, layers = 4, 2
   n_params = n_qubits * layers
   params = np.random.uniform(0, 2*np.pi, n_params)
   
   var_circuit = variational_circuit(n_qubits, layers, params)
   print(f"Variational circuit parameter count: {n_params}")

Best Practices
==============

Code Organization
-----------------

1. **Use chain API**: Improve code readability

   .. code-block:: python

      # Recommended
      result = (
          tq.Circuit(2).h(0).cnot(0, 1)
          .device(shots=1024)
          .run()
      )

2. **Use descriptive names**: Use meaningful variable names

   .. code-block:: python

      bell_state = tq.Circuit(2).h(0).cnot(0, 1)
      ghz_state = tq.Circuit(3).h(0).cnot(0, 1).cnot(1, 2)

3. **Add comments**: Explain complex quantum algorithms

   .. code-block:: python

      # Prepare Bell state |00⟩ + |11⟩
      circuit = tq.Circuit(2)
      circuit.h(0)        # Create superposition
      circuit.cnot(0, 1)  # Create entanglement

Error Handling
--------------

.. code-block:: python

   try:
       circuit = tq.Circuit(2).h(0).cnot(0, 3)  # Error: qubit 3 doesn't exist
       result = circuit.run()
   except Exception as e:
       print(f"Circuit execution error: {e}")

Performance Tips
----------------

1. **Choose appropriate device type**:
   
   - Small circuits (<20 qubits): use ``statevector``
   - Need noise simulation: use ``density_matrix``
   - Large circuits: use ``mps``

2. **Optimize shots count**:
   
   - Debugging: use fewer shots (100-1000)
   - Production: use sufficient shots (1000-10000)

Next Steps
==========

Congratulations! You've mastered the basics of TyxonQ quantum circuits. Next, we recommend:

- :doc:`first_chemistry` - Learn quantum chemistry applications
- :doc:`basic_concepts` - Deep dive into quantum computing concepts
- :doc:`../user_guide/compiler/index` - Learn about compiler optimizations
- :doc:`../examples/basic_examples` - View more examples

FAQ
===

**Q: How do I view the quantum state of a circuit?**

A: Use the statevector simulator to get the exact quantum state:

.. code-block:: python

   circuit = tq.Circuit(2).h(0).cnot(0, 1)
   
   # Get statevector (don't add measurements)
   result = circuit.device(device="statevector").run()
   # Result contains quantum state information

**Q: How do I reuse a circuit?**

A: Circuit objects can be executed multiple times:

.. code-block:: python

   circuit = tq.Circuit(2).h(0).cnot(0, 1)
   
   # Execute multiple times
   result1 = circuit.run()
   result2 = circuit.run()

**Q: What quantum gates are supported?**

A: TyxonQ supports a complete universal gate set, including Pauli gates, rotation gates, Hadamard gates, CNOT gates, etc. See the API documentation :doc:`../api/core/index` for details.

Related Resources
=================

- :doc:`/api/core/index` - Core API reference
- :doc:`/examples/basic_examples` - Basic examples
- :doc:`/user_guide/devices/index` - Devices and simulators
- :doc:`/user_guide/compiler/index` - Compiler optimizations
