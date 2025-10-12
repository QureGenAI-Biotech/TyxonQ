========
Core API
========

Complete API reference for TyxonQ's core quantum computing primitives: circuits, operations, and measurements.

.. contents:: Contents
   :depth: 3
   :local:

Overview
========

The Core API provides the fundamental building blocks for quantum programming:

🕸️ **Circuit** (``tyxonq.Circuit``)
   Quantum circuit construction and manipulation

🎯 **Hamiltonian** (``tyxonq.Hamiltonian``)
   Hamiltonian operators for quantum systems

⚙️ **Operations**
   Quantum gates and measurements

📊 **IR (Intermediate Representation)**
   Low-level circuit representation

Circuit Class
=============

.. autoclass:: tyxonq.Circuit
   :members:
   :undoc-members:
   :show-inheritance:

The Circuit class is the primary interface for building and executing quantum circuits.

**Constructor**:

.. code-block:: python

   Circuit(
       num_qubits: int,              # Number of qubits
       ops: List = None,              # Initial operations
       name: str = None               # Circuit name
   )

**Basic Gates**:

.. code-block:: python

   from tyxonq import Circuit
   
   # Create 3-qubit circuit
   circuit = Circuit(3)
   
   # Single-qubit gates
   circuit.h(0)           # Hadamard
   circuit.x(1)           # Pauli-X
   circuit.y(2)           # Pauli-Y
   circuit.z(0)           # Pauli-Z
   circuit.s(1)           # S gate
   circuit.t(2)           # T gate
   
   # Rotation gates
   circuit.rx(0, theta=0.5)   # Rotation around X
   circuit.ry(1, theta=1.0)   # Rotation around Y
   circuit.rz(2, theta=1.5)   # Rotation around Z
   
   # Two-qubit gates
   circuit.cx(0, 1)       # CNOT (Control-X)
   circuit.cy(0, 2)       # Control-Y
   circuit.cz(1, 2)       # Control-Z
   circuit.swap(0, 1)     # SWAP

**Measurements**:

.. code-block:: python

   # Measure in Z basis
   circuit.measure_z(0)
   circuit.measure_z([1, 2])  # Measure multiple qubits
   
   # Measure all qubits
   circuit.measure_all()

**Circuit Composition**:

.. code-block:: python

   # Method chaining
   circuit = (
       Circuit(2)
       .h(0)
       .cx(0, 1)
       .measure_all()
   )
   
   # Concatenate circuits
   circuit1 = Circuit(2).h(0)
   circuit2 = Circuit(2).cx(0, 1)
   combined = circuit1 + circuit2

**Execution**:

.. code-block:: python

   # Run on local simulator
   result = circuit.run()
   print(result)  # {'00': 512, '11': 512}
   
   # Run with configuration
   result = (
       circuit
       .compile(optimization_level=2)
       .device(provider="local", device="statevector", shots=1024)
       .run()
   )

**Key Methods**:

- ``run(**opts)`` - Execute circuit and return results
- ``compile(**opts)`` - Compile circuit for target device
- ``device(**opts)`` - Configure device settings
- ``postprocessing(**opts)`` - Configure postprocessing
- ``to_qasm()`` - Export to OpenQASM
- ``draw()`` - Visualize circuit

Gate Reference
==============

Single-Qubit Gates
------------------

**Pauli Gates**:

.. code-block:: python

   circuit.x(qubit)   # X gate (bit flip)
   circuit.y(qubit)   # Y gate
   circuit.z(qubit)   # Z gate (phase flip)

**Hadamard Gate**:

.. code-block:: python

   circuit.h(qubit)   # Creates superposition

**Phase Gates**:

.. code-block:: python

   circuit.s(qubit)    # S gate (π/2 phase)
   circuit.sd(qubit)   # S† gate
   circuit.t(qubit)    # T gate (π/4 phase)
   circuit.td(qubit)   # T† gate

**Rotation Gates**:

.. code-block:: python

   circuit.rx(qubit, theta)   # Rotation around X-axis
   circuit.ry(qubit, theta)   # Rotation around Y-axis
   circuit.rz(qubit, theta)   # Rotation around Z-axis
   
   # Parameterized rotation
   from tyxonq import Parameter
   theta = Parameter('θ')
   circuit.rx(0, theta)

Two-Qubit Gates
---------------

**CNOT (Controlled-X)**:

.. code-block:: python

   circuit.cx(control, target)    # CNOT gate
   circuit.cnot(control, target)  # Alias

**Other Controlled Gates**:

.. code-block:: python

   circuit.cy(control, target)   # Controlled-Y
   circuit.cz(control, target)   # Controlled-Z
   circuit.ch(control, target)   # Controlled-H

**SWAP Gate**:

.. code-block:: python

   circuit.swap(qubit1, qubit2)  # Swap two qubits

**Controlled Rotations**:

.. code-block:: python

   circuit.crx(control, target, theta)   # Controlled-Rx
   circuit.cry(control, target, theta)   # Controlled-Ry
   circuit.crz(control, target, theta)   # Controlled-Rz

Multi-Qubit Gates
-----------------

**Toffoli (CCX)**:

.. code-block:: python

   circuit.ccx(control1, control2, target)  # Toffoli gate
   circuit.toffoli(control1, control2, target)  # Alias

**Controlled-SWAP**:

.. code-block:: python

   circuit.cswap(control, target1, target2)  # Fredkin gate

Measurements
============

Z-Basis Measurement
-------------------

.. code-block:: python

   # Measure single qubit
   circuit.measure_z(0)
   
   # Measure multiple qubits
   circuit.measure_z([0, 1, 2])
   
   # Measure all qubits
   circuit.measure_all()

X and Y Basis
-------------

.. code-block:: python

   # Measure in X basis (apply H then measure Z)
   circuit.h(0)
   circuit.measure_z(0)
   
   # Measure in Y basis (apply S†H then measure Z)
   circuit.sd(0)
   circuit.h(0)
   circuit.measure_z(0)

Hamiltonian Class
=================

.. autoclass:: tyxonq.Hamiltonian
   :members:
   :undoc-members:
   :show-inheritance:

The Hamiltonian class represents quantum operators.

**Constructor**:

.. code-block:: python

   from tyxonq import Hamiltonian
   
   # From Pauli strings
   ham = Hamiltonian([
       (0.5, [("Z", 0)]),                    # 0.5 * Z_0
       (0.3, [("Z", 0), ("Z", 1)]),          # 0.3 * Z_0 Z_1
       (0.2, [("X", 0), ("X", 1)]),          # 0.2 * X_0 X_1
   ])

**Common Hamiltonians**:

.. code-block:: python

   # Ising model
   def ising_hamiltonian(n_qubits, J=-1.0, h=0.5):
       terms = []
       # ZZ interactions
       for i in range(n_qubits - 1):
           terms.append((J, [("Z", i), ("Z", i+1)]))
       # Transverse field
       for i in range(n_qubits):
           terms.append((h, [("X", i)]))
       return Hamiltonian(terms)
   
   ham = ising_hamiltonian(4)

**Operations**:

.. code-block:: python

   # Hamiltonian algebra
   ham1 = Hamiltonian([(1.0, [("Z", 0)])])
   ham2 = Hamiltonian([(0.5, [("X", 0)])])
   
   # Addition
   ham_sum = ham1 + ham2
   
   # Scalar multiplication
   ham_scaled = 2.0 * ham1

**Expectation Values**:

.. code-block:: python

   from tyxonq import Circuit, Hamiltonian
   
   # Prepare state
   circuit = Circuit(2).h(0).cx(0, 1)
   
   # Define observable
   ham = Hamiltonian([
       (1.0, [("Z", 0), ("Z", 1)])
   ])
   
   # Compute expectation
   expval = circuit.expectation(ham)
   print(f"<H> = {expval}")

Common Patterns
===============

Bell State Preparation
----------------------

.. code-block:: python

   from tyxonq import Circuit
   
   # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
   bell_circuit = Circuit(2).h(0).cx(0, 1)
   
   result = bell_circuit.run(shots=1024)
   print(result)  # ~50% '00', ~50% '11'

GHZ State
---------

.. code-block:: python

   # GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
   ghz = (
       Circuit(3)
       .h(0)
       .cx(0, 1)
       .cx(0, 2)
   )

Quantum Fourier Transform
-------------------------

.. code-block:: python

   import numpy as np
   
   def qft(n_qubits):
       circuit = Circuit(n_qubits)
       
       for j in range(n_qubits):
           circuit.h(j)
           for k in range(j + 1, n_qubits):
               angle = np.pi / (2 ** (k - j))
               circuit.crz(k, j, angle)
       
       # Reverse qubit order
       for i in range(n_qubits // 2):
           circuit.swap(i, n_qubits - i - 1)
       
       return circuit
   
   qft_circuit = qft(4)

Variational Circuit
-------------------

.. code-block:: python

   from tyxonq import Circuit, Parameter
   import numpy as np
   
   # Parameterized circuit
   def variational_circuit(n_qubits, n_layers):
       params = [Parameter(f'θ_{i}') for i in range(n_qubits * n_layers)]
       circuit = Circuit(n_qubits)
       
       for layer in range(n_layers):
           # Rotation layer
           for i in range(n_qubits):
               idx = layer * n_qubits + i
               circuit.ry(i, params[idx])
           
           # Entanglement layer
           for i in range(n_qubits - 1):
               circuit.cx(i, i + 1)
       
       return circuit
   
   vqc = variational_circuit(4, 3)

Advanced Features
=================

Conditional Operations
----------------------

.. code-block:: python

   # Mid-circuit measurement
   circuit = Circuit(2)
   circuit.h(0)
   circuit.measure_z(0)  # Measure qubit 0
   
   # Conditional gate (if measurement result is 1)
   circuit.x(1).c_if(0, 1)

Circuit Inversion
-----------------

.. code-block:: python

   # Create circuit
   circuit = Circuit(2).h(0).cx(0, 1)
   
   # Invert (create adjoint)
   inverse = circuit.inverse()
   
   # Should return to |00⟩
   combined = circuit + inverse

Circuit Decomposition
---------------------

.. code-block:: python

   # Decompose circuit into basis gates
   circuit = Circuit(2).h(0).cry(0, 1, theta=0.5)
   
   decomposed = circuit.compile(
       basis_gates=['h', 'rx', 'ry', 'rz', 'cx']
   )

See Also
========

- :doc:`/user_guide/core/index` - Core Module Guide
- :doc:`/getting_started/first_circuit` - First Circuit Tutorial  
- :doc:`/examples/basic_examples` - Circuit Examples
- :doc:`/api/compiler/index` - Compiler API
