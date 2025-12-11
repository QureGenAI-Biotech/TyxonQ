===========
Quick Start
===========

Get up and running with TyxonQ in 5 minutes! This quick start guide will show you the basics of creating and running quantum circuits.

Prerequisites
=============

Make sure you have TyxonQ installed. If not, see :doc:`installation`.

.. code-block:: bash

   pip install tyxonq

Your First Quantum Circuit
===========================

Let's create a simple Bell state - a maximally entangled quantum state.

Step 1: Import TyxonQ
---------------------

.. code-block:: python

   import tyxonq as tq
   import numpy as np

Step 2: Create a Circuit
-------------------------

.. code-block:: python

   # Create a circuit with 2 qubits
   circuit = tq.Circuit(2)
   
   # Apply Hadamard gate to qubit 0
   circuit.h(0)
   
   # Apply CNOT gate (control=0, target=1)
   circuit.cnot(0, 1)
   
   # Add measurements
   circuit.measure_all()
   
   print(circuit)

Output:

.. code-block:: text

   Circuit(2 qubits, 3 operations):
   H(0)
   CNOT(0, 1)
   Measure(0, 1)

Step 3: Compile and Execute
----------------------------

.. code-block:: python

   # Compile the circuit
   compiled_circuit = circuit.compile()
   
   # Execute on statevector simulator
   result = compiled_circuit.device('statevector').run(shots=1000)
   
   # View results
   print(result.counts)

Output:

.. code-block:: python

   {'00': 502, '11': 498}

The results show that we've created a Bell state - measuring either |00⟩ or |11⟩ with approximately equal probability.

Understanding the Code
======================

Circuit Creation
----------------

.. code-block:: python

   circuit = tq.Circuit(n_qubits)

Creates a quantum circuit with the specified number of qubits.

Gates
-----

TyxonQ supports all standard quantum gates:

.. code-block:: python

   circuit.h(qubit)          # Hadamard gate
   circuit.x(qubit)          # Pauli-X gate
   circuit.y(qubit)          # Pauli-Y gate
   circuit.z(qubit)          # Pauli-Z gate
   circuit.rx(qubit, theta)  # Rotation around X-axis
   circuit.ry(qubit, theta)  # Rotation around Y-axis
   circuit.rz(qubit, theta)  # Rotation around Z-axis
   circuit.cnot(control, target)  # Controlled-NOT
   circuit.cz(control, target)    # Controlled-Z

Measurements
------------

.. code-block:: python

   circuit.measure(qubit)     # Measure specific qubit
   circuit.measure_all()      # Measure all qubits

Compilation
-----------

.. code-block:: python

   compiled = circuit.compile()

The compiler optimizes your circuit and prepares it for execution.

Execution
---------

.. code-block:: python

   result = compiled.device('statevector').run(shots=1000)

Executes the circuit on the specified device (simulator or hardware).

More Examples
=============

Example 1: Superposition
------------------------

Create a superposition of all basis states:

.. code-block:: python

   circuit = tq.Circuit(3)
   for i in range(3):
       circuit.h(i)
   circuit.measure_all()
   
   result = circuit.compile().device('statevector').run(shots=1000)
   print(result.counts)

Example 2: Phase Kickback
--------------------------

Demonstrate quantum phase kickback:

.. code-block:: python

   circuit = tq.Circuit(2)
   
   # Prepare |+⟩ ⊗ |-⟩
   circuit.h(0)
   circuit.x(1)
   circuit.h(1)
   
   # Apply CNOT
   circuit.cnot(0, 1)
   
   # Measure
   circuit.h(0)
   circuit.measure_all()
   
   result = circuit.compile().device('statevector').run(shots=1000)
   print(result.counts)

Example 3: Quantum Teleportation
---------------------------------

A simple quantum teleportation circuit:

.. code-block:: python

   circuit = tq.Circuit(3)
   
   # Prepare Bell pair between qubits 1 and 2
   circuit.h(1)
   circuit.cnot(1, 2)
   
   # Prepare state to teleport on qubit 0 (optional, here we use |+⟩)
   circuit.h(0)
   
   # Bell measurement on qubits 0 and 1
   circuit.cnot(0, 1)
   circuit.h(0)
   circuit.measure([0, 1])
   
   # Conditional operations on qubit 2 (classical control)
   circuit.cz(1, 2)
   circuit.cnot(0, 2)
   
   # Measure final state
   circuit.measure(2)
   
   result = circuit.compile().device('statevector').run(shots=1000)
   print(result.counts)

Working with Different Backends
================================

TyxonQ supports multiple numerical backends:

NumPy Backend (Default)
-----------------------

.. code-block:: python

   import tyxonq as tq
   tq.set_backend('numpy')
   
   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cnot(0, 1)

PyTorch Backend
---------------

Enable automatic differentiation:

.. code-block:: python

   tq.set_backend('pytorch')
   
   # Same circuit code works with PyTorch backend
   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cnot(0, 1)

GPU Acceleration
----------------

Use CuPy for GPU acceleration:

.. code-block:: python

   tq.set_backend('cupy')
   
   # Circuits automatically use GPU
   circuit = tq.Circuit(10)  # Larger circuits benefit more from GPU
   for i in range(10):
       circuit.h(i)

Using Different Simulators
===========================

Statevector Simulator
---------------------

.. code-block:: python

   result = circuit.compile().device('statevector').run(shots=1000)

Density Matrix Simulator
-------------------------

For simulations with noise:

.. code-block:: python

   result = circuit.compile().device('density_matrix').run(shots=1000)

MPS Simulator
-------------

For large circuits with limited entanglement:

.. code-block:: python

   result = circuit.compile().device('mps').run(shots=1000)

Next Steps
==========

Now that you've learned the basics, explore:

- :doc:`first_circuit` - Deep dive into circuit construction
- :doc:`first_chemistry` - Quantum chemistry applications
- :doc:`../user_guide/index` - Comprehensive user guides
- :doc:`../tutorials/index` - Step-by-step tutorials
- :doc:`../examples/index` - More example code

For detailed API documentation:

- :doc:`../api/core/circuit` - Circuit API reference
- :doc:`../api/devices/index` - Device backends
- :doc:`../api/compiler/index` - Compilation pipeline
