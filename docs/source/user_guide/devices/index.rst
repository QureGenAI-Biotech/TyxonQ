==================
Device Abstraction
==================

TyxonQ's device layer provides a unified interface for accessing various quantum simulators and real hardware. Through abstraction design, users can seamlessly switch between different execution backends.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

Core design principles of the TyxonQ device abstraction layer:

- **Unified Interface**: All devices implement the same API interface
- **Transparent Execution**: Users don't need to worry about underlying implementation details
- **Flexible Configuration**: Support for various execution parameters and device-specific options
- **Session Management**: Provides elegant resource management and task scheduling

Simulator Types
===============

Statevector Simulator
---------------------

**Principle**: Uses complete state vector representation of quantum states.

.. code-block:: python

   import tyxonq as tq
   
   circuit = tq.Circuit(2).h(0).cnot(0, 1)
   
   # Use statevector simulator
   result = circuit.device('statevector').run(shots=1000)
   
   # Get exact state vector
   state_result = circuit.device('statevector').run()
   print(f"State vector: {state_result}")

Density Matrix Simulator
------------------------

**Principle**: Uses density matrix to represent quantum states, supporting mixed states and noise.

.. code-block:: python

   # Use density matrix simulator
   result = circuit.device(
       'density_matrix',
       noise_model={'depolarizing': {'p': 0.01}}
   ).run(shots=1000)

MPS Simulator
-------------

**Principle**: Uses Matrix Product State representation, suitable for large-scale circuits.

.. code-block:: python

   # Large-scale circuit
   circuit = tq.Circuit(50)
   for i in range(49):
       circuit.h(i)
   
   # Use MPS simulator
   result = circuit.device(
       'mps',
       max_bond_dim=100
   ).run(shots=1000)

Hardware Devices
================

TyxonQ Processor
----------------

.. code-block:: python

   # Connect to TyxonQ hardware
   result = circuit.device(
       provider='tyxonq',
       device='processor_s2',
       shots=1024
   ).run()

Session Management
==================

.. code-block:: python

   from tyxonq.devices import Session
   
   # Create session
   with Session(device='statevector', shots=1000) as session:
       result1 = circuit1.run()
       result2 = circuit2.run()
       result3 = circuit3.run()

Performance Comparison
======================

.. list-table:: Simulator Performance Comparison
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Simulator Type
     - Memory Complexity
     - Max Qubits
     - Precision
     - Best Application
   * - Statevector
     - O(2^n)
     - ~30-35
     - Exact
     - Small-scale simulation
   * - Density Matrix
     - O(4^n)
     - ~15-20
     - Exact (noisy)
     - Noise research
   * - MPS
     - O(χ×n)
     - 50+
     - Approximate
     - Large system simulation

Noise Simulation
================

TyxonQ provides comprehensive noise simulation for realistic NISQ algorithm development:

.. code-block:: python

   # Add noise to your circuit with one line
   result = circuit.with_noise("depolarizing", p=0.05).run(shots=1024)

**Supported Noise Models**:

- **Depolarizing**: Uniform Pauli errors (X, Y, Z)
- **Amplitude Damping**: Energy relaxation (T₁)
- **Phase Damping**: Decoherence (T₂)
- **Pauli Channel**: Asymmetric error rates

For complete details, see :doc:`noise_simulation`.

Sections
========

.. toctree::
   :maxdepth: 1

   noise_simulation

Related Resources
=================

- :doc:`/api/devices/index` - Devices API Reference
- :doc:`../numerics/index` - Numerical Computation Backend
- :doc:`/cloud_services/index` - Cloud Device Access
- :doc:`noise_simulation` - Complete Noise Simulation Guide