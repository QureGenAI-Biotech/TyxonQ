==============
Basic Concepts
==============

Welcome to the fundamental concepts of quantum computing and the TyxonQ framework! This page introduces core quantum computing principles and TyxonQ's architectural design.

.. contents:: Table of Contents
   :depth: 2
   :local:

Quantum Computing Fundamentals
===============================

Qubits
------

A quantum bit, or **qubit**, is the fundamental unit of quantum information, analogous to a classical bit. Unlike classical bits which exist in definite states of 0 or 1, qubits can exist in superposition states.

**Mathematical Representation**

The state of a single qubit can be represented as:

.. math::

   |\psi\rangle = \alpha|0\rangle + \beta|1\rangle

where :math:`\alpha` and :math:`\beta` are complex amplitudes satisfying the normalization condition :math:`|\alpha|^2 + |\beta|^2 = 1`. The squared magnitudes :math:`|\alpha|^2` and :math:`|\beta|^2` represent the probabilities of measuring the qubit in states :math:`|0\rangle` and :math:`|1\rangle`, respectively.

**Representation in TyxonQ**

.. code-block:: python

   import tyxonq as tq
   
   # Create a single-qubit circuit
   circuit = tq.Circuit(1)
   
   # View the initial state |0⟩
   result = circuit.device('statevector').run()
   print("Initial state:", result.statevector)
   # Output: [1.+0.j, 0.+0.j]

Quantum Superposition
---------------------

Quantum superposition allows qubits to exist simultaneously in multiple classical states, which is the source of quantum computational power. This fundamentally quantum property enables quantum parallelism.

.. code-block:: python

   # Create a superposition state
   circuit = tq.Circuit(1).h(0)  # Apply Hadamard gate
   
   result = circuit.device('statevector').run()
   print("Superposition state:", result.statevector)
   # Output: [0.707+0.j, 0.707+0.j] (approximately 1/√2)

The Hadamard gate creates an equal superposition, transforming :math:`|0\rangle` into :math:`(|0\rangle + |1\rangle)/\sqrt{2}`. When measured, this state has a 50% probability of yielding 0 or 1.

Quantum Entanglement
--------------------

Quantum entanglement is a non-classical correlation between multiple qubits that cannot be explained by classical probability theory. Entangled qubits exhibit correlations that persist regardless of the distance separating them.

.. code-block:: python

   # Create a Bell state (maximally entangled state)
   circuit = tq.Circuit(2).h(0).cnot(0, 1)
   
   result = circuit.device('statevector').run()
   print("Bell state:", result.statevector)
   # Output: [0.707+0.j, 0.+0.j, 0.+0.j, 0.707+0.j]
   # Represents (|00⟩ + |11⟩)/√2

This Bell state demonstrates perfect correlation: measuring the first qubit immediately determines the second qubit's state, even though each individual qubit is in a completely mixed state.

Quantum Measurement
-------------------

Measurement is the process of extracting classical information from a quantum state. Unlike classical observation, quantum measurement fundamentally disturbs the system, causing the quantum state to collapse into a definite eigenstate of the measured observable.

.. code-block:: python

   # Measure a Bell state
   circuit = tq.Circuit(2).h(0).cnot(0, 1)
   circuit.measure_z(0).measure_z(1)
   
   result = circuit.compile().device('statevector').run(shots=1000)
   print("Measurement results:", result.counts)
   # Expect approximately 50% '00' and 50% '11'

The measurement collapses the entangled state into either :math:`|00\rangle` or :math:`|11\rangle` with equal probability, never :math:`|01\rangle` or :math:`|10\rangle`, demonstrating the quantum correlation.

TyxonQ Core Architecture
========================

Architectural Overview
----------------------

TyxonQ employs a layered architecture design, providing complete abstraction from high-level algorithms to low-level hardware:

.. mermaid::

   graph TD
       A[Application Layer] --> B[Compiler Layer]
       B --> C[Device Abstraction Layer]
       C --> D[Numerics Backend Layer]
       D --> E[Hardware/Simulators]
       
       A1[Quantum Chemistry] --> A
       A2[Optimization Algorithms] --> A
       A3[Machine Learning] --> A
       
       B1[Decomposition] --> B
       B2[Rewriting] --> B
       B3[Optimization] --> B
       
       C1[Simulators] --> C
       C2[Hardware Drivers] --> C
       
       D1[NumPy] --> D
       D2[PyTorch] --> D
       D3[CuPy] --> D

Intermediate Representation (IR)
---------------------------------

TyxonQ uses an intermediate representation to uniformly handle quantum circuits. The IR serves as a stable contract between different framework components, enabling:

- **Portability**: Circuits can be compiled for different backends
- **Optimization**: Compiler passes can transform the IR
- **Introspection**: Circuit structure can be analyzed programmatically

.. code-block:: python

   # Circuit intermediate representation
   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cnot(0, 1)
   
   # Inspect IR structure
   print(f"Number of qubits: {circuit.num_qubits}")
   print(f"Operations: {circuit.ops}")
   print(f"Instruction sequence: {circuit.instructions}")

The IR maintains the logical structure of quantum operations while being independent of the eventual execution backend.

Compilation Pipeline
--------------------

TyxonQ's compiler transforms high-level circuits into device-executable forms through a multi-stage pipeline:

.. mermaid::

   graph LR
       A[Source Circuit] --> B[Decomposition Stage]
       B --> C[Rewriting Stage] 
       C --> D[Simplification Stage]
       D --> E[Scheduling Stage]
       E --> F[Target Code]

Each stage has a specific purpose:

- **Decomposition**: Breaks down high-level gates into device-native gates
- **Rewriting**: Applies circuit transformations and optimizations
- **Simplification**: Removes redundant gates and merges operations
- **Scheduling**: Allocates measurement resources and organizes execution

Device Abstraction
------------------

TyxonQ provides a unified device interface supporting multiple execution backends:

.. list-table:: Device Type Comparison
   :header-rows: 1
   :widths: 20 30 25 25

   * - Device Type
     - Use Cases
     - Advantages
     - Limitations
   * - statevector
     - Small-scale exact simulation
     - Fully exact
     - Exponential memory growth
   * - density_matrix
     - Noise simulation
     - Supports mixed states
     - Higher memory requirements
   * - mps
     - Large-scale approximate simulation
     - Linear memory scaling
     - Best for low-entanglement systems

This abstraction allows you to write quantum programs once and execute them on different backends by simply changing the device specification.

Execution Model
===============

Fluent API
----------

TyxonQ supports method chaining, providing a concise programming interface that reads naturally:

.. code-block:: python

   # Complete fluent execution
   result = (
       tq.Circuit(2)
       .h(0).cnot(0, 1)           # Build circuit
       .compile()                  # Compile and optimize
       .device('statevector')      # Device configuration
       .run(shots=1024)            # Execute
   )

This fluent style enables expressive quantum programs while maintaining clarity. Each method returns the appropriate object for the next operation in the chain.

The Concept of Shots
--------------------

"Shots" represents the number of times a circuit is executed to statistically sample measurement outcomes. More shots provide more accurate estimates of probabilities at the cost of increased runtime.

.. code-block:: python

   # Impact of different shot counts
   circuit = tq.Circuit(1).h(0).measure_z(0)
   
   # Few shots - results may be unstable
   result_10 = circuit.compile().device('statevector').run(shots=10)
   
   # Many shots - results more stable
   result_1000 = circuit.compile().device('statevector').run(shots=1000)
   
   print("10 measurements:", result_10.counts)
   print("1000 measurements:", result_1000.counts)

For a circuit creating perfect superposition, 10 shots might give 3:7 (30%:70%), while 1000 shots will be closer to 500:500 (50%:50%). The relationship follows standard statistical sampling theory.

Numerics Backend System
=======================

TyxonQ supports multiple numerical computation backends, allowing users to select based on their needs:

.. code-block:: python

   # NumPy backend (default)
   tq.set_backend('numpy')
   result_numpy = circuit.compile().device('statevector').run()
   
   # PyTorch backend (supports automatic differentiation)
   tq.set_backend('pytorch')
   result_torch = circuit.compile().device('statevector').run()
   
   # CuPy backend (GPU acceleration)
   tq.set_backend('cupy')
   result_gpu = circuit.compile().device('statevector').run()

Backend Selection Guidelines:

- **NumPy**: Best for CPU-only systems, good performance for moderate-sized circuits
- **PyTorch**: Required for gradient-based optimization with automatic differentiation
- **CuPy**: Optimal for large circuits on GPU-enabled systems

The backend abstraction ensures your quantum programs work identically across all backends.

Next Steps
==========

Now that you understand the basic concepts, continue your learning journey:

- :doc:`first_circuit` - Create your first quantum circuit in depth
- :doc:`first_chemistry` - Introduction to quantum chemistry applications
- :doc:`../user_guide/core/index` - Deep dive into core modules
- :doc:`../examples/basic_examples` - Explore more examples
- :doc:`../tutorials/beginner/index` - Step-by-step beginner tutorials

Concept Comparison Table
========================

.. list-table:: Classical vs Quantum Computing
   :header-rows: 1
   :widths: 30 35 35

   * - Concept
     - Classical Computing
     - Quantum Computing
   * - Information Unit
     - Bit (0 or 1)
     - Qubit (superposition state)
   * - Logic Gates
     - Boolean gates (AND, OR, NOT)
     - Quantum gates (H, CNOT, RX, etc.)
   * - Parallelism
     - Spatial parallelism (multiprocessor)
     - Quantum parallelism (superposition)
   * - Correlation
     - Classical correlation
     - Quantum entanglement
   * - Information Extraction
     - Direct readout
     - Quantum measurement (probabilistic)
   * - State Evolution
     - Deterministic transitions
     - Unitary evolution
   * - Computational Model
     - Turing machine
     - Quantum circuit model

Understanding these fundamental differences is key to leveraging quantum computing's unique capabilities.
