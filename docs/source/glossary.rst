========
Glossary
========

========
Glossary
========

Quantum Computing Terms
=======================

.. glossary::

   Qubit
      Quantum bit, the basic unit of quantum information. In TyxonQ, qubits are indexed starting from 0.

   Quantum Gate
      Unitary operation acting on one or more qubits. Examples: Hadamard (H), CNOT (CX), Pauli gates (X, Y, Z).

   Quantum Circuit
      Sequence of quantum gates applied to qubits. Represented by the ``Circuit`` class in TyxonQ.

   Superposition
      Quantum state that is a linear combination of basis states (e.g., |0⟩ and |1⟩).

   Entanglement
      Quantum correlation between qubits where the state of one qubit depends on the state of another.

   Measurement
      Process of reading qubit state, collapsing superposition to a classical bit value (0 or 1).

   Observable
      Hermitian operator representing a physical quantity to be measured. Often represented as a weighted sum of Pauli operators.

   Hamiltonian
      Hermitian operator representing the energy of a quantum system. In TyxonQ, represented by the ``Hamiltonian`` class.

   Shot
      Single execution of a quantum circuit with measurement. Multiple shots provide statistical sampling.

   Statevector
      Complex vector representing the quantum state in the computational basis.

   Density Matrix
      Matrix representation of quantum state, supporting mixed states and noise.

TyxonQ Architecture
===================

.. glossary::

   Circuit
      Core IR (Intermediate Representation) class in TyxonQ for constructing quantum circuits. Defined in ``src/tyxonq/core/ir/circuit.py``.

   Compilation
      Process of transforming circuits for specific targets. TyxonQ supports native and Qiskit compilation engines.

   Native Compiler
      TyxonQ's built-in compiler with basic optimization passes. Implemented in ``src/tyxonq/compiler/compile_engine/native/``.

   Qiskit Compiler
      Compilation backend using Qiskit's transpilation framework. Requires Qiskit installation.

   Numeric Backend
      Backend for numerical tensor operations. TyxonQ supports: NumPy (CPU), PyTorch (GPU), CuPyNumeric (distributed GPU).

   Runtime
      Execution environment for quantum circuits. Options: ``numeric`` (simulator), ``device`` (hardware).

   Postprocessing
      Operations applied to measurement results: error mitigation, classical shadows, expectation value computation.

   Chain API
      Fluent interface for circuit workflows: ``c.compile().device().run()``.

Compilation Terms
=================

.. glossary::

   Compilation Pass
      Transformation applied during compilation. Examples: ``measurement_rewrite``, ``shot_scheduler``, ``auto_measure``.

   Optimization Level
      Degree of circuit optimization in Qiskit compiler. Ranges from 0 (no optimization) to 3 (heavy optimization).

   Basis Gates
      Set of native gates supported by hardware. Circuits are decomposed into basis gates during compilation.

   Qubit Mapping
      Assignment of logical qubits to physical qubits on hardware.

   Gate Decomposition
      Breaking down complex gates into sequences of basis gates.

   Circuit Depth
      Maximum number of sequential gate layers in a circuit. Lower depth reduces decoherence errors.

Quantum Chemistry Terms
=======================

.. glossary::

   VQE
      Variational Quantum Eigensolver. Hybrid quantum-classical algorithm for finding ground state energies.

   UCCSD
      Unitary Coupled Cluster Singles and Doubles. Quantum chemistry ansatz for electronic structure.

   HEA
      Hardware-Efficient Ansatz. Parameterized circuit designed for near-term quantum devices.

   Ansatz
      Parameterized quantum circuit template used in variational algorithms.

   Active Space
      Subset of molecular orbitals included in quantum simulation. Reduces computational cost.

   Molecular Integral
      One-electron (``int1e``) and two-electron (``int2e``) integrals describing molecular Hamiltonian.

   Fermion-to-Qubit Mapping
      Transformation from fermionic operators to qubit operators. TyxonQ supports Jordan-Wigner and Bravyi-Kitaev mappings.

   Second Quantization
      Formalism using creation/annihilation operators to represent many-body quantum systems.

   Hartree-Fock
      Mean-field approximation providing initial guess for quantum chemistry calculations.

   FCI
      Full Configuration Interaction. Exact solution within active space, used as reference.

Algorithm Terms
===============

.. glossary::

   QAOA
      Quantum Approximate Optimization Algorithm. For combinatorial optimization problems.

   Trotter
      Method for approximating time evolution operator through product formula.

   Parameter Shift Rule
      Technique for computing gradients on quantum hardware by evaluating circuits at shifted parameter values.

   Classical Shadows
      Protocol for efficiently learning properties of quantum states from measurement data.

   Readout Error Mitigation
      Technique to correct measurement errors using calibration matrices.

   Expectation Value
      Average value of an observable over quantum state: ⟨ψ|H|ψ⟩.

Cloud and Hardware Terms
========================

.. glossary::

   Homebrew_S2
      TyxonQ's 13-qubit superconducting quantum processor operated by QureGenAI.

   TyxonQ Cloud API
      REST API for accessing quantum hardware. Endpoint: ``https://api.tyxonq.com/qau-cloud/tyxonq/``.

   Optimization Flags
      Hardware compilation options: ``o=1`` (qubit mapping), ``o=2`` (gate decomposition), ``o=4`` (initial mapping).

   Pulse-Level Control
      Direct microwave pulse programming using TQASM 0.2 format.

   TQASM
      TyxonQ Assembly. Domain-specific language for quantum circuits. Version 0.2 supports pulse-level programming.

   Waveform
      Parametric pulse shape for pulse-level control. Examples: CosineDrag, Gaussian, Sine.

   Defcal
      Pulse calibration definition in TQASM 0.2. Defines custom gate implementations.

   Frame
      Microwave control channel in pulse programming. Created with ``newframe``.

   TyxonQTask
      Object representing submitted quantum job. Has attributes: ``id``, ``device``, ``status``.

Libraries and Utilities
=======================

.. glossary::

   Hamiltonian Encoding
      Process of representing classical Hamiltonians as quantum operators.

   Ising Model
      Spin model used in combinatorial optimization, mapped to quantum Hamiltonians.

   Pauli String
      Product of Pauli operators (I, X, Y, Z) acting on multiple qubits.

   Optimizer
      Classical optimization algorithm for variational methods. Examples: COBYLA, L-BFGS-B.

   MPS
      Matrix Product State. Tensor network representation for low-entanglement states.

   Molecule
      Class representing molecular system with atomic coordinates, basis set, charge, and spin.

   PySCF
      Python-based quantum chemistry package. TyxonQ integrates with PySCF for molecular calculations.

Programming Concepts
====================

.. glossary::

   Param
      Symbolic parameter for parameterized circuits. Created with ``Param(name)``.

   Gate Operation
      Single gate instruction in circuit. Stored as tuple: ``(gate_type, qubits, parameters)``.

   Compilation Result
      Output of compilation containing optimized circuit, metadata, and performance statistics.

   Device Descriptor
      Configuration object specifying target device, shot count, and execution options.

   Session
      Execution context maintaining device state and configuration across multiple runs.

   Vectorization
      Parallelization strategy for batch circuit execution. Policies: ``auto``, ``force``, ``disable``.

Error Handling
==============

.. glossary::

   NotImplementedError
      Raised when feature is not yet implemented or unsupported by backend.

   ValueError
      Raised for invalid parameter values or circuit configurations.

   ConnectionError
      Raised when cloud API connection fails.

   TimeoutError
      Raised when task execution exceeds timeout limit.

   RuntimeError
      Raised for general execution errors, including cloud API failures.

File Formats
============

.. glossary::

   OpenQASM
      Open Quantum Assembly Language. Standard format for quantum circuits (version 2.0).

   TQASM 0.2
      TyxonQ's assembly format supporting pulse-level control.

   JSON
      JavaScript Object Notation. Used for API requests/responses and circuit serialization.

Common Abbreviations
====================

.. glossary::

   NISQ
      Noisy Intermediate-Scale Quantum. Current era of quantum computing with limited qubits and coherence.

   IR
      Intermediate Representation. Abstract representation of circuits between high-level code and execution.

   API
      Application Programming Interface. Set of functions for interacting with TyxonQ.

   GPU
      Graphics Processing Unit. Used for accelerated simulations with PyTorch/CuPy backends.

   CPU
      Central Processing Unit. Default execution using NumPy backend.

   REST
      Representational State Transfer. Architecture for cloud API.

   BNF
      Backus-Naur Form. Notation for defining syntax grammar (used in TQASM specification).