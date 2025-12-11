===========
Core Module
===========

The Core module provides the foundational abstractions and data structures for the TyxonQ framework, including the Intermediate Representation (IR), type system, and error handling. These components form the basis of the entire framework, offering stable and reliable interfaces for upper-layer applications and compilers.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

The TyxonQ Core module employs a layered architecture design, providing comprehensive abstraction levels for quantum computing:

- **Intermediate Representation (IR)**: Unified quantum circuit and pulse representation
- **Type System**: Strong type constraints ensuring code safety
- **Error Handling**: Complete exception hierarchy and error recovery mechanisms
- **Operation Definitions**: Standardized quantum gates and measurement operations

Core Architecture
=================

.. mermaid::

   graph TD
       A[User Interface Layer] --> B[Core IR Layer]
       B --> C[Type System Layer]
       C --> D[Operation Definition Layer]
       D --> E[Error Handling Layer]
       
       B1[Circuit IR] --> B
       B2[Pulse IR] --> B
       
       C1[Problem Types] --> C
       C2[Qubit Indexing] --> C
       C3[Parameter Types] --> C
       
       E1[TyxonQError] --> E
       E2[CompilationError] --> E
       E3[DeviceExecutionError] --> E

Intermediate Representation (IR)
=================================

Circuit IR
----------

The :class:`Circuit` class is the core data structure representing quantum circuits, providing a complete interface for circuit construction and manipulation.

**Key Features**

- **Immutable Design**: All operations return new Circuit instances
- **Method Chaining**: Supports fluent API design pattern
- **Metadata Support**: Can attach arbitrary metadata information
- **Serialization Support**: JSON format import and export

**Data Structure**

.. code-block:: python

   @dataclass
   class Circuit:
       num_qubits: int                           # Number of qubits
       ops: List[Any]                           # List of operations
       metadata: Dict[str, Any]                 # Metadata dictionary
       instructions: List[Tuple[str, Tuple]]    # Instruction sequence

**Core Methods**

.. list-table:: Main Methods of Circuit Class
   :header-rows: 1
   :widths: 25 35 40

   * - Method
     - Functionality
     - Example
   * - ``with_metadata()``
     - Add metadata
     - ``circuit.with_metadata(author="Alice")``
   * - ``extended()``
     - Extend circuit
     - ``circuit.extended([("h", 0)])``
   * - ``compose()``
     - Compose circuits
     - ``circuit.compose(other, indices=[1])``
   * - ``remap_qubits()``
     - Remap qubits
     - ``circuit.remap_qubits({0: 2, 1: 3})``
   * - ``inverse()``
     - Inverse circuit
     - ``circuit.inverse()``

**Usage Example**

.. code-block:: python

   import tyxonq as tq
   
   # Create a circuit
   circuit = tq.Circuit(3)
   
   # Build with method chaining
   circuit = (
       circuit
       .h(0)
       .cnot(0, 1)
       .cnot(1, 2)
       .with_metadata(description="GHZ state preparation")
   )
   
   # View circuit information
   print(f"Number of qubits: {circuit.num_qubits}")
   print(f"Number of operations: {len(circuit.ops)}")
   print(f"Gate summary: {circuit.gate_summary()}")
   print(f"Metadata: {circuit.metadata}")

Pulse IR
--------

The :class:`PulseSchedule` class represents pulse-level quantum control sequences, supporting hardware-level precise control.

**Key Features**

- **Time Precision**: Nanosecond-level time control
- **Multi-Channel Support**: Simultaneous control of multiple qubits
- **Waveform Definition**: Custom pulse waveforms
- **Synchronization Mechanism**: Precise timing coordination

.. code-block:: python

   from tyxonq.core.ir.pulse import PulseSchedule, PulseInstruction
   
   # Create pulse schedule
   schedule = PulseSchedule(sampling_rate_hz=1e9)
   
   # Add pulse instruction
   schedule.append(PulseInstruction(
       channel="drive_0",
       start_time=0.0,
       duration=20e-9,  # 20 nanoseconds
       waveform="gaussian"
   ))
   
   print(f"Total duration: {schedule.duration_seconds():.2e} seconds")

Type System
===========

Basic Types
-----------

TyxonQ defines a strict type system to ensure code safety:

.. code-block:: python

   from tyxonq.core.types import Problem
   from typing import Literal, Dict, Any
   
   # Problem type definition
   class Problem:
       kind: Literal["hamiltonian", "circuit", "pulse", "custom"]
       payload: Dict[str, Any]

**Supported Problem Types**

.. list-table:: Problem Type Descriptions
   :header-rows: 1
   :widths: 20 80

   * - Type
     - Description
   * - hamiltonian
     - Hamiltonian-related problems
   * - circuit
     - Quantum circuit problems
   * - pulse
     - Pulse control problems
   * - custom
     - Custom problem types

Qubit Indexing
--------------

Qubit indices use integer types and support negative indexing (counting from the end):

.. code-block:: python

   circuit = tq.Circuit(4)
   
   # Forward indexing
   circuit.h(0)    # First qubit
   circuit.h(3)    # Fourth qubit
   
   # Negative indexing (if supported)
   # circuit.h(-1)   # Last qubit

Operation Definitions
=====================

Quantum Gate Operations
-----------------------

TyxonQ supports a complete universal quantum gate set:

**Single-Qubit Gates**

.. code-block:: python

   # Pauli gates
   circuit.x(0)    # Pauli-X gate
   circuit.y(0)    # Pauli-Y gate
   circuit.z(0)    # Pauli-Z gate
   
   # Hadamard gate
   circuit.h(0)    # Create superposition
   
   # Phase gates
   circuit.s(0)    # S gate (π/2 phase)
   circuit.t(0)    # T gate (π/4 phase)
   
   # Parameterized rotation gates
   circuit.rx(0, theta)    # Rotation around X-axis
   circuit.ry(0, theta)    # Rotation around Y-axis
   circuit.rz(0, theta)    # Rotation around Z-axis

**Two-Qubit Gates**

.. code-block:: python

   # Controlled gates
   circuit.cnot(0, 1)    # Controlled-NOT gate
   circuit.cx(0, 1)      # Alias for CNOT
   circuit.cz(0, 1)      # Controlled-Z gate
   circuit.cy(0, 1)      # Controlled-Y gate
   
   # SWAP gate
   circuit.swap(0, 1)    # Swap two qubits
   
   # Parameterized two-qubit gates
   circuit.rxx(0, 1, theta)    # XX rotation
   circuit.rzz(0, 1, theta)    # ZZ rotation

Measurement Operations
----------------------

.. code-block:: python

   # Z-basis measurement
   circuit.measure_z(0)
   
   # Batch measurement
   circuit.add_measure(0, 1, 2)
   
   # Reset operation
   circuit.reset(0)
   
   # Barrier operation (prevents optimization across)
   circuit.add_barrier(0, 1)

Error Handling
==============

Exception Hierarchy
-------------------

TyxonQ defines a complete exception handling hierarchy:

.. mermaid::

   graph TD
       A[TyxonQError] --> B[CompilationError]
       A --> C[DeviceExecutionError]
       A --> D[ValidationError]
       A --> E[ConfigurationError]
       
       B --> B1[CircuitCompilationError]
       B --> B2[PulseCompilationError]
       
       C --> C1[DeviceConnectionError]
       C --> C2[DeviceTimeoutError]
       
       D --> D1[QubitIndexError]
       D --> D2[ParameterValidationError]

**Usage Example**

.. code-block:: python

   from tyxonq.core.errors import TyxonQError, CompilationError
   
   try:
       # Quantum circuit operation that may fail
       circuit = tq.Circuit(2)
       circuit.cnot(0, 5)  # Error: qubit index out of range
   except ValidationError as e:
       print(f"Validation error: {e}")
   except CompilationError as e:
       print(f"Compilation error: {e}")
   except TyxonQError as e:
       print(f"TyxonQ error: {e}")

Circuit Validation
------------------

All circuits undergo automatic validation during creation:

.. code-block:: python

   # Validate qubit indices
   circuit = tq.Circuit(2)
   try:
       circuit.h(3)  # Index out of range
   except Exception as e:
       print(f"Index error: {e}")
   
   # Validate gate parameters
   try:
       circuit.rx(0, "invalid")  # Parameter type error
   except Exception as e:
       print(f"Parameter error: {e}")

Best Practices
==============

Circuit Construction Guidelines
--------------------------------

1. **Use method chaining**: Improves code readability

   .. code-block:: python

      # Recommended
      circuit = (
          tq.Circuit(3)
          .h(0)
          .cnot(0, 1)
          .cnot(1, 2)
          .with_metadata(description="GHZ state")
      )

2. **Appropriate error handling**: Catch specific exceptions

   .. code-block:: python

      try:
          result = circuit.run()
      except DeviceExecutionError:
          # Retry or use another device
          result = circuit.device('statevector').run()

3. **Reasonable use of metadata**: Facilitates debugging and documentation

   .. code-block:: python

      circuit = circuit.with_metadata(
          description="Bell state preparation",
          author="Alice",
          version="1.0"
      )

Performance Optimization
------------------------

1. **Avoid frequent circuit rebuilding**:

   .. code-block:: python

      # Inefficient
      for i in range(n):
          circuit = circuit.h(i)
      
      # Efficient
      ops = [("h", i) for i in range(n)]
      circuit = circuit.extended(ops)

2. **Reasonable use of immutability**:

   .. code-block:: python

      # If multiple modifications are needed, consider one-time operations
      base_circuit = tq.Circuit(n)
      variants = [
          base_circuit.extended(ops1),
          base_circuit.extended(ops2),
          base_circuit.extended(ops3)
      ]

Related Resources
=================

- :doc:`/api/core/index` - Core API Detailed Reference
- :doc:`../compiler/index` - Compiler User Guide
- :doc:`../devices/index` - Device Execution Guide
- :doc:`/getting_started/first_circuit` - Circuit Getting Started Tutorial
