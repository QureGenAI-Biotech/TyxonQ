==================
Compiler Pipeline
==================

The TyxonQ compiler transforms high-level quantum circuits into low-level representations that can execute on target devices. Through multi-stage optimization and transformation, the compiler ensures circuits run efficiently on various quantum devices.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

The TyxonQ compiler uses a multi-stage pipeline architecture, with each stage responsible for specific optimization and transformation tasks:

- **Decomposition Stage**: Decomposes high-level quantum gates into basic gate sets
- **Rewriting Stage**: Optimizes circuit structure and measurement strategies
- **Simplification Stage**: Eliminates redundant operations and reduces circuit complexity
- **Scheduling Stage**: Manages measurement allocation and shot budgets
- **Gradient Processing**: Generates parameter shift circuits for gradient optimization

Compiler Architecture
=====================

Pipeline Framework
------------------

.. mermaid::

   graph LR
       A[Source Circuit] --> B[Decomposition Stage]
       B --> C[Rewriting Stage]
       C --> D[Simplification Stage]
       D --> E[Scheduling Stage]
       E --> F[Gradient Processing]
       F --> G[Target Code]

Compiler Engines
----------------

TyxonQ supports multiple compiler engines:

.. list-table:: Compiler Engine Comparison
   :header-rows: 1
   :widths: 20 25 30 25

   * - Engine Name
     - Target Format
     - Use Cases
     - Features
   * - native
     - TyxonQ IR
     - Native simulators
     - Complete optimization pipeline
   * - qiskit
     - Qiskit QuantumCircuit
     - IBM hardware/Qiskit integration
     - Qiskit ecosystem compatibility
   * - openqasm
     - OpenQASM 2.0
     - Universal quantum hardware
     - Cross-platform compatibility

Usage Examples
--------------

.. code-block:: python

   import tyxonq as tq
   
   circuit = tq.Circuit(2).h(0).cx(0, 1)
   
   # Use default compiler (Circuit.compile() 返回 self)
   compiled = circuit.compile()
   print(f"Compiled source: {compiled._compiled_source}")
   
   # Specify compiler engine
   compiled = circuit.compile(compile_engine='qiskit')
   
   # Set optimization level
   compiled = circuit.compile(
       compile_engine='native',
       options={'optimization_level': 2}
   )
   
   print(f"Original circuit gates: {len(circuit.ops)}")
   print(f"Compiled gates: {len(compiled.ops)}")

Compilation Stages
==================

Decomposition Stage
-------------------

**Purpose**: Decompose high-level quantum gates into basic gate sets supported by the target device.

**Supported Decompositions**

.. code-block:: python

   # RX gate decomposition: RX(θ) = H · RZ(θ) · H
   circuit = tq.Circuit(1).rx(0, 0.5)
   compiled = circuit.compile()  # 返回 Circuit 对象
   decomposed_ir = compiled
   
   # RY gate decomposition: RY(θ) = S† · H · RZ(θ) · H · S
   circuit = tq.Circuit(1).ry(0, 0.5)
   compiled = circuit.compile()  # 返回 Circuit 对象
   decomposed_ir = compiled
   
   # RZZ gate decomposition: RZZ(θ) = CNOT · RZ(θ) · CNOT
   circuit = tq.Circuit(2).rzz(0, 1, 0.5)
   compiled = circuit.compile()  # 返回 Circuit 对象
   decomposed_ir = compiled

Rewriting Stage
---------------

**Purpose**: Optimize circuit structure and measurement strategies.

**Main Rewriting Rules**

1. **Measurement Optimization**: Automatically group compatible measurements
2. **Gate Transformation**: Standardize gate representation and apply algebraic identities
3. **Auto-Measurement Insertion**: Add measurements to circuits lacking them
4. **Gate Merging and Pruning**: Merge adjacent gates and remove redundant operations

.. code-block:: python

   # Gate merging example
   circuit = (
       tq.Circuit(2)
       .rz(0, 0.1).rz(0, 0.2)  # Mergeable RZ gates
       .x(1).x(1)              # Canceling X gates
   )
   
   compiled = circuit.compile()
   print(f"Optimized gate count: {len(compiled.ops)}")

Simplification Stage
--------------------

**Light Cone Simplification**

Light cone simplification removes operations that don't affect measurement results through backward analysis:

.. code-block:: python

   # Circuit with redundant operations
   circuit = (
       tq.Circuit(4)
       .h(0).h(1).h(2).h(3)        # Apply H gates to all qubits
       .cnot(0, 1).cnot(2, 3)       # Entangling operations
       .measure_z(0).measure_z(1)   # Only measure first two qubits
   )
   
   # Compilation automatically applies light cone simplification
   simplified = circuit.compile()
   
   print(f"Original circuit gates: {len(circuit.ops)}")
   print(f"Simplified gates: {len(simplified.ops)}")
   # Expected to remove operations related to qubits 2, 3

Scheduling Stage
----------------

**Shot Scheduler**

The shot scheduler manages measurement allocation and execution planning:

.. code-block:: python

   # Circuit with measurement grouping
   circuit = (
       tq.Circuit(3)
       .h(0).cx(0, 1).cx(1, 2)
       .measure_z(0).measure_z(1).measure_z(2)
   )
   
   # Automatic scheduling during execution
   result = circuit.run(shots=1000)
   # result 是 list of unified results

Gradient Processing Stage
-------------------------

**Parameter Shift Rule**

The parameter shift rule computes gradients through finite differences:

.. code-block:: python

   import numpy as np
   
   # Parameterized circuit
   theta = np.array([0.1, 0.2, 0.3])
   circuit = (
       tq.Circuit(2)
       .ry(0, theta[0])
       .cnot(0, 1)
       .rz(1, theta[1])
       .ry(1, theta[2])
   )
   
   # Gradient computation is automatically handled during optimization

Best Practices
==============

Compilation Options
-------------------

1. **Choose Appropriate Optimization Level**

   .. code-block:: python

      # Debugging phase: low optimization
      debug_compiled = circuit.compile(options={'optimization_level': 0})
      
      # Production environment: high optimization
      prod_compiled = circuit.compile(options={'optimization_level': 3})

2. **Target Device Adaptation**

   .. code-block:: python

      # Optimize for specific device
      ibm_compiled = circuit.compile(
          compile_engine='qiskit',
          target='ibm_cairo'
      )

Performance Optimization
------------------------

1. **Compilation Caching**: Reuse compilation results
2. **Batch Compilation**: Compile multiple similar circuits at once
3. **Incremental Optimization**: Start debugging from low optimization levels

Related Resources
=================

- :doc:`/api/compiler/index` - Compiler API Reference
- :doc:`../devices/index` - Device Execution Guide
- :doc:`../core/index` - Core Module Introduction
- :doc:`/examples/optimization_examples` - Optimization Examples
