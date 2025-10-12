============
Compiler API
============

Complete API reference for TyxonQ's quantum circuit compiler and optimization framework.

.. contents:: Contents
   :depth: 3
   :local:

Overview
========

The Compiler API provides circuit transformation and optimization:

🔧 **Compilation** (``tyxonq.compiler.compile``)
   High-level compilation interface

⚙️ **Optimization Stages**
   Decomposition, optimization, layout, and scheduling

🎯 **Device Targeting**
   Hardware-specific compilation

Main Compilation Function
=========================

.. autofunction:: tyxonq.compiler.compile

**Signature**:

.. code-block:: python

   def compile(
       circuit,                      # Circuit to compile
       optimization_level: int = 1,  # 0, 1, 2, or 3
       basis_gates: List[str] = None,  # Target gate set
       compile_engine: str = "default",  # Compiler backend
       **options
   ) -> Circuit

**Example**:

.. code-block:: python

   from tyxonq import Circuit
   from tyxonq.compiler import compile
   
   # Create circuit
   circuit = Circuit(3)
   circuit.h(0)
   circuit.cx(0, 1)
   circuit.cx(1, 2)
   
   # Compile with optimization
   compiled = compile(
       circuit,
       optimization_level=2,
       basis_gates=['h', 'rx', 'ry', 'rz', 'cx']
   )

Optimization Levels
===================

Level 0: No Optimization
------------------------

Only basic translation, no optimization.

.. code-block:: python

   compiled = circuit.compile(optimization_level=0)

**Use when**:
- Debugging
- Testing specific gate sequences
- Circuit already optimized

Level 1: Light Optimization
---------------------------

Basic optimizations: gate fusion, single-qubit optimization.

.. code-block:: python

   compiled = circuit.compile(optimization_level=1)  # Default

**Optimizations**:
- Adjacent single-qubit gate fusion
- Redundant gate removal
- Simple commutation rules

**Use when**:
- General purpose compilation
- Balance between speed and quality

Level 2: Medium Optimization
----------------------------

More aggressive optimizations.

.. code-block:: python

   compiled = circuit.compile(optimization_level=2)

**Optimizations**:
- All Level 1 optimizations
- Two-qubit gate optimization
- Template matching and substitution
- Commutativity-based reordering

**Use when**:
- Production workloads
- Circuit depth matters

Level 3: Heavy Optimization
---------------------------

Maximum optimization, may be slow.

.. code-block:: python

   compiled = circuit.compile(optimization_level=3)

**Optimizations**:
- All Level 2 optimizations  
- Exhaustive template search
- Advanced synthesis techniques
- Multiple optimization passes

**Use when**:
- Critical applications
- Offline compilation acceptable
- Maximum quality needed

Basis Gate Translation
======================

Common Gate Sets
----------------

**IBM Gate Set**:

.. code-block:: python

   ibm_basis = ['id', 'rz', 'sx', 'x', 'cx']
   compiled = circuit.compile(basis_gates=ibm_basis)

**Google Gate Set**:

.. code-block:: python

   google_basis = ['sqrt_x', 'rz', 'cz']
   compiled = circuit.compile(basis_gates=google_basis)

**Universal Set**:

.. code-block:: python

   universal_basis = ['h', 'rx', 'ry', 'rz', 'cx']
   compiled = circuit.compile(basis_gates=universal_basis)

Custom Decomposition
--------------------

.. code-block:: python

   # Decompose to specific gates
   custom_basis = ['h', 't', 'cx']
   compiled = circuit.compile(
       basis_gates=custom_basis,
       optimization_level=2
   )

Compiler Engines
================

Native Compiler
---------------

TyxonQ's built-in compiler.

.. code-block:: python

   compiled = circuit.compile(
       compile_engine="native",  # or "default"
       optimization_level=2
   )

**Features**:
- Fast compilation
- Integrated with TyxonQ IR
- Supports all TyxonQ features

Qiskit Compiler
---------------

Use Qiskit's transpiler.

.. code-block:: python

   compiled = circuit.compile(
       compile_engine="qiskit",
       optimization_level=2
   )

**Features**:
- Mature optimization passes
- Hardware layout optimization
- Extensive basis gate support

**Requirements**: Qiskit must be installed

Compilation Stages
==================

Decomposition
-------------

Break down high-level gates into basis gates.

.. code-block:: python

   from tyxonq.compiler.stages.decompose import decompose
   
   # Decompose multi-controlled gates
   decomposed = decompose(
       circuit,
       basis_gates=['h', 'rx', 'ry', 'rz', 'cx']
   )

Optimization
------------

Reduce circuit depth and gate count.

.. code-block:: python

   from tyxonq.compiler.stages.optimize import optimize
   
   optimized = optimize(
       circuit,
       optimization_level=2
   )

Layout
------

Map virtual qubits to physical qubits.

.. code-block:: python

   from tyxonq.compiler.stages.layout import layout
   
   # Map to device topology
   mapped = layout(
       circuit,
       coupling_map=[(0, 1), (1, 2), (2, 3)]  # Linear chain
   )

Scheduling
----------

Schedule gates considering device constraints.

.. code-block:: python

   from tyxonq.compiler.stages.scheduling import schedule
   
   scheduled = schedule(
       circuit,
       dt=0.1,  # Time step
       basis_gates=['h', 'rx', 'ry', 'rz', 'cx']
   )

Advanced Features
=================

Custom Compilation Pipeline
---------------------------

.. code-block:: python

   from tyxonq.compiler import CompilePlan
   from tyxonq.compiler.stages import (
       DecomposePass,
       OptimizePass,
       LayoutPass
   )
   
   # Build custom pipeline
   plan = CompilePlan([
       DecomposePass(basis_gates=['h', 'cx']),
       OptimizePass(level=2),
       LayoutPass(coupling_map=device_topology)
   ])
   
   # Execute
   compiled = plan.execute(circuit)

Device-Specific Compilation
---------------------------

.. code-block:: python

   # Compile for specific device
   from tyxonq.devices import get_device_rule
   
   device_rule = get_device_rule("ibmq_manila")
   
   compiled = circuit.compile(
       device_rule=device_rule,
       optimization_level=2
   )

Circuit Unrolling
-----------------

.. code-block:: python

   # Unroll parameterized circuits
   from tyxonq import Circuit, Parameter
   import numpy as np
   
   theta = Parameter('θ')
   circuit = Circuit(2).ry(0, theta).cx(0, 1)
   
   # Bind parameters
   bound = circuit.bind_parameters({theta: np.pi/4})
   
   # Compile bound circuit
   compiled = bound.compile(optimization_level=2)

Common Workflows
================

Basic Compilation
-----------------

.. code-block:: python

   from tyxonq import Circuit
   
   # Build circuit
   circuit = (
       Circuit(3)
       .h(0)
       .cx(0, 1)
       .cx(1, 2)
       .measure_all()
   )
   
   # Compile and run
   result = circuit.compile(optimization_level=2).run()

Device-Targeted Compilation
---------------------------

.. code-block:: python

   # Target specific device
   circuit = Circuit(5).h(0)
   for i in range(4):
       circuit.cx(i, i+1)
   
   compiled = circuit.compile(
       optimization_level=2,
       basis_gates=['rz', 'sx', 'x', 'cx'],
       coupling_map=[(0,1), (1,2), (2,3), (3,4)]
   )

Chained Compilation
-------------------

.. code-block:: python

   # Use method chaining
   result = (
       Circuit(3)
       .h(0)
       .cx(0, 1)
       .cx(1, 2)
       .compile(optimization_level=2)
       .device(provider="local", device="statevector")
       .run()
   )

Compilation Metrics
===================

Circuit Statistics
------------------

.. code-block:: python

   # Get circuit metrics
   original = Circuit(3).h(0).cx(0, 1).cx(1, 2)
   compiled = original.compile(optimization_level=2)
   
   print(f"Original depth: {original.depth()}")
   print(f"Compiled depth: {compiled.depth()}")
   print(f"Original gates: {original.size()}")
   print(f"Compiled gates: {compiled.size()}")
   
   # Calculate reduction
   depth_reduction = 1 - compiled.depth() / original.depth()
   gate_reduction = 1 - compiled.size() / original.size()
   
   print(f"Depth reduction: {depth_reduction:.1%}")
   print(f"Gate reduction: {gate_reduction:.1%}")

Optimization Benchmarking
-------------------------

.. code-block:: python

   import time
   
   circuit = Circuit(10)
   for i in range(10):
       circuit.h(i)
       if i > 0:
           circuit.cx(i-1, i)
   
   # Benchmark different levels
   for level in [0, 1, 2, 3]:
       start = time.time()
       compiled = circuit.compile(optimization_level=level)
       duration = time.time() - start
       
       print(f"Level {level}:")
       print(f"  Time: {duration:.3f}s")
       print(f"  Depth: {compiled.depth()}")
       print(f"  Gates: {compiled.size()}")

See Also
========

- :doc:`/user_guide/compiler/index` - Compiler User Guide
- :doc:`/api/core/index` - Core API
- :doc:`/api/devices/index` - Devices API
- :doc:`/examples/basic_examples` - Compilation Examples
