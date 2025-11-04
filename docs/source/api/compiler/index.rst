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
       circuit,                           # Circuit or PulseProgram to compile
       compile_engine: str = "default",  # Compiler backend: "native", "qiskit", "pulse"
       output: str = "ir",               # Output format: "ir", "qasm2", "qasm3", "tqasm", "pulse_ir"
       device_params: Dict = None,       # Device parameters for pulse compilation
       options: Dict = None,             # Additional compilation options
       **kwargs
   ) -> Union[Circuit, str]

**Output Format Options**:

- ``"ir"`` (default): TyxonQ internal IR. Auto-converts when pulse operations detected
- ``"qasm2"``: OpenQASM 2.0 format
- ``"qasm3"`` / ``"openqasm3"``: OpenQASM 3.0 format with pulse extensions
- ``"tqasm"`` / ``"tqasm0.2"``: TQASM 0.2 format (cloud-ready)
- ``"pulse_ir"``: TyxonQ pulse IR (preserves waveform objects)
- ``"tyxonq_homebrew_tqasm"``: TQASM 0.2 for homebrew_s2 device

**Example 1: Gate Circuit Compilation**:

.. code-block:: python

   from tyxonq import Circuit
   from tyxonq.compiler import compile
   
   # Create circuit
   circuit = Circuit(3)
   circuit.h(0)
   circuit.cx(0, 1)
   circuit.cx(1, 2)
   
   # Basic compilation (returns CompileResult dict)
   result = compile(circuit)
   circuit_ir = result["circuit"]       # IR object
   compiled_source = result["compiled_source"]  # None or source string
   
   # Specify optimization
   result = compile(
       circuit,
       options={'optimization_level': 2}
   )
   circuit_ir = result["circuit"]

**Example 2: Pulse Circuit Compilation**:

.. code-block:: python

   # Create circuit with pulse operations
   circuit = Circuit(2)
   circuit.h(0)
   circuit.cx(0, 1)
   
   # Enable pulse compilation with device parameters
   circuit.use_pulse(device_params={
       "qubit_freq": [5.0e9, 5.1e9],
       "anharmonicity": [-330e6, -320e6]
   })
   
   # Compile to QASM3 (returns CompileResult dict)
   result = compile(circuit, output="qasm3")
   qasm3_code = result["compiled_source"]  # QASM3 string with defcal
   circuit_ir = result["circuit"]           # Original circuit IR
   
   # Or compile to pulse IR (preserves waveform objects)
   result = compile(circuit, output="pulse_ir")
   pulse_ir = result["compiled_source"]  # PulseProgram IR object
   circuit_ir = result["circuit"]

**Example 3: Cloud Submission (homebrew_s2)**:

.. code-block:: python

   # Setup for cloud execution
   circuit = Circuit(2)
   circuit.device(provider="tyxonq", device="homebrew_s2")
   circuit.h(0)
   circuit.cx(0, 1)
   
   # Use pulse mode
   circuit.use_pulse(device_params={
       "qubit_freq": [5.0e9, 5.1e9],
       "anharmonicity": [-330e6, -320e6]
   })
   
   # Compile to TQASM (auto-converts to tyxonq_homebrew_tqasm for homebrew_s2)
   result = compile(circuit, output="tqasm")
   tqasm_code = result["compiled_source"]  # TQASM string (QASM2 compatible)
   
   # Submit to cloud
   result = circuit.run(shots=100)

Pulse Compilation Function
==========================

.. autofunction:: tyxonq.compiler.compile_pulse

**Signature**:

.. code-block:: python

   def compile_pulse(
       pulse_program,                      # PulseProgram to compile
       output: str = "pulse_ir",         # Output format: "pulse_ir", "tqasm", "openqasm3"
       device_params: Dict = None,        # Device parameters (qubit_freq, anharmonicity, etc.)
       calibrations: Dict = None,         # Custom pulse calibrations
       device: str = None,                # Target device for format selection
       options: Dict = None,              # Compilation options (inline_pulses, etc.)
       **kwargs
   ) -> PulseCompileResult

**Returns**:

``PulseCompileResult`` TypedDict with three fields:

.. code-block:: python

   {
       "pulse_program": PulseProgram,              # Original PulseProgram IR
       "compiled_pulse_schedule": Optional[str],   # Compiled schedule (TQASM/QASM3) or IR object
       "metadata": Dict[str, Any]                  # Compilation metadata
   }

**Output Format Options**:

- ``"pulse_ir"`` (default): TyxonQ native Pulse IR
  - compiled_pulse_schedule: PulseProgram IR object
- ``"tqasm"`` / ``"tqasm0.2"``: TQASM 0.2 format (cloud-ready)
  - compiled_pulse_schedule: TQASM string with defcal
- ``"openqasm3"``: OpenQASM 3.0 with pulse extensions
  - compiled_pulse_schedule: QASM3 string with defcal

**Example 1: Basic Pulse Compilation**:

.. code-block:: python

   from tyxonq.core.ir.pulse import PulseProgram
   from tyxonq.compiler import compile_pulse
   
   # Create pulse program
   prog = PulseProgram(1)
   prog.drag(0, amp=1.0, duration=160, sigma=40, beta=0.2, qubit_freq=5.0e9)
   
   # Compile to Pulse IR (default)
   result = compile_pulse(prog, device_params={
       "qubit_freq": [5.0e9],
       "anharmonicity": [-330e6]
   })
   pulse_ir = result["pulse_program"]           # PulseProgram IR
   schedule = result["compiled_pulse_schedule"] # None (IR format)

**Example 2: Compile to TQASM for Cloud**:

.. code-block:: python

   # Compile to TQASM with full defcal definitions
   result = compile_pulse(
       prog,
       output="tqasm",
       device_params={
           "qubit_freq": [5.0e9],
           "anharmonicity": [-330e6]
       },
       options={"inline_pulses": True}  # Include full defcal definitions
   )
   tqasm_code = result["compiled_pulse_schedule"]  # TQASM string with defcal
   
   # Submit to cloud
   circuit = Circuit(1)
   task = circuit.device(provider="tyxonq", device="homebrew_s2").run(
       source=tqasm_code,
       shots=100
   )

Optimization Levels
===================

Automatic Compilation Rules
---------------------------

The compiler applies automatic rules to optimize output format based on circuit type and device:

**Rule 1: Gate-only circuits on homebrew_s2**

.. code-block:: python

   circuit = Circuit(2)
   circuit.device(provider="tyxonq", device="homebrew_s2")
   circuit.h(0)
   circuit.cx(0, 1)
   
   result = compile(circuit)  # output="ir" (default)
   # Result: Automatically converted to qasm2 format

**Rule 2: Pulse-mode circuits (auto-detection)**

.. code-block:: python

   circuit = Circuit(2)
   circuit.h(0)
   circuit.use_pulse(device_params={...})  # Enable pulse mode
   circuit.cx(0, 1)
   
   result = compile(circuit, output="ir")
   # Result: Automatically detects pulse operations, converts to openqasm3

**Rule 3: Pulse circuits on homebrew_s2**

.. code-block:: python

   circuit = Circuit(2)
   circuit.device(provider="tyxonq", device="homebrew_s2")
   circuit.h(0)
   circuit.use_pulse(device_params={...})  # Enable pulse mode
   circuit.cx(0, 1)
   
   result = compile(circuit, output="ir")
   # Result: Automatically converts to tyxonq_homebrew_tqasm (TQASM 0.2)

Detailed Compilation Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pulse Operation Detection**:

When compiling circuits with ``output="ir"``, the compiler automatically detects pulse operations:

.. code-block:: python

   # These trigger pulse detection:
   - circuit.use_pulse()  # Explicit pulse mode
   - Pulse operations in circuit.ops ("pulse", "pulse_inline", etc.)
   - mode="pulse_only" in options

**Auto-conversion Logic**:

1. If output="ir" and pulse operations detected:
   
   - For homebrew_s2: convert to ``tyxonq_homebrew_tqasm``
   - For other devices: convert to ``openqasm3``

2. If output is explicit ("tqasm", "qasm3", etc.), use as-is with device check
3. If output="qasm2", keep as-is (gate-only)

Optimization Levels
-------------------

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
- :doc:`/api/core/index` - Core API
- :doc:`/api/devices/index` - Devices API
- :doc:`/examples/basic_examples` - Compilation Examples
