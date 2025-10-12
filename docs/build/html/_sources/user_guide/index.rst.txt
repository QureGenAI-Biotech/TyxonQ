==========
User Guide
==========

Comprehensive guides for using the TyxonQ quantum computing framework. The User Guide covers all major components from basic quantum circuits to advanced optimization techniques, providing both theoretical background and practical examples.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

The TyxonQ User Guide is organized into six main sections, each covering a critical aspect of the framework:

🔧 **Core Module**
   Fundamental data structures including Circuit IR, type system, and quantum operations

⚙️ **Compiler Pipeline**
   Multi-stage compilation system transforming high-level circuits to optimized implementations

💻 **Device Abstraction**
   Unified interface for quantum simulators and hardware with transparent execution

🔢 **Numerics Backend**
   Flexible numerical computation supporting NumPy, PyTorch, and CuPy backends

📊 **Postprocessing**
   Advanced analysis tools including error mitigation, classical shadows, and metrics

🚀 **Advanced Topics**
   Performance optimization, custom implementations, and framework extension

.. mermaid::

   graph TB
       A[User Applications] --> B[Advanced Topics]
       B --> C[Postprocessing]
       C --> D[Device Abstraction]
       D --> E[Compiler Pipeline]
       E --> F[Core Module]
       F --> G[Numerics Backend]
       
       B1[Performance Tuning] --> B
       B2[Custom Backends] --> B
       B3[Framework Extension] --> B
       
       C1[Error Mitigation] --> C
       C2[Classical Shadows] --> C
       C3[Quantum Metrics] --> C
       
       D1[Simulators] --> D
       D2[Hardware] --> D
       D3[Session Management] --> D
       
       E1[Decomposition] --> E
       E2[Optimization] --> E
       E3[Scheduling] --> E
       
       F1[Circuit IR] --> F
       F2[Operations] --> F
       F3[Type System] --> F
       
       G1[NumPy] --> G
       G2[PyTorch] --> G
       G3[CuPy] --> G

Quick Start Guide
=================

For new users, we recommend following this path through the User Guide:

1. **Start with Core Module** - Understand the fundamental Circuit IR and operations
2. **Learn Device Abstraction** - Run your first quantum circuits on simulators
3. **Explore Compiler Pipeline** - Optimize circuits for better performance
4. **Use Numerics Backend** - Choose the right computational backend for your needs
5. **Apply Postprocessing** - Analyze results with advanced techniques
6. **Master Advanced Topics** - Customize and extend the framework

Common Workflows
================

Quantum Algorithm Development
-----------------------------

.. code-block:: python

   import tyxonq as tq
   
   # 1. Core: Build quantum circuit
   circuit = (
       tq.Circuit(3)
       .h(0)
       .cnot(0, 1)
       .cnot(1, 2)
       .with_metadata(name="GHZ State")
   )
   
   # 2. Compiler: Optimize for target device
   optimized = circuit.compile(optimization_level=2)
   
   # 3. Device: Execute on simulator
   result = optimized.device('statevector').run(shots=1000)
   
   # 4. Postprocessing: Analyze results
   from tyxonq.postprocessing import counts_to_expval
   expectation = counts_to_expval(result)

Variational Algorithm Implementation
------------------------------------

.. code-block:: python

   import numpy as np
   from tyxonq.numerics import set_backend
   from tyxonq.postprocessing import ReadoutMitigator
   
   # 1. Numerics: Enable automatic differentiation
   set_backend('pytorch')
   
   # 2. Core: Parameterized circuit
   def variational_circuit(params):
       return (
           tq.Circuit(4)
           .ry(0, params[0])
           .cnot(0, 1)
           .ry(1, params[1])
           .cnot(1, 2)
           .ry(2, params[2])
           .cnot(2, 3)
       )
   
   # 3. Compiler: Automatic optimization
   params = np.random.rand(3)
   circuit = variational_circuit(params)
   compiled = circuit.compile()
   
   # 4. Device: Execute with error mitigation
   mitigator = ReadoutMitigator(num_qubits=4)
   raw_result = compiled.run(shots=1000)
   mitigated_result = mitigator.apply(raw_result)

Best Practices Summary
======================

Circuit Construction
--------------------

- Use method chaining for readable circuit construction
- Add meaningful metadata for debugging and documentation
- Validate circuits early in development
- Use appropriate abstraction levels (Circuit IR vs direct operations)

Performance Optimization
------------------------

- Choose the right numerical backend for your use case
- Apply compiler optimizations appropriate to your target device
- Use vectorization for parameter sweeps
- Consider memory constraints for large quantum systems

Error Handling and Debugging
----------------------------

- Implement proper exception handling for device failures
- Use statevector simulator for debugging before hardware deployment
- Apply error mitigation techniques for noisy intermediate-scale quantum (NISQ) devices
- Validate results with classical shadows for large systems

Sections Overview
=================

.. toctree::
   :maxdepth: 2

   core/index
   compiler/index
   devices/index
   numerics/index
   postprocessing/index
   advanced/index

See Also
========

- :doc:`../getting_started/index` - Getting Started Guide
- :doc:`../libraries/index` - Libraries Documentation
- :doc:`../tutorials/index` - Step-by-step Tutorials
- :doc:`../examples/index` - Complete Examples
- :doc:`../api/index` - API Reference
