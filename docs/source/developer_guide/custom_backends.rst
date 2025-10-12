Numeric Engines
===============

TyxonQ uses different numeric engines for quantum circuit simulation and computation. This guide covers the available engines and their usage.

.. contents:: Table of Contents
   :local:
   :depth: 3

Engine Architecture
==================

TyxonQ supports multiple simulation engines:

.. code-block:: text

    Simulators
    ├── StatevectorEngine
    ├── DensityMatrixEngine  
    └── MatrixProductStateEngine

Available Engines
================

Statevector Engine
-----------------

The most commonly used engine for pure state simulation:

.. code-block:: python

    import tyxonq as tq
    
    # Use statevector simulator
    circuit = tq.Circuit(2).h(0).cx(0, 1)
    result = (
        circuit
        .device(provider="simulator", device="statevector")
        .run(shots=1024)
    )

**Features:**
- Memory: O(2^n)
- Fast execution for pure states
- Supports all quantum gates
- Best for small to medium circuits (up to ~20 qubits)

Density Matrix Engine
--------------------

For mixed state simulation and noise modeling:

.. code-block:: python

    # Use density matrix simulator
    result = (
        circuit
        .device(provider="simulator", device="density_matrix")
        .run(shots=1024)
    )

**Features:**
- Memory: O(4^n) 
- Supports noise channels
- Can simulate mixed states
- More expensive than statevector

Matrix Product State Engine
--------------------------

For simulating larger quantum circuits with limited entanglement:

.. code-block:: python

    # Use MPS simulator
    result = (
        circuit
        .device(provider="simulator", device="matrix_product_state")
        .run(shots=1024)
    )

**Features:**
- Memory: O(poly(n)) for low entanglement
- Can handle larger systems
- Performance depends on entanglement structure

Numeric Backend System
=====================

TyxonQ uses a unified ArrayBackend protocol for different numeric libraries:

.. code-block:: python

    import tyxonq as tq
    
    # Set numeric backend
    tq.set_backend("numpy")        # NumPy backend
    tq.set_backend("pytorch")      # PyTorch backend
    tq.set_backend("cupynumeric")  # CuPy backend

Backend Features
---------------

**NumPy Backend:**
- Default backend
- CPU computation
- Wide compatibility
- Stable and well-tested

**PyTorch Backend:**
- GPU acceleration support
- Automatic differentiation
- ML framework integration
- Gradient computation

**CuPy Backend:**
- GPU-accelerated NumPy-like interface
- CUDA support
- High performance computing

Engine Implementation
====================

Each engine follows the same interface:

.. code-block:: python

    class Engine:
        name = "engine_name"
        capabilities = {"supports_shots": True}
        
        def __init__(self, backend_name=None):
            from tyxonq.numerics.api import get_backend
            self.backend = get_backend(backend_name)
        
        def run(self, circuit, shots=None, **kwargs):
            # Implementation using self.backend
            pass

Execution Flow
=============

The typical execution flow is:

1. **Circuit Construction**: Build quantum circuit
2. **Compilation**: Apply optimization passes
3. **Engine Selection**: Choose appropriate engine
4. **Execution**: Run on selected engine
5. **Result Processing**: Handle measurement outcomes

.. code-block:: python

    # Complete workflow
    import tyxonq as tq
    
    # Set backend
    tq.set_backend("numpy")
    
    # Build circuit
    circuit = tq.Circuit(2).h(0).cx(0, 1)
    
    # Execute with specific engine
    result = (
        circuit
        .compile()
        .device(provider="simulator", device="statevector", shots=1024)
        .run()
    )

Performance Considerations
=========================

**Choosing the Right Engine:**

- **Small circuits (< 15 qubits)**: Use statevector
- **Noise modeling**: Use density_matrix
- **Large circuits with low entanglement**: Use matrix_product_state
- **GPU acceleration needed**: Set PyTorch or CuPy backend

**Memory Requirements:**

.. code-block:: text

    Qubits | Statevector | Density Matrix | MPS (bond dim=64)
    -------|-------------|----------------|------------------
    10     | 8 KB        | 64 KB          | ~10 KB
    15     | 256 KB      | 2 MB           | ~15 KB
    20     | 8 MB        | 128 MB         | ~20 KB
    25     | 256 MB      | 4 GB           | ~25 KB

Best Practices
=============

1. **Start Small**: Test with statevector for small problems
2. **Profile Memory**: Monitor memory usage for larger circuits
3. **Use Appropriate Backend**: Match backend to your needs
4. **Benchmark**: Compare different engines for your use case
5. **Consider Noise**: Use density_matrix when noise is important

.. note::
   TyxonQ's numeric engines are designed for flexibility and performance.
   Choose the appropriate engine based on your specific requirements.

.. seealso::
   
   - :doc:`architecture_overview` - TyxonQ architecture overview
   - :doc:`custom_devices` - Device development guide
   - :doc:`testing_guidelines` - Testing best practices
