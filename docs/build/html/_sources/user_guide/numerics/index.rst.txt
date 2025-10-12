================
Numerics Backend
================

The Numerics Backend system provides a unified array interface across different computational frameworks, enabling seamless switching between NumPy, PyTorch, CuPy, and other backends for quantum simulations.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

The TyxonQ numerics backend system offers:

- **Unified Interface**: ArrayBackend protocol for framework-agnostic code
- **Multiple Backends**: Support for NumPy (CPU), PyTorch (GPU/autodiff), CuPy (GPU)
- **Context Management**: Easy backend switching with context managers
- **Vectorization**: Automatic vectorization with vmap support
- **Type Safety**: Runtime type checking and validation

Available Backends
==================

NumPy Backend
-------------

**Purpose**: Default CPU-based numerical computations

.. code-block:: python

   import tyxonq as tq
   from tyxonq.numerics import set_backend
   
   # Set NumPy backend (default)
   set_backend('numpy')
   
   # Run simulation
   circuit = tq.Circuit(2).h(0).cnot(0, 1)
   result = circuit.run()

**Features:**
- Stable and reliable
- No additional dependencies
- Good for small to medium circuits
- Widely compatible

PyTorch Backend
---------------

**Purpose**: GPU acceleration and automatic differentiation

.. code-block:: python

   from tyxonq.numerics import set_backend
   
   # Set PyTorch backend
   set_backend('pytorch')
   
   # Enable GPU if available
   import torch
   if torch.cuda.is_available():
       set_backend('pytorch', device='cuda')
   
   # Run simulation with automatic differentiation
   circuit = tq.Circuit(2).ry(0, theta).cnot(0, 1)
   result = circuit.run()

**Features:**
- GPU acceleration
- Automatic differentiation
- Integration with PyTorch ecosystem
- Ideal for variational algorithms

CuPy Backend
------------

**Purpose**: High-performance GPU computations

.. code-block:: python

   from tyxonq.numerics import set_backend
   
   # Set CuPy backend for GPU acceleration
   set_backend('cupy')
   
   # Run large-scale simulation
   circuit = tq.Circuit(25).h(range(25))
   result = circuit.run()

**Features:**
- Maximum GPU performance
- NumPy-compatible API
- Large-scale simulations
- Optimized for NVIDIA GPUs

Context Management
==================

Temporary Backend Switching
---------------------------

.. code-block:: python

   from tyxonq.numerics import backend_context
   
   # Default backend (NumPy)
   result1 = circuit1.run()
   
   # Temporarily use PyTorch
   with backend_context('pytorch'):
       result2 = circuit2.run()
   
   # Back to NumPy
   result3 = circuit3.run()

Global Backend Configuration
----------------------------

.. code-block:: python

   from tyxonq.numerics import set_backend, get_backend
   
   # Check current backend
   current = get_backend()
   print(f"Current backend: {current}")
   
   # Set global backend
   set_backend('pytorch')
   
   # All subsequent computations use PyTorch
   result = circuit.run()

Vectorization
=============

Automatic Vectorization
-----------------------

The backend system provides automatic vectorization through ``vmap``:

.. code-block:: python

   import numpy as np
   from tyxonq.numerics import vectorize_or_fallback
   
   # Function to vectorize
   def run_circuit(theta):
       circuit = tq.Circuit(2).ry(0, theta).cnot(0, 1)
       return circuit.run()
   
   # Vectorize over parameter array
   thetas = np.linspace(0, np.pi, 10)
   vectorized_run = vectorize_or_fallback(run_circuit)
   results = vectorized_run(thetas)

Vectorization Policies
----------------------

.. code-block:: python

   # Auto: Use vmap if available, fallback to loop
   vectorized_auto = vectorize_or_fallback(func, policy='auto')
   
   # Force: Always use backend vmap (error if unavailable)
   vectorized_force = vectorize_or_fallback(func, policy='force')
   
   # Off: Disable vectorization
   vectorized_off = vectorize_or_fallback(func, policy='off')

Best Practices
==============

Choosing the Right Backend
--------------------------

.. list-table:: Backend Selection Guide
   :header-rows: 1
   :widths: 25 25 25 25

   * - Use Case
     - Recommended Backend
     - Reason
     - Considerations
   * - Small circuits (<20 qubits)
     - NumPy
     - Simple, reliable
     - No GPU needed
   * - Variational algorithms
     - PyTorch
     - Automatic differentiation
     - GPU beneficial
   * - Large simulations (>25 qubits)
     - CuPy
     - Maximum GPU performance
     - Requires NVIDIA GPU
   * - Debugging
     - NumPy
     - Easy inspection
     - Slower but clearer

Performance Tips
----------------

1. **Use GPU for large circuits**:

   .. code-block:: python

      # For circuits with >15 qubits
      if circuit.num_qubits > 15:
          set_backend('pytorch', device='cuda')

2. **Batch computations**:

   .. code-block:: python

      # Instead of loop
      # for theta in thetas:
      #     result = run_circuit(theta)
      
      # Use vectorization
      results = vectorize_or_fallback(run_circuit)(thetas)

3. **Memory management**:

   .. code-block:: python

      import torch
      
      # Clear GPU cache periodically
      if torch.cuda.is_available():
          torch.cuda.empty_cache()

Related Resources
=================

- :doc:`/api/numerics/index` - Numerics API Reference
- :doc:`../devices/index` - Device Execution Guide
- :doc:`/examples/hybrid_gpu_pipeline` - GPU Acceleration Examples
- :doc:`/examples/vmap_randomness` - Vectorization Examples
