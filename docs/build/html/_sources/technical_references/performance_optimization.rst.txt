Performance Optimization
========================

=========================
Performance Optimization
=========================

This document provides practical guidance for optimizing TyxonQ applications across different execution paths and use cases.

General Optimization Principles
================================

1. **Choose the Right Path**: Device path for hardware validation, numeric path for fast iteration
2. **Leverage Measurement Grouping**: Reduce circuit executions through compiler-driven grouping
3. **Select Appropriate Backend**: NumPy for development, PyTorch for gradients, CuPyNumeric for GPU
4. **Minimize Circuit Depth**: Use compiler passes for gate reduction
5. **Optimize Shot Allocation**: Variance-weighted scheduling for efficient sampling

Device Path Optimization
========================

1. Measurement Grouping
-----------------------

**Problem**: Hamiltonian with N Pauli terms requires N separate circuit executions

**Solution**: Compiler groups commuting terms for simultaneous measurement

**Example**:

.. code-block:: python

   import tyxonq as tq
   
   # Build circuit
   c = tq.Circuit(4).h(0).cx(0,1).cx(1,2).cx(2,3)
   
   # Enable measurement grouping
   compiled = c.compile(passes=["measurement_rewrite"])
   
   # Inspect grouping metadata
   print(compiled.metadata["measurement_groups"])
   # Output: Fewer groups than original Pauli terms

**Performance Gain**: 5-10x reduction in circuit executions for typical molecular Hamiltonians

2. Shot Scheduling
------------------

**Problem**: Uniform shot allocation wastes resources on low-variance terms

**Solution**: Variance-weighted shot scheduling

**Example**:

.. code-block:: python

   # Automatic shot scheduling
   compiled = c.compile(passes=["measurement_rewrite", "shot_scheduler"])
   
   # Inspect shot plan
   print(compiled.metadata["shot_plan"])
   # Output: Higher shots for high-variance groups

**Performance Gain**: 2-3x reduction in total shots for target accuracy

3. Light-cone Simplification
----------------------------

**Problem**: Gates outside measurement light-cone waste resources

**Solution**: Compiler prunes unnecessary gates

**Example**:

.. code-block:: python

   compiled = c.compile(passes=["lightcone"])
   print(f"Original ops: {len(c.ops)}")
   print(f"Simplified ops: {len(compiled.circuit.ops)}")

**Performance Gain**: 10-30% reduction in gate count for sparse measurements

4. Hardware-Specific Compilation
---------------------------------

**Use Qiskit Transpiler**: For IBM hardware, leverage Qiskit's optimizations

.. code-block:: python

   # Use Qiskit transpiler for hardware-specific optimization
   compiled = c.compile(
       engine="qiskit",
       output="qasm",
       optimization_level=3
   )

**Performance Gain**: Hardware-specific gate set, connectivity-aware routing

Numeric Path Optimization
=========================

1. Backend Selection
--------------------

**NumPy Backend**
  - **Use for**: Development, small systems (n < 12 qubits)
  - **Performance**: Moderate, CPU-bound
  
**PyTorch Backend**
  - **Use for**: Gradient-based optimization, ML integration
  - **Performance**: Automatic differentiation, GPU support
  
**CuPyNumeric Backend**
  - **Use for**: Large systems requiring GPU acceleration
  - **Performance**: High throughput on CUDA GPUs

**Example**:

.. code-block:: python

   import tyxonq as tq
   
   # Development: NumPy
   tq.set_backend("numpy")
   
   # Gradient optimization: PyTorch
   tq.set_backend("pytorch")
   
   # Large-scale GPU: CuPyNumeric
   tq.set_backend("cupynumeric")

2. Statevector Simulation
--------------------------

**Engine Selection**:

- **Statevector**: Fast, O(2^n) memory, for pure states
- **Density Matrix**: O(4^n) memory, for mixed states/noise
- **MPS**: O(poly(n)) memory, for low-entanglement systems

**Example**:

.. code-block:: python

   # Statevector for pure states (fastest)
   result = c.device(provider="simulator", device="statevector").run()
   
   # MPS for large low-entanglement systems
   result = c.device(provider="simulator", device="mps", bond_dim=128).run()

**Performance Guide**:

- n ≤ 12: Statevector (< 1 GB memory)
- 12 < n ≤ 20: Statevector with GPU (CuPyNumeric)
- n > 20: MPS for low-entanglement, or cluster simulation

3. Tensor Contraction Optimization
-----------------------------------

**Use opt_einsum**: Optimized tensor contraction paths

.. code-block:: python

   import opt_einsum as oe
   
   # TyxonQ automatically uses opt_einsum for complex contractions
   # Manual optimization example:
   path, path_info = oe.contract_path('ij,jk,kl->il', A, B, C, optimize='optimal')

**Performance Gain**: 2-10x speedup for complex tensor networks

Quantum Chemistry Optimization
===============================

1. Active Space Approximation
------------------------------

**Problem**: Full molecular Hamiltonian requires too many qubits

**Solution**: Select chemically important orbitals

**Example**:

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   from pyscf import gto, scf
   
   mol = gto.Mole()
   mol.atom = 'H 0 0 0; H 0 0 0.74'
   mol.basis = '6-31g'  # Larger basis
   mol.build()
   
   mf = scf.RHF(mol).run()
   
   # Use active space (4 electrons, 4 orbitals)
   uccsd = UCCSD(mol, n_elec_active=4, n_orb_active=4)

**Performance Gain**: Reduces qubits from 10+ to 4, enables larger molecules

2. Ansatz Selection
-------------------

**UCCSD**
  - **Accuracy**: High
  - **Cost**: Expensive (many parameters, deep circuits)
  - **Use for**: Small molecules requiring high accuracy

**HEA**
  - **Accuracy**: Moderate
  - **Cost**: Efficient (shallow circuits)
  - **Use for**: Larger molecules, hardware execution

**k-UpCCGSD**
  - **Accuracy**: High with multiple tries
  - **Cost**: Expensive but parallelizable
  - **Use for**: Production calculations

**pUCCD**
  - **Accuracy**: Moderate (paired excitations only)
  - **Cost**: Efficient
  - **Use for**: Strongly correlated systems

3. Gradient Computation
-----------------------

**Parameter-Shift Rule** (Device Path)
  - **Method**: Evaluate circuit at shifted parameters
  - **Cost**: 2 * n_params circuit evaluations
  - **Use for**: Hardware execution, noisy simulation

**Automatic Differentiation** (Numeric Path)
  - **Method**: PyTorch autograd
  - **Cost**: Single forward + backward pass
  - **Use for**: Algorithm development, fast iteration

**Example**:

.. code-block:: python

   import tyxonq as tq
   from tyxonq.applications.chem import UCCSD
   
   # Numeric path: autograd
   tq.set_backend("pytorch")
   uccsd = UCCSD(mol, runtime="numeric")
   energy, grad = uccsd.energy_and_grad(params)  # Fast autograd
   
   # Device path: parameter-shift
   uccsd = UCCSD(mol, runtime="device")
   energy, grad = uccsd.energy_and_grad(params, shots=1024)  # Hardware-realistic

4. Cloud Offloading
-------------------

**Problem**: PySCF HF/MP2/CCSD computations are slow on local CPU

**Solution**: Offload to cloud GPU nodes

**Example**:

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   
   uccsd = UCCSD(mol)
   
   # Offload classical kernels to cloud GPU
   energy = uccsd.kernel(
       runtime="device",
       classical_provider="cloud",
       classical_device="gpu_node",
       shots=4096
   )

**Performance Gain**: 5-20x speedup for HF/integral computation on GPU

Memory Management
=================

1. Statevector Memory
---------------------

**Memory Requirements**:

.. code-block:: python

   # Memory = 2^n * 16 bytes (complex128)
   # n=10: 16 KB
   # n=20: 16 MB
   # n=30: 16 GB
   # n=40: 16 TB (not feasible)

**Optimization**: Use MPS for n > 25 with low entanglement

2. Batch Processing
-------------------

**Problem**: Multiple circuits overwhelm memory

**Solution**: Batch execution with memory cleanup

.. code-block:: python

   results = []
   for batch in circuit_batches:
       batch_results = device.batch_run(batch, shots=1024)
       results.extend(batch_results)
       # Memory cleanup happens between batches

3. Caching Strategy
-------------------

**TyxonQ Caching**: Hamiltonians, MPO matrices, CI mappings

**Clear Cache**:

.. code-block:: python

   from tyxonq.applications.chem import clear_cache
   
   # Clear chemistry caches between runs
   clear_cache()

Parallel Execution
==================

1. Circuit-Level Parallelism
-----------------------------

**Use batch_run**: Execute multiple circuits in parallel

.. code-block:: python

   # Sequential
   results = [device.run(c, shots=1024) for c in circuits]
   
   # Parallel (if supported by device)
   results = device.batch_run(circuits, shots=1024)

2. Shot-Level Parallelism
--------------------------

**Hardware**: Automatically parallelized by QPU

**Simulator**: Multi-threading for shot sampling

.. code-block:: python

   # NumPy backend uses multi-threading for shots
   import os
   os.environ["OMP_NUM_THREADS"] = "8"  # 8 threads

3. Parameter Optimization Parallelism
--------------------------------------

**VQE with multiple initial guesses**: Run in parallel

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor
   
   def optimize_with_init(init_params):
       uccsd = UCCSD(mol)
       return uccsd.kernel(init_params=init_params)
   
   # Parallel optimization with different initializations
   with ProcessPoolExecutor(max_workers=4) as executor:
       results = list(executor.map(optimize_with_init, init_param_list))

Benchmarking and Profiling
===========================

1. Timing Framework
-------------------

**Separate Staging and Execution**:

.. code-block:: python

   import time
   
   # Staging time (compilation, setup)
   start = time.time()
   compiled = c.compile(passes=["measurement_rewrite", "shot_scheduler"])
   staging_time = time.time() - start
   
   # Execution time (circuit running)
   start = time.time()
   result = compiled.device(provider="simulator", device="statevector", shots=4096).run()
   execution_time = time.time() - start
   
   print(f"Staging: {staging_time:.3f}s, Execution: {execution_time:.3f}s")

2. Performance Metrics
----------------------

**Key Metrics**:

- **Circuit executions**: Fewer is better (via grouping)
- **Total shots**: Fewer for target accuracy (via scheduling)
- **Gate count**: Lower depth (via simplification)
- **Memory usage**: Within hardware limits
- **Wall-clock time**: End-to-end latency

3. Profiling Tools
------------------

**Python Profiling**:

.. code-block:: bash

   # Profile TyxonQ script
   python -m cProfile -o profile.stats your_script.py
   
   # Analyze with snakeviz
   pip install snakeviz
   snakeviz profile.stats

**GPU Profiling** (PyTorch/CuPy):

.. code-block:: python

   import torch
   
   # PyTorch profiler
   with torch.profiler.profile() as prof:
       energy = uccsd.kernel(runtime="numeric")
   
   print(prof.key_averages().table(sort_by="cuda_time_total"))

Common Performance Pitfalls
============================

1. **Not using measurement grouping**: 5-10x slowdown
2. **Uniform shot allocation**: 2-3x wasted shots
3. **Wrong backend for use case**: NumPy for large systems → OOM
4. **Deep UCCSD on hardware**: Circuit too deep for current devices
5. **Ignoring active space**: Full molecule → too many qubits
6. **No caching cleanup**: Memory leaks in long runs
7. **Sequential circuit execution**: Miss parallelization opportunities

Performance Best Practices
===========================

1. **Start with numeric path**: Fast iteration for algorithm development
2. **Validate with device path**: Ensure hardware-realistic behavior
3. **Use appropriate backend**: NumPy → PyTorch → CuPyNumeric as needed
4. **Enable all compiler passes**: Measurement grouping, shot scheduling, simplification
5. **Profile before optimizing**: Measure, don't guess
6. **Leverage cloud offloading**: For heavy classical kernels
7. **Batch circuit execution**: Amortize overhead
8. **Monitor memory usage**: Avoid OOM with MPS or active space

Conclusion
==========

TyxonQ's architecture provides multiple optimization levers:

- **Compiler-driven**: Measurement grouping, shot scheduling
- **Execution flexibility**: Device vs. numeric paths
- **Backend selection**: NumPy/PyTorch/CuPyNumeric
- **Domain-specific**: Active space, ansatz selection, cloud offloading

By understanding these mechanisms and applying them appropriately, users can achieve:

- **5-10x reduction** in circuit executions (measurement grouping)
- **2-3x reduction** in total shots (shot scheduling)
- **10-100x speedup** in gradient computation (autograd vs. parameter-shift)
- **5-20x speedup** in classical kernels (cloud GPU offloading)

For specific optimization recommendations, consult the examples in ``examples/`` and benchmarks in ``scripts/``.

Performance Optimization documentation will be added soon.
