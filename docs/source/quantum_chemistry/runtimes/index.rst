========================================
Runtime Systems for Quantum Chemistry
========================================

TyxonQ provides two runtime modes for executing quantum chemistry algorithms: **Device Runtime** for quantum hardware and simulators, and **Numeric Runtime** for exact classical simulation.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

The runtime system determines how quantum circuits are executed and energy expectation values are computed. Choosing the right runtime is crucial for balancing:

⚡ **Speed**: Numeric runtime is faster for small systems
🎯 **Accuracy**: Numeric runtime provides exact results (no shot noise)
📡 **Hardware Access**: Device runtime enables real quantum device execution
🔬 **Noise Modeling**: Device runtime includes noise and error effects

Runtime Modes
=============

Device Runtime
==============

Overview
--------

Device runtime executes quantum circuits on real quantum hardware or shot-based simulators. It:

- Compiles circuits for target hardware
- Uses shot-based sampling for measurements
- Supports noise modeling and error mitigation
- Enables execution on real quantum devices

**When to Use Device Runtime**:

- Testing algorithms on actual quantum hardware
- Studying the impact of noise and decoherence
- Validating error mitigation strategies
- Preparing for production quantum computing

Basic Usage
-----------

.. code-block:: python

   from tyxonq.applications.chem import Molecule, HEA
   
   # Define molecule
   h2 = Molecule(
       atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]],
       basis="sto-3g"
   )
   
   # Create HEA with device runtime
   hea = HEA(
       molecule=h2,
       layers=2,
       runtime="device"  # Enable device runtime
   )
   
   # Run with default settings (2048 shots)
   energy = hea.kernel()
   print(f"Energy (device): {energy:.6f} Hartree")

Configuration Options
---------------------

**Shot Count**:

.. code-block:: python

   # High precision (more shots = less noise)
   energy_high = hea.kernel(shots=8192)
   
   # Medium precision (balance speed/accuracy)
   energy_med = hea.kernel(shots=2048)  # Default
   
   # Quick estimation (faster but noisier)
   energy_low = hea.kernel(shots=512)
   
   # Exact (uses statevector, no sampling)
   energy_exact = hea.kernel(shots=0)  # Falls back to numeric

**Provider Selection**:

.. code-block:: python

   # Local simulator (default)
   hea.kernel(provider="local", device="statevector")
   
   # Alternative local simulators
   hea.kernel(provider="local", device="mps")  # Matrix product state
   hea.kernel(provider="local", device="density_matrix")  # Density matrix

**Device Options**:

.. list-table:: Available Local Simulators
   :header-rows: 1
   :widths: 30 30 40

   * - Device
     - Type
     - Best For
   * - ``statevector``
     - State vector simulator
     - Small systems (≤20 qubits), fast exact simulation
   * - ``mps`` / ``matrix_product_state``
     - Matrix product state
     - Larger systems with low entanglement
   * - ``density_matrix``
     - Density matrix simulator
     - Open quantum systems, mixed states

Advanced Features
-----------------

**Parameter-Shift Gradient**:

Device runtime uses parameter-shift rule for gradient computation:

.. code-block:: python

   # Get energy and gradient
   params = hea.init_guess
   energy, grad = hea.energy_and_grad(params, shots=2048)
   
   print(f"Energy: {energy:.6f}")
   print(f"Gradient norm: {np.linalg.norm(grad):.6f}")

**Measurement Grouping**:

Automatically groups Pauli strings by measurement basis:

.. code-block:: python

   # Hamiltonian is automatically grouped
   # X basis: all X,Y measurements
   # Z basis: all Z,I measurements
   # Reduces total circuit executions
   
   hamiltonian = h2.get_hamiltonian(mapping="jordan_wigner")
   print(f"Total Hamiltonian terms: {len(hamiltonian)}")
   # Device runtime groups these efficiently

Numeric Runtime
===============

Overview
--------

Numeric runtime uses exact classical simulation methods for quantum circuits. It:

- Computes exact expectation values (no sampling noise)
- Supports multiple simulation backends
- Provides fast gradients via automatic differentiation
- Enables rapid algorithm prototyping

**When to Use Numeric Runtime**:

- Algorithm development and debugging
- Obtaining noise-free baseline results
- Small to medium-sized molecules (≤20 qubits)
- Parameter optimization with exact gradients

Basic Usage
-----------

.. code-block:: python

   from tyxonq.applications.chem import Molecule, UCCSD
   
   # Define molecule
   h2o = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="sto-3g"
   )
   
   # Create UCCSD with numeric runtime
   uccsd = UCCSD(
       molecule=h2o,
       init_method="mp2",
       runtime="numeric"  # Enable numeric runtime
   )
   
   # Run optimization (exact, no shot noise)
   energy = uccsd.kernel(method="BFGS")
   print(f"Energy (numeric): {energy:.6f} Hartree")

Simulation Engines
------------------

**Statevector Engine** (Default):

.. code-block:: python

   # Fast, exact statevector simulation
   hea = HEA(
       molecule=h2,
       layers=2,
       runtime="numeric",
       numeric_engine="statevector"  # Default
   )

**Characteristics**:

- Memory: O(2^n) for n qubits
- Speed: Very fast for ≤20 qubits
- Best for: Small to medium molecules
- Scaling: Up to ~25 qubits on typical workstations

Performance Optimization
------------------------

.. code-block:: python

   # For small molecules: use statevector
   if n_qubits <= 20:
       runtime = "numeric"
       engine = "statevector"
   
   # For medium molecules: use numeric runtime with shots=0
   elif 20 < n_qubits <= 30:
       runtime = "device"
       shots = 0  # Exact simulation without sampling
   
   # For large molecules: use device with shots
   else:
       runtime = "device"
       shots = 2048  # Shot-based sampling

Gradient Computation
--------------------

Numeric runtime provides exact gradients:

.. code-block:: python

   # Energy and exact gradient
   energy, grad = uccsd.energy_and_grad(params)
   
   # Use with scipy optimizers
   from scipy.optimize import minimize
   
   def objective(x):
       e, g = uccsd.energy_and_grad(x)
       return e, g
   
   result = minimize(
       lambda x: objective(x)[0],
       x0=uccsd.init_guess,
       jac=lambda x: objective(x)[1],
       method="L-BFGS-B"
   )

Runtime Comparison
==================

Accuracy Comparison
-------------------

.. code-block:: python

   from tyxonq.applications.chem import Molecule, HEA
   import numpy as np
   
   h2 = Molecule(
       atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]],
       basis="sto-3g"
   )
   
   # Numeric runtime (exact)
   hea_numeric = HEA(molecule=h2, layers=2, runtime="numeric")
   e_numeric = hea_numeric.kernel()
   
   # Device runtime with different shot counts
   hea_device = HEA(molecule=h2, layers=2, runtime="device")
   
   shot_counts = [512, 1024, 2048, 4096, 8192]
   errors = []
   
   for shots in shot_counts:
       e_device = hea_device.kernel(shots=shots)
       error = abs(e_device - e_numeric)
       errors.append(error)
       print(f"Shots: {shots:5d}, Error: {error:.6f} Hartree")
   
   # Statistical error scales as 1/sqrt(shots)
   expected_scaling = [errors[0] * np.sqrt(shot_counts[0]/s) for s in shot_counts]

Speed Comparison
----------------

.. code-block:: python

   import time
   
   # Benchmark numeric runtime
   start = time.time()
   e_numeric = hea_numeric.kernel()
   time_numeric = time.time() - start
   
   # Benchmark device runtime
   start = time.time()
   e_device = hea_device.kernel(shots=2048)
   time_device = time.time() - start
   
   print(f"Numeric runtime: {time_numeric:.2f} seconds")
   print(f"Device runtime:  {time_device:.2f} seconds")
   print(f"Speedup: {time_device/time_numeric:.1f}x")

**Typical Performance** (for small molecules):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - System Size
     - Numeric (statevector)
     - Device (2048 shots)
     - Speedup
   * - 4 qubits (H2)
     - 0.1s
     - 1.5s
     - 15x
   * - 8 qubits (H2O)
     - 0.3s
     - 3.0s
     - 10x
   * - 12 qubits (NH3)
     - 1.0s
     - 5.0s
     - 5x
   * - 16 qubits
     - 5.0s
     - 10.0s
     - 2x

Practical Workflows
===================

Development Workflow
--------------------

.. code-block:: python

   from tyxonq.applications.chem import Molecule, UCCSD
   
   # Step 1: Develop with numeric runtime
   mol = Molecule(atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]], basis="sto-3g")
   
   uccsd_dev = UCCSD(molecule=mol, init_method="mp2", runtime="numeric")
   e_dev = uccsd_dev.kernel(method="BFGS")
   print(f"Development result: {e_dev:.6f} Hartree")
   
   # Step 2: Validate with device runtime (local simulator)
   uccsd_val = UCCSD(molecule=mol, init_method="mp2", runtime="device")
   e_val = uccsd_val.kernel(shots=4096, provider="local")
   print(f"Validation result: {e_val:.6f} Hartree")
   
   # Step 3: Deploy to real hardware
   uccsd_prod = UCCSD(molecule=mol, init_method="mp2", runtime="device")
   e_prod = uccsd_prod.kernel(shots=8192, provider="ibm", device="ibmq_manila")
   print(f"Production result: {e_prod:.6f} Hartree")

Hybrid Workflow
---------------

.. code-block:: python

   # Use numeric runtime for optimization
   hea_numeric = HEA(molecule=mol, layers=3, runtime="numeric")
   optimal_params = hea_numeric.kernel()
   
   # Transfer optimal parameters to device runtime
   hea_device = HEA(molecule=mol, layers=3, runtime="device")
   hea_device.params = hea_numeric.params  # Transfer parameters
   
   # Evaluate on device
   final_energy = hea_device.energy(hea_device.params, shots=8192)
   print(f"Final energy on device: {final_energy:.6f} Hartree")

Best Practices
==============

Choosing Runtime
----------------

**Use Numeric Runtime when**:

✅ Developing new algorithms
✅ Need exact results for validation
✅ System size ≤ 20 qubits
✅ Optimizing with gradient-based methods
✅ Rapid prototyping

**Use Device Runtime when**:

✅ Testing on real hardware
✅ Studying noise effects
✅ Preparing for production
✅ System too large for classical simulation
✅ Validating error mitigation

Optimizing Performance
----------------------

.. code-block:: python

   # Strategy 1: Use numeric runtime for initial optimization
   hea = HEA(molecule=mol, layers=2, runtime="numeric")
   hea.scipy_minimize_options = {"maxiter": 200, "gtol": 1e-6}
   optimal_energy = hea.kernel(method="L-BFGS-B")
   
   # Strategy 2: Reduce shots for device runtime during optimization
   hea_device = HEA(molecule=mol, layers=2, runtime="device")
   hea_device.scipy_minimize_options = {"maxiter": 50}  # Fewer iterations
   hea_device.kernel(shots=1024, method="COBYLA")  # Lower shots
   
   # Strategy 3: Final evaluation with high shots
   final_energy = hea_device.energy(hea_device.params, shots=8192)

Error Handling
--------------

.. code-block:: python

   try:
       # Attempt numeric runtime
       hea = HEA(molecule=large_mol, layers=3, runtime="numeric")
       energy = hea.kernel()
   except MemoryError:
       print("System too large for statevector, switching to device runtime")
       hea = HEA(molecule=large_mol, layers=3, runtime="device")
       energy = hea.kernel(shots=2048)

Advanced Topics
===============

.. toctree::
   :maxdepth: 2

   architecture
   runtime_optimization
   numerical_methods

Related Resources
=================

- :doc:`architecture` - **Architecture Guide**: Visual overview of dual-path execution and caching strategy
- :doc:`runtime_optimization` - **Runtime Optimization**: Caching strategies, RDM fixes, NumPy 2.0 compatibility
- :doc:`numerical_methods` - **Numerical Methods**: Engine selection, gradient computation, performance optimization
- :doc:`../fundamentals/index` - Quantum Chemistry Fundamentals
- :doc:`../algorithms/index` - Quantum Chemistry Algorithms
- :doc:`../molecule/index` - Molecule Class Guide
- :doc:`/user_guide/devices/index` - Device System Details
- :doc:`/examples/chemistry_examples` - Practical Examples
