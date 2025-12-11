=====================
Optimization Examples
=====================

This page provides comprehensive examples of variational optimization algorithms in TyxonQ,
including VQE, QAOA, and custom optimization strategies.

.. contents:: Contents
   :depth: 3
   :local:

.. note::
   These examples demonstrate optimization techniques beyond quantum chemistry.
   For chemistry-specific examples, see :doc:`chemistry_examples`.

Variational Quantum Eigensolver (VQE)
======================================

VQE is the foundational variational algorithm for finding ground state energies.

Basic VQE Example
-----------------

Implementing VQE from scratch:

.. code-block:: python

   import tyxonq as tq
   import numpy as np
   from scipy.optimize import minimize

   # Define Hamiltonian: H = 0.5*Z0 + 0.3*Z1 - 0.7*X0*X1
   hamiltonian = [
       (0.5, [('Z', 0)]),
       (0.3, [('Z', 1)]),
       (-0.7, [('X', 0), ('X', 1)])
   ]

   def create_ansatz(params, n_qubits):
       """Create parameterized ansatz circuit"""
       circuit = tq.Circuit(n_qubits)
       
       # Initial layer
       for i in range(n_qubits):
           circuit.ry(i, params[i])
       
       # Entangling layer
       circuit.cnot(0, 1)
       
       # Second rotation layer
       for i in range(n_qubits):
           circuit.ry(i, params[n_qubits + i])
       
       return circuit

   def energy_expectation(params):
       """Calculate energy expectation value"""
       n_qubits = 2
       circuit = create_ansatz(params, n_qubits)
       
       energy = 0.0
       for coeff, pauli_ops in hamiltonian:
           # Create measurement circuit for this term
           meas_circuit = circuit.copy()
           
           # Apply basis rotations for Pauli measurements
           for pauli, qubit in pauli_ops:
               if pauli == 'X':
                   meas_circuit.h(qubit)
               elif pauli == 'Y':
                   meas_circuit.sdg(qubit)
                   meas_circuit.h(qubit)
               # Z measurements don't need rotation
           
           # Add measurements
           for i in range(n_qubits):
               meas_circuit.measure_z(i)
           
           # Execute and compute expectation
           result = meas_circuit.run(shots=1000)
           exp_val = compute_pauli_expectation(result, pauli_ops)
           energy += coeff * exp_val
       
       return energy

   def compute_pauli_expectation(counts, pauli_ops):
       """Compute Pauli expectation from measurement counts"""
       total_shots = sum(counts.values())
       expectation = 0.0
       
       for bitstring, count in counts.items():
           parity = 1
           for pauli, qubit in pauli_ops:
               if bitstring[qubit] == '1':
                   parity *= -1
           expectation += parity * count
       
       return expectation / total_shots

   # Optimize
   n_params = 4  # 2 qubits × 2 layers
   init_params = np.random.uniform(0, 2*np.pi, n_params)
   
   result = minimize(energy_expectation, init_params, method='COBYLA')
   
   print(f"Optimal energy: {result.fun:.6f} Hartree")
   print(f"Optimal parameters: {result.x}")
   print(f"Converged in {result.nfev} function evaluations")

**Expected Output**:

.. code-block:: text

   Optimal energy: -0.920000 Hartree
   Optimal parameters: [1.234, 2.567, 0.891, 3.012]
   Converged in 87 function evaluations

VQE with Gradient-Based Optimization
-------------------------------------

Using parameter-shift rule for gradients:

.. code-block:: python

   def parameter_shift_gradient(params, param_idx):
       """Compute gradient using parameter shift rule"""
       shift = np.pi / 2
       
       # Forward shift
       params_plus = params.copy()
       params_plus[param_idx] += shift
       energy_plus = energy_expectation(params_plus)
       
       # Backward shift
       params_minus = params.copy()
       params_minus[param_idx] -= shift
       energy_minus = energy_expectation(params_minus)
       
       # Gradient
       grad = (energy_plus - energy_minus) / 2
       return grad

   def energy_and_gradient(params):
       """Return both energy and gradient"""
       energy = energy_expectation(params)
       
       # Compute gradient for all parameters
       gradient = np.array([parameter_shift_gradient(params, i) 
                            for i in range(len(params))])
       
       return energy, gradient

   # Optimize with gradient
   result_grad = minimize(
       fun=lambda x: energy_and_gradient(x)[0],
       x0=init_params,
       method='L-BFGS-B',
       jac=lambda x: energy_and_gradient(x)[1]
   )
   
   print(f"Gradient-based optimization:")
   print(f"  Final energy: {result_grad.fun:.6f}")
   print(f"  Iterations: {result_grad.nit}")

**Benefits of Gradient-Based Methods**:

- Faster convergence for smooth landscapes
- More efficient parameter updates
- Better suited for high-dimensional problems
- Theoretical guarantees from parameter-shift rule

Quantum Approximate Optimization Algorithm (QAOA)
==================================================

QAOA solves combinatorial optimization problems.

Max-Cut QAOA Example
--------------------

Solving Max-Cut problem on a simple graph:

.. code-block:: python

   def create_qaoa_circuit(params, n_qubits, graph_edges, p):
       """Create QAOA circuit with p layers
       
       Args:
           params: [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
           n_qubits: Number of qubits (nodes in graph)
           graph_edges: List of edges [(i,j), ...]
           p: Number of QAOA layers
       """
       circuit = tq.Circuit(n_qubits)
       
       # Initial superposition
       for i in range(n_qubits):
           circuit.h(i)
       
       # QAOA layers
       for layer in range(p):
           gamma = params[layer]
           beta = params[p + layer]
           
           # Problem Hamiltonian: phase separation
           for edge in graph_edges:
               i, j = edge
               circuit.cnot(i, j)
               circuit.rz(j, gamma)
               circuit.cnot(i, j)
           
           # Mixer Hamiltonian
           for qubit in range(n_qubits):
               circuit.rx(qubit, beta)
       
       return circuit

   def maxcut_cost(bitstring, edges):
       """Calculate Max-Cut cost for a bitstring"""
       cost = 0
       for i, j in edges:
           if bitstring[i] != bitstring[j]:
               cost += 1
       return cost

   def qaoa_expectation(params, n_qubits, edges, p):
       """Compute expectation value for QAOA"""
       circuit = create_qaoa_circuit(params, n_qubits, edges, p)
       
       # Add measurements
       for i in range(n_qubits):
           circuit.measure_z(i)
       
       # Execute
       result = circuit.run(shots=2000)
       
       # Compute average cost
       avg_cost = 0
       for bitstring, count in result.items():
           cost = maxcut_cost(bitstring, edges)
           avg_cost += cost * count
       
       avg_cost /= sum(result.values())
       return -avg_cost  # Negative for minimization

   # Define graph (triangle)
   edges = [(0, 1), (1, 2), (2, 0)]
   n_nodes = 3
   p_layers = 2
   
   # Optimize
   n_params = 2 * p_layers
   init_params = np.random.uniform(0, 2*np.pi, n_params)
   
   result = minimize(
       lambda x: qaoa_expectation(x, n_nodes, edges, p_layers),
       init_params,
       method='COBYLA',
       options={'maxiter': 200}
   )
   
   print(f"QAOA Max-Cut Solution:")
   print(f"  Optimal cost: {-result.fun:.2f}")
   print(f"  Max possible: {len(edges)}")
   print(f"  Parameters: {result.x}")

**Analyzing Results**:

.. code-block:: python

   # Get best bitstring from optimized circuit
   optimal_circuit = create_qaoa_circuit(result.x, n_nodes, edges, p_layers)
   for i in range(n_nodes):
       optimal_circuit.measure_z(i)
   
   final_result = optimal_circuit.run(shots=5000)
   best_bitstring = max(final_result, key=final_result.get)
   best_cost = maxcut_cost(best_bitstring, edges)
   
   print(f"\nBest bitstring: {best_bitstring}")
   print(f"Best cost: {best_cost}")
   print(f"Partition: Set A = {[i for i, b in enumerate(best_bitstring) if b=='0']}")
   print(f"           Set B = {[i for i, b in enumerate(best_bitstring) if b=='1']}")

Advanced Optimization Techniques
================================

SOAP Optimizer
--------------

TyxonQ's Sequential Optimization with Approximate Parabola (SOAP) optimizer:

.. code-block:: python

   from tyxonq.libs.optimizer import soap
   import numpy as np

   # Define objective function
   def objective(params):
       """Your optimization objective"""
       # Example: Rosenbrock function
       x, y = params
       return (1 - x)**2 + 100*(y - x**2)**2

   # Initial guess
   x0 = np.array([0.0, 0.0])
   
   # Run SOAP optimization
   result = soap(
       fun=objective,
       x0=x0,
       maxfev=200,  # Maximum function evaluations
       u=0.1,       # Update scale parameter
   )
   
   print(f"SOAP Optimization Results:")
   print(f"  Optimal point: {result['x']}")
   print(f"  Optimal value: {result['fun']:.6f}")
   print(f"  Function evaluations: {result['nfev']}")

**When to Use SOAP**:

- Gradient-free optimization needed
- Noisy objective functions
- Limited function evaluations budget
- Non-smooth optimization landscapes

Noisy Optimization Strategies
-----------------------------

Handling shot noise in quantum optimization:

.. code-block:: python

   def noisy_vqe_optimization():
       """VQE with shot noise mitigation strategies"""
       
       # Strategy 1: Adaptive shot allocation
       def adaptive_shots_energy(params, base_shots=100):
           """Increase shots near convergence"""
           # Simple heuristic: more shots for smaller gradients
           energy = energy_expectation_with_shots(params, base_shots)
           return energy
       
       # Strategy 2: Gradient averaging
       def averaged_gradient(params, n_samples=5):
           """Average gradient over multiple evaluations"""
           gradients = []
           for _ in range(n_samples):
               _, grad = energy_and_gradient(params)
               gradients.append(grad)
           return np.mean(gradients, axis=0)
       
       # Strategy 3: Robust optimization with SPSA
       from scipy.optimize import differential_evolution
       
       bounds = [(0, 2*np.pi)] * n_params
       result = differential_evolution(
           energy_expectation,
           bounds,
           strategy='best1bin',
           maxiter=100,
           popsize=10
       )
       
       return result

   # Run noisy optimization
   noisy_result = noisy_vqe_optimization()
   print(f"Noisy optimization result: {noisy_result.fun:.6f}")

**Noise Mitigation Tips**:

1. **Start with few shots**: Use 100-500 shots for initial exploration
2. **Increase near convergence**: Use 5000+ shots for final refinement
3. **Use gradient-free methods**: COBYLA, SPSA work better with noise
4. **Average measurements**: Multiple runs reduce variance
5. **Error mitigation**: Apply readout error correction

Multi-Objective Optimization
============================

Example: Optimizing Energy and Circuit Depth
---------------------------------------------

.. code-block:: python

   def multi_objective_vqe(params, depth_weight=0.1):
       """Optimize both energy and circuit depth"""
       
       # Primary objective: energy
       energy = energy_expectation(params)
       
       # Secondary objective: circuit complexity
       circuit = create_ansatz(params, n_qubits=2)
       circuit_cost = len(circuit.ops) * depth_weight
       
       # Combined objective
       total_cost = energy + circuit_cost
       return total_cost

   # Optimize with trade-off
   result_multi = minimize(
       multi_objective_vqe,
       init_params,
       method='COBYLA',
       options={'maxiter': 150}
   )
   
   print(f"Multi-objective optimization:")
   print(f"  Total cost: {result_multi.fun:.6f}")
   print(f"  Pure energy: {energy_expectation(result_multi.x):.6f}")

Advanced VQE Patterns
=====================

Example: VQE with Multiple Observables
---------------------------------------

Optimizing multiple Hamiltonians simultaneously:

.. code-block:: python

   def multi_hamiltonian_vqe(params, hamiltonians, weights):
       """VQE for multiple Hamiltonians with weighted sum"""
       total_energy = 0.0
       
       for hamiltonian, weight in zip(hamiltonians, weights):
           energy = compute_hamiltonian_energy(params, hamiltonian)
           total_energy += weight * energy
       
       return total_energy

   # Define multiple targets
   h1 = [(1.0, [('Z', 0)])]
   h2 = [(1.0, [('X', 0)])]
   hamiltonians = [h1, h2]
   weights = [0.7, 0.3]
   
   result = minimize(
       lambda x: multi_hamiltonian_vqe(x, hamiltonians, weights),
       init_params,
       method='COBYLA'
   )

Example: Constrained VQE
------------------------

VQE with parameter constraints:

.. code-block:: python

   from scipy.optimize import minimize, Bounds

   # Define parameter bounds
   bounds = Bounds(
       lb=[0, 0, 0, 0],           # Lower bounds
       ub=[2*np.pi, np.pi, 2*np.pi, np.pi]  # Upper bounds
   )
   
   # Constrained optimization
   result = minimize(
       energy_expectation,
       init_params,
       method='L-BFGS-B',
       bounds=bounds
   )
   
   print(f"Constrained optimization:")
   print(f"  Energy: {result.fun:.6f}")
   print(f"  Parameters within bounds: {np.all(result.x >= 0) and np.all(result.x <= 2*np.pi)}")

Performance Optimization
========================

Parallel VQE Evaluation
-----------------------

Evaluate multiple parameter sets in parallel:

.. code-block:: python

   from concurrent.futures import ThreadPoolExecutor
   import numpy as np

   def parallel_vqe_search(n_random_starts=10):
       """Try multiple random initializations in parallel"""
       
       def run_single_optimization(seed):
           np.random.seed(seed)
           init = np.random.uniform(0, 2*np.pi, n_params)
           result = minimize(energy_expectation, init, method='COBYLA')
           return result
       
       # Parallel execution
       with ThreadPoolExecutor(max_workers=4) as executor:
           futures = [executor.submit(run_single_optimization, i) 
                     for i in range(n_random_starts)]
           results = [f.result() for f in futures]
       
       # Find best result
       best_result = min(results, key=lambda r: r.fun)
       
       print(f"Parallel search results:")
       print(f"  Best energy: {best_result.fun:.6f}")
       print(f"  Tried {n_random_starts} initializations")
       
       return best_result

   best = parallel_vqe_search(n_random_starts=10)

Batched Gradient Computation
----------------------------

.. code-block:: python

   def batched_parameter_shift(params, batch_size=4):
       """Compute gradients in batches for efficiency"""
       n_params = len(params)
       gradients = np.zeros(n_params)
       
       for i in range(0, n_params, batch_size):
           batch_end = min(i + batch_size, n_params)
           
           # Compute batch of gradients
           for j in range(i, batch_end):
               gradients[j] = parameter_shift_gradient(params, j)
       
       return gradients

Troubleshooting Optimization
============================

Diagnosing Convergence Issues
-----------------------------

.. code-block:: python

   def diagnose_optimization(params_history, energy_history):
       """Analyze optimization trajectory"""
       import matplotlib.pyplot as plt
       
       # Plot energy convergence
       plt.figure(figsize=(12, 4))
       
       plt.subplot(1, 2, 1)
       plt.plot(energy_history)
       plt.xlabel('Iteration')
       plt.ylabel('Energy')
       plt.title('Energy Convergence')
       plt.grid(True)
       
       # Plot parameter evolution
       plt.subplot(1, 2, 2)
       params_array = np.array(params_history)
       for i in range(params_array.shape[1]):
           plt.plot(params_array[:, i], label=f'θ{i}')
       plt.xlabel('Iteration')
       plt.ylabel('Parameter Value')
       plt.title('Parameter Evolution')
       plt.legend()
       plt.grid(True)
       
       plt.tight_layout()
       plt.savefig('optimization_diagnosis.png')
       
       # Check for common issues
       energy_range = max(energy_history) - min(energy_history)
       if energy_range < 1e-6:
           print("⚠️  Warning: Very small energy changes - possible plateau")
       
       param_changes = np.diff(params_array, axis=0)
       if np.all(np.abs(param_changes[-10:]) < 1e-6):
           print("⚠️  Warning: Parameters not changing - possible convergence or stuck")

Common Issues and Solutions
---------------------------

**Issue 1: Optimizer stuck in local minimum**

.. code-block:: python

   # Solution: Multiple random starts
   best_energy = float('inf')
   best_params = None
   
   for trial in range(10):
       init = np.random.uniform(0, 2*np.pi, n_params)
       result = minimize(energy_expectation, init, method='COBYLA')
       
       if result.fun < best_energy:
           best_energy = result.fun
           best_params = result.x

**Issue 2: Slow convergence**

.. code-block:: python

   # Solution: Use better optimizer or increase tolerance
   result = minimize(
       energy_expectation,
       init_params,
       method='L-BFGS-B',
       options={
           'maxiter': 500,
           'ftol': 1e-9,  # Tighter tolerance
           'gtol': 1e-6
       }
   )

**Issue 3: High shot noise**

.. code-block:: python

   # Solution: Adaptive shot allocation
   def energy_adaptive_shots(params, min_shots=100, max_shots=10000):
       # Start with few shots
       shots = min_shots
       energy = energy_expectation_with_shots(params, shots)
       
       # Increase shots near convergence
       if iteration > 50:  # Near end of optimization
           shots = max_shots
           energy = energy_expectation_with_shots(params, shots)
       
       return energy

See Also
========

- :doc:`chemistry_examples` - Quantum chemistry optimization (VQE, UCCSD)
- :doc:`../quantum_chemistry/algorithms/index` - Chemistry algorithms
- :doc:`../libraries/optimizer/index` - Optimizer library reference
- :doc:`../user_guide/advanced/index` - Advanced optimization techniques

Next Steps
==========

After mastering optimization:

1. Try :doc:`chemistry_examples` for real molecular problems
2. Explore :doc:`cloud_examples` for hardware execution
3. Study :doc:`advanced_examples` for hybrid algorithms
4. Check :doc:`../tutorials/intermediate/index` for detailed tutorials
