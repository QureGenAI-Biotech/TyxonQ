======================
Intermediate Tutorials
======================

Ready to go deeper? This tutorial covers variational algorithms, quantum chemistry, and optimization techniques.

.. contents:: Tutorial Contents
   :depth: 3
   :local:

.. note::
   This tutorial assumes you've completed the :doc:`../beginner/index` tutorial.

Prerequisites
=============

Before starting, make sure you understand:

✅ Basic quantum gates (X, H, CNOT)  
✅ Superposition and entanglement  
✅ Circuit measurement and results  
✅ Working with multiple qubits  

If you need a refresher, review the :doc:`../beginner/index` tutorial first.

Variational Quantum Algorithms
===============================

What are Variational Algorithms?
---------------------------------

**Key idea**: Use quantum circuits with adjustable parameters to solve optimization problems.

**The process**:
1. 🚀 Create a parameterized quantum circuit (ansatz)
2. 🎯 Measure the circuit to get a cost function value
3. 📋 Use classical optimization to adjust parameters
4. 🔁 Repeat until convergence

.. code-block:: python

   import tyxonq as tq
   import numpy as np
   from scipy.optimize import minimize

   def create_variational_circuit(params):
       """Create circuit with adjustable parameters"""
       circuit = tq.Circuit(2)
       
       # Parameterized gates
       circuit.ry(0, params[0])  # params[0] controls rotation
       circuit.ry(1, params[1])  # params[1] controls rotation
       circuit.cnot(0, 1)        # Entanglement
       circuit.ry(0, params[2])  # More parameters
       circuit.ry(1, params[3])
       
       return circuit

   # Test with random parameters
   test_params = np.random.uniform(0, 2*np.pi, 4)
   test_circuit = create_variational_circuit(test_params)
   print(f"Created circuit with {len(test_params)} parameters")

Your First VQE Algorithm
-------------------------

**VQE (Variational Quantum Eigensolver)** finds the ground state energy of a quantum system.

**Step 1: Define the problem**

.. code-block:: python

   # Simple Hamiltonian: H = 0.5*Z0 + 0.3*Z1 - 0.7*X0*X1
   hamiltonian = [
       (0.5, [('Z', 0)]),
       (0.3, [('Z', 1)]),
       (-0.7, [('X', 0), ('X', 1)])
   ]

**Step 2: Implement energy calculation**

.. code-block:: python

   def calculate_energy(params):
       """Calculate energy expectation value"""
       circuit = create_variational_circuit(params)
       total_energy = 0.0
       
       for coeff, pauli_ops in hamiltonian:
           meas_circuit = circuit.copy()
           
           # Apply basis rotations
           for pauli, qubit in pauli_ops:
               if pauli == 'X':
                   meas_circuit.h(qubit)
               elif pauli == 'Y':
                   meas_circuit.sdg(qubit)
                   meas_circuit.h(qubit)
           
           meas_circuit.measure_z(0)
           meas_circuit.measure_z(1)
           
           result = meas_circuit.run(shots=1000)
           expectation = compute_pauli_expectation(result, pauli_ops)
           total_energy += coeff * expectation
       
       return total_energy

   def compute_pauli_expectation(counts, pauli_ops):
       """Compute expectation from measurement counts"""
       total_shots = sum(counts.values())
       expectation = 0.0
       
       for bitstring, count in counts.items():
           parity = 1
           for pauli, qubit in pauli_ops:
               if bitstring[qubit] == '1':
                   parity *= -1
           expectation += parity * count
       
       return expectation / total_shots

**Step 3: Run optimization**

.. code-block:: python

   # Optimize using scipy
   initial_params = np.random.uniform(0, 2*np.pi, 4)
   
   result = minimize(
       calculate_energy,
       initial_params,
       method='COBYLA',
       options={'maxiter': 100}
   )
   
   print(f"Final energy: {result.fun:.6f}")
   print(f"Optimal parameters: {result.x}")
   print(f"Function evaluations: {result.nfev}")

Building Better Ansatz Circuits
--------------------------------

**Hardware Efficient Ansatz (HEA)**:

.. code-block:: python

   def create_hea_circuit(params, n_qubits, layers):
       """Hardware-efficient ansatz with multiple layers"""
       circuit = tq.Circuit(n_qubits)
       param_idx = 0
       
       # Initial rotation layer
       for i in range(n_qubits):
           circuit.ry(i, params[param_idx])
           param_idx += 1
       
       # Repeated layers
       for layer in range(layers):
           # Entanglement layer
           for i in range(n_qubits - 1):
               circuit.cnot(i, i + 1)
           
           # Rotation layer
           for i in range(n_qubits):
               circuit.ry(i, params[param_idx])
               param_idx += 1
       
       return circuit
   
   # Example: 3 qubits, 2 layers
   n_qubits = 3
   layers = 2
   n_params = n_qubits * (layers + 1)
   
   hea_params = np.random.uniform(0, 2*np.pi, n_params)
   hea_circuit = create_hea_circuit(hea_params, n_qubits, layers)
   
   print(f"HEA circuit: {n_qubits} qubits, {layers} layers, {n_params} parameters")

Quantum Chemistry Basics
========================

Why Quantum Chemistry?
----------------------

**The challenge**: Molecules are quantum systems, so classical computers struggle with:

❌ **Exponential scaling**: N electrons need 2^N computational resources  
❌ **Correlation effects**: Electrons interact in complex ways  
❌ **Ground state problems**: Finding lowest energy configurations  

**The quantum advantage**:

✅ **Natural representation**: Qubits naturally represent electron states  
✅ **Efficient simulation**: Quantum computers handle quantum systems well  
✅ **Polynomial scaling**: Much better than classical methods  

Your First Molecule: H₂
-----------------------

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   from pyscf import gto
   
   # Step 1: Define the molecule
   mol = gto.Mole()
   mol.atom = '''
   H 0.0 0.0 0.0
   H 0.0 0.0 0.74
   '''
   mol.basis = 'sto-3g'
   mol.build()
   
   print(f"Total electrons: {mol.nelectron}")
   
   # Step 2: Create quantum chemistry solver
   uccsd = UCCSD(molecule=mol, init_method="mp2")
   
   print(f"Qubits needed: {uccsd.n_qubits}")
   print(f"Parameters: {uccsd.n_params}")
   
   # Step 3: Run the calculation
   energy = uccsd.kernel()
   
   # Step 4: Analyze results
   print(f"Ground state energy: {energy:.8f} Hartree")
   print(f"Hartree-Fock energy: {uccsd.e_hf:.8f} Hartree")
   print(f"FCI energy (exact):  {uccsd.e_fci:.8f} Hartree")
   
   fci_error = abs(energy - uccsd.e_fci) * 1000
   print(f"Error from exact: {fci_error:.3f} mHartree")
   
   if fci_error < 1.0:
       print("✅ Chemical accuracy achieved!")

Comparing Different Algorithms
------------------------------

.. code-block:: python

   from tyxonq.applications.chem import HEA, PUCCD
   import time
   
   # Lithium hydride molecule
   mol = gto.Mole()
   mol.atom = 'Li 0 0 0; H 0 0 1.6'
   mol.basis = 'sto-3g'
   mol.build()
   
   algorithms = {
       'UCCSD': lambda: UCCSD(molecule=mol, init_method="mp2"),
       'HEA': lambda: HEA(molecule=mol, layers=3),
       'PUCCD': lambda: PUCCD(molecule=mol, init_method="mp2")
   }
   
   print("Algorithm Comparison (LiH):")
   print("Method | Energy     | Time  | Params")
   print("-" * 35)
   
   for name, create_alg in algorithms.items():
       start_time = time.time()
       alg = create_alg()
       energy = alg.kernel()
       end_time = time.time()
       
       print(f"{name:6s} | {energy:9.6f} | {end_time-start_time:4.1f}s | {alg.n_params:6d}")

Optimization Techniques
=======================

Choosing the Right Optimizer
----------------------------

.. code-block:: python

   # Test different optimizers on a simple problem
   def test_function(x):
       return (x[0] - 1)**2 + (x[1] - 2)**2
   
   optimizers = ['COBYLA', 'L-BFGS-B', 'Powell']
   start_point = np.array([0.0, 0.0])
   
   print("Optimizer Comparison:")
   print("Method    | Final Value | Iterations")
   print("-" * 30)
   
   for method in optimizers:
       result = minimize(test_function, start_point, method=method)
       print(f"{method:9s} | {result.fun:10.6f} | {result.nfev:9d}")
   
   print("\nRecommendations:")
   print("🎯 COBYLA: Best for noisy quantum measurements")
   print("🎯 L-BFGS-B: Best when gradients are available")
   print("🎯 Powell: Good balance for smooth functions")

Parameter-Shift Gradients
-------------------------

.. code-block:: python

   def parameter_shift_gradient(circuit_func, params, param_idx):
       """Compute gradient using parameter-shift rule"""
       shift = np.pi / 2
       
       # Forward evaluation
       params_plus = params.copy()
       params_plus[param_idx] += shift
       energy_plus = circuit_func(params_plus)
       
       # Backward evaluation
       params_minus = params.copy()
       params_minus[param_idx] -= shift
       energy_minus = circuit_func(params_minus)
       
       # Gradient
       return (energy_plus - energy_minus) / 2
   
   def compute_all_gradients(circuit_func, params):
       """Compute gradients for all parameters"""
       gradients = np.zeros_like(params)
       for i in range(len(params)):
           gradients[i] = parameter_shift_gradient(circuit_func, params, i)
       return gradients
   
   # Example: use gradients in optimization
   test_params = np.array([0.1, 0.2, 0.3, 0.4])
   gradients = compute_all_gradients(calculate_energy, test_params)
   
   print(f"Parameters: {test_params}")
   print(f"Gradients:  {gradients}")

Practical Projects
==================

Project 1: Custom VQE Implementation
------------------------------------

.. code-block:: python

   class MyVQE:
       def __init__(self, hamiltonian, ansatz_func, n_params):
           self.hamiltonian = hamiltonian
           self.ansatz_func = ansatz_func
           self.n_params = n_params
           self.history = []
       
       def energy(self, params, shots=1000):
           """Calculate energy expectation value"""
           circuit = self.ansatz_func(params)
           
           total_energy = 0.0
           for coeff, pauli_ops in self.hamiltonian:
               exp_val = self._measure_expectation(circuit, pauli_ops, shots)
               total_energy += coeff * exp_val
           
           self.history.append((params.copy(), total_energy))
           return total_energy
       
       def optimize(self, initial_params, method='COBYLA'):
           """Run optimization"""
           result = minimize(
               self.energy,
               initial_params,
               method=method,
               options={'maxiter': 200}
           )
           return result
   
   # Define your ansatz
   def my_ansatz(params):
       circuit = tq.Circuit(2)
       circuit.ry(0, params[0])
       circuit.ry(1, params[1])
       circuit.cnot(0, 1)
       circuit.ry(0, params[2])
       circuit.ry(1, params[3])
       return circuit
   
   # Use your custom VQE
   my_vqe = MyVQE(hamiltonian, my_ansatz, 4)
   initial = np.random.uniform(0, 2*np.pi, 4)
   
   result = my_vqe.optimize(initial)
   print(f"My VQE result: {result.fun:.6f}")
   print(f"Optimization steps: {len(my_vqe.history)}")

Next Steps
==========

Congratulations! You've learned:

✅ **Variational algorithms**: VQE and optimization  
✅ **Quantum chemistry**: Molecular calculations  
✅ **Optimization techniques**: Gradients and methods  
✅ **Practical implementation**: Custom VQE solver  

What's Next?
------------

1. **Advanced Tutorial**: Hybrid algorithms and quantum ML
2. **Real Hardware**: Deploy on quantum computers
3. **Research Projects**: Explore cutting-edge applications

See Also
========

- :doc:`../advanced/index` - Advanced quantum algorithms
- :doc:`../../examples/optimization_examples` - More VQE examples
- :doc:`../../examples/chemistry_examples` - Chemistry applications
- :doc:`../../user_guide/advanced/index` - Advanced features guide
