=================
Advanced Examples
=================

This page showcases advanced quantum computing techniques, hybrid algorithms, and complex applications using TyxonQ.

.. contents:: Contents
   :depth: 3
   :local:

.. note::
   These examples assume familiarity with basic quantum computing concepts and TyxonQ fundamentals.

Hybrid Quantum-Classical Algorithms
===================================

Quantum-Classical Neural Networks
----------------------------------

Integrating quantum circuits into machine learning:

.. code-block:: python

   import tyxonq as tq
   import numpy as np
   import torch
   import torch.nn as nn

   class QuantumLayer(nn.Module):
       """Quantum layer for hybrid neural network"""
       
       def __init__(self, n_qubits, n_layers):
           super().__init__()
           self.n_qubits = n_qubits
           self.n_layers = n_layers
           self.n_params = n_qubits * n_layers * 3
           
           # Trainable quantum parameters
           self.params = nn.Parameter(
               torch.randn(self.n_params) * 0.1
           )
       
       def create_circuit(self, x, params):
           """Create parameterized quantum circuit"""
           circuit = tq.Circuit(self.n_qubits)
           
           # Encode input data
           for i in range(self.n_qubits):
               circuit.ry(i, x[i])
           
           # Variational layers
           idx = 0
           for layer in range(self.n_layers):
               # Rotation gates
               for i in range(self.n_qubits):
                   circuit.rx(i, params[idx])
                   circuit.ry(i, params[idx + 1])
                   circuit.rz(i, params[idx + 2])
                   idx += 3
               
               # Entanglement
               for i in range(self.n_qubits - 1):
                   circuit.cnot(i, i + 1)
           
           return circuit
       
       def forward(self, x):
           """Forward pass through quantum layer"""
           batch_size = x.shape[0]
           outputs = []
           
           for i in range(batch_size):
               circuit = self.create_circuit(
                   x[i].detach().numpy(),
                   self.params.detach().numpy()
               )
               
               # Measure expectations
               circuit.measure_z(0)
               result = circuit.run(shots=1000)
               
               # Compute expectation <Z>
               exp_z = sum(
                   (-1 if bs[0] == '1' else 1) * count 
                   for bs, count in result.items()
               ) / 1000
               
               outputs.append(exp_z)
           
           return torch.tensor(outputs, dtype=torch.float32)

   # Build hybrid model
   class HybridQNN(nn.Module):
       def __init__(self):
           super().__init__()
           self.classical1 = nn.Linear(4, 4)
           self.quantum = QuantumLayer(n_qubits=4, n_layers=2)
           self.classical2 = nn.Linear(1, 2)
       
       def forward(self, x):
           x = torch.relu(self.classical1(x))
           x = self.quantum(x).unsqueeze(1)
           x = self.classical2(x)
           return x

   # Training
   model = HybridQNN()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
   criterion = nn.CrossEntropyLoss()

   # Example training loop
   for epoch in range(10):
       optimizer.zero_grad()
       # ... training code ...

**Applications**:

- Quantum-enhanced feature extraction
- Hybrid image classification
- Time series prediction with quantum layers

Variational Quantum Linear Solver (VQLS)
-----------------------------------------

Solving linear systems Ax = b variationally:

.. code-block:: python

   def vqls_cost_function(params, A, b):
       """Cost function for VQLS: C = ||Ax - b||^2"""
       
       # Create ansatz for x
       circuit_x = create_ansatz(params)
       
       # Compute <ψ|A†A|ψ>
       circuit_aa = circuit_x.copy()
       apply_operator(circuit_aa, A.T @ A)
       expectation_aa = measure_expectation(circuit_aa)
       
       # Compute <ψ|A†b>
       circuit_ab = circuit_x.copy()
       apply_operator(circuit_ab, A.T)
       expectation_ab = measure_overlap(circuit_ab, b)
       
       # Cost: ||Ax - b||^2 = <Ax|Ax> - 2Re<Ax|b> + ||b||^2
       cost = expectation_aa - 2 * expectation_ab.real + np.linalg.norm(b)**2
       return cost

   def solve_linear_system(A, b, n_layers=3):
       """Solve Ax = b using VQLS"""
       n_qubits = int(np.log2(len(b)))
       n_params = n_qubits * n_layers * 3
       
       init_params = np.random.uniform(0, 2*np.pi, n_params)
       
       result = minimize(
           lambda p: vqls_cost_function(p, A, b),
           init_params,
           method='COBYLA'
       )
       
       return result

Quantum Error Correction
========================

Surface Code Example
--------------------

Implementing basic surface code:

.. code-block:: python

   def create_surface_code_circuit(distance=3):
       """Create surface code circuit for error detection
       
       Args:
           distance: Code distance (odd number)
       """
       # Calculate qubit requirements
       n_data_qubits = distance ** 2
       n_ancilla_x = (distance - 1) * distance // 2
       n_ancilla_z = (distance - 1) * distance // 2
       total_qubits = n_data_qubits + n_ancilla_x + n_ancilla_z
       
       circuit = tq.Circuit(total_qubits)
       
       # Initialize data qubits
       for i in range(n_data_qubits):
           circuit.h(i)  # Prepare |+⟩ state
       
       # X-type stabilizers
       for i in range(n_ancilla_x):
           ancilla_idx = n_data_qubits + i
           circuit.h(ancilla_idx)
           
           # Apply CNOT to neighboring data qubits
           for neighbor in get_x_neighbors(i, distance):
               circuit.cnot(ancilla_idx, neighbor)
           
           circuit.h(ancilla_idx)
           circuit.measure_z(ancilla_idx)
       
       # Z-type stabilizers
       for i in range(n_ancilla_z):
           ancilla_idx = n_data_qubits + n_ancilla_x + i
           
           # Apply CZ to neighboring data qubits
           for neighbor in get_z_neighbors(i, distance):
               circuit.cz(ancilla_idx, neighbor)
           
           circuit.measure_z(ancilla_idx)
       
       return circuit

   def detect_errors(syndrome):
       """Decode syndrome measurements to detect errors"""
       # Implement minimum-weight perfect matching
       # or other decoding algorithm
       pass

**Applications**:

- Fault-tolerant quantum computing
- Long-duration quantum computations
- Protecting quantum states

Repetition Code for Bit Flip Errors
------------------------------------

.. code-block:: python

   def three_qubit_bit_flip_code():
       """Implement 3-qubit bit flip code"""
       circuit = tq.Circuit(3)
       
       # Encode: |ψ⟩ → |ψψψ⟩
       circuit.cnot(0, 1)
       circuit.cnot(0, 2)
       
       # Simulate error on qubit 1
       circuit.x(1)  # Bit flip error
       
       # Error detection
       circuit.cnot(0, 1)
       circuit.cnot(0, 2)
       circuit.ccx(1, 2, 0)  # Toffoli for correction
       
       return circuit

Advanced Ansatz Design
======================

Adaptive VQE with Gradient-Based Ansatz
----------------------------------------

.. code-block:: python

   class AdaptiveVQE:
       """Adaptive VQE with growing ansatz"""
       
       def __init__(self, hamiltonian, n_qubits):
           self.hamiltonian = hamiltonian
           self.n_qubits = n_qubits
           self.operators = []
           self.params = []
       
       def compute_gradients(self):
           """Compute gradients for all possible operators"""
           operator_pool = self.generate_operator_pool()
           gradients = {}
           
           for op in operator_pool:
               # Compute gradient for adding this operator
               grad = self.compute_operator_gradient(op)
               gradients[op] = abs(grad)
           
           return gradients
       
       def grow_ansatz(self, threshold=1e-3):
           """Add operator with largest gradient"""
           gradients = self.compute_gradients()
           
           # Find operator with largest gradient
           best_op = max(gradients, key=gradients.get)
           
           if gradients[best_op] > threshold:
               self.operators.append(best_op)
               self.params.append(0.0)  # Initialize parameter
               return True
           return False
       
       def optimize(self, max_cycles=10):
           """Adaptive optimization loop"""
           for cycle in range(max_cycles):
               print(f"Cycle {cycle}:")
               
               # Grow ansatz
               if not self.grow_ansatz():
                   print("Converged - no operator improves energy")
                   break
               
               # Optimize current ansatz
               result = minimize(
                   self.energy_function,
                   self.params,
                   method='L-BFGS-B'
               )
               self.params = result.x
               
               print(f"  Added operator, energy: {result.fun:.6f}")
           
           return result.fun

Unitary Coupled Cluster with Generalized Excitations
-----------------------------------------------------

.. code-block:: python

   def generalized_ucc_ansatz(params, n_qubits, max_rank=3):
       """UCC ansatz with up to 3-body excitations"""
       circuit = tq.Circuit(n_qubits)
       idx = 0
       
       # Singles
       for i in range(0, n_qubits, 2):
           for a in range(1, n_qubits, 2):
               if i < a:
                   apply_single_excitation(circuit, i, a, params[idx])
                   idx += 1
       
       # Doubles
       for i in range(0, n_qubits-1, 2):
           for j in range(i+2, n_qubits, 2):
               for a in range(1, n_qubits-1, 2):
                   for b in range(a+2, n_qubits, 2):
                       apply_double_excitation(
                           circuit, i, j, a, b, params[idx]
                       )
                       idx += 1
       
       # Triples (if max_rank >= 3)
       if max_rank >= 3:
           # ... triple excitation logic ...
           pass
       
       return circuit

Advanced Measurement Strategies
===============================

Deterministic State Preparation and Measurement (DSPM)
------------------------------------------------------

.. code-block:: python

   def dspm_protocol(target_state, n_measurements=100):
       """Deterministically prepare and measure quantum states"""
       n_qubits = int(np.log2(len(target_state)))
       
       # Prepare target state
       prep_circuit = state_preparation_circuit(target_state)
       
       # Measurement strategy
       measurement_bases = generate_measurement_bases(n_qubits)
       
       results = {}
       for basis in measurement_bases:
           circuit = prep_circuit.copy()
           
           # Apply basis rotations
           for qubit, rotation in enumerate(basis):
               if rotation == 'X':
                   circuit.h(qubit)
               elif rotation == 'Y':
                   circuit.sdg(qubit)
                   circuit.h(qubit)
           
           # Measure
           circuit.measure_all()
           result = circuit.run(shots=n_measurements)
           results[tuple(basis)] = result
       
       # Tomographic reconstruction
       reconstructed_state = reconstruct_state(results)
       return reconstructed_state

Adaptive Measurement Optimization
---------------------------------

.. code-block:: python

   def adaptive_measurement(hamiltonian, circuit, shot_budget=10000):
       """Allocate shots adaptively across Hamiltonian terms"""
       
       # Initial equal allocation
       n_terms = len(hamiltonian)
       shots_per_term = shot_budget // n_terms
       
       # Measure each term
       variances = []
       for term_idx, (coeff, pauli_ops) in enumerate(hamiltonian):
           result = measure_pauli_term(
               circuit, pauli_ops, shots=shots_per_term
           )
           variance = compute_variance(result)
           variances.append(variance)
       
       # Reallocate shots based on variance
       total_variance = sum(variances)
       optimized_shots = [
           int(shot_budget * var / total_variance) 
           for var in variances
       ]
       
       # Re-measure with optimized allocation
       energy = 0
       for (coeff, pauli_ops), shots in zip(hamiltonian, optimized_shots):
           result = measure_pauli_term(circuit, pauli_ops, shots=shots)
           energy += coeff * compute_expectation(result)
       
       return energy

Quantum Simulation
==================

Trotterized Time Evolution
--------------------------

.. code-block:: python

   def trotter_evolution(hamiltonian, time, n_steps, n_qubits):
       """Simulate time evolution using Trotterization"""
       dt = time / n_steps
       circuit = tq.Circuit(n_qubits)
       
       # Initial state preparation
       for i in range(n_qubits):
           circuit.h(i)
       
       # Trotter steps
       for step in range(n_steps):
           # Apply exp(-iHt) ≈ ∏ exp(-iH_k dt)
           for coeff, pauli_ops in hamiltonian:
               apply_pauli_evolution(
                   circuit, pauli_ops, coeff * dt
               )
       
       return circuit

   def apply_pauli_evolution(circuit, pauli_ops, theta):
       """Apply exp(-i θ P) where P is Pauli string"""
       # Diagonalize Pauli string
       for pauli, qubit in pauli_ops:
           if pauli == 'X':
               circuit.h(qubit)
           elif pauli == 'Y':
               circuit.sdg(qubit)
               circuit.h(qubit)
       
       # Multi-controlled rotation
       qubits = [q for _, q in pauli_ops]
       apply_multi_controlled_rz(circuit, qubits, 2 * theta)
       
       # Undiagonalize
       for pauli, qubit in reversed(pauli_ops):
           if pauli == 'X':
               circuit.h(qubit)
           elif pauli == 'Y':
               circuit.h(qubit)
               circuit.s(qubit)

Quantum Phase Estimation
------------------------

.. code-block:: python

   def quantum_phase_estimation(unitary_circuit, n_precision_qubits):
       """Estimate eigenvalues using QPE"""
       n_target = unitary_circuit.n_qubits
       n_total = n_precision_qubits + n_target
       
       circuit = tq.Circuit(n_total)
       
       # Prepare eigenstate on target qubits
       # (assume already prepared)
       
       # Create superposition on precision qubits
       for i in range(n_precision_qubits):
           circuit.h(i)
       
       # Controlled unitary operations
       for i in range(n_precision_qubits):
           power = 2 ** (n_precision_qubits - 1 - i)
           
           # Apply U^(2^i) controlled by qubit i
           for _ in range(power):
               apply_controlled_unitary(
                   circuit,
                   control=i,
                   target_qubits=range(n_precision_qubits, n_total),
                   unitary=unitary_circuit
               )
       
       # Inverse QFT on precision qubits
       apply_inverse_qft(circuit, range(n_precision_qubits))
       
       # Measure precision qubits
       for i in range(n_precision_qubits):
           circuit.measure_z(i)
       
       result = circuit.run(shots=5000)
       
       # Extract phase
       measured_int = max(result, key=result.get)
       phase = int(measured_int[:n_precision_qubits], 2) / (2**n_precision_qubits)
       
       return phase * 2 * np.pi

Quantum Machine Learning
========================

Quantum Kernel Methods
----------------------

.. code-block:: python

   def quantum_kernel(x1, x2, n_qubits):
       """Compute quantum kernel K(x1, x2)"""
       circuit = tq.Circuit(n_qubits)
       
       # Encode x1
       for i in range(n_qubits):
           circuit.ry(i, x1[i])
           circuit.rz(i, x1[i]**2)
       
       # Entangle
       for i in range(n_qubits - 1):
           circuit.cnot(i, i + 1)
       
       # Encode x2 (inverse)
       for i in range(n_qubits):
           circuit.rz(i, -x2[i]**2)
           circuit.ry(i, -x2[i])
       
       # Reverse entanglement
       for i in range(n_qubits - 2, -1, -1):
           circuit.cnot(i, i + 1)
       
       # Measure overlap
       circuit.measure_all()
       result = circuit.run(shots=2000)
       
       # Probability of |000...0⟩
       zero_state = '0' * n_qubits
       kernel_value = result.get(zero_state, 0) / 2000
       
       return kernel_value

   def quantum_svm(X_train, y_train, X_test):
       """Quantum support vector machine"""
       from sklearn.svm import SVC
       
       # Build kernel matrix
       n_train = len(X_train)
       K_train = np.zeros((n_train, n_train))
       
       for i in range(n_train):
           for j in range(i, n_train):
               K_train[i, j] = quantum_kernel(X_train[i], X_train[j], n_qubits=4)
               K_train[j, i] = K_train[i, j]
       
       # Train SVM with precomputed kernel
       svm = SVC(kernel='precomputed')
       svm.fit(K_train, y_train)
       
       # Predict
       n_test = len(X_test)
       K_test = np.zeros((n_test, n_train))
       for i in range(n_test):
           for j in range(n_train):
               K_test[i, j] = quantum_kernel(X_test[i], X_train[j], n_qubits=4)
       
       predictions = svm.predict(K_test)
       return predictions

Quantum Generative Adversarial Networks
----------------------------------------

.. code-block:: python

   class QuantumGenerator:
       def __init__(self, n_qubits, n_layers):
           self.n_qubits = n_qubits
           self.n_layers = n_layers
           self.n_params = n_qubits * n_layers * 3
           self.params = np.random.uniform(0, 2*np.pi, self.n_params)
       
       def generate(self, noise):
           """Generate quantum state from noise"""
           circuit = tq.Circuit(self.n_qubits)
           
           # Encode noise
           for i in range(self.n_qubits):
               circuit.ry(i, noise[i])
           
           # Variational layers
           idx = 0
           for layer in range(self.n_layers):
               for i in range(self.n_qubits):
                   circuit.rx(i, self.params[idx])
                   circuit.ry(i, self.params[idx + 1])
                   circuit.rz(i, self.params[idx + 2])
                   idx += 3
               
               for i in range(self.n_qubits - 1):
                   circuit.cnot(i, i + 1)
           
           # Measure
           circuit.measure_all()
           result = circuit.run(shots=100)
           return result

Noise Simulation for NISQ Algorithms
=====================================

TyxonQ provides production-ready noise simulation for developing realistic quantum algorithms. This section demonstrates how to use noise models to simulate real quantum hardware behavior.

Basic Noise Simulation
----------------------

Adding depolarizing noise to a Bell state:

.. code-block:: python

   import tyxonq as tq
   import numpy as np

   # Create Bell state circuit
   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cnot(0, 1)

   # Compare ideal vs noisy execution
   ideal_result = circuit.run(shots=1024)
   noisy_result = circuit.with_noise("depolarizing", p=0.05).run(shots=1024)

   print("Ideal:", ideal_result.counts)
   print("Noisy:", noisy_result.counts)
   # Noisy result will show some '01' and '10' errors

Comparing Different Noise Models
---------------------------------

Test how different noise types affect a GHZ state:

.. code-block:: python

   def compare_noise_models():
       """Compare all noise models on a 3-qubit GHZ state."""
       
       # Create GHZ state
       circuit = tq.Circuit(3)
       circuit.h(0)
       circuit.cnot(0, 1)
       circuit.cnot(1, 2)
       
       # Test different noise models
       noise_configs = [
           ("Ideal", None, {}),
           ("Depolarizing", "depolarizing", {"p": 0.05}),
           ("Amplitude Damping", "amplitude_damping", {"gamma": 0.1}),
           ("Phase Damping", "phase_damping", {"l": 0.1}),
           ("Pauli (asymmetric)", "pauli", {"px": 0.02, "py": 0.02, "pz": 0.06})
       ]
       
       results = {}
       for name, noise_type, params in noise_configs:
           if noise_type is None:
               result = circuit.run(shots=2048)
           else:
               result = circuit.with_noise(noise_type, **params).run(shots=2048)
           
           # Calculate GHZ fidelity
           total = sum(result.counts.values())
           ghz_fidelity = (
               result.counts.get('000', 0) + result.counts.get('111', 0)
           ) / total
           
           results[name] = {
               'counts': result.counts,
               'fidelity': ghz_fidelity
           }
           print(f"{name:30s} GHZ fidelity: {ghz_fidelity:.4f}")
       
       return results

   # Run comparison
   results = compare_noise_models()

VQE with Realistic Noise
------------------------

Variational Quantum Eigensolver with noise simulation:

.. code-block:: python

   from scipy.optimize import minimize
   from tyxonq.libs.hamiltonian_encoding import PauliSum

   # Define Hamiltonian
   hamiltonian = PauliSum()
   hamiltonian.add_term('ZZ', [0, 1], -0.8)
   hamiltonian.add_term('Z', [0], 0.2)
   hamiltonian.add_term('Z', [1], 0.2)

   def vqe_ansatz(params):
       """Parameterized quantum circuit for VQE."""
       circuit = tq.Circuit(2)
       circuit.ry(0, params[0])
       circuit.ry(1, params[1])
       circuit.cnot(0, 1)
       circuit.ry(0, params[2])
       circuit.ry(1, params[3])
       return circuit

   def energy_evaluation(params, noise_level=0.0):
       """Evaluate energy with optional noise."""
       circuit = vqe_ansatz(params)
       
       if noise_level > 0:
           result = (
               circuit.with_noise("depolarizing", p=noise_level)
                      .run(shots=4096)
           )
       else:
           result = circuit.run(shots=0)  # Exact simulation
       
       # Compute Hamiltonian expectation value
       energy = hamiltonian.expectation(result)
       return energy

   # Compare optimization with and without noise
   init_params = np.random.rand(4) * 2 * np.pi

   # Ideal case
   result_ideal = minimize(
       lambda p: energy_evaluation(p, noise_level=0.0),
       init_params,
       method='COBYLA'
   )

   # Noisy case (5% depolarizing noise)
   result_noisy = minimize(
       lambda p: energy_evaluation(p, noise_level=0.05),
       init_params,
       method='COBYLA'
   )

   print(f"Ideal VQE energy: {result_ideal.fun:.6f} Hartree")
   print(f"Noisy VQE energy: {result_noisy.fun:.6f} Hartree")
   print(f"Energy error due to noise: {abs(result_noisy.fun - result_ideal.fun):.6f}")

Realistic Hardware Simulation
-----------------------------

Model a superconducting qubit device with T₁ and T₂ relaxation:

.. code-block:: python

   def simulate_hardware_noise(circuit, hardware_params):
       """
       Simulate circuit with realistic hardware noise.
       
       Args:
           circuit: Quantum circuit to simulate
           hardware_params: Dict with 'T1', 'T2', 'gate_time' in seconds
       
       Returns:
           Noisy simulation result
       """
       T1 = hardware_params['T1']  # e.g., 100e-6 (100 μs)
       T2 = hardware_params['T2']  # e.g., 80e-6 (80 μs)
       gate_time = hardware_params['gate_time']  # e.g., 50e-9 (50 ns)
       
       # Calculate noise parameters
       gamma = 1 - np.exp(-gate_time / T1)  # Amplitude damping
       lambda_val = 1 - np.exp(-gate_time / T2)  # Phase damping
       
       print(f"Hardware parameters:")
       print(f"  T₁ = {T1*1e6:.1f} μs")
       print(f"  T₂ = {T2*1e6:.1f} μs")
       print(f"  Gate time = {gate_time*1e9:.1f} ns")
       print(f"  γ (amplitude damping) = {gamma:.6f}")
       print(f"  λ (phase damping) = {lambda_val:.6f}")
       
       # Apply combined noise (simplified: use depolarizing as approximation)
       # For more accurate simulation, apply both T1 and T2 sequentially
       p_total = gamma + lambda_val
       result = circuit.with_noise("depolarizing", p=p_total).run(shots=2048)
       
       return result

   # Example: Simulate IBM-like hardware
   circuit = tq.Circuit(4)
   circuit.h(0)
   for i in range(3):
       circuit.cnot(i, i+1)

   ibm_params = {
       'T1': 100e-6,      # 100 microseconds
       'T2': 80e-6,       # 80 microseconds
       'gate_time': 50e-9 # 50 nanoseconds
   }

   result = simulate_hardware_noise(circuit, ibm_params)
   print(f"\nResult: {result.counts}")

Noise-Aware Circuit Optimization
---------------------------------

Optimize circuit depth to minimize noise impact:

.. code-block:: python

   def noise_aware_compilation(circuit, noise_level):
       """
       Compile circuit with noise awareness.
       
       Trade-off: Deeper optimized circuits may have fewer gates
       but same noise accumulation if gate count reduction is minimal.
       """
       from tyxonq.compiler import optimize_circuit
       
       # Get different optimization levels
       opt_levels = [0, 1, 2, 3]
       results = {}
       
       for level in opt_levels:
           optimized = optimize_circuit(
               circuit,
               optimization_level=level
           )
           
           gate_count = len(optimized.ops)
           depth = calculate_circuit_depth(optimized)
           
           # Simulate with noise
           noisy_result = (
               optimized.with_noise("depolarizing", p=noise_level)
                        .run(shots=2048)
           )
           
           # Calculate success probability (circuit-dependent metric)
           success_prob = evaluate_success_metric(noisy_result)
           
           results[level] = {
               'gates': gate_count,
               'depth': depth,
               'success_prob': success_prob
           }
           
           print(f"Level {level}: {gate_count} gates, "
                 f"depth {depth}, success {success_prob:.3f}")
       
       # Choose level with best success probability
       best_level = max(results.items(), key=lambda x: x[1]['success_prob'])[0]
       print(f"\nBest optimization level: {best_level}")
       
       return results

**See Also**:

- :doc:`../user_guide/devices/noise_simulation` - Complete noise simulation guide
- :doc:`../user_guide/postprocessing/index` - Error mitigation techniques

Performance Optimization
========================

Circuit Cutting for Large Systems
----------------------------------

.. code-block:: python

   def circuit_cutting(large_circuit, max_qubits=10):
       """Split large circuit into smaller subcircuits"""
       if large_circuit.n_qubits <= max_qubits:
           return [large_circuit]
       
       # Find optimal cut points
       cut_points = find_cut_points(large_circuit, max_qubits)
       
       # Create subcircuits
       subcircuits = []
       for start, end in cut_points:
           subcircuit = extract_subcircuit(
               large_circuit,
               qubit_range=range(start, end)
           )
           subcircuits.append(subcircuit)
       
       # Execute subcircuits
       sub_results = [sc.run(shots=2000) for sc in subcircuits]
       
       # Reconstruct full result
       full_result = reconstruct_from_subcircuits(
           sub_results, cut_points
       )
       
       return full_result

Dynamic Circuit Optimization
----------------------------

.. code-block:: python

   def dynamic_circuit_opt(circuit, backend='statevector'):
       """Optimize circuit during runtime"""
       from tyxonq.compiler import optimize_circuit
       
       # Compile with aggressive optimization
       optimized = optimize_circuit(
           circuit,
           optimization_level=3,
           target_backend=backend
       )
       
       # Dynamic gate merging
       optimized = merge_consecutive_rotations(optimized)
       
       # Remove redundant gates
       optimized = remove_identity_gates(optimized)
       
       # Optimize for specific topology
       optimized = optimize_for_topology(optimized, backend)
       
       return optimized

See Also
========

- :doc:`basic_examples` - Fundamental examples
- :doc:`chemistry_examples` - Quantum chemistry applications
- :doc:`optimization_examples` - VQE and QAOA
- :doc:`cloud_examples` - Cloud execution
- :doc:`../user_guide/advanced/index` - Advanced features guide

Further Reading
===============

- Quantum error correction: Surface codes, Shor code
- Variational algorithms: ADAPT-VQE, QNSPSA
- Quantum simulation: Hamiltonian simulation, quantum dynamics
- Quantum machine learning: QSVM, quantum neural networks
- Hybrid algorithms: Quantum-classical optimization
