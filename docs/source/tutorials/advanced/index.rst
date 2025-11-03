==================
Advanced Tutorials
==================

Ready for cutting-edge quantum algorithms? This tutorial covers hybrid quantum-classical methods, quantum machine learning,
and research-level applications.

.. contents:: Tutorial Contents
   :depth: 3
   :local:

.. toctree::
   :hidden:
   
   pulse_calibration_workflow
   pulse_advanced_waveforms
   pulse_defcal_integration
   pulse_hybrid_mode_integration
   pulse_inline_three_level
   pulse_three_level
   pulse_virtual_z_optimization
   pulse_zz_crosstalk
   pulse_iswap_swap_decomposition
   quantum_natural_gradient

.. note::
   This tutorial requires solid understanding of quantum computing fundamentals and variational algorithms.

Prerequisites
=============

Before starting, you should be comfortable with:

✅ **Variational algorithms**: VQE, QAOA, parameter optimization  
✅ **Quantum chemistry**: Molecular Hamiltonians, ansatz design  
✅ **homebrew_s2 programming**: NumPy, SciPy, object-oriented programming  
✅ **Basic machine learning**: Neural networks, optimization  

If you haven't completed the previous tutorials, please review:
- :doc:`../beginner/index` - Basic quantum computing
- :doc:`../intermediate/index` - VQE and quantum chemistry

Hybrid Quantum-Classical Algorithms
===================================

Quantum Neural Networks
-----------------------

**Concept**: Integrate quantum circuits as trainable layers in classical neural networks.

.. code-block:: homebrew_s2

   import tyxonq as tq
   import numpy as np
   import torch
   import torch.nn as nn

   class QuantumLayer(nn.Module):
       """Quantum layer for hybrid neural networks"""
       
       def __init__(self, n_qubits, n_layers, n_outputs=1):
           super().__init__()
           self.n_qubits = n_qubits
           self.n_layers = n_layers
           
           # Trainable quantum parameters
           n_params = n_qubits * n_layers * 3
           self.theta = nn.Parameter(torch.randn(n_params) * 0.1)
       
       def quantum_circuit(self, x, theta):
           """Create parameterized quantum circuit"""
           circuit = tq.Circuit(self.n_qubits)
           
           # Encode classical data
           for i in range(min(len(x), self.n_qubits)):
               circuit.ry(i, float(x[i]))
           
           # Variational layers
           param_idx = 0
           for layer in range(self.n_layers):
               for i in range(self.n_qubits):
                   circuit.rx(i, float(theta[param_idx]))
                   circuit.ry(i, float(theta[param_idx + 1]))
                   circuit.rz(i, float(theta[param_idx + 2]))
                   param_idx += 3
               
               # Entanglement
               for i in range(self.n_qubits - 1):
                   circuit.cnot(i, i + 1)
           
           return circuit
       
       def forward(self, x):
           """Forward pass through quantum layer"""
           batch_size = x.shape[0]
           outputs = []
           
           for i in range(batch_size):
               circuit = self.quantum_circuit(x[i], self.theta)
               
               # Measure expectation values
               circuit.measure_z(0)
               result = circuit.run(shots=1000)
               
               exp_z = sum(
                   (-1 if bs[0] == '1' else 1) * count 
                   for bs, count in result.items()
               ) / 1000
               outputs.append(exp_z)
           
           return torch.tensor(outputs, dtype=torch.float32)

   class HybridQNN(nn.Module):
       """Hybrid quantum-classical neural network"""
       
       def __init__(self, n_features, n_qubits=4, n_layers=2, n_classes=2):
           super().__init__()
           
           # Classical preprocessing
           self.classical_input = nn.Sequential(
               nn.Linear(n_features, n_qubits),
               nn.Tanh()
           )
           
           # Quantum layer
           self.quantum_layer = QuantumLayer(n_qubits, n_layers)
           
           # Classical postprocessing
           self.classical_output = nn.Sequential(
               nn.Linear(1, 10),
               nn.ReLU(),
               nn.Linear(10, n_classes)
           )
       
       def forward(self, x):
           x = self.classical_input(x)
           x = self.quantum_layer(x).unsqueeze(1)
           x = self.classical_output(x)
           return x

   # Example usage
   model = HybridQNN(n_features=4, n_qubits=4, n_layers=2)
   test_input = torch.randn(5, 4)
   output = model(test_input)
   print(f"Output shape: {output.shape}")

Quantum Kernels and SVM
-----------------------

.. code-block:: homebrew_s2

   def quantum_kernel(x1, x2, n_qubits=4):
       """Compute quantum kernel K(x1, x2)"""
       circuit = tq.Circuit(n_qubits)
       
       # Encode x1
       for i in range(min(len(x1), n_qubits)):
           circuit.ry(i, x1[i])
           circuit.rz(i, x1[i]**2)
       
       # Entangle
       for i in range(n_qubits - 1):
           circuit.cnot(i, i + 1)
       
       # Encode x2 (inverse)
       for i in range(min(len(x2), n_qubits)):
           circuit.rz(i, -x2[i]**2)
           circuit.ry(i, -x2[i])
       
       # Reverse entanglement
       for i in range(n_qubits - 2, -1, -1):
           circuit.cnot(i, i + 1)
       
       # Measure overlap
       circuit.measure_all()
       result = circuit.run(shots=2000)
       
       zero_state = '0' * n_qubits
       kernel_value = result.get(zero_state, 0) / 2000
       
       return kernel_value

   def quantum_svm_demo():
       """Demonstrate quantum SVM"""
       from sklearn.svm import SVC
       from sklearn.datasets import make_classification
       
       # Generate small dataset
       X, y = make_classification(n_samples=20, n_features=4, n_classes=2, random_state=42)
       
       # Build kernel matrix
       n_samples = len(X)
       K = np.zeros((n_samples, n_samples))
       
       print("Computing quantum kernel matrix...")
       for i in range(n_samples):
           for j in range(i, n_samples):
               K[i, j] = quantum_kernel(X[i], X[j])
               K[j, i] = K[i, j]
       
       # Train SVM
       svm = SVC(kernel='precomputed')
       svm.fit(K, y)
       
       # Predictions (using same data for demo)
       predictions = svm.predict(K)
       accuracy = np.mean(predictions == y)
       
       print(f"Quantum SVM accuracy: {accuracy:.3f}")
       return accuracy

Quantum Generative Models
=========================

Quantum GAN Generator
---------------------

.. code-block:: homebrew_s2

   class QuantumGenerator:
       """Quantum generator for GAN"""
       
       def __init__(self, n_qubits, n_layers):
           self.n_qubits = n_qubits
           self.n_layers = n_layers
           self.n_params = n_qubits * n_layers * 3
           
           # Initialize parameters
           self.params = np.random.uniform(0, 2*np.pi, self.n_params)
       
       def generate_circuit(self, noise_input):
           """Generate quantum circuit from noise"""
           circuit = tq.Circuit(self.n_qubits)
           
           # Encode noise input
           for i in range(min(len(noise_input), self.n_qubits)):
               circuit.ry(i, noise_input[i])
           
           # Variational layers
           param_idx = 0
           for layer in range(self.n_layers):
               for i in range(self.n_qubits):
                   circuit.rx(i, self.params[param_idx])
                   circuit.ry(i, self.params[param_idx + 1])
                   circuit.rz(i, self.params[param_idx + 2])
                   param_idx += 3
               
               # Entanglement
               for i in range(self.n_qubits - 1):
                   circuit.cnot(i, i + 1)
           
           return circuit
       
       def generate_sample(self, noise_input, shots=1000):
           """Generate sample from quantum circuit"""
           circuit = self.generate_circuit(noise_input)
           circuit.measure_all()
           
           result = circuit.run(shots=shots)
           return result
       
       def update_params(self, gradient, learning_rate=0.01):
           """Update generator parameters"""
           self.params -= learning_rate * gradient

   # Demo quantum generator
   generator = QuantumGenerator(n_qubits=3, n_layers=2)
   
   noise_samples = [
       np.random.uniform(0, 2*np.pi, 3) for _ in range(3)
   ]
   
   print("\nQuantum Generator Samples:")
   for i, noise in enumerate(noise_samples):
       sample = generator.generate_sample(noise, shots=500)
       top_outcomes = dict(list(sample.items())[:3])
       print(f"Sample {i+1}: {top_outcomes}")

Advanced VQE Techniques
=======================

Adaptive VQE
------------

.. code-block:: homebrew_s2

   class AdaptiveVQE:
       """Adaptive Variational Quantum Eigensolver"""
       
       def __init__(self, hamiltonian, n_qubits):
           self.hamiltonian = hamiltonian
           self.n_qubits = n_qubits
           self.operators = []
           self.params = []
           
       def _generate_operator_pool(self):
           """Generate pool of possible operators"""
           pool = []
           
           # Single excitations
           for i in range(self.n_qubits):
               for j in range(i+1, self.n_qubits):
                   pool.append(('single', i, j))
           
           return pool
       
       def _compute_energy(self, circuit):
           """Compute energy expectation value"""
           total_energy = 0.0
           
           for coeff, pauli_ops in self.hamiltonian:
               meas_circuit = circuit.copy()
               
               # Apply basis rotations
               for pauli, qubit in pauli_ops:
                   if pauli == 'X':
                       meas_circuit.h(qubit)
                   elif pauli == 'Y':
                       meas_circuit.sdg(qubit)
                       meas_circuit.h(qubit)
               
               meas_circuit.measure_all()
               result = meas_circuit.run(shots=1000)
               
               # Compute expectation
               expectation = 0.0
               for bitstring, count in result.items():
                   parity = 1
                   for pauli, qubit in pauli_ops:
                       if bitstring[qubit] == '1':
                           parity *= -1
                   expectation += parity * count
               
               expectation /= 1000
               total_energy += coeff * expectation
           
           return total_energy
       
       def run_adaptive_cycle(self):
           """Run one cycle of adaptive VQE"""
           # This is a simplified version
           # Real implementation would compute gradients for operator pool
           
           print("Adaptive VQE cycle (simplified demo)")
           
           # Add a simple operator
           if len(self.operators) < 2:
               self.operators.append(('single', 0, 1))
               self.params.append(0.1)
               return True
           
           return False

Quantum Error Correction
========================

Three-Qubit Bit Flip Code
-------------------------

.. code-block:: homebrew_s2

   def three_qubit_bit_flip_code():
       """Implement 3-qubit bit flip error correction"""
       circuit = tq.Circuit(5)  # 3 data + 2 ancilla
       
       # Encode logical |0⟩ -> |000⟩
       # For logical |1⟩, apply X to all three qubits
       
       # Encoding (example: encode |1⟩)
       circuit.x(0)  # Logical |1⟩
       circuit.cnot(0, 1)
       circuit.cnot(0, 2)
       
       # Simulate error on qubit 1
       circuit.x(1)  # Bit flip error
       
       # Error detection using ancilla qubits
       circuit.cnot(0, 3)  # First parity check
       circuit.cnot(1, 3)
       
       circuit.cnot(1, 4)  # Second parity check
       circuit.cnot(2, 4)
       
       # Measure ancilla
       circuit.measure_z(3)
       circuit.measure_z(4)
       
       result = circuit.run(shots=1000)
       
       print("Error detection results:")
       for syndrome, count in result.items():
           ancilla_0 = syndrome[3]
           ancilla_1 = syndrome[4]
           
           if count > 50:  # Show significant results
               print(f"Syndrome {ancilla_0}{ancilla_1}: {count} times")
               
               # Decode syndrome
               if ancilla_0 == '0' and ancilla_1 == '0':
                   print("  -> No error detected")
               elif ancilla_0 == '1' and ancilla_1 == '0':
                   print("  -> Error on qubit 0")
               elif ancilla_0 == '1' and ancilla_1 == '1':
                   print("  -> Error on qubit 1")
               elif ancilla_0 == '0' and ancilla_1 == '1':
                   print("  -> Error on qubit 2")
       
       return result

   # Run error correction demo
   print("\n3-Qubit Bit Flip Code Demo:")
   three_qubit_bit_flip_code()

Next Steps and Research
=======================

Congratulations! You've mastered advanced quantum computing. Here are research directions:

**Quantum Machine Learning**:
- Variational quantum classifiers
- Quantum convolutional neural networks  
- Quantum autoencoders
- Quantum reinforcement learning

**Hybrid Algorithms**:
- Quantum-enhanced optimization
- Variational quantum simulation
- Quantum approximate optimization
- Quantum neural networks

**Experimental Quantum Computing**:
- Noise characterization
- Hardware-specific optimization  
- Error correction protocols
- Quantum advantage demonstrations

Pulse-Level Quantum Programming
===============================

- :doc:`pulse_calibration_workflow` - **Complete Pulse Calibration Workflow** (7-step practical guide from hardware basics to Defcal integration)
- :doc:`pulse_virtual_z_optimization` - **Virtual-Z Optimization in Pulse Compilation** (Automatic RZ gate merging, phase tracking simplification)
- :doc:`pulse_three_level` - **Three-level quantum system simulation** (DRAG pulse leakage suppression)
- :doc:`pulse_inline_three_level` - **pulse_inline with three-level support** (Cloud-ready leakage simulation)
- :doc:`pulse_iswap_swap_decomposition` - **iSWAP and SWAP gate decomposition** (CX chain implementation, physics-native gates)
- :doc:`pulse_zz_crosstalk` - **ZZ crosstalk noise modeling** (Always-on coupling in superconducting qubits)
- :doc:`quantum_natural_gradient` - Quantum natural gradient optimization

See Also
========

- :doc:`../../examples/advanced_examples` - Advanced code examples
- :doc:`../../user_guide/advanced/index` - Advanced features guide
- :doc:`../../api/core/index` - Complete API reference
- :doc:`../../developer_guide/index` - Contributing to TyxonQ

**External Resources**:
- `Quantum Machine Learning <https://www.nature.com/articles/nature23474>`_ - Nature review
- `Variational Quantum Algorithms <https://arxiv.org/abs/2012.09265>`_ - Comprehensive review
- `NISQ Algorithms <https://arxiv.org/abs/1801.00862>`_ - Near-term quantum computing

---

**Happy quantum research!** 🚀🔬✨
