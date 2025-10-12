==============
Basic Examples
==============

This section provides fundamental usage examples of TyxonQ to help you get started quickly.

.. contents:: Examples List
   :depth: 2
   :local:

Bell State Creation
===================

Create the simplest entangled state:

.. code-block:: python

   import tyxonq as tq
   
   def create_bell_state():
       """Create Bell state (|00⟩ + |11⟩)/√2"""
       circuit = tq.Circuit(2)
       circuit.h(0)        # Create superposition
       circuit.cnot(0, 1)  # Create entanglement
       
       # Add measurements
       circuit.measure_z(0)
       circuit.measure_z(1)
       
       return circuit
   
   # Execute circuit
   bell_circuit = create_bell_state()
   result = bell_circuit.run(shots=1000)
   
   print("Measurement results:", result)
   # Expected output: {'00': ~500, '11': ~500}

**Explanation**:

- The Hadamard gate creates superposition on qubit 0
- CNOT entangles qubits 0 and 1
- Measurements collapse to either |00⟩ or |11⟩ with equal probability

GHZ State Generation
====================

Create multi-qubit GHZ states:

.. code-block:: python

   def create_ghz_state(n_qubits):
       """Create n-qubit GHZ state"""
       circuit = tq.Circuit(n_qubits)
       circuit.h(0)  # Superposition on first qubit
       
       # Entangle all qubits
       for i in range(n_qubits - 1):
           circuit.cnot(i, i + 1)
       
       return circuit
   
   # Create 3-qubit GHZ state
   ghz3 = create_ghz_state(3)
   
   # Add measurements
   for i in range(3):
       ghz3.measure_z(i)
   
   result = ghz3.run(shots=1000)
   print("GHZ state results:", result)
   # Expected output: {'000': ~500, '111': ~500}

**GHZ State Properties**:

- Maximally entangled multi-qubit state
- Measurement outcomes are perfectly correlated
- Used in quantum communication and error correction

Parameterized Circuits
======================

Build flexible circuits with parameterized gates:

.. code-block:: python

   import numpy as np
   
   def parametrized_circuit(theta):
       """Parameterized quantum circuit"""
       circuit = tq.Circuit(2)
       
       # Parameterized rotation gates
       circuit.ry(0, theta[0])
       circuit.ry(1, theta[1])
       
       # Entangling gate
       circuit.cnot(0, 1)
       
       # More parameterized gates
       circuit.rz(0, theta[2])
       circuit.rz(1, theta[3])
       
       return circuit
   
   # Use random parameters
   params = np.random.uniform(0, 2*np.pi, 4)
   circuit = parametrized_circuit(params)
   
   print(f"Using parameters: {params}")
   print(f"Number of gates: {len(circuit.ops)}")

**Use Cases**:

- Variational quantum algorithms (VQE, QAOA)
- Quantum machine learning
- Circuit optimization and training

Quantum Fourier Transform
==========================

Implement simplified QFT algorithm:

.. code-block:: python

   def qft_circuit(n_qubits):
       """Quantum Fourier Transform circuit"""
       circuit = tq.Circuit(n_qubits)
       
       for j in range(n_qubits):
           circuit.h(j)
           for k in range(j + 1, n_qubits):
               angle = np.pi / (2 ** (k - j))
               # Simplified: using RZ gate instead of controlled phase gate
               circuit.rz(k, angle / 2)
       
       return circuit
   
   # Create 3-qubit QFT
   qft_3 = qft_circuit(3)
   print(f"QFT circuit gate count: {len(qft_3.ops)}")

**Applications**:

- Shor's algorithm
- Quantum phase estimation
- Signal processing

Random Circuit Sampling
========================

Generate and test random circuits:

.. code-block:: python

   def random_circuit(n_qubits, depth, seed=None):
       """Generate random quantum circuit"""
       if seed is not None:
           np.random.seed(seed)
       
       circuit = tq.Circuit(n_qubits)
       gates = ['h', 'x', 'y', 'z']
       
       for layer in range(depth):
           # Random single-qubit gates
           for qubit in range(n_qubits):
               gate = np.random.choice(gates)
               if gate == 'h':
                   circuit.h(qubit)
               elif gate == 'x':
                   circuit.x(qubit)
               elif gate == 'y':
                   circuit.y(qubit)
               elif gate == 'z':
                   circuit.z(qubit)
           
           # Random two-qubit gates
           if n_qubits > 1:
               for _ in range(n_qubits // 2):
                   q1, q2 = np.random.choice(n_qubits, 2, replace=False)
                   circuit.cnot(q1, q2)
       
       return circuit
   
   # Generate random circuit
   random_circ = random_circuit(4, 3, seed=42)
   print(f"Random circuit gate count: {len(random_circ.ops)}")

**Use Cases**:

- Quantum supremacy experiments
- Benchmarking quantum devices
- Testing circuit compilation

Circuit Composition Example
============================

Combine multiple circuits into more complex ones:

.. code-block:: python

   def create_complex_circuit():
       """Compose multiple basic circuits"""
       # Basic circuit 1: Bell state
       bell = tq.Circuit(2).h(0).cnot(0, 1)
       
       # Basic circuit 2: Single-qubit rotation
       rotation = tq.Circuit(1).rz(0, np.pi/4)
       
       # Compose circuits
       # Apply rotation circuit to the second qubit of bell circuit
       combined = bell.compose(rotation, indices=[1])
       
       return combined
   
   complex_circuit = create_complex_circuit()
   print(f"Combined circuit gate count: {len(complex_circuit.ops)}")
   print(f"Circuit summary: {complex_circuit.gate_summary()}")

**Benefits**:

- Modular circuit design
- Reusable circuit components
- Easier debugging and testing

Circuit Visualization and Analysis
===================================

Analyze circuit properties and results:

.. code-block:: python

   def analyze_circuit(circuit):
       """Analyze circuit properties"""
       print(f"Circuit Analysis:")
       print(f"  Number of qubits: {circuit.num_qubits}")
       print(f"  Total gate count: {len(circuit.ops)}")
       print(f"  Gate type statistics: {circuit.gate_summary()}")
       
       # Check for measurements
       has_measurement = any('measure' in str(op).lower() for op in circuit.ops)
       print(f"  Contains measurements: {has_measurement}")
       
       return circuit
   
   # Analyze Bell state circuit
   bell = create_bell_state()
   analyze_circuit(bell)

**Output Example**:

.. code-block:: text

   Circuit Analysis:
     Number of qubits: 2
     Total gate count: 4
     Gate type statistics: {'H': 1, 'CNOT': 1, 'MEASURE_Z': 2}
     Contains measurements: True

Performance Testing Example
============================

Test performance across different devices:

.. code-block:: python

   import time
   
   def performance_test():
       """Performance testing example"""
       # Create test circuit
       circuit = create_ghz_state(10)
       for i in range(10):
           circuit.measure_z(i)
       
       devices = ['statevector', 'density_matrix']
       
       for device in devices:
           try:
               start_time = time.time()
               result = circuit.device(device).run(shots=100)
               end_time = time.time()
               
               print(f"{device}: {end_time - start_time:.4f} seconds")
           except Exception as e:
               print(f"{device}: Failed - {e}")
   
   # Run performance test
   performance_test()

**Expected Output**:

.. code-block:: text

   statevector: 0.0234 seconds
   density_matrix: 0.1456 seconds

**Insights**:

- Statevector simulation is fastest for pure states
- Density matrix needed for mixed states and noise
- Scale testing helps choose appropriate backend

Complete Application Example
=============================

A comprehensive example combining multiple concepts:

.. code-block:: python

   def complete_example():
       """Complete TyxonQ application example"""
       print("Starting complete example...")
       
       # 1. Create parameterized circuit
       params = [np.pi/4, np.pi/3, np.pi/6]
       circuit = tq.Circuit(3)
       
       # Initialize superposition
       for i in range(3):
           circuit.h(i)
       
       # Parameterized gates
       circuit.ry(0, params[0])
       circuit.ry(1, params[1])
       circuit.ry(2, params[2])
       
       # Entangling operations
       circuit.cnot(0, 1)
       circuit.cnot(1, 2)
       
       # Measurements
       for i in range(3):
           circuit.measure_z(i)
       
       # 2. Execute and analyze
       print(f"\nCircuit Information:")
       analyze_circuit(circuit)
       
       # 3. Execute on different devices
       print(f"\nExecution Results:")
       result = circuit.device('statevector').run(shots=1000)
       print(f"Statevector results: {result}")
       
       return circuit, result
   
   # Run complete example
   final_circuit, final_result = complete_example()

**Expected Output**:

.. code-block:: text

   Starting complete example...
   
   Circuit Information:
     Number of qubits: 3
     Total gate count: 11
     Gate type statistics: {'H': 3, 'RY': 3, 'CNOT': 2, 'MEASURE_Z': 3}
     Contains measurements: True
   
   Execution Results:
   Statevector results: {'000': 245, '001': 123, '010': 98, ...}

Advanced Circuit Patterns
==========================

Example: Circuit with Barriers
-------------------------------

Use barriers to organize circuit execution:

.. code-block:: python

   def circuit_with_barriers():
       """Circuit using barriers for organization"""
       circuit = tq.Circuit(3)
       
       # Preparation layer
       circuit.h(0)
       circuit.h(1)
       circuit.h(2)
       circuit.barrier()  # Mark section boundary
       
       # Entanglement layer
       circuit.cnot(0, 1)
       circuit.cnot(1, 2)
       circuit.barrier()
       
       # Measurement layer
       circuit.measure_z(0)
       circuit.measure_z(1)
       circuit.measure_z(2)
       
       return circuit
   
   barrier_circuit = circuit_with_barriers()

**Benefits of Barriers**:

- Visual organization in circuit diagrams
- Prevent unwanted gate optimization across sections
- Document circuit structure

Example: Conditional Operations
--------------------------------

Implement classical feedback based on measurements:

.. code-block:: python

   def conditional_circuit():
       """Circuit with conditional operations"""
       circuit = tq.Circuit(2)
       
       # Prepare first qubit
       circuit.h(0)
       circuit.measure_z(0)
       
       # Conditional X gate on second qubit
       # (requires circuit.conditional or if_measure methods)
       # Simplified example:
       circuit.x(1)  # Would be conditional on measurement of qubit 0
       
       return circuit

Tips and Best Practices
========================

1. **Start Simple**: Begin with small circuits and gradually increase complexity
2. **Use Visualization**: Analyze circuits before execution
3. **Choose Right Device**: 
   
   - Statevector: Fast, exact for pure states
   - Density matrix: Handles noise and mixed states
   - MPS: Efficient for low-entanglement circuits

4. **Monitor Performance**: Test circuit scaling early
5. **Modular Design**: Break complex circuits into reusable components

Troubleshooting Common Issues
==============================

**Issue 1: Circuit too deep**

.. code-block:: python

   # Solution: Use circuit compilation
   from tyxonq.compiler import compile_circuit
   
   deep_circuit = create_complex_circuit()
   optimized = compile_circuit(deep_circuit, optimization_level=3)
   print(f"Original gates: {len(deep_circuit.ops)}")
   print(f"Optimized gates: {len(optimized.ops)}")

**Issue 2: Memory errors for large circuits**

.. code-block:: python

   # Solution: Use MPS simulator for larger systems
   large_circuit = create_ghz_state(20)
   result = large_circuit.device('mps').run(shots=1000)

**Issue 3: Slow execution**

.. code-block:: python

   # Solution: Reduce shots for testing
   test_circuit = random_circuit(5, 10)
   quick_result = test_circuit.run(shots=100)  # Fast testing
   final_result = test_circuit.run(shots=10000)  # Production

See Also
========

- :doc:`/getting_started/first_circuit` - First circuit tutorial
- :doc:`chemistry_examples` - Quantum chemistry examples
- :doc:`optimization_examples` - Optimization algorithms
- :doc:`/user_guide/core/index` - Core module guide
- :doc:`/api/core/index` - Core API reference

Next Steps
==========

After mastering basic circuits:

1. Explore :doc:`chemistry_examples` for quantum chemistry applications
2. Learn :doc:`optimization_examples` for variational algorithms
3. Try :doc:`cloud_examples` for cloud execution
4. Study :doc:`advanced_examples` for complex applications
