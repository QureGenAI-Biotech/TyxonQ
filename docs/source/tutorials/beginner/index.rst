==================
Beginner Tutorials
==================

Welcome to TyxonQ! This tutorial will guide you through your first steps in quantum computing,
from installation to running your first quantum circuits.

.. contents:: Tutorial Contents
   :depth: 3
   :local:

.. note::
   No prior quantum computing experience required! We'll explain everything step by step.

Getting Started
===============

What is TyxonQ?
---------------

TyxonQ is a powerful quantum computing framework that allows you to:

🔬 **Simulate quantum circuits** on classical computers  
⚗️ **Solve chemistry problems** using quantum algorithms  
☁️ **Run on real quantum hardware** via cloud providers  
🧠 **Build quantum machine learning** models  

**Why TyxonQ?**

- **Easy to learn**: Simple, Pythonic API
- **Fast simulation**: Optimized quantum simulators
- **Real hardware**: Access to quantum computers
- **Rich ecosystem**: Chemistry, optimization, ML tools

Installation Guide
------------------

**Step 1: Install Python**

TyxonQ requires Python 3.8 or higher. Check your version:

.. code-block:: bash

   python --version
   # Should show Python 3.8+ 

**Step 2: Install TyxonQ**

.. code-block:: bash

   # Install from PyPI
   pip install tyxonq
   
   # Or install development version
   pip install git+https://github.com/TyxonQ/TyxonQ.git

**Step 3: Verify Installation**

.. code-block:: python

   import tyxonq as tq
   print(f"TyxonQ version: {tq.__version__}")
   
   # Test basic functionality
   circuit = tq.Circuit(1)
   circuit.h(0)
   print("✅ TyxonQ installed successfully!")

**Troubleshooting Installation**:

- **Issue**: ``ModuleNotFoundError: No module named 'tyxonq'``  
  **Solution**: Make sure you're using the correct Python environment

- **Issue**: Import errors with dependencies  
  **Solution**: Update pip and try again: ``pip install --upgrade pip``

Your First Quantum Circuit
===========================

Understanding Qubits
--------------------

A **qubit** is the basic unit of quantum information, like a quantum version of a classical bit.

**Classical bit**: Can be 0 or 1  
**Qubit**: Can be 0, 1, or both at the same time (superposition)!

.. code-block:: python

   import tyxonq as tq
   
   # Create a quantum circuit with 1 qubit
   circuit = tq.Circuit(1)
   
   print(f"Number of qubits: {circuit.n_qubits}")
   print(f"Initial state: |0⟩")  # All qubits start in |0⟩ state

Basic Quantum Gates
-------------------

**Gates** are operations that manipulate qubits. Let's learn the most important ones:

### X Gate (NOT Gate)

Flips qubit from |0⟩ to |1⟩, or vice versa:

.. code-block:: python

   circuit = tq.Circuit(1)
   circuit.x(0)  # Apply X gate to qubit 0
   
   # Measure the result
   circuit.measure_z(0)
   result = circuit.run(shots=100)
   print(result)  # Should show {'1': 100}

### H Gate (Hadamard Gate)

Creates **superposition** - qubit becomes 50% |0⟩ and 50% |1⟩:

.. code-block:: python

   circuit = tq.Circuit(1)
   circuit.h(0)  # Apply Hadamard gate
   circuit.measure_z(0)
   
   result = circuit.run(shots=1000)
   print(result)  # Should show ~{'0': 500, '1': 500}

**🤔 What just happened?**

The Hadamard gate put our qubit in superposition. When we measure it, we randomly get either 0 or 1, each with 50% probability.

### CNOT Gate (Controlled-NOT)

Creates **entanglement** between two qubits:

.. code-block:: python

   circuit = tq.Circuit(2)  # 2 qubits needed
   circuit.h(0)        # Put first qubit in superposition
   circuit.cnot(0, 1)  # CNOT: control=0, target=1
   
   # Measure both qubits
   circuit.measure_z(0)
   circuit.measure_z(1)
   
   result = circuit.run(shots=1000)
   print(result)  # Should show ~{'00': 500, '11': 500}

**🤯 Magic!** The qubits are now entangled. When qubit 0 is measured as 0, qubit 1 is always 0. When qubit 0 is 1, qubit 1 is always 1.

Step-by-Step: Creating Your First Circuit
------------------------------------------

Let's build a complete quantum circuit step by step:

**Goal**: Create a Bell state (maximally entangled 2-qubit state)

.. code-block:: python

   import tyxonq as tq
   
   # Step 1: Create circuit with 2 qubits
   print("Step 1: Creating circuit...")
   circuit = tq.Circuit(2)
   print(f"Created circuit with {circuit.n_qubits} qubits")
   
   # Step 2: Apply Hadamard to first qubit
   print("\nStep 2: Creating superposition...")
   circuit.h(0)
   print("Applied H gate to qubit 0")
   print("State: (|00⟩ + |10⟩)/√2")
   
   # Step 3: Apply CNOT gate
   print("\nStep 3: Creating entanglement...")
   circuit.cnot(0, 1)
   print("Applied CNOT gate")
   print("Final state: (|00⟩ + |11⟩)/√2 <- This is a Bell state!")
   
   # Step 4: Add measurements
   print("\nStep 4: Adding measurements...")
   circuit.measure_z(0)  # Measure qubit 0
   circuit.measure_z(1)  # Measure qubit 1
   print("Added measurements to both qubits")
   
   # Step 5: Run the circuit
   print("\nStep 5: Running the circuit...")
   result = circuit.run(shots=1000)
   
   # Step 6: Analyze results
   print("\nStep 6: Results analysis:")
   print(f"Results: {result}")
   
   prob_00 = result.get('00', 0) / 1000
   prob_11 = result.get('11', 0) / 1000
   
   print(f"Probability of |00⟩: {prob_00:.1%}")
   print(f"Probability of |11⟩: {prob_11:.1%}")
   
   if prob_00 > 0.4 and prob_11 > 0.4:
       print("✅ Success! You created a Bell state!")
   else:
       print("🤔 Something went wrong. Try running again.")

Understanding Results
---------------------

**What do the results mean?**

When you run the circuit above, you should see something like:

.. code-block:: text

   Results: {'00': 496, '11': 504}
   Probability of |00⟩: 49.6%
   Probability of |11⟩: 50.4%
   ✅ Success! You created a Bell state!

**Key insights**:

1. **No '01' or '10'**: The qubits are perfectly correlated
2. **~50/50 split**: Random, but always correlated
3. **Bell state achieved**: Maximum entanglement!

Working with Multiple Qubits
=============================

Scaling Up
----------

Let's work with more qubits:

.. code-block:: python

   # Create a 3-qubit circuit
   circuit = tq.Circuit(3)
   
   # Apply gates to different qubits
   circuit.h(0)        # Hadamard on qubit 0
   circuit.x(1)        # X gate on qubit 1  
   circuit.cnot(0, 2)  # CNOT from qubit 0 to qubit 2
   
   # Measure all qubits
   for i in range(3):
       circuit.measure_z(i)
   
   result = circuit.run(shots=500)
   print(result)
   # Expected: {'010': ~250, '111': ~250}

**Understanding the result**:
- Qubit 1 is always 1 (due to X gate)
- Qubits 0 and 2 are entangled (due to CNOT after H gate)
- So we get either '010' or '111'

GHZ State (3-Qubit Entanglement)
---------------------------------

Let's create a famous 3-qubit entangled state:

.. code-block:: python

   def create_ghz_state(n_qubits):
       """Create GHZ state: (|000...⟩ + |111...⟩)/√2"""
       circuit = tq.Circuit(n_qubits)
       
       # Step 1: Create superposition on first qubit
       circuit.h(0)
       
       # Step 2: Entangle all other qubits with first qubit
       for i in range(1, n_qubits):
           circuit.cnot(0, i)
       
       return circuit
   
   # Create 3-qubit GHZ state
   ghz_circuit = create_ghz_state(3)
   
   # Add measurements
   for i in range(3):
       ghz_circuit.measure_z(i)
   
   # Run and check results
   result = ghz_circuit.run(shots=1000)
   print(f"GHZ state results: {result}")
   
   # Should see ~{'000': 500, '111': 500}
   if '000' in result and '111' in result:
       print("✅ GHZ state created successfully!")
       print("All qubits are entangled together!")

Quantum Circuit Analysis
========================

Inspecting Your Circuit
-----------------------

Let's learn how to analyze circuits:

.. code-block:: python

   # Create a sample circuit
   circuit = tq.Circuit(3)
   circuit.h(0)
   circuit.cnot(0, 1)
   circuit.cnot(1, 2)
   circuit.x(0)
   circuit.measure_z(0)
   circuit.measure_z(1)
   circuit.measure_z(2)
   
   # Analyze the circuit
   print("Circuit Analysis:")
   print(f"  Number of qubits: {circuit.n_qubits}")
   print(f"  Number of operations: {len(circuit.ops)}")
   
   # Count different types of gates
   gate_counts = {}
   for op in circuit.ops:
       gate_type = type(op).__name__
       gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
   
   print(f"  Gate breakdown: {gate_counts}")

Visualization (Optional)
------------------------

If you want to visualize your circuits:

.. code-block:: python

   # Print circuit as text (built-in)
   print("Circuit diagram:")
   print(circuit)  # This shows a text representation
   
   # For fancier visualization (if matplotlib is installed):
   try:
       import matplotlib.pyplot as plt
       circuit.draw()  # Creates a visual diagram
       plt.show()
   except ImportError:
       print("Install matplotlib for circuit visualization: pip install matplotlib")

Simulator Backends
==================

Choosing the Right Simulator
-----------------------------

TyxonQ offers different simulators for different needs:

.. code-block:: python

   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cnot(0, 1)
   circuit.measure_z(0)
   circuit.measure_z(1)
   
   # Option 1: Statevector simulator (fastest, exact)
   result1 = circuit.device('statevector').run(shots=1000)
   print(f"Statevector: {result1}")
   
   # Option 2: Density matrix simulator (handles noise)
   result2 = circuit.device('density_matrix').run(shots=1000)
   print(f"Density matrix: {result2}")
   
   # Option 3: MPS simulator (for larger circuits)
   result3 = circuit.device('mps').run(shots=1000)
   print(f"MPS: {result3}")

**When to use each**:

- **Statevector**: Pure states, fast simulation, ≤20 qubits
- **Density matrix**: Mixed states, noise simulation, ≤15 qubits  
- **MPS**: Low-entanglement circuits, ≤50 qubits

Testing Different Shot Counts
-----------------------------

.. code-block:: python

   circuit = tq.Circuit(1)
   circuit.h(0)  # 50/50 probability
   circuit.measure_z(0)
   
   # Test with different shot counts
   shot_counts = [10, 100, 1000, 10000]
   
   for shots in shot_counts:
       result = circuit.run(shots=shots)
       prob_0 = result.get('0', 0) / shots
       prob_1 = result.get('1', 0) / shots
       
       print(f"Shots: {shots:5d} | P(0): {prob_0:.3f} | P(1): {prob_1:.3f}")
   
   print("\n💡 More shots = more accurate probabilities")

Practical Exercises
===================

Exercise 1: Superposition States
---------------------------------

**Task**: Create different superposition states and measure their probabilities.

.. code-block:: python

   # Your code here:
   # 1. Create a circuit with 1 qubit
   # 2. Apply H gate
   # 3. Measure and run with 1000 shots
   # 4. Check that you get ~50/50 results
   
   # Solution:
   circuit = tq.Circuit(1)
   circuit.h(0)
   circuit.measure_z(0)
   result = circuit.run(shots=1000)
   
   print(f"Exercise 1 result: {result}")
   # Should be close to {'0': 500, '1': 500}

Exercise 2: Two-Qubit States
-----------------------------

**Task**: Create a state where both qubits are definitely in |1⟩.

.. code-block:: python

   # Your code here:
   # Hint: Use X gates
   
   # Solution:
   circuit = tq.Circuit(2)
   circuit.x(0)  # Set qubit 0 to |1⟩
   circuit.x(1)  # Set qubit 1 to |1⟩
   circuit.measure_z(0)
   circuit.measure_z(1)
   result = circuit.run(shots=100)
   
   print(f"Exercise 2 result: {result}")
   # Should be {'11': 100}

Exercise 3: Custom Bell States
-------------------------------

**Task**: Create the Bell state |Φ⁻⟩ = (|01⟩ + |10⟩)/√2

.. code-block:: python

   # Hint: Start with |Φ⁺⟩ = (|00⟩ + |11⟩)/√2, then apply X to one qubit
   
   # Solution:
   circuit = tq.Circuit(2)
   circuit.h(0)        # Create superposition
   circuit.cnot(0, 1)  # Create |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
   circuit.x(1)        # Flip second qubit: (|01⟩ + |10⟩)/√2
   
   circuit.measure_z(0)
   circuit.measure_z(1)
   result = circuit.run(shots=1000)
   
   print(f"Exercise 3 result: {result}")
   # Should see ~{'01': 500, '10': 500}

Troubleshooting Common Issues
=============================

Issue 1: Unexpected Results
---------------------------

**Problem**: "My circuit should give 50/50, but I get 70/30"

**Solutions**:

.. code-block:: python

   # ❌ Wrong: Too few shots
   result = circuit.run(shots=10)  # Not enough for statistics
   
   # ✅ Right: Use more shots
   result = circuit.run(shots=1000)
   
   # ❌ Wrong: Forgot to add measurements
   circuit = tq.Circuit(1)
   circuit.h(0)
   # No measure_z() call!
   
   # ✅ Right: Always add measurements
   circuit.measure_z(0)

Issue 2: Import Errors
----------------------

**Problem**: ``ImportError`` or ``ModuleNotFoundError``

**Solutions**:

.. code-block:: python

   # Check your installation
   try:
       import tyxonq as tq
       print(f"✅ TyxonQ {tq.__version__} loaded successfully")
   except ImportError as e:
       print(f"❌ Import failed: {e}")
       print("Solution: pip install tyxonq")

Issue 3: Performance Problems
-----------------------------

**Problem**: "My circuit is too slow"

**Solutions**:

.. code-block:: python

   # ❌ Slow: Too many qubits for statevector
   big_circuit = tq.Circuit(25)  # Will be very slow!
   
   # ✅ Fast: Use appropriate simulator
   circuit = tq.Circuit(10)
   circuit.device('mps')  # Better for larger circuits
   
   # ✅ Fast: Reduce shots for testing
   result = circuit.run(shots=100)  # Instead of 10000

Next Steps
==========

Congratulations! 🎉 You've completed the beginner tutorial. You now know:

✅ How to install and set up TyxonQ  
✅ Basic quantum gates (X, H, CNOT)  
✅ Creating superposition and entanglement  
✅ Measuring quantum circuits  
✅ Working with multiple qubits  
✅ Analyzing circuit results  

What's Next?
------------

1. **Intermediate Tutorial**: Learn about variational algorithms and optimization
2. **Chemistry Examples**: Apply quantum computing to real molecules
3. **Cloud Tutorial**: Run circuits on real quantum hardware
4. **API Reference**: Deep dive into TyxonQ's features

Recommended Learning Path
-------------------------

**Next tutorials to try**:

.. code-block:: text

   📚 You are here: Beginner Tutorial ✅
   📚 Next: Intermediate Tutorial 🚧
   📚 Then: Advanced Tutorial 🚧
   📚 Finally: Real projects! 🚀

**Example projects to build**:

- Random number generator using quantum superposition
- Simple quantum game (quantum coin flip)
- Quantum teleportation protocol
- Basic quantum machine learning

Useful Resources
================

**Documentation**:
- :doc:`../intermediate/index` - Intermediate tutorials
- :doc:`../../examples/basic_examples` - More code examples
- :doc:`../../user_guide/core/index` - Complete feature guide
- :doc:`../../api/core/index` - API reference

**External Learning**:
- `Qiskit Textbook <https://qiskit.org/textbook/>`_ - Quantum computing theory
- `Microsoft Q# Kata <https://github.com/Microsoft/QuantumKatas>`_ - Practice problems
- `Quantum Computing: An Applied Approach <https://www.springer.com/gp/book/9783030239213>`_ - Comprehensive textbook

**Community**:
- GitHub Issues: Report bugs and ask questions
- Discussions: Share your quantum circuits
- Stack Overflow: Tag your questions with 'tyxonq'

---

**Happy quantum computing!** 🚀✨

.. note::
   Remember: Quantum computing can be counterintuitive at first. Don't worry if some concepts seem strange - even Einstein found quantum mechanics "spooky"! Keep experimenting and asking questions.
