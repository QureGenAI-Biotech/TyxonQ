==============
Postprocessing
==============

The Postprocessing module provides tools for analyzing measurement results, including expectation value calculation, error mitigation, classical shadows, and metrics analysis.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

TyxonQ postprocessing capabilities include:

- **Expectation Values**: Convert measurement counts to expectation values
- **Error Mitigation**: Readout error correction and noise mitigation
- **Classical Shadows**: Scalable quantum state tomography
- **Metrics and Analysis**: Fidelity, entropy, and other quantum metrics
- **Noise Simulation**: Classical noise modeling on count data

Expectation Value Calculation
==============================

Basic Usage
-----------

.. code-block:: python

   import tyxonq as tq
   from tyxonq.postprocessing import counts_to_expval
   
   # Run circuit and get counts
   circuit = tq.Circuit(2).h(0).cnot(0, 1)
   counts = circuit.run(shots=1000)
   
   # Calculate expectation value
   expval = counts_to_expval(counts)
   print(f"Expectation value: {expval}")

Pauli Observable Expectations
------------------------------

.. code-block:: python

   from tyxonq.postprocessing import pauli_expval
   
   # Define Pauli observable (e.g., Z0*Z1)
   observable = "ZZ"
   
   # Calculate expectation
   expval = pauli_expval(counts, observable)
   print(f"<{observable}>: {expval}")

Error Mitigation
================

Readout Error Mitigation
------------------------

**Purpose**: Correct systematic errors in qubit readout

.. code-block:: python

   from tyxonq.postprocessing import ReadoutMitigator
   
   # Create calibration circuits
   mitigator = ReadoutMitigator(num_qubits=3)
   
   # Run calibration
   calibration_results = mitigator.calibrate(device='statevector', shots=1000)
   
   # Apply mitigation to measurement results
   raw_counts = circuit.run(shots=1000)
   mitigated_counts = mitigator.apply(raw_counts)
   
   print(f"Raw counts: {raw_counts}")
   print(f"Mitigated counts: {mitigated_counts}")

**How it works**:

.. mermaid::

   graph LR
       A[Calibration Circuits] --> B[Measure Error Matrix]
       B --> C[Construct Inverse Matrix]
       D[Raw Counts] --> E[Apply Inverse Matrix]
       C --> E
       E --> F[Corrected Counts]

Noise-Adaptive Processing
-------------------------

Apply noise models to count data for what-if analysis:

.. code-block:: python

   from tyxonq.postprocessing import apply_bitflip_counts, apply_depolarizing_counts
   
   # Simulate bit-flip noise
   noisy_counts_bitflip = apply_bitflip_counts(counts, p_flip=0.01)
   
   # Simulate depolarizing noise
   noisy_counts_depol = apply_depolarizing_counts(counts, p_depol=0.05)
   
   print(f"Original: {counts}")
   print(f"With bit-flip: {noisy_counts_bitflip}")
   print(f"With depolarizing: {noisy_counts_depol}")

Classical Shadows
=================

**Purpose**: Efficient quantum state property estimation without full tomography

Basic Workflow
--------------

.. code-block:: python

   from tyxonq.postprocessing import random_pauli_basis, estimate_expectation_pauli_product
   
   # Generate random measurement bases
   num_shots = 1000
   bases = random_pauli_basis(num_qubits=3, num_shots=num_shots)
   
   # Measure circuit in random bases
   results = []
   for basis in bases:
       # Apply basis rotation and measure
       measured_circuit = circuit.copy().rotate_to_basis(basis).measure_all()
       result = measured_circuit.run(shots=1)
       results.append(result)
   
   # Estimate expectation of target observable
   target_observable = "ZZI"  # Z_0 * Z_1 * I_2
   expval_estimate = estimate_expectation_pauli_product(
       results, bases, target_observable
   )
   
   print(f"Estimated <{target_observable}>: {expval_estimate}")

Advantages
----------

- **Scalability**: Works for large systems
- **Sample efficiency**: Fewer measurements for low-weight observables
- **Unbiased**: Provides unbiased estimates
- **Flexible**: Can estimate many observables from same data

Metrics and Analysis
====================

Fidelity Calculation
--------------------

.. code-block:: python

   from tyxonq.postprocessing import state_fidelity
   
   # Calculate fidelity between two states
   state1 = circuit1.device('statevector').run()
   state2 = circuit2.device('statevector').run()
   
   fidelity = state_fidelity(state1, state2)
   print(f"Fidelity: {fidelity}")

Entropy Measures
----------------

.. code-block:: python

   from tyxonq.postprocessing import von_neumann_entropy
   
   # Calculate von Neumann entropy
   density_matrix = circuit.device('density_matrix').run()
   entropy = von_neumann_entropy(density_matrix)
   
   print(f"Entropy: {entropy}")

Count Statistics
----------------

.. code-block:: python

   from tyxonq.postprocessing import counts_statistics
   
   counts = circuit.run(shots=10000)
   stats = counts_statistics(counts)
   
   print(f"Total shots: {stats['total_shots']}")
   print(f"Unique outcomes: {stats['num_outcomes']}")
   print(f"Most probable: {stats['most_probable']}")
   print(f"Entropy: {stats['shannon_entropy']}")

Advanced Techniques
===================

Zero-Noise Extrapolation
------------------------

.. code-block:: python

   from tyxonq.postprocessing import zero_noise_extrapolation
   
   # Run at different noise levels
   noise_levels = [0.0, 0.01, 0.02, 0.03]
   expvals = []
   
   for noise in noise_levels:
       result = circuit.device(
           'density_matrix',
           noise_model={'depolarizing': {'p': noise}}
       ).run(shots=1000)
       expval = counts_to_expval(result)
       expvals.append(expval)
   
   # Extrapolate to zero noise
   zne_expval = zero_noise_extrapolation(noise_levels, expvals)
   print(f"Zero-noise extrapolated value: {zne_expval}")

Probabilistic Error Cancellation
---------------------------------

.. code-block:: python

   from tyxonq.postprocessing import probabilistic_error_cancellation
   
   # Define ideal and noisy operations
   ideal_circuit = tq.Circuit(2).h(0).cnot(0, 1)
   noisy_circuit = ideal_circuit.device(
       'density_matrix',
       noise_model={'depolarizing': {'p': 0.01}}
   )
   
   # Apply PEC
   mitigated_result = probabilistic_error_cancellation(
       noisy_circuit,
       num_samples=1000
   )

Best Practices
==============

Error Mitigation Guidelines
---------------------------

1. **Calibrate regularly**: Readout errors drift over time
2. **Use appropriate shots**: More shots for better mitigation
3. **Validate mitigation**: Compare with ideal simulations
4. **Combine techniques**: Use multiple mitigation methods together

Performance Tips
----------------

1. **Batch processing**:

   .. code-block:: python

      # Process multiple count dictionaries efficiently
      all_expvals = [counts_to_expval(c) for c in count_list]

2. **Caching calibration**:

   .. code-block:: python

      # Save and load calibration data
      mitigator.save_calibration('calibration.json')
      mitigator.load_calibration('calibration.json')

3. **Parallel processing**:

   .. code-block:: python

      from concurrent.futures import ProcessPoolExecutor
      
      with ProcessPoolExecutor() as executor:
          results = list(executor.map(process_counts, count_list))

Related Resources
=================

- :doc:`/api/postprocessing/index` - Postprocessing API Reference
- :doc:`../devices/index` - Device Execution Guide
- :doc:`/examples/readout_mitigation` - Error Mitigation Examples
- :doc:`/examples/noise_controls_demo` - Noise Modeling Examples
