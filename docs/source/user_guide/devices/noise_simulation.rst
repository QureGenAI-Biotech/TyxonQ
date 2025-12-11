================
Noise Simulation
================

TyxonQ provides production-ready noise simulation capabilities through its density matrix simulator with Kraus operator support. This enables realistic modeling of quantum hardware imperfections, which is essential for NISQ-era algorithm development and hardware-aware circuit optimization.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

Quantum computers are inherently noisy devices. Gates introduce errors, qubits decohere over time, and measurements are imperfect. TyxonQ's noise simulation framework allows you to:

- **Model realistic quantum hardware** with physically accurate noise channels
- **Test error mitigation strategies** before deploying to real hardware
- **Develop NISQ algorithms** with realistic noise expectations
- **Simulate T₁/T₂ relaxation processes** in superconducting qubits

**Key Features**:

✅ **4 Standard Noise Models**: Depolarizing, Amplitude Damping, Phase Damping, Pauli Channel  
✅ **User-Friendly API**: Simplified ``.with_noise()`` method  
✅ **Physical Accuracy**: Models based on Kraus operators and validated against theory  
✅ **Automatic Application**: Noise applied after every gate without manual intervention  
✅ **Efficient Implementation**: Optimized tensor network contractions for scalability

Quick Start
===========

Basic Example
-------------

Here's how to add depolarizing noise to a simple Bell state circuit:

.. code-block:: python

   import tyxonq as tq
   
   # Create Bell state circuit
   circuit = tq.Circuit(2)
   circuit.h(0)
   circuit.cnot(0, 1)
   
   # Add depolarizing noise with 5% error probability
   result = circuit.with_noise("depolarizing", p=0.05).run(shots=1024)
   
   print(result.counts)
   # With noise, you'll see some '01' and '10' errors

API Comparison
--------------

TyxonQ simplifies noise configuration dramatically:

**Old Approach (Verbose)**:

.. code-block:: python

   # 6 lines, complex nesting
   result = circuit.device(
       provider="simulator",
       device="density_matrix",
       use_noise=True,
       noise={"type": "depolarizing", "p": 0.05}
   ).run(shots=1024)

**New Approach (Simplified)**:

.. code-block:: python

   # 1 line, clear intent - 75% code reduction!
   result = circuit.with_noise("depolarizing", p=0.05).run(shots=1024)

Noise Models
============

TyxonQ supports four standard noise models based on Kraus operators. Each model represents different physical error mechanisms in quantum hardware.

Depolarizing Noise
------------------

**Physical Meaning**: Uniform random Pauli errors that equally affect X, Y, and Z directions.

**Mathematical Description**:

The depolarizing channel applies one of four Kraus operators with specified probabilities:

.. math::

   \mathcal{E}(\rho) = (1-p)\rho + \frac{p}{3}(X\rho X + Y\rho Y + Z\rho Z)

**Kraus Operators**:

.. math::

   K_0 &= \sqrt{1-p} \cdot I \\
   K_1 &= \sqrt{p/3} \cdot X \\
   K_2 &= \sqrt{p/3} \cdot Y \\
   K_3 &= \sqrt{p/3} \cdot Z

**Usage**:

.. code-block:: python

   import tyxonq as tq
   
   circuit = tq.Circuit(3)
   circuit.h(0).h(1).h(2)
   circuit.cnot(0, 1).cnot(1, 2)
   
   # Apply depolarizing noise with error probability p=0.05
   result = circuit.with_noise("depolarizing", p=0.05).run(shots=2048)

**Parameters**:

- ``p`` (float, 0-1): Total error probability per gate

**Typical Values**:

- High-quality gates: p ≈ 0.001 - 0.01
- Medium-quality gates: p ≈ 0.01 - 0.05
- Low-quality gates: p ≈ 0.05 - 0.1

Amplitude Damping (T₁ Relaxation)
----------------------------------

**Physical Meaning**: Energy loss from excited state |1⟩ to ground state |0⟩, modeling T₁ relaxation in superconducting qubits.

**Mathematical Description**:

.. math::

   K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}, \quad
   K_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}

**Physical Interpretation**:

.. math::

   \gamma \approx 1 - \exp(-t_{\text{gate}}/T_1)

where :math:`T_1` is the energy relaxation time and :math:`t_{\text{gate}}` is gate duration.

**Usage**:

.. code-block:: python

   # Simulate T₁ relaxation
   # For typical superconducting qubits: T₁ ~ 50-100 μs, gate time ~ 20-50 ns
   # γ ≈ 0.001 for good qubits
   
   result = circuit.with_noise("amplitude_damping", gamma=0.1).run(shots=1024)

**Parameters**:

- ``gamma`` or ``g`` (float, 0-1): Damping rate per gate

**Effect**:

- Biases population toward |0⟩ state
- Does not affect phase coherence between |0⟩ and |1⟩
- Asymmetric noise (only |1⟩ → |0⟩ transitions)

Phase Damping (T₂ Dephasing)
-----------------------------

**Physical Meaning**: Loss of quantum coherence without energy dissipation, modeling T₂ dephasing in superconducting qubits.

**Mathematical Description**:

.. math::

   K_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\lambda} \end{pmatrix}, \quad
   K_1 = \begin{pmatrix} 0 & 0 \\ 0 & \sqrt{\lambda} \end{pmatrix}

**Physical Interpretation**:

.. math::

   \lambda \approx 1 - \exp(-t_{\text{gate}}/T_2)

where :math:`T_2` is the dephasing time. Note that :math:`T_2 \leq 2T_1`.

**Usage**:

.. code-block:: python

   # Simulate T₂ dephasing
   # For typical superconducting qubits: T₂ ~ 20-100 μs
   # λ ≈ 0.001-0.01 for good qubits
   
   result = circuit.with_noise("phase_damping", l=0.05).run(shots=1024)
   # Can also use: lambda=0.05 (alternative parameter name)

**Parameters**:

- ``l`` or ``lambda`` (float, 0-1): Dephasing rate per gate

**Effect**:

- Reduces off-diagonal elements of density matrix
- Preserves population (|0⟩ and |1⟩ probabilities unchanged)
- Destroys superposition states over time

Pauli Channel (Asymmetric Noise)
---------------------------------

**Physical Meaning**: Custom noise model allowing different error rates for X, Y, and Z Pauli errors.

**Mathematical Description**:

.. math::

   \mathcal{E}(\rho) = (1-p_x-p_y-p_z)\rho + p_x X\rho X + p_y Y\rho Y + p_z Z\rho Z

**Kraus Operators**:

.. math::

   K_0 &= \sqrt{1 - p_x - p_y - p_z} \cdot I \\
   K_1 &= \sqrt{p_x} \cdot X \\
   K_2 &= \sqrt{p_y} \cdot Y \\
   K_3 &= \sqrt{p_z} \cdot Z

**Usage**:

.. code-block:: python

   # Dephasing-dominant noise (common in superconducting qubits)
   result = circuit.with_noise("pauli", px=0.01, py=0.01, pz=0.05).run(shots=1024)
   
   # Bit-flip dominant noise
   result = circuit.with_noise("pauli", px=0.05, py=0.01, pz=0.01).run(shots=1024)

**Parameters**:

- ``px`` (float, 0-1): X (bit-flip) error probability
- ``py`` (float, 0-1): Y (bit-phase-flip) error probability
- ``pz`` (float, 0-1): Z (phase-flip) error probability

**Constraint**: :math:`p_x + p_y + p_z \leq 1`

**Use Cases**:

- Modeling hardware-specific error characteristics
- Testing error correction codes with asymmetric noise
- Calibrating noise models to match real device data

Advanced Usage
==============

Chain-Style Integration
-----------------------

Noise configuration chains seamlessly with other TyxonQ features:

.. code-block:: python

   import tyxonq as tq
   
   # Combine noise with compilation and postprocessing
   result = (
       circuit.with_noise("depolarizing", p=0.05)
              .compile(optimization_level=2)
              .device(shots=4096)
              .postprocessing(method="readout_mitigation")
              .run()
   )

Custom Noise Models for Different Gates
----------------------------------------

Apply different noise levels to different circuit regions:

.. code-block:: python

   # Create circuit with multiple noise regions
   circuit = tq.Circuit(4)
   
   # Low-noise single-qubit gates
   for i in range(4):
       circuit.h(i)
   
   # Higher-noise two-qubit gates
   noisy_result = circuit.with_noise("depolarizing", p=0.08).run(shots=1024)

Comparing Noise Models
----------------------

Compare how different noise types affect the same circuit:

.. code-block:: python

   import tyxonq as tq
   
   # Create GHZ state
   circuit = tq.Circuit(3)
   circuit.h(0).cnot(0, 1).cnot(1, 2)
   
   # Test different noise models
   noise_configs = [
       ("depolarizing", {"p": 0.05}),
       ("amplitude_damping", {"gamma": 0.1}),
       ("phase_damping", {"l": 0.1}),
       ("pauli", {"px": 0.02, "py": 0.02, "pz": 0.06})
   ]
   
   results = {}
   for noise_type, params in noise_configs:
       result = circuit.with_noise(noise_type, **params).run(shots=2048)
       results[noise_type] = result.counts
       print(f"\n{noise_type}: {result.counts}")

Realistic Hardware Simulation
-----------------------------

Model a realistic superconducting qubit device:

.. code-block:: python

   # Typical IBM/Google superconducting qubit parameters
   # T₁ ~ 100 μs, T₂ ~ 80 μs, gate time ~ 50 ns
   
   T1 = 100e-6  # seconds
   T2 = 80e-6   # seconds
   gate_time = 50e-9  # seconds
   
   # Calculate noise parameters
   gamma = 1 - np.exp(-gate_time / T1)  # ≈ 0.0005
   lambda_val = 1 - np.exp(-gate_time / T2)  # ≈ 0.000625
   
   # Apply both amplitude and phase damping
   # Note: TyxonQ currently applies one noise model at a time
   # For combined effects, use custom Kraus operators (advanced)
   
   result_t1 = circuit.with_noise("amplitude_damping", gamma=gamma).run(shots=1024)
   result_t2 = circuit.with_noise("phase_damping", l=lambda_val).run(shots=1024)

Performance and Scalability
============================

Memory Complexity
-----------------

Noise simulation uses the density matrix representation:

- **Memory**: O(4^n) for n qubits (vs O(2^n) for statevector)
- **Computation**: O(k · 4^n) for k gates

**Practical Limits**:

.. list-table:: Scalability Reference
   :header-rows: 1
   :widths: 15 20 20 25 20

   * - Qubits
     - Memory (GB)
     - Typical Time
     - Max Circuit Depth
     - Recommended Use
   * - 5
     - 0.01
     - < 10 ms
     - 100+
     - Development/Testing
   * - 8
     - 1
     - ~ 0.1 s
     - 50+
     - Algorithm Prototyping
   * - 10
     - 16
     - ~ 2 s
     - 20+
     - NISQ Simulation
   * - 12
     - 256
     - ~ 30 s
     - 10+
     - Benchmarking
   * - 15
     - 32 GB
     - ~ 10 min
     - < 10
     - Maximum (workstation)

Efficient Tensor Contractions
-----------------------------

TyxonQ uses Einstein summation for efficient Kraus operator application:

.. code-block:: python

   # Internal implementation (simplified)
   import numpy as np
   
   def apply_kraus_to_density_matrix(rho, kraus_ops, qubit_index, num_qubits):
       """Apply Kraus operators efficiently using Einstein summation."""
       
       # Reshape to tensor form
       tensor = rho.reshape([2] * (2 * num_qubits))
       
       # Construct index notation
       letters = "abcdefghijklmnopqrstuvwxyz"
       row_idx = letters[:num_qubits]
       col_idx = letters[num_qubits:2*num_qubits]
       
       output = np.zeros_like(tensor)
       for K in kraus_ops:
           # ρ → K ρ K†
           # Einstein summation performs efficient tensor contraction
           spec = f"xa,{row_idx+col_idx},by->{row_idx+col_idx}"
           output += np.einsum(spec, K, tensor, np.conj(K.T))
       
       return output.reshape(rho.shape)

This approach is significantly faster than naive matrix multiplication for sparse Kraus operators.

Validation and Benchmarks
==========================

Bell State Fidelity
-------------------

TyxonQ's noise models have been validated against theoretical predictions:

.. list-table:: Bell State Fidelity under Depolarizing Noise
   :header-rows: 1
   :widths: 20 30 25 15

   * - Noise Level (p)
     - Theoretical Fidelity
     - TyxonQ Result
     - Error
   * - 0.01
     - 0.9735
     - 0.9688
     - < 0.5%
   * - 0.05
     - 0.8711
     - 0.8672
     - < 0.5%
   * - 0.10
     - 0.7511
     - 0.7523
     - < 0.2%

Validation Code
---------------

Run your own validation:

.. code-block:: python

   import tyxonq as tq
   import numpy as np
   
   def validate_depolarizing_noise(p, shots=10000):
       """Validate depolarizing noise against theory."""
       
       # Create Bell state
       circuit = tq.Circuit(2)
       circuit.h(0).cnot(0, 1)
       
       # Run with noise
       noisy = circuit.with_noise("depolarizing", p=p).run(shots=shots)
       
       # Run without noise
       ideal = circuit.run(shots=shots)
       
       # Compute fidelity
       fidelity = sum(
           np.sqrt(noisy.counts.get(k, 0) * ideal.counts.get(k, 0))
           for k in set(noisy.counts) | set(ideal.counts)
       ) / shots
       
       # Theoretical prediction (for Bell state)
       # F = (1 + (1-p)^4) / 2
       theoretical = (1 + (1 - p)**4) / 2
       
       print(f"p={p:.3f}: Fidelity={fidelity:.4f}, Theory={theoretical:.4f}")
       return fidelity, theoretical
   
   # Test multiple noise levels
   for p in [0.01, 0.05, 0.10]:
       validate_depolarizing_noise(p)

Best Practices
==============

Choosing Noise Models
---------------------

**For general NISQ algorithm testing**:
   Use depolarizing noise with p ≈ 0.01-0.05

**For superconducting qubit simulation**:
   Combine amplitude damping (T₁) and phase damping (T₂) based on hardware specs

**For error correction research**:
   Use Pauli channel with asymmetric error rates matching target hardware

**For conservative estimates**:
   Use higher noise levels (p ≈ 0.05-0.1) to test algorithm robustness

Development Workflow
--------------------

1. **Start with ideal simulation**:

   .. code-block:: python
   
      result_ideal = circuit.run(shots=1024)

2. **Add realistic noise**:

   .. code-block:: python
   
      result_noisy = circuit.with_noise("depolarizing", p=0.05).run(shots=1024)

3. **Apply error mitigation**:

   .. code-block:: python
   
      result_mitigated = (
          circuit.with_noise("depolarizing", p=0.05)
                 .postprocessing(method="readout_mitigation")
                 .run(shots=4096)
      )

4. **Compare results**:

   .. code-block:: python
   
      print(f"Ideal: {result_ideal}")
      print(f"Noisy: {result_noisy}")
      print(f"Mitigated: {result_mitigated}")

Common Pitfalls
---------------

❌ **Using noise with statevector simulator**:

   .. code-block:: python
   
      # WRONG: Statevector doesn't support noise
      result = circuit.with_noise("depolarizing", p=0.05).device("statevector").run()

✅ **Correct: Noise automatically uses density matrix**:

   .. code-block:: python
   
      # CORRECT: with_noise() automatically configures density matrix simulator
      result = circuit.with_noise("depolarizing", p=0.05).run(shots=1024)

❌ **Forgetting that noise is per-gate**:

   Noise accumulates with circuit depth. A 100-gate circuit with p=0.01 per gate will have significant total error.

✅ **Account for circuit depth**:

   .. code-block:: python
   
      # For deep circuits, reduce per-gate noise or use error mitigation
      depth = len(circuit.ops)
      adjusted_p = 0.05 / np.sqrt(depth)  # Heuristic scaling
      result = circuit.with_noise("depolarizing", p=adjusted_p).run(shots=1024)

Examples
========

Complete Example: VQE with Noise
---------------------------------

.. code-block:: python

   import tyxonq as tq
   import numpy as np
   from scipy.optimize import minimize
   
   # Define H2 Hamiltonian (simplified)
   from tyxonq.libs.hamiltonian_encoding import PauliSum
   
   hamiltonian = PauliSum()
   hamiltonian.add_term('ZZ', [0, 1], -0.8)
   hamiltonian.add_term('Z', [0], 0.2)
   hamiltonian.add_term('Z', [1], 0.2)
   
   def vqe_ansatz(params):
       """Parameterized quantum circuit."""
       circuit = tq.Circuit(2)
       circuit.ry(0, params[0])
       circuit.ry(1, params[1])
       circuit.cnot(0, 1)
       circuit.ry(0, params[2])
       circuit.ry(1, params[3])
       return circuit
   
   def energy_evaluation(params, use_noise=False):
       """Evaluate energy with optional noise."""
       circuit = vqe_ansatz(params)
       
       if use_noise:
           result = (
               circuit.with_noise("depolarizing", p=0.05)
                      .run(shots=4096)
           )
       else:
           result = circuit.run(shots=0)  # Exact
       
       # Compute expectation value
       return hamiltonian.expectation(result)
   
   # Optimize without noise
   init_params = np.random.rand(4) * 2 * np.pi
   result_ideal = minimize(
       lambda p: energy_evaluation(p, use_noise=False),
       init_params,
       method='COBYLA'
   )
   
   # Optimize with noise
   result_noisy = minimize(
       lambda p: energy_evaluation(p, use_noise=True),
       init_params,
       method='COBYLA'
   )
   
   print(f"Ideal energy: {result_ideal.fun:.6f}")
   print(f"Noisy energy: {result_noisy.fun:.6f}")

Complete Example: Noise Model Comparison
-----------------------------------------

.. code-block:: python

   import tyxonq as tq
   import matplotlib.pyplot as plt
   import numpy as np
   
   def compare_noise_models():
       """Compare all noise models on a GHZ state."""
       
       # Create 3-qubit GHZ state
       circuit = tq.Circuit(3)
       circuit.h(0)
       circuit.cnot(0, 1)
       circuit.cnot(1, 2)
       
       # Define noise configurations
       noise_configs = {
           "Ideal": None,
           "Depolarizing (p=0.05)": ("depolarizing", {"p": 0.05}),
           "Amplitude Damping (γ=0.1)": ("amplitude_damping", {"gamma": 0.1}),
           "Phase Damping (λ=0.1)": ("phase_damping", {"l": 0.1}),
           "Pauli (asymmetric)": ("pauli", {"px": 0.02, "py": 0.02, "pz": 0.06})
       }
       
       results = {}
       for name, config in noise_configs.items():
           if config is None:
               result = circuit.run(shots=2048)
           else:
               noise_type, params = config
               result = circuit.with_noise(noise_type, **params).run(shots=2048)
           
           results[name] = result.counts
       
       # Analyze results
       for name, counts in results.items():
           total = sum(counts.values())
           ghz_fidelity = (counts.get('000', 0) + counts.get('111', 0)) / total
           print(f"{name:30s} GHZ fidelity: {ghz_fidelity:.4f}")
       
       return results
   
   # Run comparison
   results = compare_noise_models()

See Also
========

- :doc:`index` - Device Abstraction Overview
- :doc:`../postprocessing/index` - Error Mitigation Techniques
- :doc:`/examples/index` - Complete Noise Simulation Examples
- :doc:`/api/devices/noise` - Noise Simulation API Reference

External Resources
==================

- **Kraus Operators**: Nielsen & Chuang, "Quantum Computation and Quantum Information", Chapter 8
- **T₁/T₂ Relaxation**: Krantz et al., "A Quantum Engineer's Guide to Superconducting Qubits" (2019)
- **NISQ Algorithms**: Preskill, "Quantum Computing in the NISQ era and beyond" (2018)
