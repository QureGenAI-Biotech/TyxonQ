==========================
Quantum Chemistry Algorithms
==========================

TyxonQ implements a comprehensive suite of quantum chemistry algorithms for solving molecular electronic structure problems.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

TyxonQ supports the following major quantum chemistry algorithms:

🎯 **Variational Quantum Eigensolver (VQE)**
   Foundation algorithm for finding ground state energies

🧬 **UCCSD (Unitary Coupled Cluster Singles and Doubles)**
   High-accuracy ansatz based on coupled cluster theory

⚡ **HEA (Hardware Efficient Ansatz)**
   Shallow circuit design optimized for near-term quantum devices

🔧 **k-UpCCGSD (k-layer Unitary Pair Coupled Cluster)**
   Flexible multi-layer ansatz with tunable depth

⚛️ **PUCCD (Paired Unitary Coupled Cluster Doubles)**
   Specialized for strongly correlated electron pairs

Variational Quantum Eigensolver (VQE)
======================================

Algorithm Principle
-------------------

VQE solves for ground state energy using the variational principle:

.. math::

   E_0 = \min_{\theta} \langle \psi(\theta) | H | \psi(\theta) \rangle

.. code-block:: python

   from tyxonq.applications.chem import HEA
   
   # Create HEA algorithm
   hea = HEA(molecule=molecule, layers=2)
   result = hea.kernel(method="COBYLA")

UCCSD Algorithm
===============

Theoretical Foundation
----------------------

UCCSD (Unitary Coupled Cluster Singles and Doubles) is one of the most accurate ansatze for quantum chemistry, based on the classical coupled cluster method adapted for quantum computers.

**Ansatz Form**:

.. math::

   |\psi(\theta)\rangle = e^{T - T^\dagger} |\text{HF}\rangle

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   
   uccsd = UCCSD(molecule=molecule, init_method="mp2")
   result = uccsd.kernel(method="BFGS")

Hardware Efficient Ansatz (HEA)
================================

Design Philosophy
-----------------

HEA is designed specifically for near-term quantum devices (NISQ era), using a shallow circuit structure that:

- Minimizes gate count and circuit depth
- Uses native gates available on quantum hardware
- Provides good balance between expressivity and trainability

**Layer Structure**:

- **Rotation layer**: Single-qubit RY gates with trainable parameters
- **Entanglement layer**: CNOT gates connecting neighboring qubits
- **Repeating pattern**: Alternating rotation and entanglement layers

**Circuit Architecture**:

.. code-block:: python

   # Create HEA with 3 layers
   hea = HEA(
       molecule=molecule,
       layers=3,
       runtime="device"
   )
   
   # Run optimization
   energy = hea.kernel(method="COBYLA")
   print(f"Ground state energy: {energy:.6f} Hartree")

k-UpCCGSD Algorithm
===================

Overview
--------

k-UpCCGSD (k-layer Unitary Pair Coupled Cluster Generalized Singles and Doubles) provides a flexible framework that balances accuracy and computational cost through adjustable circuit depth.

**Key Features**:

- Tunable number of excitation layers (k parameter)
- Includes both generalized singles and doubles excitations
- Supports paired excitation patterns for efficiency
- Provides excellent flexibility for different molecular systems

.. code-block:: python

   from tyxonq.applications.chem import KUPCCGSD
   
   # Create k-UpCCGSD with 2 layers
   kupccgsd = KUPCCGSD(
       molecule=molecule,
       k_layers=2,
       init_method="mp2"
   )
   
   # Optimize
   energy = kupccgsd.kernel(method="BFGS")
   print(f"Optimized energy: {energy:.6f} Hartree")

PUCCD Algorithm
===============

Specialized for Strong Correlation
----------------------------------

PUCCD (Paired Unitary Coupled Cluster Doubles) focuses exclusively on paired double excitations, making it particularly effective for:

- **Strongly correlated systems**: Molecules with significant electron correlation
- **Dissociation curves**: Bond breaking scenarios
- **Superconducting materials**: Systems with pairing phenomena
- **Resource efficiency**: Reduces qubit count using hard-core boson mapping

**Hard-Core Boson (HCB) Mapping**:

PUCCD uses HCB mapping where each spatial orbital is represented by a single qubit representing the electron pair occupation, reducing qubit requirements by half compared to standard fermion mapping.

.. code-block:: python

   from tyxonq.applications.chem import PUCCD
   
   # Create PUCCD instance
   puccd = PUCCD(
       molecule=molecule,
       init_method="mp2"
   )
   
   # Run calculation
   energy = puccd.kernel()
   print(f"PUCCD energy: {energy:.6f} Hartree")
   
   # Calculate reduced density matrices
   rdm1 = puccd.make_rdm1(puccd.params)
   rdm2 = puccd.make_rdm2(puccd.params)

Algorithm Comparison
====================

.. list-table:: Algorithm Feature Comparison
   :header-rows: 1
   :widths: 18 18 18 18 28

   * - Algorithm
     - Parameters
     - Circuit Depth
     - Accuracy
     - Best Use Case
   * - **UCCSD**
     - O(N⁴)
     - Deep
     - Very High
     - Chemical accuracy for small molecules
   * - **HEA**
     - O(N·L)
     - Shallow
     - Medium
     - NISQ devices, quick prototyping
   * - **k-UpCCGSD**
     - Tunable
     - Medium
     - High
     - Flexible depth-accuracy tradeoff
   * - **PUCCD**
     - O(N²)
     - Medium
     - High
     - Strong correlation, reduced qubits

Execution Runtimes
==================

TyxonQ supports two execution modes for all quantum chemistry algorithms:

Device Runtime
--------------

Executes circuits on real quantum devices or high-fidelity simulators using shot-based measurements.

**Characteristics**:

- Circuit compilation and optimization
- Shot-based sampling (default: 2048 shots)
- Noise modeling (when available)
- Real device execution support

**When to use**:

- Testing algorithms on actual quantum hardware
- Studying the impact of noise and errors
- Validating circuit compilation strategies

Numeric Runtime
---------------

Uses exact numerical methods (statevector or tensor network simulation) for fast, noiseless calculations.

**Characteristics**:

- Exact expectation values (no shot noise)
- Fast gradient computation
- Efficient for small to medium systems
- Multiple backends: statevector, tensornetwork

**When to use**:

- Algorithm development and debugging
- Obtaining baseline results without noise
- Rapid prototyping and testing

**Example Usage**:

.. code-block:: python

   # Device runtime (with shots)
   hea_device = HEA(molecule=mol, runtime="device")
   energy_device = hea_device.kernel(shots=2048, provider="local")
   
   # Numeric runtime (exact)
   hea_numeric = HEA(molecule=mol, runtime="numeric", numeric_engine="statevector")
   energy_numeric = hea_numeric.kernel()
   
   print(f"Device energy: {energy_device:.6f} Hartree")
   print(f"Numeric energy: {energy_numeric:.6f} Hartree")

Best Practices
==============

Algorithm Selection
-------------------

**Choose UCCSD when**:

- Chemical accuracy (<1 kcal/mol) is required
- System size is small enough (≤20 qubits)
- Classical validation is needed

**Choose HEA when**:

- Working with NISQ devices with limited coherence
- Quick results are more important than high accuracy
- Circuit depth must be minimized

**Choose k-UpCCGSD when**:

- Need to balance accuracy and circuit resources
- Exploring different depth-accuracy tradeoffs
- System has moderate correlation

**Choose PUCCD when**:

- Strong electron correlation is present
- Qubit count is limited
- Studying bond dissociation or pairing phenomena

Parameter Initialization
------------------------

Proper initialization significantly improves convergence:

.. code-block:: python

   # Use MP2 initialization (recommended for closed-shell)
   uccsd = UCCSD(molecule, init_method="mp2")
   
   # Use CCSD initialization (higher accuracy start point)
   uccsd = UCCSD(molecule, init_method="ccsd")
   
   # Zero initialization (for open-shell or testing)
   uccsd = UCCSD(molecule, init_method="zeros")

**Initialization Guidelines**:

- **MP2**: Fast, good for weakly correlated systems
- **CCSD**: More accurate, slower, best for strongly correlated
- **Zeros**: Use for open-shell systems or when classical methods fail

Optimizer Configuration
-----------------------

.. code-block:: python

   # For gradient-based algorithms (UCCSD, k-UpCCGSD, PUCCD)
   uccsd.scipy_minimize_options = {
       "maxiter": 200,
       "gtol": 1e-6,
       "ftol": 1e-9
   }
   energy = uccsd.kernel(method="BFGS")
   
   # For gradient-free algorithms (HEA on noisy devices)
   hea.scipy_minimize_options = {
       "maxiter": 100,
       "rhobeg": 0.1
   }
   energy = hea.kernel(method="COBYLA")

**Optimizer Recommendations**:

- **L-BFGS-B**: Best for numeric runtime with exact gradients
- **COBYLA**: Robust for noisy gradients (device runtime)
- **SLSQP**: Good balance with constraints

Active Space Selection
----------------------

For large molecules, use active space approximation:

.. code-block:: python

   # Define active space: 4 electrons in 4 orbitals
   uccsd = UCCSD(
       molecule,
       active_space=(4, 4),
       init_method="mp2"
   )
   
   # Manually select active orbitals
   uccsd = UCCSD(
       molecule,
       active_space=(4, 4),
       active_orbital_indices=[2, 3, 4, 5],  # HOMO-1, HOMO, LUMO, LUMO+1
       init_method="mp2"
   )

Practical Example Workflow
---------------------------

.. code-block:: python

   from tyxonq.applications.chem import Molecule, UCCSD, HEA
   
   # 1. Define molecule
   h2o = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="sto-3g"
   )
   
   # 2. Quick estimation with HEA (numeric runtime)
   hea = HEA(molecule=h2o, layers=2, runtime="numeric")
   e_hea = hea.kernel()
   print(f"HEA energy: {e_hea:.6f} Hartree")
   
   # 3. High-accuracy calculation with UCCSD
   uccsd = UCCSD(molecule=h2o, init_method="mp2", runtime="numeric")
   e_uccsd = uccsd.kernel(method="BFGS")
   print(f"UCCSD energy: {e_uccsd:.6f} Hartree")
   
   # 4. Compare with classical references
   print(f"HF energy: {h2o.hf_energy:.6f} Hartree")
   print(f"Correlation energy (UCCSD): {e_uccsd - h2o.hf_energy:.6f} Hartree")

Related Resources
=================

- :doc:`../fundamentals/index` - Quantum Chemistry Fundamentals
- :doc:`../molecule/index` - Molecule Class Usage Guide
- :doc:`../runtimes/index` - Runtime System Details
- :doc:`/examples/chemistry_examples` - Quantum Chemistry Examples
- :doc:`/api/applications/index` - Applications API Reference
