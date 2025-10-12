========================================
Quantum Chemistry Examples
========================================

This page provides practical, executable examples for quantum chemistry calculations using TyxonQ.
Each example includes complete code, explanations, and expected outputs.

.. contents:: Contents
   :depth: 3
   :local:

.. note::
   All examples are tested and can be run directly. Make sure TyxonQ is properly installed.

Getting Started
===============

Example 1: Your First Molecule - H₂ Ground State
-------------------------------------------------

This is the simplest quantum chemistry calculation - finding the ground state energy of hydrogen molecule.

.. code-block:: python

   import tyxonq as tq
   from tyxonq.applications.chem import UCCSD
   from pyscf import gto

   # Define H2 molecule
   mol = gto.Mole()
   mol.atom = '''
   H 0.0 0.0 0.0
   H 0.0 0.0 0.74
   '''
   mol.basis = 'sto-3g'
   mol.build()

   # Run UCCSD calculation
   uccsd = UCCSD(molecule=mol, init_method="mp2")
   energy = uccsd.kernel()
   
   print(f"Ground state energy: {energy:.8f} Hartree")
   print(f"HF energy:  {uccsd.e_hf:.8f} Hartree")
   print(f"FCI energy: {uccsd.e_fci:.8f} Hartree")

**Expected Output**:

.. code-block:: text

   Ground state energy: -1.13727498 Hartree
   HF energy:  -1.11671398 Hartree
   FCI energy: -1.13727498 Hartree

**Explanation**:

- UCCSD provides chemical accuracy for small molecules
- The energy is lower than Hartree-Fock (HF), capturing electron correlation
- For H₂ with minimal basis, UCCSD achieves FCI (exact) accuracy

Example 2: Using Hardware Efficient Ansatz (HEA)
-------------------------------------------------

HEA is optimized for near-term quantum devices with shallow circuit depth.

.. code-block:: python

   from tyxonq.applications.chem import HEA
   from pyscf import gto

   # Define molecule
   mol = gto.Mole()
   mol.atom = 'H 0 0 0; H 0 0 0.74'
   mol.basis = 'sto-3g'
   mol.build()

   # Create HEA with 3 layers
   hea = HEA(molecule=mol, layers=3)
   
   # Optimize with COBYLA method
   energy = hea.kernel(method="COBYLA")
   
   print(f"Optimized energy: {energy:.8f} Hartree")
   print(f"Number of parameters: {hea.n_params}")
   print(f"Circuit depth: ~{3 * hea.n_qubits} gates")

**Key Features of HEA**:

- Shallow circuit: suitable for NISQ devices
- Fewer parameters: ``(layers + 1) × n_qubits``
- Fast optimization: COBYLA works well for noisy gradients

Molecule Construction
=====================

Example 3: Different Ways to Define Molecules
----------------------------------------------

**Method 1: Using PySCF Mole Object**

.. code-block:: python

   from pyscf import gto
   from tyxonq.applications.chem import UCCSD

   mol = gto.Mole()
   mol.atom = '''
   O 0.0 0.0 0.0
   H 0.0 0.757 0.587
   H 0.0 -0.757 0.587
   '''
   mol.basis = '6-31g'
   mol.charge = 0
   mol.spin = 0
   mol.build()

   uccsd = UCCSD(molecule=mol)

**Method 2: Direct Parameters**

.. code-block:: python

   from tyxonq.applications.chem import HEA

   # Pass atom specification directly
   hea = HEA(
       atom="H 0 0 0; H 0 0 0.74",
       basis="sto-3g",
       charge=0,
       spin=0,
       layers=2
   )

**Method 3: Using Pre-built Molecules**

.. code-block:: python

   # If you have predefined molecule configurations
   from pyscf import gto
   
   # LiH molecule
   mol_lih = gto.Mole()
   mol_lih.atom = 'Li 0 0 0; H 0 0 1.6'
   mol_lih.basis = 'sto-3g'
   mol_lih.build()

Algorithm Comparison
====================

Example 4: Comparing Different Ansatze
---------------------------------------

Let's compare UCCSD, HEA, and PUCCD on the same molecule.

.. code-block:: python

   from tyxonq.applications.chem import UCCSD, HEA, PUCCD
   from pyscf import gto
   import time

   # Define LiH molecule
   mol = gto.Mole()
   mol.atom = 'Li 0 0 0; H 0 0 1.6'
   mol.basis = 'sto-3g'
   mol.build()

   # UCCSD - High accuracy
   print("\n=== UCCSD ===")
   t0 = time.time()
   uccsd = UCCSD(molecule=mol, init_method="mp2")
   e_uccsd = uccsd.kernel()
   t_uccsd = time.time() - t0
   print(f"Energy: {e_uccsd:.8f} Hartree")
   print(f"Parameters: {uccsd.n_params}")
   print(f"Time: {t_uccsd:.2f}s")

   # HEA - Hardware efficient
   print("\n=== HEA ===")
   t0 = time.time()
   hea = HEA(molecule=mol, layers=3)
   e_hea = hea.kernel()
   t_hea = time.time() - t0
   print(f"Energy: {e_hea:.8f} Hartree")
   print(f"Parameters: {hea.n_params}")
   print(f"Time: {t_hea:.2f}s")

   # PUCCD - Paired electrons
   print("\n=== PUCCD ===")
   t0 = time.time()
   puccd = PUCCD(molecule=mol, init_method="mp2")
   e_puccd = puccd.kernel()
   t_puccd = time.time() - t0
   print(f"Energy: {e_puccd:.8f} Hartree")
   print(f"Parameters: {puccd.n_params}")
   print(f"Time: {t_puccd:.2f}s")

   # Comparison
   print("\n=== Comparison ===")
   print(f"FCI reference: {uccsd.e_fci:.8f} Hartree")
   print(f"UCCSD error: {abs(e_uccsd - uccsd.e_fci)*1000:.4f} mHartree")
   print(f"HEA error:   {abs(e_hea - uccsd.e_fci)*1000:.4f} mHartree")
   print(f"PUCCD error: {abs(e_puccd - uccsd.e_fci)*1000:.4f} mHartree")

**Expected Results**:

.. list-table:: Algorithm Comparison for LiH
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Algorithm
     - Energy (Hartree)
     - Parameters
     - Error (mH)
     - Best For
   * - UCCSD
     - -7.8823
     - ~20
     - <0.1
     - Chemical accuracy
   * - HEA
     - -7.8815
     - ~24
     - ~0.8
     - NISQ devices
   * - PUCCD
     - -7.8820
     - ~10
     - ~0.3
     - Paired systems

Active Space Calculations
=========================

Example 5: Using Active Space for Larger Molecules
---------------------------------------------------

For larger molecules, use active space to reduce computational cost.

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   from pyscf import gto

   # Define water molecule with larger basis
   mol = gto.Mole()
   mol.atom = '''
   O 0.0 0.0 0.0
   H 0.0 0.757 0.587
   H 0.0 -0.757 0.587
   '''
   mol.basis = '6-31g'
   mol.build()

   # Use active space: 4 electrons in 4 orbitals
   uccsd = UCCSD(
       molecule=mol,
       active_space=(4, 4),  # (n_electrons, n_orbitals)
       init_method="mp2"
   )
   
   energy = uccsd.kernel()
   print(f"Active space energy: {energy:.8f} Hartree")
   print(f"Number of qubits: {uccsd.n_qubits}")
   print(f"Full space would need: {mol.nelectron} qubits")

**Active Space Selection Tips**:

- Include frontier orbitals (HOMO/LUMO)
- For bond breaking: include bonding and antibonding orbitals
- Balance: larger space = more accurate, more expensive
- Common choices: (2,2), (4,4), (6,6) for small molecules

Example 6: Custom Orbital Selection
------------------------------------

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   from pyscf import gto

   mol = gto.Mole()
   mol.atom = 'N 0 0 0; N 0 0 1.1'
   mol.basis = 'cc-pvdz'
   mol.build()

   # Select specific orbitals (0-based indexing)
   # For N2: select π and π* orbitals
   uccsd = UCCSD(
       molecule=mol,
       active_space=(6, 6),
       active_orbital_indices=[4, 5, 6, 7, 8, 9],  # Manually select orbitals
       init_method="mp2"
   )
   
   energy = uccsd.kernel()
   print(f"Custom active space energy: {energy:.8f} Hartree")

Advanced Features
=================

Example 7: Energy and Gradient Calculation
-------------------------------------------

.. code-block:: python

   from tyxonq.applications.chem import HEA
   import numpy as np

   # Create HEA instance
   hea = HEA(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", layers=2)
   
   # Random parameters
   params = np.random.random(hea.n_params)
   
   # Calculate energy only
   energy = hea.energy(params)
   print(f"Energy: {energy:.8f} Hartree")
   
   # Calculate energy and gradient simultaneously
   energy, gradient = hea.energy_and_grad(params)
   print(f"Energy: {energy:.8f} Hartree")
   print(f"Gradient norm: {np.linalg.norm(gradient):.6f}")
   print(f"Gradient: {gradient}")

**Use Cases**:

- Custom optimizers requiring gradients
- Analyzing parameter landscape
- Implementing advanced optimization algorithms

Example 8: Accessing Quantum Circuit
-------------------------------------

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   import numpy as np

   # Create UCCSD instance
   uccsd = UCCSD(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
   
   # Get the parameterized circuit
   params = np.zeros(uccsd.n_params)  # Start from HF state
   circuit = uccsd.get_circuit(params)
   
   print(f"Number of qubits: {circuit.n}")
   print(f"Circuit operations: {len(circuit.ops)}")
   
   # Visualize circuit (if you want)
   # print(circuit)

Example 9: Reduced Density Matrices
------------------------------------

Access 1- and 2-electron reduced density matrices (RDMs).

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   import numpy as np

   # Run UCCSD calculation
   uccsd = UCCSD(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
   energy = uccsd.kernel()
   
   # Get optimized parameters
   params = uccsd.params
   
   # Calculate 1-RDM
   rdm1 = uccsd.make_rdm1(params)
   print("1-electron RDM shape:", rdm1.shape)
   print("1-RDM:\n", rdm1)
   
   # Calculate 2-RDM
   rdm2 = uccsd.make_rdm2(params)
   print("\n2-electron RDM shape:", rdm2.shape)
   
   # Verify electron number from RDM1
   n_elec_from_rdm = np.trace(rdm1)
   print(f"\nElectrons from RDM1: {n_elec_from_rdm:.6f}")
   print(f"Expected electrons: {uccsd.n_elec}")

**Applications of RDMs**:

- Calculate molecular properties (dipole moment, etc.)
- Analyze electron correlation
- Interface with other quantum chemistry tools
- Verify calculation correctness

Optimization Control
====================

Example 10: Custom Optimization Settings
-----------------------------------------

.. code-block:: python

   from tyxonq.applications.chem import HEA

   hea = HEA(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", layers=3)
   
   # Set custom scipy optimization options
   hea.scipy_minimize_options = {
       'maxiter': 500,
       'ftol': 1e-8,
       'gtol': 1e-6
   }
   
   # Run with specific optimizer
   energy = hea.kernel(method="L-BFGS-B")
   
   # Access optimization results
   print(f"Final energy: {energy:.8f} Hartree")
   print(f"Converged: {hea.opt_res['success']}")
   print(f"Iterations: {hea.opt_res['nit']}")
   print(f"Message: {hea.opt_res['message']}")

Example 11: Different Initialization Methods
---------------------------------------------

.. code-block:: python

   from tyxonq.applications.chem import UCCSD

   mol_config = {"atom": "H 0 0 0; H 0 0 0.74", "basis": "sto-3g"}
   
   # Method 1: MP2 initialization (default, recommended)
   uccsd_mp2 = UCCSD(**mol_config, init_method="mp2")
   e_mp2 = uccsd_mp2.kernel()
   
   # Method 2: CCSD initialization (more accurate initial guess)
   uccsd_ccsd = UCCSD(**mol_config, init_method="ccsd")
   e_ccsd = uccsd_ccsd.kernel()
   
   # Method 3: Zero initialization (start from HF state)
   uccsd_zeros = UCCSD(**mol_config, init_method="zeros")
   e_zeros = uccsd_zeros.kernel()
   
   print("Initialization comparison:")
   print(f"MP2 init:   {e_mp2:.8f} Hartree, {uccsd_mp2.opt_res['nit']} iterations")
   print(f"CCSD init:  {e_ccsd:.8f} Hartree, {uccsd_ccsd.opt_res['nit']} iterations")
   print(f"Zero init:  {e_zeros:.8f} Hartree, {uccsd_zeros.opt_res['nit']} iterations")

**Recommendation**: Use ``"mp2"`` for most cases - good balance of accuracy and speed.

Practical Applications
======================

Example 12: Potential Energy Surface Scan
------------------------------------------

Calculate energy at different bond lengths (dissociation curve).

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   from pyscf import gto
   import numpy as np
   import matplotlib.pyplot as plt

   # Scan H2 bond lengths
   distances = np.linspace(0.5, 3.0, 15)
   energies_uccsd = []
   energies_hf = []
   
   for d in distances:
       mol = gto.Mole()
       mol.atom = f'H 0 0 0; H 0 0 {d}'
       mol.basis = 'sto-3g'
       mol.build()
       
       uccsd = UCCSD(molecule=mol, init_method="mp2")
       e = uccsd.kernel()
       
       energies_uccsd.append(e)
       energies_hf.append(uccsd.e_hf)
       print(f"d = {d:.2f} Å: E = {e:.6f} Hartree")
   
   # Plot results
   plt.figure(figsize=(10, 6))
   plt.plot(distances, energies_hf, 'o-', label='Hartree-Fock', alpha=0.7)
   plt.plot(distances, energies_uccsd, 's-', label='UCCSD', alpha=0.7)
   plt.xlabel('H-H Distance (Å)', fontsize=12)
   plt.ylabel('Energy (Hartree)', fontsize=12)
   plt.title('H₂ Potential Energy Surface', fontsize=14)
   plt.legend(fontsize=11)
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('h2_pes.png', dpi=300)
   plt.show()

Example 13: Molecular Property Calculation
-------------------------------------------

Calculate dipole moment from RDMs.

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   from pyscf import gto
   import numpy as np

   # Water molecule
   mol = gto.Mole()
   mol.atom = '''
   O 0.0 0.0 0.0
   H 0.0 0.757 0.587
   H 0.0 -0.757 0.587
   '''
   mol.basis = 'sto-3g'
   mol.build()

   # Run UCCSD
   uccsd = UCCSD(molecule=mol, active_space=(4, 4))
   energy = uccsd.kernel()
   
   # Get 1-RDM
   rdm1 = uccsd.make_rdm1(uccsd.params)
   
   # Calculate dipole moment (requires molecular orbital integrals)
   # This is a simplified example
   print(f"Ground state energy: {energy:.8f} Hartree")
   print(f"1-RDM trace (electron number): {np.trace(rdm1):.6f}")

Troubleshooting
===============

Common Issues and Solutions
---------------------------

**Issue 1: Optimization doesn't converge**

.. code-block:: python

   # Solution: Increase max iterations and try different optimizer
   hea = HEA(atom="...", basis="sto-3g", layers=2)
   hea.scipy_minimize_options = {'maxiter': 1000}
   energy = hea.kernel(method="COBYLA")  # Try COBYLA for noisy landscapes

**Issue 2: Out of memory for large molecules**

.. code-block:: python

   # Solution: Use active space reduction
   uccsd = UCCSD(
       atom="...",
       basis="sto-3g",
       active_space=(6, 6),  # Reduce from full space
       init_method="mp2"
   )

**Issue 3: Energy higher than expected**

.. code-block:: python

   # Solution: Increase ansatz expressivity
   # For HEA: increase layers
   hea = HEA(atom="...", basis="sto-3g", layers=5)  # More layers
   
   # Or switch to more accurate ansatz
   uccsd = UCCSD(atom="...", basis="sto-3g")  # Higher accuracy

Performance Tips
----------------

1. **Start small**: Test with minimal basis (sto-3g) before using larger basis sets
2. **Use MP2 init**: Provides good starting point for optimization
3. **Active space**: Essential for molecules >4 atoms
4. **Algorithm choice**:
   
   - Quick testing: HEA with 2-3 layers
   - Chemical accuracy: UCCSD
   - Paired systems: PUCCD

5. **Optimization**: COBYLA for quick results, L-BFGS-B for accuracy

See Also
========

- :doc:`../quantum_chemistry/algorithms/index` - Algorithm theory and details
- :doc:`../quantum_chemistry/molecule/index` - Molecule construction
- :doc:`../quantum_chemistry/runtimes/index` - Runtime systems and devices
- :doc:`../api/applications/index` - Complete API reference
- :doc:`../tutorials/beginner/index` - Step-by-step tutorials

Next Steps
==========

After mastering these examples:

1. Explore :doc:`optimization_examples` for VQE and other optimization tasks
2. Try :doc:`cloud_examples` to run on real quantum hardware
3. Check :doc:`advanced_examples` for complex applications
4. Read :doc:`../quantum_chemistry/aidd/index` for drug discovery applications

.. tip::
   All code examples are available in the TyxonQ repository under ``examples/quantum_chemistry/``.
   You can download and run them directly!
