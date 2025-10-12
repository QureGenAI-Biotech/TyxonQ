===============
Molecule Module
===============

The Molecule module provides a comprehensive interface for defining molecular systems, computing molecular properties, and integrating with quantum chemistry algorithms.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

TyxonQ's Molecule class is built on top of PySCF (Python-based Simulations of Chemistry Framework), providing:

🧪 **Molecular Definition**
   Flexible ways to define molecular geometry and electronic structure

⚛️ **Quantum Properties**
   Automatic calculation of Hartree-Fock, MP2, and other reference energies

🔧 **Hamiltonian Generation**
   Convert molecular systems into qubit Hamiltonians for quantum algorithms

🔄 **PySCF Integration**
   Seamless integration with PySCF's extensive capabilities

Creating Molecules
==================

From Atomic Coordinates
-----------------------

The most straightforward way to define a molecule is by specifying atomic symbols and coordinates:

.. code-block:: python

   from tyxonq.applications.chem import Molecule
   
   # Water molecule (H2O)
   h2o = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="sto-3g",
       unit="Angstrom"
   )
   
   # Hydrogen molecule (H2)
   h2 = Molecule(
       atoms=[
           ["H", [0.0, 0.0, 0.0]],
           ["H", [0.0, 0.0, 0.74]]
       ],
       basis="6-31g"
   )

**Coordinate Units**:

- ``"Angstrom"``: Default, convenient for molecular structures
- ``"Bohr"``: Atomic units, used in quantum chemistry calculations

Predefined Molecules
--------------------

TyxonQ provides common molecules for quick testing:

.. code-block:: python

   from tyxonq.applications.chem.molecule import (
       h2, h2o, nh3, ch4, 
       h_chain, h_ring, water
   )
   
   # Use predefined H2 molecule
   my_h2 = h2
   print(f"HF energy: {my_h2.hf_energy}")
   
   # Create hydrogen chain with custom bond length
   h4_chain = h_chain(n_h=4, bond_distance=0.8)
   
   # Create water with custom geometry
   custom_water = water(bond_length=0.96, bond_angle=104.5, basis="cc-pvdz")

**Available Predefined Molecules**:

- **Diatomics**: ``h2``, ``n2``, ``co``, ``lih``
- **Polyatomics**: ``h2o``, ``nh3``, ``ch4``, ``bh3``, ``hcn``
- **Clusters**: ``h_chain``, ``h_ring``, ``h_square``, ``h_cube``

With Charge and Spin
--------------------

.. code-block:: python

   # H3+ ion (charge = +1)
   h3_plus = Molecule(
       atoms=[
           ["H", [0.0, 0.0, 0.0]],
           ["H", [0.0, 0.0, 0.8]],
           ["H", [0.0, 0.0, 1.6]]
       ],
       charge=1,
       spin=0,  # Singlet
       basis="sto-3g"
   )
   
   # Oxygen atom (triplet state)
   o_atom = Molecule(
       atoms=[["O", [0.0, 0.0, 0.0]]],
       charge=0,
       spin=2,  # Triplet (2 unpaired electrons)
       basis="6-31g"
   )

Basis Sets
==========

Common Basis Sets
-----------------

TyxonQ supports all PySCF basis sets:

.. list-table:: Recommended Basis Sets
   :header-rows: 1
   :widths: 20 30 25 25

   * - Basis Set
     - Type
     - Speed
     - Use Case
   * - ``sto-3g``
     - Minimal
     - Very Fast
     - Quick testing, qualitative results
   * - ``3-21g``
     - Small split-valence
     - Fast
     - Preliminary calculations
   * - ``6-31g``
     - Split-valence
     - Medium
     - General purpose
   * - ``6-31g*``
     - Polarized
     - Medium
     - Better accuracy
   * - ``cc-pvdz``
     - Correlation-consistent
     - Slow
     - High accuracy
   * - ``cc-pvtz``
     - Correlation-consistent
     - Very Slow
     - Production calculations

.. code-block:: python

   # Minimal basis - fastest
   mol_minimal = Molecule(atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]], basis="sto-3g")
   
   # Polarized basis - better accuracy
   mol_polarized = Molecule(atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]], basis="6-31g*")
   
   # Correlation-consistent - high accuracy
   mol_ccpvdz = Molecule(atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]], basis="cc-pvdz")

Molecular Properties
====================

Basic Properties
----------------

.. code-block:: python

   h2o = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="sto-3g"
   )
   
   # Basic electronic properties
   print(f"Number of electrons: {h2o.n_electrons}")
   print(f"Number of orbitals: {h2o.n_orbitals}")
   print(f"Number of spin orbitals: {h2o.n_spin_orbitals}")
   print(f"Number of qubits required: {h2o.n_qubits}")
   
   # Molecular symmetry
   print(f"Point group: {h2o.symmetry}")
   print(f"Nuclear repulsion energy: {h2o.nuclear_repulsion:.6f} Hartree")

Energies and Reference Calculations
-----------------------------------

.. code-block:: python

   # Hartree-Fock energy (automatically computed)
   print(f"HF energy: {h2o.hf_energy:.6f} Hartree")
   
   # MP2 correlation energy
   print(f"MP2 energy: {h2o.mp2_energy:.6f} Hartree")
   print(f"MP2 correlation: {h2o.mp2_energy - h2o.hf_energy:.6f} Hartree")
   
   # CCSD energy (if available)
   if hasattr(h2o, 'ccsd_energy'):
       print(f"CCSD energy: {h2o.ccsd_energy:.6f} Hartree")
   
   # FCI energy (exact within basis set)
   if hasattr(h2o, 'fci_energy'):
       print(f"FCI energy: {h2o.fci_energy:.6f} Hartree")

Orbital Information
-------------------

.. code-block:: python

   # Molecular orbital coefficients
   mo_coeffs = h2o.mo_coefficients
   print(f"MO coefficient shape: {mo_coeffs.shape}")
   
   # Orbital energies
   mo_energies = h2o.mo_energies
   print(f"HOMO energy: {mo_energies[h2o.n_electrons//2 - 1]:.6f} Hartree")
   print(f"LUMO energy: {mo_energies[h2o.n_electrons//2]:.6f} Hartree")
   
   # HOMO-LUMO gap
   print(f"HOMO-LUMO gap: {h2o.homo_lumo_gap:.6f} Hartree")
   print(f"HOMO-LUMO gap: {h2o.homo_lumo_gap * 27.2114:.2f} eV")

Hamiltonian Generation
======================

Qubit Hamiltonians
------------------

Convert molecular Hamiltonian to qubit operators for quantum algorithms:

.. code-block:: python

   # Jordan-Wigner mapping
   ham_jw = h2o.get_hamiltonian(mapping="jordan_wigner")
   print(f"JW Hamiltonian terms: {len(ham_jw)}")
   
   # Bravyi-Kitaev mapping
   ham_bk = h2o.get_hamiltonian(mapping="bravyi_kitaev")
   print(f"BK Hamiltonian terms: {len(ham_bk)}")
   
   # Parity mapping (reduces qubits by 2)
   ham_parity = h2o.get_hamiltonian(mapping="parity")
   print(f"Parity Hamiltonian terms: {len(ham_parity)}")
   
   # Examine Hamiltonian terms
   for coeff, pauli_string in ham_jw[:5]:
       print(f"{coeff:.6f} * {pauli_string}")

**Mapping Comparison**:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Mapping
     - Qubits
     - Gate Count
     - Best For
   * - Jordan-Wigner
     - 2N
     - Medium
     - General purpose, intuitive
   * - Bravyi-Kitaev
     - 2N
     - Lower
     - Circuit optimization
   * - Parity
     - 2N-2
     - Lowest
     - Qubit-limited systems

Active Space Approximation
--------------------------

For large molecules, use active space to reduce computational cost:

.. code-block:: python

   # Large molecule example
   from tyxonq.applications.chem import Molecule
   
   benzene = Molecule(
       atoms=[
           ["C", [0.0, 1.398, 0.0]],
           ["C", [1.210, 0.699, 0.0]],
           ["C", [1.210, -0.699, 0.0]],
           ["C", [0.0, -1.398, 0.0]],
           ["C", [-1.210, -0.699, 0.0]],
           ["C", [-1.210, 0.699, 0.0]],
           # ... hydrogens
       ],
       basis="sto-3g"
   )
   
   # Use UCCSD with active space
   from tyxonq.applications.chem import UCCSD
   
   uccsd = UCCSD(
       molecule=benzene,
       active_space=(6, 6),  # 6 electrons in 6 orbitals
       init_method="mp2"
   )

Integration with Algorithms
============================

Direct Usage with VQE
---------------------

.. code-block:: python

   from tyxonq.applications.chem import Molecule, HEA, UCCSD
   
   # Define molecule
   h2 = Molecule(
       atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]],
       basis="sto-3g"
   )
   
   # Use with HEA
   hea = HEA(molecule=h2, layers=2, runtime="numeric")
   e_hea = hea.kernel()
   
   # Use with UCCSD
   uccsd = UCCSD(molecule=h2, init_method="mp2", runtime="numeric")
   e_uccsd = uccsd.kernel()
   
   print(f"HF energy:    {h2.hf_energy:.6f} Hartree")
   print(f"HEA energy:   {e_hea:.6f} Hartree")
   print(f"UCCSD energy: {e_uccsd:.6f} Hartree")

Potential Energy Surfaces
-------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   
   # Scan H2 dissociation curve
   distances = np.linspace(0.4, 3.0, 15)
   energies = []
   
   for d in distances:
       h2 = Molecule(
           atoms=[["H", [0, 0, 0]], ["H", [0, 0, d]]],
           basis="6-31g"
       )
       
       # Use VQE to compute energy
       hea = HEA(molecule=h2, layers=2, runtime="numeric")
       energy = hea.kernel()
       energies.append(energy)
   
   # Plot potential energy curve
   plt.figure(figsize=(10, 6))
   plt.plot(distances, energies, 'o-', linewidth=2)
   plt.xlabel('H-H Distance (Angstrom)', fontsize=12)
   plt.ylabel('Energy (Hartree)', fontsize=12)
   plt.title('H2 Potential Energy Curve (VQE)', fontsize=14)
   plt.grid(True, alpha=0.3)
   plt.show()

Advanced Topics
===============

Custom Integrals
----------------

For advanced users, you can work directly with molecular integrals:

.. code-block:: python

   # Access one-electron integrals
   int1e = h2o.int1e  # Kinetic + nuclear attraction
   print(f"One-electron integral shape: {int1e.shape}")
   
   # Access two-electron integrals
   int2e = h2o.int2e  # Electron-electron repulsion
   print(f"Two-electron integral shape: {int2e.shape}")
   
   # Core energy (nuclear repulsion)
   e_core = h2o.e_core
   print(f"Core energy: {e_core:.6f} Hartree")

Cloud Execution
---------------

For large molecules or expensive basis sets, use cloud HF/MP2 calculations:

.. code-block:: python

   # Local execution (default)
   mol_local = Molecule(
       atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]],
       basis="cc-pvdz",
       classical_provider="local"
   )
   
   # Cloud execution for heavy calculations
   large_mol = Molecule(
       atoms=large_molecular_structure,
       basis="cc-pvtz",
       classical_provider="cloud",  # Use cloud resources
       classical_device="hf_server"
   )

Common Workflows
================

Quick Energy Calculation
------------------------

.. code-block:: python

   from tyxonq.applications.chem import Molecule, UCCSD
   
   # Step 1: Define molecule
   mol = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="6-31g"
   )
   
   # Step 2: Check HF baseline
   print(f"HF energy: {mol.hf_energy:.6f} Hartree")
   
   # Step 3: Run VQE-UCCSD
   uccsd = UCCSD(molecule=mol, init_method="mp2", runtime="numeric")
   vqe_energy = uccsd.kernel()
   
   # Step 4: Compare results
   print(f"VQE energy: {vqe_energy:.6f} Hartree")
   print(f"Correlation energy: {vqe_energy - mol.hf_energy:.6f} Hartree")

Error Handling
==============

.. code-block:: python

   from tyxonq.applications.chem import Molecule
   
   try:
       # Invalid geometry (atoms too close)
       bad_mol = Molecule(
           atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.01]]],  # Too close!
           basis="sto-3g"
       )
   except Exception as e:
       print(f"Error: {e}")
   
   try:
       # Invalid basis set
       bad_basis = Molecule(
           atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]],
           basis="invalid-basis"
       )
   except ValueError as e:
       print(f"Basis set error: {e}")
   
   try:
       # Invalid charge/spin combination
       bad_spin = Molecule(
           atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]],
           charge=0,
           spin=1  # Odd spin with even electrons!
       )
   except ValueError as e:
       print(f"Spin error: {e}")

Related Resources
=================

- :doc:`../fundamentals/index` - Quantum Chemistry Fundamentals
- :doc:`../algorithms/index` - Quantum Chemistry Algorithms
- :doc:`/api/applications/index` - Molecule API Reference
- :doc:`/examples/chemistry_examples` - Practical Examples
