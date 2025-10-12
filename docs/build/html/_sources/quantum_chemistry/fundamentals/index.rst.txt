=========================
Quantum Chemistry Fundamentals
=========================

Welcome to quantum chemistry fundamentals with TyxonQ! This page introduces core concepts in quantum chemistry and their implementation in quantum computing.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

Quantum chemistry is one of the most important application domains for quantum computing. TyxonQ provides a complete quantum chemistry module supporting the entire workflow from molecular modeling to algorithm implementation:

🧪 **Molecular Hamiltonians**
   Quantum mechanical description of electronic structure problems

⚛️ **Second Quantization**
   Algebraic representation of fermionic systems

🔄 **Fermion-to-Qubit Mappings**
   Transform fermionic operators into qubit operations

📊 **Variational Methods**
   Theoretical foundation for VQE and related quantum algorithms

Electronic Structure Problem
============================

Molecular Hamiltonian
---------------------

Under the Born-Oppenheimer approximation, the electronic Hamiltonian of a molecule can be written as:

.. math::

   H = -\\sum_i \\frac{1}{2}\\nabla_i^2 - \\sum_{i,A} \\frac{Z_A}{r_{iA}} + \\sum_{i<j} \\frac{1}{r_{ij}} + \\sum_{A<B} \\frac{Z_A Z_B}{R_{AB}}

**Physical Meaning of Each Term**:

- **First term**: Electronic kinetic energy
- **Second term**: Electron-nuclear attraction potential
- **Third term**: Electron-electron repulsion potential
- **Fourth term**: Nuclear-nuclear repulsion (constant term)

Born-Oppenheimer Approximation
------------------------------

The Born-Oppenheimer approximation assumes that nuclei are stationary relative to electrons, allowing separation of the molecular Hamiltonian into electronic and nuclear motion parts:

.. math::

   \\Psi_{total}(r,R) \\approx \\psi_{elec}(r;R) \\cdot \\chi_{nuclear}(R)

This approximation is justified by:

- Electron mass << Nuclear mass
- Electron velocity >> Nuclear velocity

**Application in TyxonQ**:

.. code-block:: python

   from tyxonq.applications.chem import Molecule
   
   # Define molecular geometry (fixed nuclear coordinates)
   h2o = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="sto-3g"
   )
   
   print(f"Total electrons: {h2o.n_electrons}")
   print(f"HF energy: {h2o.hf_energy:.6f} Hartree")

Basis Set Theory
----------------

In quantum chemistry calculations, molecular orbitals are represented as linear combinations of basis functions:

.. math::

   \\psi_i = \\sum_{\\mu} c_{\\mu i} \\phi_\\mu

**Common Basis Set Types**:

.. list-table:: Basis Set Comparison
   :header-rows: 1
   :widths: 20 30 25 25

   * - Basis Set
     - Description
     - Computational Cost
     - Accuracy
   * - STO-3G
     - Minimal basis
     - Very Low
     - Qualitative
   * - 6-31G
     - Split-valence basis
     - Medium
     - Quantitative
   * - cc-pVDZ
     - Correlation-consistent basis
     - High
     - High accuracy
   * - aug-cc-pVTZ
     - Augmented correlation-consistent
     - Very High
     - Very high accuracy

.. code-block:: python

   # Compare accuracy of different basis sets
   molecules = {
       "sto-3g": Molecule(atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]], basis="sto-3g"),
       "6-31g": Molecule(atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]], basis="6-31g"),
       "cc-pvdz": Molecule(atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]], basis="cc-pvdz")
   }
   
   for basis, mol in molecules.items():
       print(f"{basis}: HF energy = {mol.hf_energy:.6f} Hartree")

Second Quantization
===================

Fermionic Operators
-------------------

In the second quantization representation, electrons are described using creation and annihilation operators:

- $a_p^\\dagger$: Creates an electron in orbital p
- $a_p$: Annihilates an electron in orbital p

**Anticommutation Relations**:

.. math::

   \\{a_p^\\dagger, a_q\\} = \\delta_{pq}
   
   \\{a_p^\\dagger, a_q^\\dagger\\} = 0
   
   \\{a_p, a_q\\} = 0

Electronic Hamiltonian
----------------------

The electronic Hamiltonian in second quantized form:

.. math::

   H = \\sum_{pq} h_{pq} a_p^\\dagger a_q + \\frac{1}{2}\\sum_{pqrs} h_{pqrs} a_p^\\dagger a_q^\\dagger a_s a_r

where:

- $h_{pq}$: One-electron integrals (kinetic energy + nuclear attraction)
- $h_{pqrs}$: Two-electron integrals (electron repulsion)

**Integral Calculations**:

.. math::

   h_{pq} = \\int \\phi_p^*(r) \\left(-\\frac{1}{2}\\nabla^2 - \\sum_A \\frac{Z_A}{|r-R_A|}\\right) \\phi_q(r) dr
   
   h_{pqrs} = \\int\\int \\phi_p^*(r_1)\\phi_q^*(r_2) \\frac{1}{|r_1-r_2|} \\phi_r(r_1)\\phi_s(r_2) dr_1 dr_2

Fermion-to-Qubit Mappings
=========================

Jordan-Wigner Transformation
----------------------------

The Jordan-Wigner transformation is the most intuitive fermion-to-qubit mapping:

.. math::

   a_j^\\dagger = \\left(\\bigotimes_{k=0}^{j-1} Z_k\\right) \\otimes \\sigma_j^+
   
   a_j = \\left(\\bigotimes_{k=0}^{j-1} Z_k\\right) \\otimes \\sigma_j^-

where $\\sigma^+ = (X + iY)/2$ and $\\sigma^- = (X - iY)/2$.

**TyxonQ Implementation**:

.. code-block:: python

   # Get Hamiltonian with Jordan-Wigner mapping
   hamiltonian = h2o.get_hamiltonian(mapping="jordan_wigner")
   
   print(f"Number of Hamiltonian terms: {len(hamiltonian)}")
   print(f"Required qubits: {h2o.n_qubits}")
   
   # Display first few terms
   for i, (coeff, pauli_string) in enumerate(hamiltonian[:3]):
       print(f"Term {i+1}: {coeff:.6f} * {pauli_string}")

Bravyi-Kitaev Transformation
----------------------------

The Bravyi-Kitaev transformation reduces the number of quantum gates through a binary tree structure:

.. math::

   a_j^\\dagger = \\frac{1}{2}\\left(\\bigotimes_{k \\in P(j)} Z_k\\right) \\otimes \\left(\\bigotimes_{k \\in Q(j)} X_k\\right) \\otimes (X_j - iY_j)

where P(j) and Q(j) are index sets defined based on binary representation.

**Mapping Comparison**:

.. code-block:: python

   # Compare different mapping methods
   mappings = ['jordan_wigner', 'bravyi_kitaev', 'parity']
   
   for mapping in mappings:
       ham = h2o.get_hamiltonian(mapping=mapping)
       print(f"{mapping}: {len(ham)} terms")

Parity Transformation
---------------------

The Parity transformation is based on parity encoding:

.. math::

   |n_0, n_1, \\ldots, n_{N-1}\\rangle \\rightarrow |p_0, p_1, \\ldots, p_{N-1}\\rangle

where $p_j = n_0 \\oplus n_1 \\oplus \\cdots \\oplus n_j$.

Variational Methods Fundamentals
=================================

Variational Principle
---------------------

The variational principle in quantum mechanics states that for any trial wavefunction $|\\psi\\rangle$:

.. math::

   E_0 \\leq \\frac{\\langle\\psi|H|\\psi\\rangle}{\\langle\\psi|\\psi\\rangle}

This is the theoretical foundation of the VQE algorithm.

**VQE Algorithm Workflow**:

.. mermaid::

   graph TD
       A[Initialize parameters θ] --> B[Construct ansatz |ψ(θ)⟩]
       B --> C[Measure energy expectation ⟨H⟩]
       C --> D[Classical optimizer updates θ]
       D --> E{Converged?}
       E -->|No| B
       E -->|Yes| F[Output ground state energy]

Ansatz Circuit Design
---------------------

**Hardware Efficient Ansatz (HEA)**:

.. code-block:: python

   from tyxonq.applications.chem import HEA
   
   # Create HEA algorithm instance
   hea = HEA(
       molecule=h2o,
       layers=3,
       runtime="numeric"
   )
   
   print(f"HEA parameter count: {hea.n_params}")
   print(f"Estimated circuit depth: {3 * h2o.n_qubits * 2}")

**UCCSD Ansatz**:

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   
   # Create UCCSD algorithm instance
   uccsd = UCCSD(
       molecule=h2o,
       init_method="mp2"
   )
   
   print(f"UCCSD parameter count: {uccsd.n_params}")
   print(f"Number of excitation operators: {len(uccsd.excitation_ops)}")

Mean-Field Methods
==================

Hartree-Fock Theory
-------------------

The Hartree-Fock method is the most fundamental mean-field approximation:

.. math::

   |\\Psi_{HF}\\rangle = |\\phi_1 \\phi_2 \\cdots \\phi_N\\rangle

**Self-Consistent Field Equations**:

.. math::

   F|\\phi_i\\rangle = \\epsilon_i|\\phi_i\\rangle

where the Fock operator is $F = h + \\sum_j (J_j - K_j)$.

**HF Calculations in TyxonQ**:

.. code-block:: python

   # Get HF results
   print(f"HF energy: {h2o.hf_energy:.6f} Hartree")
   print(f"HOMO-LUMO gap: {h2o.homo_lumo_gap:.6f} Hartree")
   
   # Get molecular orbital coefficients
   mo_coeffs = h2o.mo_coefficients
   print(f"Number of molecular orbitals: {mo_coeffs.shape}")

Electron Correlation
--------------------

The electron correlation energy is defined as the difference between the exact ground state energy and the HF energy:

.. math::

   E_{corr} = E_{exact} - E_{HF}

**Comparison of Correlation Methods**:

.. list-table:: Accuracy Comparison of Correlation Methods
   :header-rows: 1
   :widths: 25 25 25 25

   * - Method
     - Computational Complexity
     - Correlation Energy Recovery
     - Application Range
   * - MP2
     - O(N^5)
     - ~80-90%
     - Weakly correlated systems
   * - CCSD
     - O(N^6)
     - ~95%
     - Moderately correlated systems
   * - CCSD(T)
     - O(N^7)
     - ~99%
     - Strongly correlated systems
   * - VQE-UCCSD
     - Polynomial (quantum)
     - ~95%
     - Quantum algorithms

Practical Application Examples
==============================

Hydrogen Molecule Dissociation
------------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   
   # Scan H2 bond length
   distances = np.linspace(0.5, 3.0, 11)
   hf_energies = []
   vqe_energies = []
   
   for d in distances:
       h2 = Molecule(
           atoms=[["H", [0, 0, 0]], ["H", [0, 0, d]]],
           basis="sto-3g"
       )
       
       hf_energies.append(h2.hf_energy)
       
       # VQE calculation
       hea = HEA(molecule=h2, layers=2)
       vqe_energy = hea.kernel(method="COBYLA")
       vqe_energies.append(vqe_energy)
   
   # Plot potential energy curve
   plt.figure(figsize=(10, 6))
   plt.plot(distances, hf_energies, 'o-', label='HF')
   plt.plot(distances, vqe_energies, 's-', label='VQE-HEA')
   plt.xlabel('Bond Length (Bohr)')
   plt.ylabel('Energy (Hartree)')
   plt.legend()
   plt.title('H2 Potential Energy Curve')
   plt.grid(True)
   plt.show()

Water Molecule Ground State
---------------------------

.. code-block:: python

   # VQE calculation for water molecule
   from tyxonq.applications.chem import UCCSD
   
   h2o = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="6-31g"
   )
   
   # Use UCCSD method
   uccsd = UCCSD(molecule=h2o, init_method="mp2")
   
   print(f"Molecular Information:")
   print(f"  Electrons: {h2o.n_electrons}")
   print(f"  Qubits: {h2o.n_qubits}")
   print(f"  HF energy: {h2o.hf_energy:.6f} Hartree")
   
   # Execute VQE optimization
   vqe_energy = uccsd.kernel(method="BFGS")
   
   print(f"\\nVQE Results:")
   print(f"  VQE energy: {vqe_energy:.6f} Hartree")
   print(f"  Correlation energy: {vqe_energy - h2o.hf_energy:.6f} Hartree")

Related Resources
=================

- :doc:`../algorithms/index` - Quantum Chemistry Algorithms
- :doc:`../molecule/index` - Molecule Class Guide
- :doc:`/api/applications/index` - Applications API Reference
- :doc:`/examples/chemistry_examples` - Quantum Chemistry Examples
