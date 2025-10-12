=========================================
Your First Quantum Chemistry Calculation
=========================================

Welcome to the world of quantum chemistry! This guide will walk you through performing your first quantum chemistry calculation using TyxonQ's VQE (Variational Quantum Eigensolver).

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

Quantum chemistry is one of the most important application areas for quantum computing. TyxonQ provides a complete quantum chemistry module, including:

- **Molecular modeling**: Using the :class:`Molecule` class to define molecular structures
- **Hamiltonian construction**: Automatic generation of quantum Hamiltonians
- **VQE algorithms**: Including UCCSD, HEA and other ansatzes
- **Variational optimization**: Integration with various optimizers

**Key Concepts**

- **VQE algorithm**: Solving for molecular ground state energy through variational principle
- **Ansatz circuits**: Parameterized quantum circuits for preparing trial wavefunctions
- **Hamiltonian**: Quantum operator describing molecular electronic structure
- **Fermion-to-qubit mapping**: Converting fermionic operators to qubit operators

Molecular Setup and Modeling
=============================

Creating Molecule Objects
--------------------------

Use TyxonQ's :class:`Molecule` class to define molecules:

.. code-block:: python

   from tyxonq.applications.chem import Molecule
   
   # Define hydrogen molecule (H2)
   h2_molecule = Molecule(
       atoms=[
           ["H", [0.0, 0.0, 0.0]],      # First hydrogen atom
           ["H", [0.0, 0.0, 0.74]]      # Second hydrogen atom, distance 0.74 Angstrom
       ],
       basis="sto-3g",                   # Use STO-3G basis set
       charge=0,                         # Neutral molecule
       spin=0                            # Singlet state
   )
   
   print(f"Molecular charge: {h2_molecule.charge}")
   print(f"Spin: {h2_molecule.spin}")
   print(f"Number of electrons: {h2_molecule.n_electrons}")

Complex Molecule Example
-------------------------

Creating a water molecule (H2O):

.. code-block:: python

   # Water molecule with optimized geometry
   h2o_molecule = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="6-31g",                    # Use more accurate basis set
       charge=0,
       spin=0
   )
   
   print(f"H2O electrons: {h2o_molecule.n_electrons}")
   print(f"Orbitals: {h2o_molecule.n_orbitals}")

Hamiltonian Construction
========================

Electronic Hamiltonian
----------------------

In second quantization representation, the general form of the molecular Hamiltonian is:

.. math::

   H = \sum_{ij} h_{ij} a_i^\dagger a_j + \frac{1}{2}\sum_{ijkl} h_{ijkl} a_i^\dagger a_j^\dagger a_k a_l

Where:
- First term: one-electron terms (kinetic energy + nuclear attraction)
- Second term: two-electron terms (electron repulsion)

Fermion-to-Qubit Mapping
-------------------------

Convert the fermionic Hamiltonian to a qubit Hamiltonian:

.. code-block:: python

   # Get Hamiltonian (automatically uses Jordan-Wigner transformation)
   hamiltonian = h2_molecule.get_hamiltonian()
   
   print(f"Number of Hamiltonian terms: {len(hamiltonian)}")
   print(f"Required qubits: {h2_molecule.n_qubits}")
   
   # View the first term of the Hamiltonian
   if hamiltonian:
       coeff, pauli_string = hamiltonian[0]
       print(f"First term: {coeff:.6f} * {pauli_string}")

VQE Algorithm Basics
====================

Variational Principle
---------------------

VQE is based on the variational principle, finding the ground state energy by optimizing a parameterized quantum state:

.. math::

   E_0 = \min_{\theta} \langle \psi(\theta) | H | \psi(\theta) \rangle

Where $|\psi(\theta)\rangle$ is the trial wavefunction prepared by the ansatz circuit.

Running VQE Calculations
========================

Using HEA Ansatz
----------------

Hardware Efficient Ansatz (HEA) is ideal for near-term quantum devices:

.. code-block:: python

   from tyxonq.applications.chem import HEA
   
   # Create HEA algorithm instance
   hea = HEA(
       molecule=h2_molecule,
       layers=2,                      # Number of ansatz layers
       runtime="numeric",             # Use numeric backend
       mapping="jordan_wigner"        # Mapping method
   )
   
   print(f"HEA parameter count: {hea.n_params}")
   print(f"Qubits used: {hea.n_qubits}")

Executing VQE Optimization
---------------------------

.. code-block:: python

   # Run VQE optimization
   result = hea.kernel(
       method="COBYLA",               # Optimization algorithm
       options={
           "maxiter": 100,            # Maximum iterations
           "disp": True               # Display optimization progress
       }
   )
   
   print(f"VQE energy: {result:.6f} Hartree")
   print(f"HF energy: {h2_molecule.hf_energy:.6f} Hartree")
   print(f"Correlation energy: {result - h2_molecule.hf_energy:.6f} Hartree")

Complete Example: H2 Molecule VQE Calculation
==============================================

Here's a complete H2 molecule VQE calculation example:

.. code-block:: python

   import tyxonq as tq
   from tyxonq.applications.chem import Molecule, HEA
   import numpy as np
   
   def h2_vqe_calculation():
       """Complete H2 molecule VQE calculation example"""
       
       # Step 1: Define molecule
       print("Step 1: Creating hydrogen molecule...")
       h2 = Molecule(
           atoms=[
               ["H", [0.0, 0.0, 0.0]],
               ["H", [0.0, 0.0, 0.74]]
           ],
           basis="sto-3g",
           charge=0,
           spin=0
       )
       
       print(f"Molecular information:")
       print(f"  Electrons: {h2.n_electrons}")
       print(f"  Qubits: {h2.n_qubits}")
       print(f"  HF energy: {h2.hf_energy:.6f} Hartree")
       
       # Step 2: Create HEA algorithm
       print("\nStep 2: Setting up HEA algorithm...")
       hea = HEA(
           molecule=h2,
           layers=2,
           runtime="numeric"
       )
       
       print(f"HEA configuration:")
       print(f"  Layers: {hea.layers}")
       print(f"  Parameters: {hea.n_params}")
       
       # Step 3: Execute VQE optimization
       print("\nStep 3: Executing VQE optimization...")
       vqe_energy = hea.kernel(
           method="COBYLA",
           options={"maxiter": 100, "disp": False}
       )
       
       # Step 4: Result analysis
       print("\nResult analysis:")
       print(f"VQE energy: {vqe_energy:.6f} Hartree")
       print(f"HF energy:  {h2.hf_energy:.6f} Hartree")
       print(f"Correlation energy: {vqe_energy - h2.hf_energy:.6f} Hartree")
       
       return vqe_energy
   
   # Run calculation
   if __name__ == "__main__":
       result = h2_vqe_calculation()

Using UCCSD Ansatz
==================

Unitary Coupled Cluster with Singles and Doubles (UCCSD) provides more chemically accurate results:

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   
   # Create UCCSD algorithm instance
   uccsd = UCCSD(
       molecule=h2_molecule,
       runtime="numeric"
   )
   
   # Run UCCSD-VQE
   uccsd_energy = uccsd.kernel(method="BFGS")
   
   print(f"UCCSD-VQE energy: {uccsd_energy:.6f} Hartree")

Advanced Topics
===============

Custom Optimizers
-----------------

You can use custom optimizers from SciPy or other libraries:

.. code-block:: python

   from scipy.optimize import minimize
   
   # Define custom optimizer
   result = hea.kernel(
       method="SLSQP",
       options={
           "maxiter": 200,
           "ftol": 1e-6
       }
   )

Energy Landscape Analysis
--------------------------

Analyze the energy landscape:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Sample energy landscape
   angles = np.linspace(0, 2*np.pi, 50)
   energies = []
   
   for angle in angles:
       params = np.array([angle] * hea.n_params)
       energy = hea.evaluate_energy(params)
       energies.append(energy)
   
   # Plot
   plt.plot(angles, energies)
   plt.xlabel('Parameter value')
   plt.ylabel('Energy (Hartree)')
   plt.title('Energy Landscape')
   plt.show()

Next Steps
==========

Congratulations! You've successfully completed your first quantum chemistry calculation. Next, we recommend:

- :doc:`../quantum_chemistry/algorithms/index` - Learn more algorithms
- :doc:`../quantum_chemistry/fundamentals/index` - Deep dive into theory
- :doc:`../examples/chemistry_examples` - View more chemistry examples
- :doc:`../user_guide/numerics/index` - Learn about numeric backends

FAQ
===

**Q: How do I choose an appropriate basis set?**

A: Basis set selection affects calculation accuracy and efficiency:
- STO-3G: Minimal basis set, suitable for quick testing
- 6-31G: Balance between accuracy and efficiency
- cc-pVDZ: High-accuracy calculations

**Q: What if VQE doesn't converge?**

A: Try the following approaches:
- Increase maximum iteration count
- Change optimizer (e.g., BFGS, SLSQP)
- Adjust ansatz layer count
- Use better initial parameters

**Q: How do I estimate computation time?**

A: Computation time mainly depends on:
- Molecule size (number of qubits)
- Ansatz complexity (parameter count)
- Number of optimization iterations
- Backend selection (numeric vs device)

**Q: Can I run calculations on real quantum hardware?**

A: Yes, TyxonQ supports multiple quantum hardware providers. Set ``runtime="device"`` and configure the provider accordingly. Note that real hardware has limitations in qubit count and gate fidelity.

Related Resources
=================

- :doc:`/api/applications/index` - Applications API reference
- :doc:`/quantum_chemistry/index` - Quantum chemistry module
- :doc:`/examples/chemistry_examples` - Chemistry examples
- :doc:`/user_guide/postprocessing/index` - Result analysis
