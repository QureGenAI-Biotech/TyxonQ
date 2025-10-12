==================
Quantum Chemistry
==================

Quantum chemistry applications and algorithms for molecular electronic structure calculations.

.. contents:: Contents
   :depth: 2
   :local:

Overview
========

The TyxonQ quantum chemistry module provides a comprehensive framework for solving molecular electronic structure problems on quantum computers. It includes:

- **Variational Algorithms**: VQE-based methods for ground and excited states
- **Ansatz Libraries**: UCCSD, HEA, k-UpCCGSD, PUCCD implementations  
- **Molecule Class**: Interface to classical quantum chemistry codes
- **Runtime Systems**: Device and numeric execution backends
- **AIDD Applications**: AI-driven drug discovery workflows

Key Features
============

Molecular Hamiltonian Construction
----------------------------------

- Support for arbitrary molecules and basis sets
- Integration with PySCF for classical computations
- Multiple fermion-to-qubit mappings (Jordan-Wigner, Bravyi-Kitaev, Parity)
- Efficient Hamiltonian term grouping and measurement scheduling

Quantum Algorithms
------------------

- **UCCSD**: Unitary Coupled Cluster Singles and Doubles
- **HEA**: Hardware Efficient Ansatz with customizable layers
- **k-UpCCGSD**: k-layer Unitary Product Coupled Cluster
- **PUCCD**: Paired Unitary Coupled Cluster Doubles

Execution Runtimes
------------------

- **Device Runtime**: Execute on simulators or real quantum hardware
- **Numeric Runtime**: Classical tensor network simulation
- **Hybrid Workflows**: Combine quantum and classical computations

Quick Start Example
===================

.. code-block:: python

   from tyxonq.applications.chem import Molecule, UCCSD, HEA
   
   # Define water molecule
   h2o = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="sto-3g"
   )
   
   # Method 1: UCCSD algorithm
   uccsd = UCCSD(molecule=h2o, init_method="mp2")
   uccsd_energy = uccsd.kernel(method="BFGS")
   
   # Method 2: HEA algorithm
   hea = HEA(molecule=h2o, layers=2, runtime="numeric")
   hea_energy = hea.kernel(method="COBYLA")
   
   print(f"HF energy:    {h2o.hf_energy:.6f} Hartree")
   print(f"UCCSD energy: {uccsd_energy:.6f} Hartree")
   print(f"HEA energy:   {hea_energy:.6f} Hartree")

Documentation Structure
=======================

.. toctree::
   :maxdepth: 2

   fundamentals/index
   algorithms/index
   molecule/index
   runtimes/index
   aidd/index

Related Resources
=================

- :doc:`/api/applications/index` - Applications API Reference
- :doc:`/examples/chemistry_examples` - Quantum Chemistry Examples
- :doc:`/user_guide/numerics/index` - Numerics Backend Guide
