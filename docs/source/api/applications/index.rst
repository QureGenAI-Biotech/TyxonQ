================
Applications API
================

Complete API reference for TyxonQ's application modules, including quantum chemistry algorithms and utilities.

.. contents:: Contents
   :depth: 3
   :local:

Overview
========

The Applications API provides high-level interfaces for domain-specific quantum computing applications:

🧪 **Quantum Chemistry** (``tyxonq.applications.chem``)
   Complete molecular simulation framework with VQE algorithms

📊 **Future Applications**
   Optimization, machine learning, and other quantum applications (planned)

Quantum Chemistry API
=====================

Molecule Class
--------------

.. autoclass:: tyxonq.applications.chem.Molecule
   :members:
   :undoc-members:
   :show-inheritance:

The Molecule class is the foundation for quantum chemistry calculations in TyxonQ.

**Constructor**:

.. code-block:: python

   Molecule(
       atoms: List[List],           # [["Symbol", [x, y, z]], ...]
       basis: str = "sto-3g",       # Basis set name
       charge: int = 0,              # Molecular charge
       spin: int = 0,                # Spin multiplicity
       unit: str = "Angstrom"        # Coordinate units
   )

**Key Properties**:

- ``hf_energy`` (float): Hartree-Fock energy in Hartree
- ``mp2_energy`` (float): MP2 correlation energy
- ``n_electrons`` (int): Total number of electrons
- ``n_qubits`` (int): Number of qubits required
- ``homo_lumo_gap`` (float): HOMO-LUMO energy gap
- ``mo_coefficients`` (ndarray): Molecular orbital coefficients

**Key Methods**:

- ``get_hamiltonian(mapping="jordan_wigner")`` - Generate qubit Hamiltonian

**Example**:

.. code-block:: python

   from tyxonq.applications.chem import Molecule
   
   h2o = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="sto-3g"
   )
   
   print(f"HF energy: {h2o.hf_energy:.6f} Hartree")
   print(f"Qubits needed: {h2o.n_qubits}")
   
   # Get Hamiltonian
   ham = h2o.get_hamiltonian(mapping="jordan_wigner")

Algorithm Classes
-----------------

UCCSD
~~~~~

.. autoclass:: tyxonq.applications.chem.UCCSD
   :members: kernel, energy, energy_and_grad, get_ex_ops
   :show-inheritance:

**Unitary Coupled Cluster Singles and Doubles** - High-accuracy variational ansatz.

**Constructor Parameters**:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``molecule``
     - Molecule
     - Target molecular system
   * - ``init_method``
     - str
     - Initialization: "mp2", "ccsd", "zeros"
   * - ``active_space``
     - tuple
     - (n_electrons, n_orbitals) for active space
   * - ``runtime``
     - str
     - "device" or "numeric"
   * - ``pick_ex2``
     - bool
     - Screen double excitations by amplitude
   * - ``epsilon``
     - float
     - Threshold for excitation screening (default: 1e-12)

**Methods**:

- ``kernel(method="BFGS", **opts)`` - Run VQE optimization
- ``energy(params, **opts)`` - Compute energy for given parameters
- ``energy_and_grad(params, **opts)`` - Compute energy and gradient

**Example**:

.. code-block:: python

   from tyxonq.applications.chem import Molecule, UCCSD
   
   h2 = Molecule(
       atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]],
       basis="6-31g"
   )
   
   uccsd = UCCSD(
       molecule=h2,
       init_method="mp2",
       runtime="numeric"
   )
   
   # Optimize
   energy = uccsd.kernel(method="BFGS")
   print(f"UCCSD energy: {energy:.6f} Hartree")
   
   # Access results
   print(f"Parameters: {uccsd.params}")
   print(f"Optimization info: {uccsd.opt_res}")

HEA
~~~

.. autoclass:: tyxonq.applications.chem.HEA
   :members: kernel, energy, energy_and_grad, get_circuit
   :show-inheritance:

**Hardware Efficient Ansatz** - Shallow circuit for NISQ devices.

**Constructor Parameters**:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``molecule``
     - Molecule
     - Target molecular system
   * - ``layers``
     - int
     - Number of ansatz layers
   * - ``runtime``
     - str
     - "device" or "numeric"
   * - ``numeric_engine``
     - str
     - "statevector" (default)
   * - ``mapping``
     - str
     - "parity", "jordan_wigner", "bravyi_kitaev"

**Methods**:

- ``kernel(method="COBYLA", shots=2048, **opts)`` - Run VQE optimization
- ``energy(params, shots=2048, **opts)`` - Evaluate energy
- ``get_circuit(params)`` - Build ansatz circuit

**Example**:

.. code-block:: python

   from tyxonq.applications.chem import Molecule, HEA
   
   h2o = Molecule(
       atoms=[
           ["O", [0.0, 0.0, 0.0]],
           ["H", [0.757, 0.586, 0.0]],
           ["H", [-0.757, 0.586, 0.0]]
       ],
       basis="sto-3g"
   )
   
   hea = HEA(
       molecule=h2o,
       layers=3,
       runtime="numeric"
   )
   
   # Optimize with COBYLA (gradient-free)
   energy = hea.kernel(method="COBYLA")
   print(f"HEA energy: {energy:.6f} Hartree")

KUPCCGSD
~~~~~~~~

.. autoclass:: tyxonq.applications.chem.KUPCCGSD
   :members: kernel
   :show-inheritance:

**k-layer Unitary Pair Coupled Cluster** - Flexible depth-accuracy tradeoff.

**Constructor Parameters**:

- ``molecule`` (Molecule): Target molecular system
- ``k_layers`` (int): Number of excitation layers
- ``init_method`` (str): "mp2", "ccsd", or "zeros"
- ``runtime`` (str): "device" or "numeric"

**Example**:

.. code-block:: python

   from tyxonq.applications.chem import KUPCCGSD
   
   kupccgsd = KUPCCGSD(
       molecule=mol,
       k_layers=2,
       init_method="mp2"
   )
   energy = kupccgsd.kernel()

PUCCD
~~~~~

.. autoclass:: tyxonq.applications.chem.PUCCD
   :members: kernel, make_rdm1, make_rdm2
   :show-inheritance:

**Paired Unitary Coupled Cluster Doubles** - For strongly correlated systems.

**Features**:

- Uses hard-core boson (HCB) mapping
- Reduces qubit count by factor of 2
- Optimized for bond dissociation
- Supports RDM calculations

**Example**:

.. code-block:: python

   from tyxonq.applications.chem import PUCCD
   
   puccd = PUCCD(
       molecule=h2_stretched,
       init_method="mp2"
   )
   
   energy = puccd.kernel()
   
   # Get reduced density matrices
   rdm1 = puccd.make_rdm1(puccd.params)
   rdm2 = puccd.make_rdm2(puccd.params)

Predefined Molecules
--------------------

Common Molecules
~~~~~~~~~~~~~~~~

.. code-block:: python

   from tyxonq.applications.chem.molecule import (
       h2,      # H2 molecule
       h2o,     # Water
       nh3,     # Ammonia
       ch4,     # Methane
       n2,      # Nitrogen
       co,      # Carbon monoxide
       lih,     # Lithium hydride
   )
   
   # Use predefined molecule
   print(f"H2 HF energy: {h2.hf_energy:.6f} Hartree")

Molecule Builders
~~~~~~~~~~~~~~~~~

.. autofunction:: tyxonq.applications.chem.molecule.h_chain

.. autofunction:: tyxonq.applications.chem.molecule.h_ring

.. autofunction:: tyxonq.applications.chem.molecule.water

**Example**:

.. code-block:: python

   from tyxonq.applications.chem.molecule import h_chain, water
   
   # Create hydrogen chain
   h4 = h_chain(n_h=4, bond_distance=0.8)
   
   # Create custom water
   h2o_custom = water(
       bond_length=0.96,
       bond_angle=104.5,
       basis="6-31g"
   )

Common Workflows
================

Basic VQE Calculation
---------------------

.. code-block:: python

   from tyxonq.applications.chem import Molecule, UCCSD
   
   # Step 1: Define molecule
   mol = Molecule(
       atoms=[["H", [0, 0, 0]], ["H", [0, 0, 0.74]]],
       basis="6-31g"
   )
   
   # Step 2: Create algorithm
   uccsd = UCCSD(
       molecule=mol,
       init_method="mp2",
       runtime="numeric"
   )
   
   # Step 3: Run optimization
   energy = uccsd.kernel(method="BFGS")
   
   # Step 4: Analyze results
   print(f"HF energy:    {mol.hf_energy:.6f} Hartree")
   print(f"VQE energy:   {energy:.6f} Hartree")
   print(f"Correlation:  {energy - mol.hf_energy:.6f} Hartree")

Potential Energy Surface
------------------------

.. code-block:: python

   import numpy as np
   from tyxonq.applications.chem import Molecule, HEA
   
   # Scan bond lengths
   distances = np.linspace(0.5, 3.0, 11)
   energies = []
   
   for d in distances:
       h2 = Molecule(
           atoms=[["H", [0, 0, 0]], ["H", [0, 0, d]]],
           basis="sto-3g"
       )
       hea = HEA(molecule=h2, layers=2, runtime="numeric")
       energies.append(hea.kernel())
   
   # Plot
   import matplotlib.pyplot as plt
   plt.plot(distances, energies)
   plt.xlabel('Bond Length (Angstrom)')
   plt.ylabel('Energy (Hartree)')
   plt.show()

Active Space Calculation
------------------------

.. code-block:: python

   from tyxonq.applications.chem import Molecule, UCCSD
   
   # Large molecule
   benzene = Molecule(atoms=benzene_coords, basis="sto-3g")
   
   # Use active space to reduce cost
   uccsd = UCCSD(
       molecule=benzene,
       active_space=(6, 6),  # 6 electrons in 6 orbitals
       init_method="mp2"
   )
   
   energy = uccsd.kernel()

See Also
========

- :doc:`/quantum_chemistry/index` - Quantum Chemistry User Guide
- :doc:`/quantum_chemistry/algorithms/index` - Algorithm Details
- :doc:`/examples/chemistry_examples` - Complete Examples
