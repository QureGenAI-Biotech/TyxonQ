=====================
Hamiltonian Encoding
=====================

The Hamiltonian Encoding library provides essential tools for transforming fermionic operators into qubit operators and grouping Pauli terms for efficient measurement. These capabilities are fundamental for quantum chemistry applications, particularly in variational quantum eigensolvers (VQE) and quantum simulation workflows.

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

Quantum chemistry problems are naturally expressed in terms of fermionic operators that describe electrons in molecular orbitals. To simulate these systems on quantum computers, we must:

1. **Transform** fermionic operators to qubit operators (fermion-to-qubit mapping)
2. **Encode** molecular Hamiltonians as sums of Pauli operators
3. **Group** Pauli terms by measurement basis to minimize circuit executions
4. **Allocate** measurement shots efficiently across grouped terms

TyxonQ's Hamiltonian Encoding library provides comprehensive support for all these operations.

.. mermaid::

   graph TD
       A[Fermionic Hamiltonian] --> B[Fermion-to-Qubit Mapping]
       B --> C[Qubit Hamiltonian]
       C --> D[Term Grouping]
       D --> E[Measurement Bases]
       E --> F[Shot Allocation]
       F --> G[Circuit Execution]
       
       B1[Jordan-Wigner] --> B
       B2[Bravyi-Kitaev] --> B
       B3[Parity] --> B
       B4[Binary] --> B

Key Features
============

✨ **Multiple Mapping Strategies**
   Support for Jordan-Wigner, Bravyi-Kitaev, Parity, and Binary transformations

🔬 **Qubit Reduction**
   Two-qubit reduction for parity mapping, checksum encoding for binary mapping

📊 **Efficient Grouping**
   Intelligent grouping of Pauli terms to minimize measurement overhead

🎯 **Optimized I/O**
   Functions for reading and writing Pauli operators in compact formats

⚡ **Integration**
   Seamless integration with quantum chemistry algorithms and compilers

Fermion-to-Qubit Mappings
==========================

Fermionic operators describe particles that obey the Pauli exclusion principle (electrons). To simulate fermonic systems on quantum computers, we must map these operators to qubit operators. Different mapping strategies offer trade-offs between circuit depth, qubit count, and operator locality.

Theoretical Background
----------------------

Fermion Operators
~~~~~~~~~~~~~~~~~

A fermionic system is described by creation (:math:`a_p^\dagger`) and annihilation (:math:`a_p`) operators that satisfy the anticommutation relations:

.. math::

   \{a_p, a_q^\dagger\} = \delta_{pq}, \quad \{a_p, a_q\} = 0, \quad \{a_p^\dagger, a_q^\dagger\} = 0

where :math:`p, q` index spin-orbitals.

A molecular Hamiltonian in second quantization takes the form:

.. math::

   H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2}\sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s

where :math:`h_{pq}` are one-electron integrals and :math:`h_{pqrs}` are two-electron integrals.

Qubit Operators
~~~~~~~~~~~~~~~

Qubit operators are expressed as sums of Pauli strings:

.. math::

   H_{qubit} = \sum_i c_i P_i

where :math:`c_i` are coefficients and :math:`P_i` are products of Pauli matrices :math:`\{I, X, Y, Z\}`.

Jordan-Wigner Transformation
-----------------------------

The Jordan-Wigner (JW) transformation is the most straightforward fermion-to-qubit mapping, establishing a direct correspondence between spin-orbitals and qubits.

**Mapping Rules**

.. math::

   a_p^\dagger &= \left(\bigotimes_{j<p} Z_j\right) \otimes \sigma_p^+ \\
   a_p &= \left(\bigotimes_{j<p} Z_j\right) \otimes \sigma_p^-

where :math:`\sigma^+ = (X - iY)/2` and :math:`\sigma^- = (X + iY)/2`.

**Properties**

- **Locality**: Operators affecting orbitals :math:`p` involve qubits :math:`0` through :math:`p`
- **Qubit Count**: Requires :math:`n` qubits for :math:`n` spin-orbitals
- **Circuit Depth**: Linear in orbital index for general operators
- **Simplicity**: Easiest to understand and implement

**Usage in TyxonQ**

.. code-block:: python

   from tyxonq.libs.hamiltonian_encoding import fop_to_qop
   from openfermion import FermionOperator
   
   # Create a fermionic operator: a_2^+ a_0
   fop = FermionOperator('2^ 0')
   
   # Map to qubits using Jordan-Wigner
   qop = fop_to_qop(fop, mapping='jordan-wigner', n_sorb=4, n_elec=2)
   
   print(qop)
   # Output shows the corresponding Pauli strings

Bravyi-Kitaev Transformation
-----------------------------

The Bravyi-Kitaev (BK) transformation achieves better locality than Jordan-Wigner by encoding occupation numbers and parity information more efficiently.

**Key Idea**

Instead of storing occupation information sequentially, BK uses a binary tree structure where:

- Some qubits store partial parity information
- Some qubits store occupation numbers
- This reduces operator support from :math:`O(n)` to :math:`O(\log n)` for some operators

**Properties**

- **Locality**: Improved compared to Jordan-Wigner
- **Qubit Count**: Still :math:`n` qubits for :math:`n` spin-orbitals
- **Circuit Depth**: Logarithmic scaling for certain operators
- **Complexity**: More complex transformation logic

**Usage**

.. code-block:: python

   # Same fermionic operator as before
   fop = FermionOperator('2^ 0')
   
   # Map using Bravyi-Kitaev
   qop_bk = fop_to_qop(fop, mapping='bravyi-kitaev', n_sorb=4, n_elec=2)
   
   # The resulting Pauli strings will be different from JW
   print(qop_bk)

Parity Mapping with Two-Qubit Reduction
----------------------------------------

The parity mapping exploits conservation of electron number and spin symmetries to reduce the qubit count.

**Two-Qubit Reduction**

When the total electron number :math:`N` and spin :math:`S_z` are conserved, two qubits can be eliminated:

1. One qubit encodes total electron parity
2. One qubit encodes spin parity

This reduces the qubit requirement from :math:`n` to :math:`n-2`.

**Mathematical Foundation**

For a system with :math:`(N_\alpha, N_\beta)` electrons, the parity mapping uses:

.. math::

   P = \prod_{p} (1 - 2n_p)

where :math:`n_p = a_p^\dagger a_p` is the occupation number operator.

**Implementation**

.. code-block:: python

   # For a system with 4 spin-orbitals and 2 electrons
   fop = FermionOperator('1^ 0')
   
   # Parity mapping with two-qubit reduction
   qop_parity = fop_to_qop(
       fop, 
       mapping='parity', 
       n_sorb=4, 
       n_elec=2  # Can also specify (n_alpha, n_beta)
   )
   
   # This will use 2 qubits instead of 4
   print(qop_parity)

**Advantages**

- **Qubit Reduction**: Saves two qubits, significant for NISQ devices
- **Natural Symmetries**: Respects physical conservation laws
- **VQE Suitability**: Well-suited for variational algorithms

**Limitations**

- **Fixed Particle Number**: Cannot represent superpositions of different particle numbers
- **Complexity**: More complex to implement and debug

Binary Transformation
---------------------

The binary transformation with checksum code offers an intermediate approach between standard mappings and aggressive reduction.

**Checksum Encoding**

The checksum code provides one-qubit reduction while maintaining important symmetry constraints:

.. math::

   n_{qubits} = n_{spin-orbitals} - 1

**When to Use**

The binary transformation is useful when:

- Full two-qubit reduction (parity) is not applicable
- You want moderate qubit savings
- System size is modest (not too large for encoding overhead)

**Usage**

.. code-block:: python

   fop = FermionOperator('3^ 2^ 1 0')
   
   # Binary mapping with checksum
   qop_binary = fop_to_qop(
       fop,
       mapping='binary',
       n_sorb=6,
       n_elec=4
   )

API Reference: fop_to_qop
-------------------------

.. py:function:: fop_to_qop(fop, mapping, n_sorb, n_elec)

   Transform a fermionic operator to a qubit operator using the specified mapping.

   :param fop: Input fermionic operator (OpenFermion FermionOperator)
   :type fop: FermionOperator
   :param mapping: Mapping strategy: "jordan-wigner", "bravyi-kitaev", "parity", or "binary"
   :type mapping: str
   :param n_sorb: Number of spin-orbitals in the system
   :type n_sorb: int
   :param n_elec: Electron count as integer (total) or tuple (n_alpha, n_beta)
   :type n_elec: int or tuple
   :return: Transformed qubit operator
   :rtype: QubitOperator
   :raises ValueError: If mapping type is not supported

   **Example**:

   .. code-block:: python

      from openfermion import FermionOperator
      from tyxonq.libs.hamiltonian_encoding import fop_to_qop

      # Molecular hydrogen with 4 spin-orbitals, 2 electrons
      # Create a simple excitation operator
      fop = FermionOperator('1^ 0')
      
      # Try different mappings
      qop_jw = fop_to_qop(fop, 'jordan-wigner', 4, 2)
      qop_bk = fop_to_qop(fop, 'bravyi-kitaev', 4, 2)
      qop_parity = fop_to_qop(fop, 'parity', 4, 2)
      
      print(f"Jordan-Wigner: {qop_jw}")
      print(f"Bravyi-Kitaev: {qop_bk}")
      print(f"Parity: {qop_parity}")

Mapping Comparison
------------------

.. list-table:: Fermion-to-Qubit Mapping Comparison
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Property
     - Jordan-Wigner
     - Bravyi-Kitaev
     - Parity
     - Binary
   * - Qubits Required
     - :math:`n`
     - :math:`n`
     - :math:`n-2`
     - :math:`n-1`
   * - Operator Locality
     - :math:`O(n)`
     - :math:`O(\log n)`
     - :math:`O(n)`
     - :math:`O(n)`
   * - Implementation Complexity
     - Simple
     - Moderate
     - Moderate
     - Moderate
   * - Best For
     - General use
     - Low-weight ops
     - VQE, fixed :math:`N`
     - Moderate savings
   * - Circuit Depth
     - High
     - Lower
     - High
     - High

**Selection Guidelines**:

1. **Default Choice**: Jordan-Wigner for its simplicity and broad applicability
2. **Resource-Constrained**: Parity mapping when qubits are limited
3. **Deep Circuits**: Bravyi-Kitaev when circuit depth is a bottleneck
4. **Moderate Savings**: Binary when partial reduction is acceptable


Pauli Term Grouping
===================

One of the most critical optimizations for variational algorithms is grouping Pauli terms that can be measured simultaneously. This dramatically reduces the number of required circuit executions.

Motivation
----------

A molecular Hamiltonian might contain thousands of Pauli terms:

.. math::

   H = \sum_{i=1}^{N} c_i P_i

where each :math:`P_i` is a Pauli string. Naively, measuring this Hamiltonian requires :math:`N` separate circuit executions. However, Pauli operators that commute and share a common eigenbasis can be measured simultaneously.

**Example**: For a 4-qubit system:

- :math:`Z_0 Z_1` and :math:`Z_1 Z_2` **can** be measured together (both Z-basis on overlapping qubits)
- :math:`Z_0 Z_1` and :math:`X_0 X_1` **cannot** be measured together (different bases)

Grouping Strategy
-----------------

TyxonQ groups Pauli terms by their measurement basis:

1. **Extract Basis**: Determine which single-qubit basis each qubit needs (X, Y, or Z)
2. **Group by Pattern**: Terms with the same basis pattern can be measured together
3. **Separate Identity**: Extract constant (identity) terms separately

**Mathematical Foundation**

Two Pauli strings :math:`P_i` and :math:`P_j` are in the same measurement group if:

.. math::

   \forall k: \quad P_i^{(k)} \in \{I, \sigma\} \text{ and } P_j^{(k)} \in \{I, \sigma\} \Rightarrow \sigma \text{ is the same}

where :math:`P_i^{(k)}` denotes the Pauli operator on qubit :math:`k`, and :math:`\sigma \in \{X, Y, Z\}`.

API: group_hamiltonian_pauli_terms
-----------------------------------

.. py:function:: group_hamiltonian_pauli_terms(hamiltonian, n_qubits)

   Group Pauli sum terms by their measurement basis.

   :param hamiltonian: List of (coefficient, [(operator, qubit), ...]) tuples
   :type hamiltonian: List[Tuple[float, List[Tuple[str, int]]]]
   :param n_qubits: Number of qubits in the system
   :type n_qubits: int
   :return: (identity_constant, grouped_terms)
   :rtype: Tuple[float, Dict]

   The grouped_terms dictionary maps measurement bases to lists of terms:
   
   - **Key**: Tuple of basis operators (e.g., ('Z', 'Z', 'I', 'X'))
   - **Value**: List of (term_tuple, coefficient) pairs

   **Example**:

   .. code-block:: python

      from tyxonq.libs.hamiltonian_encoding import group_hamiltonian_pauli_terms

      # Define a Hamiltonian: 0.5*Z0Z1 + 0.3*Z1Z2 + 0.2*X0X1 + 0.1*I
      hamiltonian = [
          (0.5, [('Z', 0), ('Z', 1)]),
          (0.3, [('Z', 1), ('Z', 2)]),
          (0.2, [('X', 0), ('X', 1)]),
          (0.1, []),  # Identity term
      ]

      # Group terms
      identity, groups = group_hamiltonian_pauli_terms(hamiltonian, n_qubits=3)

      print(f"Identity constant: {identity}")
      print(f"Number of measurement groups: {len(groups)}")
      
      for basis, terms in groups.items():
          print(f"Basis {basis}: {len(terms)} terms")

   **Output**:

   .. code-block:: text

      Identity constant: 0.1
      Number of measurement groups: 2
      Basis ('Z', 'Z', 'I'): 1 terms
      Basis ('Z', 'Z', 'Z'): 1 terms
      Basis ('X', 'X', 'I'): 1 terms

Practical Example: VQE Hamiltonian
-----------------------------------

Let's see how grouping reduces measurements for a realistic VQE problem.

.. code-block:: python

   from tyxonq.applications.chem import Molecule
   from tyxonq.libs.hamiltonian_encoding import group_hamiltonian_pauli_terms

   # Create H2 molecule
   mol = Molecule(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
   mol.build()

   # Get Hamiltonian in Pauli form (simplified example)
   # In practice, you'd get this from molecular integrals
   hamiltonian = mol.get_hamiltonian_pauli_terms()

   # Group terms
   identity, groups = group_hamiltonian_pauli_terms(
       hamiltonian, 
       n_qubits=mol.n_qubits
   )

   print(f"Total terms: {len(hamiltonian)}")
   print(f"Measurement groups: {len(groups)}")
   print(f"Reduction factor: {len(hamiltonian) / len(groups):.2f}x")

   # Distribution of group sizes
   group_sizes = [len(terms) for terms in groups.values()]
   print(f"Average terms per group: {sum(group_sizes) / len(group_sizes):.2f}")

For typical molecules, grouping can reduce measurements by 10-100x!

Shot Allocation Strategy
========================

After grouping Pauli terms, we must decide how to distribute a fixed shot budget across measurement groups. Different strategies optimize for different objectives.

Uniform Allocation
------------------

The simplest strategy: divide shots equally among groups.

.. math::

   n_{\text{shots}}^{(g)} = \frac{N_{\text{total}}}{|G|}

where :math:`G` is the set of measurement groups.

**Pros**: Simple, unbiased
**Cons**: Ignores variance and coefficient magnitudes

Weighted by Coefficient
-----------------------

Allocate more shots to terms with larger coefficients:

.. math::

   n_{\text{shots}}^{(g)} = N_{\text{total}} \cdot \frac{\sum_{i \in g} |c_i|}{\sum_{j} |c_j|}

**Rationale**: Terms with larger coefficients contribute more to the energy expectation value.

Variance-Weighted Allocation
-----------------------------

The optimal strategy minimizes the variance of the energy estimator:

.. math::

   n_{\text{shots}}^{(g)} \propto \sigma_g \cdot \sum_{i \in g} |c_i|

where :math:`\sigma_g` is the estimated measurement variance for group :math:`g`.

This is implemented in TyxonQ's shot scheduler (see :doc:`../../api/compiler/stages/scheduling`).

Operator Encoding and I/O
==========================

TyxonQ provides utilities for encoding operators in different representations and efficient I/O.

Encoding Formats
----------------

**OpenFermion Format**

.. code-block:: python

   from openfermion import QubitOperator
   
   # Create operator
   op = QubitOperator('X0 Y1 Z2', 0.5) + QubitOperator('Z0', 0.3)

**TyxonQ List Format**

.. code-block:: python

   # Same operator as list of tuples
   op_list = [
       (0.5, [('X', 0), ('Y', 1), ('Z', 2)]),
       (0.3, [('Z', 0)]),
   ]

**Compact String Format**

.. code-block:: python

   # Compact string representation
   op_str = "0.5*X0Y1Z2 + 0.3*Z0"

I/O Functions
-------------

.. py:function:: save_pauli_operator(operator, filename)

   Save a Pauli operator to a file in compact format.

   :param operator: Pauli operator to save
   :param filename: Output file path

.. py:function:: load_pauli_operator(filename)

   Load a Pauli operator from a file.

   :param filename: Input file path
   :return: Loaded Pauli operator

**Example**:

.. code-block:: python

   from tyxonq.libs.hamiltonian_encoding import (
       save_pauli_operator,
       load_pauli_operator
   )

   # Save a Hamiltonian
   save_pauli_operator(hamiltonian, 'h2_hamiltonian.txt')

   # Load it back
   loaded_h = load_pauli_operator('h2_hamiltonian.txt')

This is particularly useful for:

- Caching expensive Hamiltonian constructions
- Sharing Hamiltonians between simulations
- Archiving computational chemistry results

Advanced Topics
===============

Gray Code and Binary Encodings
-------------------------------

For certain encoding schemes, Gray code or binary encodings provide additional optimizations:

- **Gray Code**: Minimizes bit flips when traversing computational basis states
- **Binary Encoding**: Compact representation for specific operator classes

These are primarily used internally by the compiler but can be accessed for custom applications.

Integration with Quantum Chemistry Workflows
---------------------------------------------

The Hamiltonian encoding library integrates seamlessly with TyxonQ's quantum chemistry modules:

.. mermaid::

   sequenceDiagram
       participant User
       participant Molecule
       participant PySCF
       participant Encoder
       participant Compiler
       participant Device
       
       User->>Molecule: Define molecule
       Molecule->>PySCF: Run HF calculation
       PySCF-->>Molecule: Molecular orbitals, integrals
       Molecule->>Encoder: Get fermionic Hamiltonian
       Encoder->>Encoder: Apply fermion-to-qubit mapping
       Encoder->>Encoder: Group Pauli terms
       Encoder-->>Compiler: Qubit Hamiltonian + groups
       Compiler->>Compiler: Allocate shots
       Compiler->>Device: Execute circuits
       Device-->>User: Measurement results

See Also
========

- :doc:`../circuits_library/index` - Circuit templates for Hamiltonian simulation
- :doc:`../quantum_library/index` - Low-level quantum kernels
- :doc:`../../user_guide/compiler/index` - Compiler pipeline details
- :doc:`../../quantum_chemistry/algorithms/index` - VQE and other algorithms
- :doc:`../../api/libs/hamiltonian_encoding` - Full API reference

Further Reading
===============

**Foundational Papers**

.. [JW1928] P. Jordan and E. Wigner, "Über das Paulische Äquivalenzverbot", 
   Zeitschrift für Physik, 47, 631 (1928)

.. [BK2002] S. B. Bravyi and A. Y. Kitaev, "Fermionic Quantum Computation", 
   Annals of Physics, 298, 210 (2002)

.. [Parity2017] S. Bravyi, J. M. Gambetta, A. Mezzacapo, and K. Temme, 
   "Tapering off qubits to simulate fermionic Hamiltonians", arXiv:1701.08213 (2017)

**Review Articles**

.. [Review2020] S. McArdle, S. Endo, A. Aspuru-Guzik, S. C. Benjamin, and X. Yuan, 
   "Quantum computational chemistry", Reviews of Modern Physics, 92, 015003 (2020)

**TyxonQ Technical Documentation**

- :doc:`../../technical_references/whitepaper` - TyxonQ architecture and design
- :doc:`../../technical_references/performance_optimization` - Optimization strategies

Examples
========

Complete Example: H2 Molecule VQE
----------------------------------

This example demonstrates the full workflow from molecule definition to Hamiltonian grouping:

.. code-block:: python

   import tyxonq as tq
   from tyxonq.applications.chem import Molecule
   from tyxonq.libs.hamiltonian_encoding import (
       fop_to_qop,
       group_hamiltonian_pauli_terms
   )

   # Step 1: Define molecule
   mol = Molecule(
       atom='H 0 0 0; H 0 0 0.74',
       basis='sto-3g',
       charge=0,
       spin=0
   )
   mol.build()

   # Step 2: Get molecular integrals and construct fermionic Hamiltonian
   # (This is typically done internally by VQE classes)
   fop_h = mol.get_fermionic_hamiltonian()

   # Step 3: Map to qubits
   qop_h = fop_to_qop(
       fop_h,
       mapping='parity',  # Use two-qubit reduction
       n_sorb=mol.n_orbitals * 2,
       n_elec=mol.n_electrons
   )

   # Step 4: Convert to list format for grouping
   hamiltonian_list = qop_to_list(qop_h)  # Helper function

   # Step 5: Group Pauli terms
   identity, groups = group_hamiltonian_pauli_terms(
       hamiltonian_list,
       n_qubits=mol.n_qubits
   )

   # Step 6: Analyze grouping efficiency
   print(f"Identity contribution: {identity:.6f}")
   print(f"Total Pauli terms: {len(hamiltonian_list)}")
   print(f"Measurement groups: {len(groups)}")
   print(f"Compression ratio: {len(hamiltonian_list)/len(groups):.2f}x")

   # Step 7: Create VQE circuit (simplified)
   from tyxonq.applications.chem.algorithms import UCCSD

   vqe = UCCSD(mol, mapping='parity')
   energy = vqe.kernel()

   print(f"Ground state energy: {energy:.6f} Ha")

This complete workflow showcases how Hamiltonian encoding enables efficient quantum chemistry simulations.

