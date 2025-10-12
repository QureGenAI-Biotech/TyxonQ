================
Circuits Library
================

The Circuits Library provides pre-built, parameterized quantum circuit templates for common quantum algorithms and applications. These templates enable rapid prototyping and standardized implementation of variational quantum algorithms, combinatorial optimization, and time evolution simulations.

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

Quantum algorithm development often requires implementing complex parameterized circuits with specific structures. The Circuits Library accelerates this process by providing:

✨ **Pre-configured Templates**
   Battle-tested circuit structures for VQE, QAOA, UCC, and more

🎯 **Parameterization Support**
   Flexible parameter handling for variational optimization

⚡ **Optimized Construction**
   Efficient circuit building with minimal overhead

🔧 **Customization**
   Extensible templates that can be modified for specific needs

🧬 **Chemistry Integration**
   Specialized templates for quantum chemistry applications

.. mermaid::

   graph TD
       A[Circuit Template] --> B[VQE Ansatz]
       A --> C[QAOA Circuit]
       A --> D[UCC Circuit]
       A --> E[Trotter Circuit]
       A --> F[Variational Ansatz]
       
       B --> G[Optimization Algorithm]
       C --> G
       D --> H[Quantum Chemistry]
       E --> I[Time Evolution]
       F --> G

Available Templates
===================

The library includes five main categories of circuit templates:

.. list-table:: Circuit Template Categories
   :header-rows: 1
   :widths: 25 35 40

   * - Template Type
     - Primary Use Case
     - Key Features
   * - VQE Ansatze
     - Ground state energy
     - Hardware-efficient, chemically-inspired
   * - QAOA/Ising
     - Combinatorial optimization
     - MaxCut, graph problems, Ising models
   * - UCC/UCCSD
     - Quantum chemistry
     - Fermionic excitations, chemical accuracy
   * - Trotter Circuits
     - Time evolution
     - Hamiltonian simulation, dynamics
   * - Variational Templates
     - General optimization
     - Flexible, customizable structures

VQE Ansatz Templates
====================

Variational Quantum Eigensolver (VQE) is one of the most important near-term quantum algorithms. The Circuits Library provides several ansatz templates optimized for different scenarios.

Hardware-Efficient Ansatz (HEA)
--------------------------------

The Hardware-Efficient Ansatz is designed to work well on near-term quantum devices with limited connectivity and gate fidelity.

**Structure**

A typical HEA consists of:

1. **Initial layer**: Single-qubit rotations (usually RY gates)
2. **Entangling layers**: Two-qubit gates (CNOT chains)
3. **Rotation layers**: Additional single-qubit rotations

.. code-block:: python

   from tyxonq.libs.circuits_library import build_hea_circuit
   from tyxonq import Circuit
   import numpy as np

   # Create a 4-qubit HEA with 3 layers
   n_qubits = 4
   n_layers = 3
   n_params = (n_layers + 1) * n_qubits  # RY gates
   
   # Random initial parameters
   params = np.random.randn(n_params) * 0.1
   
   # Build circuit
   circuit = build_hea_circuit(
       n_qubits=n_qubits,
       n_layers=n_layers,
       params=params,
       entangler='cx',  # Use CNOT for entanglement
       entanglement='linear'  # Linear chain topology
   )
   
   print(f"Circuit depth: {circuit.depth()}")
   print(f"Gate count: {len(circuit.ops)}")

**Parameter Structure**

For an HEA with :math:`L` layers and :math:`n` qubits:

.. math::

   N_{params} = (L + 1) \times n

Parameters are organized as:

- Layer 0: Initial RY rotations on all qubits
- Layers 1 to L: RY rotations after each entangling layer

**Customization Options**

.. code-block:: python

   # Different entangling gates
   circuit_cz = build_hea_circuit(..., entangler='cz')    # CZ gates
   circuit_swap = build_hea_circuit(..., entangler='swap')  # SWAP gates
   
   # Different entanglement topologies
   circuit_full = build_hea_circuit(..., entanglement='full')      # All-to-all
   circuit_circular = build_hea_circuit(..., entanglement='circular')  # Ring
   circuit_sca = build_hea_circuit(..., entanglement='sca')        # Shifted circular

**When to Use HEA**

- ✅ Hardware-limited environments (NISQ devices)
- ✅ Unknown Hamiltonian structure
- ✅ Rapid prototyping and testing
- ❌ Requires chemical accuracy (use UCC instead)
- ❌ Need physical interpretation of parameters

Chemically-Inspired Ansatze
----------------------------

For quantum chemistry problems, ansatze based on electronic structure theory often provide better performance.

See :doc:`../../quantum_chemistry/algorithms/index` for detailed information on:

- UCCSD (Unitary Coupled Cluster Singles and Doubles)
- k-UpCCGSD (Unitary Paired Coupled Cluster)
- PUCCD (Pair Unitary Coupled Cluster Doubles)


QAOA for Combinatorial Optimization
====================================

The Quantum Approximate Optimization Algorithm (QAOA) is designed for solving combinatorial optimization problems on near-term quantum devices.

Theoretical Background
----------------------

QAOA approximates the ground state of a cost Hamiltonian :math:`H_C` by applying alternating unitaries:

.. math::

   |\psi(\boldsymbol{\gamma}, \boldsymbol{\beta})\rangle = U_M(\beta_p) U_C(\gamma_p) \cdots U_M(\beta_1) U_C(\gamma_1) |+\rangle^{\otimes n}

where:

- :math:`U_C(\gamma) = e^{-i\gamma H_C}` is the cost unitary
- :math:`U_M(\beta) = e^{-i\beta H_M}` is the mixer unitary
- :math:`|+\rangle = H|0\rangle` is the equal superposition state
- :math:`p` is the number of QAOA layers

QAOA for Ising Models
----------------------

The library provides specialized support for Ising-type Hamiltonians:

.. math::

   H_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j

**API: qaoa_ising**

.. py:function:: qaoa_ising(num_qubits, nlayers, pauli_z_terms, weights, params, mixer='X', full_coupling=False)

   Build QAOA circuit for Ising model problems.

   :param num_qubits: Number of qubits
   :type num_qubits: int
   :param nlayers: Number of QAOA layers (p-value)
   :type nlayers: int
   :param pauli_z_terms: Z-basis Pauli terms (each as list of qubit indices)
   :type pauli_z_terms: Sequence[Sequence[int]]
   :param weights: Coefficients for each Pauli term
   :type weights: Sequence[float]
   :param params: Circuit parameters [γ₁, β₁, γ₂, β₂, ..., γₚ, βₚ]
   :type params: Sequence[float]
   :param mixer: Mixer type: 'X', 'XY', or 'ZZ'
   :type mixer: str
   :param full_coupling: Use all-to-all coupling for mixer
   :type full_coupling: bool
   :return: QAOA circuit
   :rtype: Circuit

**Example: MaxCut Problem**

.. code-block:: python

   from tyxonq.libs.circuits_library import qaoa_ising
   import numpy as np

   # Define MaxCut on a 4-node graph
   # Edges: (0,1), (1,2), (2,3), (3,0)
   num_qubits = 4
   nlayers = 3

   # Cost Hamiltonian: -0.5 * sum of Z_i Z_j for each edge
   # In Ising form: each edge contributes Z_i Z_j
   pauli_z_terms = [
       [1, 1, 0, 0],  # Z_0 Z_1
       [0, 1, 1, 0],  # Z_1 Z_2
       [0, 0, 1, 1],  # Z_2 Z_3
       [1, 0, 0, 1],  # Z_3 Z_0
   ]
   weights = [-0.5] * 4  # Coefficient for each edge

   # Initial parameters (to be optimized)
   params = np.random.rand(2 * nlayers) * 0.5

   # Build QAOA circuit
   circuit = qaoa_ising(
       num_qubits=num_qubits,
       nlayers=nlayers,
       pauli_z_terms=pauli_z_terms,
       weights=weights,
       params=params,
       mixer='X',  # Standard X mixer
       full_coupling=False
   )

   print(f"QAOA circuit with {nlayers} layers")
   print(f"Parameters: {len(params)}")
   print(f"Gates: {len(circuit.ops)}")

**Mixer Options**

The mixer Hamiltonian can take different forms:

.. list-table:: QAOA Mixer Types
   :header-rows: 1
   :widths: 15 35 50

   * - Mixer
     - Hamiltonian
     - Use Case
   * - 'X'
     - :math:`H_M = \sum_i X_i`
     - Standard QAOA, general problems
   * - 'XY'
     - :math:`H_M = \sum_{\langle i,j \rangle} (X_i X_j + Y_i Y_j)`
     - Particle-conserving, constrained problems
   * - 'ZZ'
     - :math:`H_M = \sum_{\langle i,j \rangle} Z_i Z_j`
     - Alternative mixing for hard instances

**Parameter Optimization**

QAOA parameters are typically optimized to minimize the cost function expectation value:

.. code-block:: python

   from scipy.optimize import minimize

   def cost_function(params):
       circuit = qaoa_ising(num_qubits, nlayers, pauli_z_terms, 
                           weights, params, mixer='X')
       result = circuit.compile().device('statevector').run()
       # Compute expectation value of cost Hamiltonian
       energy = compute_energy(result, pauli_z_terms, weights)
       return energy

   # Optimize
   result = minimize(cost_function, initial_params, method='COBYLA')
   optimal_params = result.x
   optimal_energy = result.fun

**Performance Tips**

1. **Layer Count**: Start with p=1, increase if needed. Typically p=3-5 is sufficient.
2. **Initialization**: Initialize γ and β around π/4 for better convergence.
3. **Mixer Selection**: Use 'XY' mixer for problems with particle conservation.
4. **Graph Structure**: Exploit graph symmetries to reduce parameter space.

UCC Circuit Templates
=====================

Unitary Coupled Cluster (UCC) circuits are the gold standard for quantum chemistry simulations, providing a chemically-motivated ansatz that can systematically approach exact results.

Theoretical Foundation
----------------------

The UCC ansatz applies a unitary transformation to a reference state (typically Hartree-Fock):

.. math::

   |\psi_{UCC}\rangle = e^{\hat{T} - \hat{T}^\dagger} |HF\rangle

where :math:`\hat{T}` is the cluster operator:

.. math::

   \hat{T} = \hat{T}_1 + \hat{T}_2 + \cdots

- :math:`\hat{T}_1 = \sum_{ia} t_i^a a_a^\dagger a_i` (single excitations)
- :math:`\hat{T}_2 = \sum_{ijab} t_{ij}^{ab} a_a^\dagger a_b^\dagger a_j a_i` (double excitations)

After fermion-to-qubit mapping, these become parameterized quantum circuits.

API: build_ucc_circuit
----------------------

.. py:function:: build_ucc_circuit(params, n_qubits, n_elec_s, ex_ops, param_ids=None, mode='fermion', init_state=None, decompose_multicontrol=False, trotter=False)

   Construct a UCC circuit from excitation operators.

   :param params: Parameter values for excitations
   :type params: Sequence[float]
   :param n_qubits: Number of qubits
   :type n_qubits: int
   :param n_elec_s: Electron count as (n_alpha, n_beta)
   :type n_elec_s: Tuple[int, int]
   :param ex_ops: Excitation operators (fermionic indices)
   :type ex_ops: Sequence[Tuple]
   :param param_ids: Map operators to parameter indices
   :type param_ids: Sequence[int] | None
   :param mode: 'fermion', 'qubit', or 'hcb'
   :type mode: str
   :param init_state: Initial state circuit (default: HF state)
   :type init_state: Circuit | None
   :param decompose_multicontrol: Decompose multi-controlled gates
   :type decompose_multicontrol: bool
   :param trotter: Use Trotterization instead of gate-level implementation
   :type trotter: bool
   :return: UCC circuit
   :rtype: Circuit

**Example: UCCSD for H2**

.. code-block:: python

   from tyxonq.libs.circuits_library import build_ucc_circuit
   from tyxonq.applications.chem import Molecule
   import numpy as np

   # Define H2 molecule
   mol = Molecule(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
   mol.build()

   # Get excitation operators for UCCSD
   # This is typically done automatically by UCCSD class
   # Here we show the manual process
   from tyxonq.applications.chem.algorithms import UCCSD
   
   uccsd = UCCSD(mol)
   ex_ops = uccsd.excitation_ops
   n_params = len(ex_ops)

   # Initial parameters (from MP2 or CCSD)
   params = np.zeros(n_params)

   # Build UCC circuit
   circuit = build_ucc_circuit(
       params=params,
       n_qubits=mol.n_qubits,
       n_elec_s=(mol.n_electrons // 2, mol.n_electrons // 2),
       ex_ops=ex_ops,
       mode='fermion',
       trotter=False  # Use gate-level implementation
   )

   print(f"UCCSD circuit for H2")
   print(f"Qubits: {circuit.num_qubits}")
   print(f"Parameters: {n_params}")
   print(f"Excitations: {len(ex_ops)}")

**Implementation Modes**

The `mode` parameter controls how excitation operators are implemented:

- **'fermion'**: Full fermionic implementation with number-preserving gates
- **'qubit'**: Qubit-based implementation after Jordan-Wigner mapping
- **'hcb'**: Hardware-compatible basis (optimized for specific devices)

**Trotterization**

For large molecules, Trotterization can reduce circuit depth at the cost of accuracy:

.. code-block:: python

   # Standard gate-level (deeper, more accurate)
   circuit_gates = build_ucc_circuit(..., trotter=False)

   # Trotterized (shallower, approximate)
   circuit_trotter = build_ucc_circuit(..., trotter=True)

**See Also**

- :doc:`../../quantum_chemistry/algorithms/uccsd` - Complete UCCSD algorithm documentation
- :doc:`../../quantum_chemistry/algorithms/kupccgsd` - k-UpCCGSD variant
- :doc:`../hamiltonian_encoding/index` - Fermion-to-qubit mappings


Trotter Circuit Templates
==========================

Trotter-Suzuki decomposition enables time evolution simulation on quantum computers by breaking down the evolution operator into manageable pieces.

Theoretical Background
----------------------

For a time-independent Hamiltonian :math:`H = \sum_j w_j P_j` (sum of Pauli terms), the time evolution operator is:

.. math::

   U(t) = e^{-iHt}

The first-order Trotter-Suzuki decomposition approximates this as:

.. math::

   U(t) \approx \left(\prod_j e^{-iw_j P_j \Delta t}\right)^{n}

where :math:`\Delta t = t/n` is the Trotter step size and :math:`n` is the number of steps.

**Trotter Error**

The approximation error scales as:

.. math::

   \|U(t) - U_{Trotter}(t)\| = O\left(\frac{t^2}{n}\right)

Higher-order Trotter formulas can reduce this error at the cost of increased circuit depth.

API: build_trotter_circuit
---------------------------

.. py:function:: build_trotter_circuit(pauli_terms, weights=None, time, steps, num_qubits=None, order='first')

   Construct a Trotterized time evolution circuit.

   :param pauli_terms: Pauli terms encoded as sequences of {0,1,2,3} for {I,X,Y,Z}
   :type pauli_terms: Sequence[Sequence[int]]
   :param weights: Coefficients for each term (default: all 1.0)
   :type weights: Sequence[float] | None
   :param time: Total evolution time
   :type time: float
   :param steps: Number of Trotter steps
   :type steps: int
   :param num_qubits: Number of qubits (inferred if not provided)
   :type num_qubits: int | None
   :param order: Trotter order ('first' only currently supported)
   :type order: str
   :return: Trotter evolution circuit
   :rtype: Circuit

**Pauli Term Encoding**

Pauli terms are encoded as integer sequences:

- 0 = I (identity)
- 1 = X
- 2 = Y
- 3 = Z

Example: :math:`X_0 Y_1 Z_2` on 3 qubits → ``[1, 2, 3]``

**Example: Heisenberg Model Evolution**

.. code-block:: python

   from tyxonq.libs.circuits_library import build_trotter_circuit
   import numpy as np

   # 1D Heisenberg chain: H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
   n_qubits = 4
   
   # Define Pauli terms
   pauli_terms = []
   weights = []
   
   for i in range(n_qubits - 1):
       # X_i X_{i+1}
       term_x = [0] * n_qubits
       term_x[i] = 1      # X on qubit i
       term_x[i+1] = 1    # X on qubit i+1
       pauli_terms.append(term_x)
       weights.append(1.0)
       
       # Y_i Y_{i+1}
       term_y = [0] * n_qubits
       term_y[i] = 2      # Y on qubit i
       term_y[i+1] = 2    # Y on qubit i+1
       pauli_terms.append(term_y)
       weights.append(1.0)
       
       # Z_i Z_{i+1}
       term_z = [0] * n_qubits
       term_z[i] = 3      # Z on qubit i
       term_z[i+1] = 3    # Z on qubit i+1
       pauli_terms.append(term_z)
       weights.append(1.0)

   # Time evolution parameters
   evolution_time = 1.0
   trotter_steps = 10

   # Build circuit
   circuit = build_trotter_circuit(
       pauli_terms=pauli_terms,
       weights=weights,
       time=evolution_time,
       steps=trotter_steps,
       num_qubits=n_qubits
   )

   print(f"Heisenberg evolution for t={evolution_time}")
   print(f"Trotter steps: {trotter_steps}")
   print(f"Circuit depth: {circuit.depth()}")
   print(f"Total gates: {len(circuit.ops)}")

**Choosing Trotter Parameters**

The tradeoff between accuracy and circuit depth:

.. list-table:: Trotter Parameter Guidelines
   :header-rows: 1
   :widths: 30 30 40

   * - Parameter
     - Effect of Increase
     - Recommendation
   * - Number of steps
     - ↑ Accuracy, ↑ Depth
     - Start with n ~ 10, increase if needed
   * - Evolution time
     - ↑ Error (for fixed n)
     - Keep :math:`t \cdot \|H\| / n < 1` for accuracy
   * - Hamiltonian norm
     - ↑ Error sensitivity
     - Normalize H or increase steps

**Advanced: Adaptive Trotterization**

For time-dependent problems or when accuracy requirements vary:

.. code-block:: python

   def adaptive_trotter_circuit(pauli_terms, weights, time, 
                                error_threshold=1e-3):
       """
       Automatically determine number of Trotter steps
       based on error threshold.
       """
       # Estimate required steps (simplified)
       H_norm = sum(abs(w) for w in weights)
       min_steps = int(np.ceil((time * H_norm)**2 / error_threshold))
       
       return build_trotter_circuit(
           pauli_terms=pauli_terms,
           weights=weights,
           time=time,
           steps=min_steps
       )

**Applications**

Trotter circuits are essential for:

- Quantum dynamics simulation
- Adiabatic state preparation
- Quantum annealing
- Time-dependent VQE
- Quantum walk implementations

Variational Circuit Utilities
==============================

The library also provides utilities for constructing custom variational circuits.

Pauli Evolution
---------------

.. py:function:: evolve_pauli_ops(pauli_string, angle)

   Generate gates for :math:`e^{-i\theta P}` where P is a Pauli string.

   :param pauli_string: Pauli operators as (qubit, symbol) tuples
   :type pauli_string: Tuple[Tuple[int, str], ...]
   :param angle: Evolution angle θ
   :type angle: float
   :return: List of gate operations
   :rtype: List[Tuple]

This is used internally by both UCC and Trotter circuits but can also be used directly for custom circuits.

**Example**:

.. code-block:: python

   from tyxonq.libs.circuits_library.variational import evolve_pauli_ops
   from tyxonq import Circuit

   # Evolve under X_0 Y_1 Z_2
   pauli_string = ((0, 'X'), (1, 'Y'), (2, 'Z'))
   angle = np.pi / 4

   ops = evolve_pauli_ops(pauli_string, angle)
   
   # Build circuit from ops
   circuit = Circuit(3, ops=ops)

Best Practices
==============

General Guidelines
------------------

1. **Start Simple**: Begin with small circuits and gradually increase complexity
2. **Parameter Initialization**: Use physically-motivated initial values when possible
3. **Circuit Verification**: Always verify circuits produce expected behavior on simple cases
4. **Performance Monitoring**: Track circuit depth and gate count

Circuit Selection Guide
-----------------------

.. mermaid::

   graph TD
       A[Circuit Selection] --> B{Problem Type?}
       B --> |Optimization| C{Graph Structure?}
       B --> |Chemistry| D{Accuracy Needed?}
       B --> |Dynamics| E[Trotter]
       
       C --> |Regular| F[QAOA-Ising]
       C --> |General| G[VQE-HEA]
       
       D --> |High| H[UCC/UCCSD]
       D --> |Medium| I[HEA]
       
       E --> J{Time Scale?}
       J --> |Short| K[Few Steps]
       J --> |Long| L[Many Steps]

**Decision Tree**:

- **Need chemical accuracy?** → UCC/UCCSD
- **Graph optimization?** → QAOA
- **Unknown structure?** → HEA
- **Time evolution?** → Trotter
- **Custom needs?** → Compose from utilities

Performance Optimization
------------------------

**Circuit Depth Reduction**:

.. code-block:: python

   # Before: Deep UCC circuit
   circuit = build_ucc_circuit(..., trotter=False)
   depth_before = circuit.depth()

   # After: Trotterized UCC (shallower)
   circuit = build_ucc_circuit(..., trotter=True)
   depth_after = circuit.depth()

   print(f"Depth reduction: {depth_before} → {depth_after}")

**Parameter Efficiency**:

.. code-block:: python

   # Screen small excitation amplitudes in UCC
   from tyxonq.applications.chem.algorithms import UCCSD

   uccsd = UCCSD(mol, epsilon=1e-3)  # Screen amplitudes < 0.001
   # This reduces the number of parameters

See Also
========

- :doc:`../../user_guide/core/index` - Circuit IR details
- :doc:`../../user_guide/compiler/index` - Circuit compilation and optimization
- :doc:`../../quantum_chemistry/algorithms/index` - Chemistry algorithms using these circuits
- :doc:`../../examples/optimization_examples` - Practical optimization examples
- :doc:`../../api/libs/circuits_library` - Complete API reference

Further Reading
===============

**QAOA**

.. [Farhi2014] E. Farhi, J. Goldstone, and S. Gutmann,  
   "A Quantum Approximate Optimization Algorithm", arXiv:1411.4028 (2014)

.. [Zhou2020] L. Zhou, S-T. Wang, S. Choi, H. Pichler, and M. D. Lukin,  
   "Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices",  
   Physical Review X, 10, 021067 (2020)

**UCC Methods**

.. [Romero2018] J. Romero, R. Babbush, J. R. McClean, et al.,  
   "Strategies for quantum computing molecular energies using the unitary coupled cluster ansatz",  
   Quantum Science and Technology, 4, 014008 (2018)

**Trotter Methods**

.. [Lloyd1996] S. Lloyd,  
   "Universal Quantum Simulators", Science, 273, 1073 (1996)

.. [Campbell2019] E. Campbell,  
   "Random Compiler for Fast Hamiltonian Simulation",  
   Physical Review Letters, 123, 070503 (2019)

