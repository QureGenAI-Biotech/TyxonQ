===============
Quantum Library
===============

The Quantum Library provides low-level computational kernels for quantum state manipulation and simulation. These kernels form the foundation of TyxonQ's simulators, offering optimized implementations for statevector, density matrix, and matrix product state (MPS) representations.

.. contents:: Table of Contents
   :depth: 3
   :local:

Overview
========

Quantum simulations require efficient implementations of fundamental operations on quantum states. The Quantum Library abstracts these operations into reusable kernels that:

⚡ **High Performance**
   Optimized tensor operations using einsum and efficient memory layouts

🔧 **Multiple Representations**
   Support for pure states, mixed states, and compressed representations

🎯 **Backend Agnostic**
   Works seamlessly with NumPy, PyTorch, and CuPy backends

📊 **Measurement Tools**
   Expectation values, sampling, and observable computation

⏱️ **Dynamics Simulation**
   Time evolution and Hamiltonian simulation capabilities

.. mermaid::

   graph TD
       A[Quantum Library] --> B[Kernels]
       A --> C[Dynamics]
       A --> D[Measurement]
       
       B --> E[Statevector]
       B --> F[Density Matrix]
       B --> G[MPS]
       B --> H[Unitary]
       B --> I[Pauli]
       
       E --> J[Simulators]
       F --> J
       G --> J
       
       C --> K[Time Evolution]
       D --> L[Expectation Values]

Library Architecture
====================

The library is organized into three main components:

.. list-table:: Quantum Library Components
   :header-rows: 1
   :widths: 25 35 40

   * - Component
     - Purpose
     - Key Features
   * - Kernels
     - State manipulation primitives
     - Statevector, density matrix, MPS operations
   * - Dynamics
     - Time evolution
     - Hamiltonian simulation, ODE integration
   * - Measurement
     - Observable computation
     - Expectation values, sampling, variance

Quantum State Kernels
=====================

Statevector Kernel
------------------

The statevector kernel provides the most straightforward quantum state representation as a complex vector in :math:`\mathbb{C}^{2^n}`.

**Theoretical Foundation**

A pure quantum state of :math:`n` qubits is represented as:

.. math::

   |\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle, \quad \sum_i |\alpha_i|^2 = 1

**Core Operations**

.. py:function:: init_statevector(num_qubits, backend)

   Initialize a statevector in the |0⟩⊗ⁿ state.

   :param num_qubits: Number of qubits
   :param backend: Numerical backend (numpy, pytorch, cupy)
   :return: Initial statevector of shape (2ⁿ,)

.. py:function:: apply_1q_statevector(backend, state, gate2, qubit, num_qubits)

   Apply a single-qubit gate to the statevector.

   :param backend: Numerical backend
   :param state: Current statevector
   :param gate2: 2×2 unitary matrix
   :param qubit: Target qubit index
   :param num_qubits: Total number of qubits
   :return: Updated statevector

.. py:function:: apply_2q_statevector(backend, state, gate4, q0, q1, num_qubits)

   Apply a two-qubit gate to the statevector.

   :param backend: Numerical backend
   :param state: Current statevector
   :param gate4: 4×4 unitary matrix
   :param q0: First qubit index
   :param q1: Second qubit index
   :param num_qubits: Total number of qubits
   :return: Updated statevector

.. py:function:: expect_z_statevector(state, qubit, num_qubits, backend)

   Compute expectation value of Z operator on a qubit.

   :param state: Current statevector
   :param qubit: Target qubit index
   :param num_qubits: Total number of qubits
   :param backend: Numerical backend
   :return: ⟨Z⟩ expectation value

**Example: Bell State Creation**

.. code-block:: python

   from tyxonq.libs.quantum_library.kernels import statevector as sv
   from tyxonq.numerics import get_backend
   import numpy as np

   # Get NumPy backend
   backend = get_backend('numpy')

   # Initialize 2-qubit state
   n_qubits = 2
   state = sv.init_statevector(n_qubits, backend)
   print(f"Initial state: {state}")  # [1, 0, 0, 0]

   # Apply Hadamard to qubit 0
   H = backend.array([[1, 1], [1, -1]]) / np.sqrt(2)
   state = sv.apply_1q_statevector(backend, state, H, 0, n_qubits)

   # Apply CNOT(0, 1)
   CNOT = backend.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0]])
   state = sv.apply_2q_statevector(backend, state, CNOT, 0, 1, n_qubits)

   print(f"Bell state: {state}")
   # Output: [0.707, 0, 0, 0.707] (approximately)

   # Compute expectation values
   exp_z0 = sv.expect_z_statevector(state, 0, n_qubits, backend)
   exp_z1 = sv.expect_z_statevector(state, 1, n_qubits, backend)
   print(f"⟨Z₀⟩ = {exp_z0}, ⟨Z₁⟩ = {exp_z1}")
   # Both should be ~0 for Bell state

**Performance Characteristics**

- **Memory**: :math:`O(2^n)` - Exponential in qubit count
- **Gate Application**: :math:`O(2^n)` - Full state update
- **Best For**: Small systems (n ≤ 20 qubits), exact simulation
- **Limitation**: Memory becomes prohibitive for large n

Density Matrix Kernel
----------------------

The density matrix kernel supports mixed quantum states, essential for simulating open quantum systems and noise.

**Theoretical Foundation**

A mixed state is represented by a density matrix:

.. math::

   \rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|, \quad \text{Tr}(\rho) = 1

For :math:`n` qubits, :math:`\rho \in \mathbb{C}^{2^n \times 2^n}` is a positive semidefinite, Hermitian matrix.

**Core Operations**

.. py:function:: init_density(num_qubits, backend)

   Initialize a density matrix in the pure |0⟩⊗ⁿ state.

   :param num_qubits: Number of qubits
   :param backend: Numerical backend
   :return: Density matrix of shape (2ⁿ, 2ⁿ)

.. py:function:: apply_1q_density(backend, rho, U, q, n)

   Apply single-qubit operation: ρ → U ρ U†

   :param backend: Numerical backend
   :param rho: Current density matrix
   :param U: 2×2 unitary matrix
   :param q: Target qubit index
   :param n: Total number of qubits
   :return: Updated density matrix

.. py:function:: apply_2q_density(backend, rho, U4, q0, q1, n)

   Apply two-qubit operation: ρ → U ρ U†

   :param backend: Numerical backend
   :param rho: Current density matrix
   :param U4: 4×4 unitary matrix
   :param q0: First qubit index
   :param q1: Second qubit index
   :param n: Total number of qubits
   :return: Updated density matrix

.. py:function:: exp_z_density(backend, rho, q, n)

   Compute expectation value: Tr(Z_q ρ)

   :param backend: Numerical backend
   :param rho: Current density matrix
   :param q: Target qubit index
   :param n: Total number of qubits
   :return: ⟨Z⟩ expectation value

**Example: Depolarizing Noise**

.. code-block:: python

   from tyxonq.libs.quantum_library.kernels import density_matrix as dm
   from tyxonq.numerics import get_backend
   import numpy as np

   backend = get_backend('numpy')
   n_qubits = 1

   # Initialize in |+⟩ state
   rho = dm.init_density(n_qubits, backend)
   H = backend.array([[1, 1], [1, -1]]) / np.sqrt(2)
   rho = dm.apply_1q_density(backend, rho, H, 0, n_qubits)

   print(f"Pure |+⟩ state:")
   print(rho)
   # [[0.5, 0.5], [0.5, 0.5]]

   # Apply depolarizing noise: ρ → (1-p)ρ + p*I/2
   p = 0.1  # 10% depolarizing
   I = backend.eye(2) / 2
   rho = (1 - p) * rho + p * I

   print(f"After noise:")
   print(rho)
   # Off-diagonal elements reduced

   # Measure purity
   purity = backend.to_numpy(backend.trace(backend.matmul(rho, rho)))
   print(f"Purity: {purity}")  # < 1 for mixed state

**When to Use Density Matrices**

- ✅ Simulating noisy quantum circuits
- ✅ Open quantum systems
- ✅ Partial trace operations
- ✅ Entanglement analysis
- ❌ Large pure state simulations (use statevector instead)

**Performance Characteristics**

- **Memory**: :math:`O(4^n)` - Quadratic compared to statevector
- **Gate Application**: :math:`O(4^n)` - More expensive than statevector
- **Best For**: Noisy simulations up to ~10 qubits
- **Trade-off**: Can represent mixed states at cost of higher memory


Matrix Product State (MPS) Kernel
----------------------------------

The MPS kernel provides a compressed representation for quantum states, enabling simulation of much larger systems than possible with full statevector.

**Theoretical Foundation**

An MPS represents a quantum state as a product of tensors:

.. math::

   |\psi\rangle = \sum_{i_1,\ldots,i_n} A^{[1]}_{i_1} A^{[2]}_{i_2} \cdots A^{[n]}_{i_n} |i_1 i_2 \cdots i_n\rangle

where each :math:`A^{[k]}_{i_k}` is a matrix of dimension :math:`\chi_{k-1} \times \chi_k`, and :math:`\chi_k` is the **bond dimension**.

**Key Advantage**: Memory scales as :math:`O(n\chi^2)` instead of :math:`O(2^n)`.

**Core Operations**

.. py:function:: init_product_state(num_qubits, bitstring)

   Initialize MPS in a computational basis state.

   :param num_qubits: Number of qubits
   :param bitstring: Initial state as list of 0s and 1s
   :return: MPSState object

.. py:function:: apply_1q(mps, U, site)

   Apply single-qubit unitary to MPS (exact, no truncation).

   :param mps: MPSState object
   :param U: 2×2 unitary matrix
   :param site: Site index
   :return: Updated MPSState

.. py:function:: apply_2q_nn(mps, U4, left_site, max_bond=None, svd_cutoff=1e-12)

   Apply two-qubit unitary to nearest-neighbor sites with SVD truncation.

   :param mps: MPSState object
   :param U4: 4×4 unitary matrix
   :param left_site: Left site of the two-qubit gate
   :param max_bond: Maximum bond dimension (truncation threshold)
   :param svd_cutoff: SVD singular value cutoff
   :return: Updated MPSState

.. py:function:: to_statevector(mps)

   Convert MPS to full statevector (for small systems only).

   :param mps: MPSState object
   :return: Full statevector

.. py:function:: bond_dims(mps)

   Get current bond dimensions of MPS.

   :param mps: MPSState object
   :return: List of bond dimensions

**Example: Large-Scale GHZ State**

.. code-block:: python

   from tyxonq.libs.quantum_library.kernels import matrix_product_state as mps
   import numpy as np

   # Create GHZ state on 50 qubits (impossible with statevector!)
   n_qubits = 50
   state = mps.init_product_state(n_qubits, [0] * n_qubits)

   # Apply H to first qubit
   H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
   state = mps.apply_1q(state, H, 0)

   # Apply CNOTs in chain
   CNOT = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])
   
   for i in range(n_qubits - 1):
       state = mps.apply_2q_nn(state, CNOT, i, max_bond=2)
       # GHZ has bond dimension 2

   # Check bond dimensions
   bonds = mps.bond_dims(state)
   print(f"Bond dimensions: {bonds}")
   # All bonds should be ≤ 2 for GHZ

   # For verification on small system
   if n_qubits <= 10:
       sv = mps.to_statevector(state)
       print(f"|000...0⟩ amplitude: {sv[0]}")
       print(f"|111...1⟩ amplitude: {sv[-1]}")
       # Should both be ~1/√2

**Bond Dimension Management**

The key to MPS efficiency is managing the bond dimension:

.. list-table:: Bond Dimension Trade-offs
   :header-rows: 1
   :widths: 20 40 40

   * - Bond Dimension
     - Accuracy
     - Computational Cost
   * - χ = 2
     - Low (only for special states)
     - :math:`O(n)` memory
   * - χ = 10-50
     - Medium (1D systems)
     - :math:`O(n \chi^2)` practical
   * - χ = 100-500
     - High (weakly entangled)
     - Expensive but feasible
   * - χ → ∞
     - Exact
     - Equivalent to statevector

**When to Use MPS**

- ✅ 1D quantum systems (spin chains)
- ✅ Shallow circuits with limited entanglement
- ✅ Time evolution with local Hamiltonians
- ✅ Systems with > 20 qubits
- ❌ Highly entangled states (all-to-all connectivity)
- ❌ Deep random circuits

Additional Kernels
==================

Unitary Kernel
--------------

Provides canonical matrix representations of quantum gates.

.. py:function:: get_unitary(gate_name, *params)

   Get unitary matrix for a named gate.

   :param gate_name: Gate name ('H', 'RZ', 'CX', etc.)
   :param params: Gate parameters (e.g., rotation angle)
   :return: Unitary matrix

**Example**:

.. code-block:: python

   from tyxonq.libs.quantum_library.kernels import unitary

   # Get standard gates
   H = unitary.get_unitary('H')
   X = unitary.get_unitary('X')
   
   # Parameterized gates
   RZ_theta = unitary.get_unitary('RZ', theta=np.pi/4)
   RY_phi = unitary.get_unitary('RY', phi=np.pi/3)

   # Two-qubit gates
   CNOT = unitary.get_unitary('CX')
   CZ = unitary.get_unitary('CZ')

Pauli Kernel
------------

Operations with Pauli matrices and Pauli strings.

.. code-block:: python

   from tyxonq.libs.quantum_library.kernels import pauli

   # Pauli matrices
   I, X, Y, Z = pauli.I, pauli.X, pauli.Y, pauli.Z

   # Pauli string operations
   pauli_string = pauli.string_to_matrix(['X', 'Y', 'Z'])  # X⊗Y⊗Z
   
   # Commutation relations
   commutes = pauli.check_commutation(['X', 'I', 'Z'], 
                                      ['X', 'Y', 'I'])

Dynamics Module
===============

The dynamics module provides tools for quantum time evolution and Hamiltonian simulation.

**Hamiltonian Time Evolution**

For a time-independent Hamiltonian :math:`H`, evolve states under:

.. math::

   |\psi(t)\rangle = e^{-iHt} |\psi(0)\rangle

.. code-block:: python

   from tyxonq.libs.quantum_library import dynamics
   from tyxonq.numerics import get_backend
   import numpy as np

   backend = get_backend('numpy')

   # Define Hamiltonian (Pauli-Z on single qubit)
   H = backend.array([[1, 0], [0, -1]])  # Z operator

   # Initial state |+⟩
   psi_0 = backend.array([1, 1]) / np.sqrt(2)

   # Evolve for time t
   t = np.pi / 4
   psi_t = dynamics.evolve_hamiltonian(psi_0, H, t, backend)

   print(f"|ψ(t)⟩ = {psi_t}")

**ODE-based Evolution**

For more complex time-dependent Hamiltonians:

.. code-block:: python

   def hamiltonian(t):
       # Time-dependent H(t) = sin(t) * X
       return np.sin(t) * np.array([[0, 1], [1, 0]])

   # Solve Schrödinger equation numerically
   times = np.linspace(0, 10, 100)
   solution = dynamics.solve_schrodinger_ode(
       psi_0, 
       hamiltonian, 
       times,
       method='RK45'
   )

Measurement Module
==================

The measurement module provides tools for computing observables and sampling.

**Expectation Values**

.. code-block:: python

   from tyxonq.libs.quantum_library import measurement

   # Compute expectation value of Pauli string
   pauli_string = [('X', 0), ('Y', 1), ('Z', 2)]
   state = ...  # Some 3-qubit state
   
   exp_val = measurement.expectation_value(
       state, 
       pauli_string,
       backend
   )

**Sampling**

.. code-block:: python

   # Sample measurement outcomes
   counts = measurement.sample(state, shots=1000, backend=backend)
   
   print(counts)
   # {'000': 245, '001': 253, '010': 251, '011': 251}

**Variance Estimation**

.. code-block:: python

   # Estimate variance of observable
   variance = measurement.variance(state, pauli_string, backend)
   
   # Uncertainty
   uncertainty = np.sqrt(variance)

Best Practices
==============

Choosing the Right Kernel
--------------------------

.. mermaid::

   graph TD
       A[Start] --> B{Pure or Mixed State?}
       B -->|Pure| C{System Size?}
       B -->|Mixed| D[Density Matrix]
       
       C -->|n ≤ 20| E[Statevector]
       C -->|n > 20| F{Entanglement?}
       
       F -->|Low| G[MPS]
       F -->|High| H[Cannot Simulate]
       
       E --> I[Exact Simulation]
       G --> J[Approximate Simulation]
       D --> K{n?}
       K -->|n ≤ 10| L[Feasible]
       K -->|n > 10| M[Too Expensive]

**Decision Guide**:

1. **Need noise simulation?** → Density Matrix (n ≤ 10)
2. **Pure state, n ≤ 20?** → Statevector (exact)
3. **Pure state, n > 20, 1D system?** → MPS (approximate)
4. **Highly entangled, n > 20?** → Cannot classically simulate efficiently

Performance Optimization
------------------------

**Memory Management**:

.. code-block:: python

   # For large MPS simulations, monitor bond dimensions
   state = mps.init_product_state(100, [0]*100)
   
   for i in range(100):
       state = mps.apply_2q_nn(state, gate, i, max_bond=50)
       
       # Periodically check bonds
       if i % 10 == 0:
           bonds = mps.bond_dims(state)
           print(f"Step {i}: max bond = {max(bonds)}")

**Backend Selection**:

.. code-block:: python

   # CPU: NumPy for prototyping
   backend = get_backend('numpy')

   # GPU: CuPy for large systems
   backend = get_backend('cupy')  # Requires GPU

   # Automatic differentiation: PyTorch
   backend = get_backend('pytorch')

See Also
========

- :doc:`../../user_guide/devices/index` - Device simulators using these kernels
- :doc:`../../user_guide/numerics/index` - Backend system details
- :doc:`../circuits_library/index` - Circuit templates
- :doc:`../../api/libs/quantum_library` - Complete API reference

Further Reading
===============

**MPS Theory**

.. [Schollwock2011] U. Schollwöck,  
   "The density-matrix renormalization group in the age of matrix product states",  
   Annals of Physics, 326, 96 (2011)

.. [Vidal2003] G. Vidal,  
   "Efficient Classical Simulation of Slightly Entangled Quantum Computations",  
   Physical Review Letters, 91, 147902 (2003)

**Quantum Simulation**

.. [Nielsen2000] M. A. Nielsen and I. L. Chuang,  
   "Quantum Computation and Quantum Information",  
   Cambridge University Press (2000)

.. [Feynman1982] R. P. Feynman,  
   "Simulating physics with computers",  
   International Journal of Theoretical Physics, 21, 467 (1982)

