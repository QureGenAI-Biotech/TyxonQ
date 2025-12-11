=======================
Numerical Methods Guide
=======================

TyxonQ's numeric runtime system provides multiple computation engines and optimization strategies for quantum chemistry calculations. This guide covers the technical details of numeric execution paths, backend selection, and performance optimization.

.. contents:: Contents
   :depth: 2
   :local:

Numeric Engine Architecture
============================

TyxonQ supports multiple numeric engines for different use cases:

.. code-block:: text

    Algorithm Layer (HEA/UCC/UCCSD)
              ↓
    Runtime Selection Layer
    ├─ runtime="numeric" → Numeric Runtime
    │   └─ numeric_engine selection
    │       ├─ statevector (default, fastest)
    │       ├─ civector (exact, PySCF-based)
    │       └─ hybrid (combines both)
    │
    └─ runtime="device" → Device Runtime
        └─ Simulated measurement sampling

Statevector Engine
==================

The default numeric engine for fast exact simulation:

Overview
--------

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   
   uccsd = UCCSD(
       hf_object,
       active_space=(4, 6),
       runtime="numeric",           # ← Select numeric runtime
       numeric_engine="statevector" # ← Select statevector engine (default)
   )

**Characteristics**:

- Memory: O(2^n) for n qubits
- Speed: Exponential scaling but practical for ≤25 qubits
- Accuracy: Exact (no sampling noise)
- Gradient support: Full autograd capability
- Backend: NumPy, PyTorch, CuPy

Performance Profile
-------------------

Typical performance on molecular systems:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 25

   * - System
     - Qubits
     - Energy (ms)
     - Gradient (ms)
   * - H2 (STO-3G)
     - 4
     - 2.5
     - 8.0
   * - H2O (STO-3G)
     - 14
     - 25
     - 75
   * - NH3 (STO-3G)
     - 16
     - 150
     - 450
   * - CH4 (STO-3G)
     - 20
     - 800
     - 2400

Implementation Details
----------------------

Energy computation with statevector backend:

.. code-block:: python

   def statevector_energy(
       circuit, 
       hamiltonian,
       backend="numpy"
   ) -> float:
       """Compute energy using statevector simulation.
       
       Args:
           circuit: Parameterized quantum circuit
           hamiltonian: Molecular Hamiltonian
           backend: NumPy/PyTorch/CuPy
           
       Returns:
           Energy expectation value
       """
       # 1. Build statevector
       psi = build_statevector(circuit, backend=backend)
       
       # 2. Apply Hamiltonian
       H_psi = hamiltonian @ psi  # Matrix-vector product
       
       # 3. Compute expectation
       energy = np.real(np.conj(psi) @ H_psi)
       
       return energy

Gradient computation:

.. code-block:: python

   def statevector_gradient(
       circuit_builder,
       params,
       hamiltonian,
       backend="pytorch"
   ) -> Tuple[float, np.ndarray]:
       """Compute energy and gradient using parameter-shift rule or autograd.
       
       If backend="pytorch" and params.requires_grad=True:
           Uses automatic differentiation (1 forward + 1 backward pass)
       Else:
           Uses parameter-shift rule (2n circuit evaluations)
       """
       if backend == "pytorch" and hasattr(params, 'requires_grad'):
           # Autograd path
           params.requires_grad_(True)
           psi = build_statevector(circuit_builder(params), backend=backend)
           energy = np.real(np.conj(psi) @ hamiltonian @ psi)
           energy.backward()
           return energy.item(), params.grad.numpy()
       else:
           # Parameter-shift rule (hardware-realistic)
           grad = np.zeros_like(params)
           base_energy = statevector_energy(circuit_builder(params), hamiltonian)
           
           for i in range(len(params)):
               # Shift forward
               params[i] += np.pi / 2
               plus = statevector_energy(circuit_builder(params), hamiltonian)
               
               # Shift backward
               params[i] -= np.pi
               minus = statevector_energy(circuit_builder(params), hamiltonian)
               
               # Gradient element
               grad[i] = (plus - minus) / 2
               
               # Restore
               params[i] += np.pi / 2
           
           return base_energy, grad

CI Vector Engine
================

Exact computation in CI vector space using PySCF:

Overview
--------

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   
   uccsd = UCCSD(
       hf_object,
       active_space=(4, 6),
       runtime="numeric",
       numeric_engine="civector"  # ← Use CI vector space
   )

**Characteristics**:

- Space: Full CI space (exact for active space)
- Memory: O(C(N, n_a) × C(N, n_b)) where N=orbitals, n_a/n_b=electrons
- Speed: Comparable to statevector, sometimes faster
- Accuracy: Exact in active space
- Ideal for: Post-HF validation, PySCF integration

When to Use
-----------

Use CI vector engine when:

✅ Validating against PySCF FCI solutions
✅ Need post-HF methods (MP2, CCSD)  
✅ Working with restricted (closed-shell) systems
✅ Comparing with traditional quantum chemistry codes
✅ Active space is small (≤12 electrons)

Implementation
---------------

CI vector computation using PySCF:

.. code-block:: python

   class CIvectorEngine:
       """Execute UCC ansatz in CI vector space using PySCF."""
       
       def __init__(self, hf_object, active_space):
           self.hf = hf_object
           self.na, self.nb = n_elec_a, n_elec_b
           self.norb = n_orb
           
           # Build CI solver
           from pyscf import fci
           self.fci_solver = fci.FCI(hf_object)
       
       def civector_from_params(self, params) -> np.ndarray:
           """Generate UCC-ansatz CI vector.
           
           For each parameter θ_i and excitation operator e_i:
               |ψ(θ)⟩ = exp(∑_i θ_i (e_i - e_i†)) |ψ_0⟩
           
           Compute as:
               |ψ(θ)⟩ = cos(θ_i)|ψ_0⟩ + sin(θ_i)(a_i|ψ_0⟩)
           """
           civector = self.init_civector.copy()
           
           for theta, ex_op in zip(params, self.excitation_ops):
               # Apply UCC evolution in CI space
               a_civector = apply_excitation_civector(civector, ex_op)
               civector = (np.cos(theta) * civector + 
                          np.sin(theta) * a_civector)
           
           return civector
       
       def energy_from_civector(self, civector) -> float:
           """Compute energy in CI space."""
           from pyscf import fci
           
           # Apply Hamiltonian in CI space
           hci = self.fci_solver.contract_1e(h1e, civector)
           hci += self.fci_solver.contract_2e(eri, civector)
           
           # Energy = ⟨ψ|H|ψ⟩
           energy = np.vdot(civector, hci)
           
           return np.real(energy)

Performance Comparison
=====================

Statevector vs CI Vector

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25 25

   * - System
     - Qubits
     - Statevector (ms)
     - CI Vector (ms)
     - Winner
   * - H2 (STO-3G)
     - 4
     - 2.5
     - 3.2
     - Statevector
   * - H2O (STO-3G)
     - 14
     - 25
     - 18
     - CI Vector
   * - NH3 (STO-3G)
     - 16
     - 150
     - 120
     - CI Vector
   * - C2H6 (STO-3G)
     - 26
     - OOM
     - 450
     - CI Vector

**Insights**:

- Statevector: Better for small systems
- CI Vector: More memory-efficient for large active spaces
- Crossover point: ~10 qubits

Backend Selection Strategy
==========================

NumPy vs PyTorch vs CuPy

NumPy Backend
-------------

Best for:

- Production deployments
- Deterministic computations
- Small to medium systems
- Simple workflows without gradients

.. code-block:: python

   from tyxonq.numerics import set_backend
   
   set_backend('numpy')
   
   # Use case: Production evaluation
   energy = uccsd.kernel()

PyTorch Backend
---------------

Best for:

- VQE optimization (automatic gradients)
- Hybrid quantum-classical training
- GPU acceleration (if available)
- Research prototyping

.. code-block:: python

   set_backend('pytorch')
   
   # Automatic differentiation support
   params = torch.tensor(..., requires_grad=True)
   energy = vqe_energy(circuit_builder(params), H)
   energy.backward()  # Gradient computation

CuPy Backend
------------

Best for:

- Large-scale simulations (>20 qubits)
- GPU-accelerated computations
- Batch parameter sweeps
- Systems with thousands of parameters

.. code-block:: python

   set_backend('cupy')
   
   # GPU acceleration
   params_gpu = cp.asarray(params)  # Transfer to GPU
   energy = compute_energy_gpu(params_gpu)

Optimization Techniques
=======================

Caching and Memoization
------------------------

Global LRU cache for repeated operations:

.. code-block:: python

   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def cached_operator_application(state_hash, operator_id):
       """Cache expensive operator applications."""
       # Expensive computation happens only once per unique state
       return apply_operator(state, operator)

Lazy Evaluation
---------------

Compute expensive values only when needed:

.. code-block:: python

   class NumericRuntime:
       @property
       def precomputed_tensors(self):
           """Lazy-load operator tensors once."""
           if not hasattr(self, '_tensors'):
               self._tensors = self._build_tensors()
           return self._tensors

Batch Processing
-----------------

Vectorize multiple evaluations:

.. code-block:: python

   # Instead of loop
   energies = []
   for theta in thetas:
       e = energy(theta)
       energies.append(e)
   
   # Use vectorization
   energies = vectorized_energy(thetas)  # Much faster

Circuit Compilation Optimization
---------------------------------

Optimize circuit structure before simulation:

.. code-block:: python

   circuit = build_ansatz(params)
   
   # Optimize circuit
   circuit = circuit.optimize(
       passes=['merge_single_qubit_gates',
               'remove_redundant_gates',
               'commutative_cancellation']
   )
   
   # Then simulate optimized circuit
   energy = statevector_energy(circuit, hamiltonian)

Numerical Precision
===================

Complex-to-Real Conversion
--------------------------

Handle numerical precision carefully:

.. code-block:: python

   def to_real_array(complex_array) -> np.ndarray:
       """Safely convert complex array to real, handling imaginary parts.
       
       In quantum mechanics, eigenvalues must be real.
       Small imaginary parts (<1e-10) indicate numerical error.
       """
       real_part = np.real(complex_array)
       imag_part = np.imag(complex_array)
       
       # Verify imaginary part is negligible
       max_imag = np.max(np.abs(imag_part))
       if max_imag > 1e-8:
           warnings.warn(f"Large imaginary component: {max_imag}")
       
       return real_part

NumPy 2.0 Compatibility
------------------------

Explicit dtype handling:

.. code-block:: python

   # ✅ Correct: Explicit conversion
   result = np.asarray(custom_array, dtype=np.float64)
   energy = np.dot(result, other_array)
   
   # ❌ Avoid: Implicit conversion
   energy = np.dot(custom_array, other_array)

Validation and Testing
======================

Cross-Validation Strategy
--------------------------

Compare results across engines:

.. code-block:: python

   def validate_engines(hf_object, active_space, params):
       """Cross-validate different numeric engines."""
       
       # Statevector computation
       uccsd_sv = UCCSD(hf_object, active_space, 
                        numeric_engine="statevector")
       e_sv = uccsd_sv.energy(params)
       
       # CI vector computation
       uccsd_ci = UCCSD(hf_object, active_space,
                        numeric_engine="civector")
       e_ci = uccsd_ci.energy(params)
       
       # PySCF reference
       e_pyscf = get_pyscf_energy(...)
       
       # Verify consistency
       assert abs(e_sv - e_ci) < 1e-6
       assert abs(e_sv - e_pyscf) < 1e-4
       
       return {
           "statevector": e_sv,
           "ci_vector": e_ci,
           "pyscf": e_pyscf
       }

Gradient Validation
-------------------

Verify gradients with finite differences:

.. code-block:: python

   def validate_gradients(circuit_builder, params, hamiltonian):
       """Validate computed gradients."""
       
       # Autograd gradient
       energy, grad_auto = compute_energy_and_grad(...)
       
       # Finite difference gradient
       grad_fd = compute_finite_difference(circuit_builder, params, H, eps=1e-5)
       
       # Verify agreement
       error = np.linalg.norm(grad_auto - grad_fd) / np.linalg.norm(grad_fd)
       
       print(f"Gradient error: {error:.2e}")
       assert error < 1e-5, "Gradient mismatch!"

Troubleshooting
===============

Memory Issues
-------------

If you encounter memory errors:

.. code-block:: python

   # Check system size
   n_qubits = circuit.num_qubits
   memory_needed = 16 * 2**n_qubits  # 16 bytes per complex number
   
   # For 25+ qubits, use CI vector engine
   if n_qubits > 20:
       uccsd = UCCSD(hf, active_space, numeric_engine="civector")

Gradient Computation Slow
--------------------------

.. code-block:: python

   # Enable caching
   uccsd.use_cache = True
   
   # Use PyTorch backend for autograd
   from tyxonq.numerics import set_backend
   set_backend('pytorch')
   
   # Batch gradients instead of computing one-by-one
   grad = compute_all_gradients_vectorized(params)

Numerical Instability
----------------------

.. code-block:: python

   # Use higher precision
   from numpy import float64
   params = np.array(params, dtype=np.float64)
   
   # Or higher precision backend
   from mpmath import mp
   mp.dps = 50  # 50 decimal places

Related Resources
=================

- :doc:`runtime_optimization` - Runtime Optimization (caching, fixes)
- :doc:`index` - Runtime Systems Overview
- :doc:`../algorithms/index` - Algorithm Implementation
- :doc:`/user_guide/numerics/index` - Backend System

