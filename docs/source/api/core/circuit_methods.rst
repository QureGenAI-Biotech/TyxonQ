Circuit Methods API Reference
==============================

This page provides a complete reference for all methods available in the :class:`Circuit` class.

.. currentmodule:: tyxonq

Quantum State Manipulation
---------------------------

state()
~~~~~~~

.. method:: Circuit.state(engine=None, backend=None, form=None)

   Get the quantum state of this circuit.

   This method executes the circuit using a state simulator and returns the
   quantum state. The simulator engine is automatically selected based on the
   circuit's device configuration, or can be explicitly specified.

   :param engine: Simulator engine to use. Options:
       
       - ``"statevector"``: Dense statevector simulation (O(2^n) memory)
       - ``"mps"`` or ``"matrix_product_state"``: MPS simulation (efficient for low entanglement)
       - ``"density_matrix"``: Density matrix simulation (supports noise)
       
       If ``None``, uses the device configured via :meth:`device()` method.
   
   :type engine: str or None
   
   :param backend: Numeric backend (numpy/pytorch). If ``None``, uses current global backend.
   :type backend: optional
   
   :param form: Output format. Options:
       
       - ``None``, ``"ket"``, or ``"tensor"``: Return backend tensor (default, preserves autograd)
       - ``"numpy"``: Return numpy array (breaks autograd)
   
   :type form: str or None
   
   :returns: Quantum state representation (depends on engine and form):
       
       - **statevector**: 1D array/tensor of shape ``[2^num_qubits]``
       - **mps**: 1D array/tensor (reconstructed from MPS)
       - **density_matrix**: 2D array/tensor of shape ``[2^num_qubits, 2^num_qubits]``
   
   :rtype: Backend tensor or numpy array

   **Examples:**

   Use default statevector engine (returns backend tensor):

   .. code-block:: python

      import torch
      import tyxonq as tq
      
      tq.set_backend("pytorch")
      c = tq.Circuit(2)
      c.h(0).cx(0, 1)
      psi = c.state()  # Returns torch.Tensor (preserves gradients)
      print(type(psi), psi.shape)
      # <class 'torch.Tensor'> torch.Size([4])

   Get numpy array (for visualization/non-differentiable use):

   .. code-block:: python

      psi_np = c.state(form="numpy")
      print(type(psi_np))
      # <class 'numpy.ndarray'>

   Configure MPS simulator via :meth:`device()`:

   .. code-block:: python

      c = tq.Circuit(10)
      c.device(provider="simulator", device="matrix_product_state", max_bond=32)
      for i in range(10): 
          c.h(i)
      psi = c.state()  # Automatically uses MPS engine

   Explicitly specify engine:

   .. code-block:: python

      psi_mps = c.state(engine="mps")

   **See Also:**
   
   - :meth:`wavefunction`: Alias with clearer quantum physics semantics
   - :doc:`/tutorials/quantum_state_manipulation`: Comprehensive tutorial on state manipulation

wavefunction()
~~~~~~~~~~~~~~

.. method:: Circuit.wavefunction(engine=None, backend=None, form=None)

   Get the quantum wavefunction of this circuit.

   This is an alias for :meth:`state()` with clearer quantum physics semantics.
   The term "wavefunction" is traditional in quantum mechanics, while "state"
   is more general (applicable to mixed states in density matrix formalism).

   :param engine: Simulator engine (see :meth:`state()`)
   :param backend: Numeric backend (see :meth:`state()`)
   :param form: Output format (see :meth:`state()`)
   :returns: Quantum wavefunction as backend tensor or numpy array

   **Examples:**

   .. code-block:: python

      import tyxonq as tq
      
      c = tq.Circuit(2)
      c.h(0).cx(0, 1)
      psi = c.wavefunction()  # |Φ+⟩ = (|00⟩ + |11⟩)/√2
      psi_np = c.wavefunction(form="numpy")

   **See Also:**
   
   - :meth:`state`: General quantum state getter (supports mixed states)

Custom Gates
------------

unitary()
~~~~~~~~~

.. method:: Circuit.unitary(*qubits, matrix)

   Apply arbitrary unitary matrix to one or more qubits.

   This is a general-purpose gate that applies a custom unitary transformation
   to the specified qubits. It's useful for:
   
   - Implementing custom gates not available in the standard gate set
   - Random quantum circuits and benchmarking
   - Variational quantum algorithms with parameterized unitaries
   - Clifford gate optimization

   :param qubits: Target qubit indices (1 or 2 qubits supported).
   :type qubits: int
   
   :param matrix: Unitary matrix as numpy array or backend tensor.
       
       - For 1 qubit: 2×2 complex matrix
       - For 2 qubits: 4×4 complex matrix
   
   :type matrix: array-like
   
   :returns: Self for method chaining
   :rtype: Circuit
   
   :raises ValueError: If qubits count is not 1 or 2, or matrix shape is invalid.

   **Examples:**

   Apply custom single-qubit gate (√X):

   .. code-block:: python

      import numpy as np
      import tyxonq as tq
      
      c = tq.Circuit(1)
      sqrt_x = np.array([[0.5+0.5j, 0.5-0.5j],
                         [0.5-0.5j, 0.5+0.5j]])
      c.unitary(0, matrix=sqrt_x)

   Apply custom two-qubit gate (iSWAP):

   .. code-block:: python

      c = tq.Circuit(2)
      iswap = np.array([[1, 0, 0, 0],
                        [0, 0, 1j, 0],
                        [0, 1j, 0, 0],
                        [0, 0, 0, 1]])
      c.unitary(0, 1, matrix=iswap)

   Use in variational circuits:

   .. code-block:: python

      from tyxonq.libs.quantum_library.kernels.gates import gate_ry
      
      param = 0.5
      c.unitary(0, matrix=gate_ry(param))

   **Performance Notes:**
   
   - Matrix is stored in circuit cache to preserve autograd graph
   - Supports both numpy arrays and PyTorch tensors
   - Gradients propagate through parametrized unitary matrices

   **See Also:**
   
   - :doc:`/tutorials/custom_gates`: Tutorial on custom gate implementation
   - :mod:`tyxonq.libs.quantum_library.kernels.gates`: Standard gate library

Expectation Values
------------------

expectation()
~~~~~~~~~~~~~

.. method:: Circuit.expectation(*pauli_ops)

   Compute expectation value of Pauli operator product.

   This method computes ⟨ψ|O|ψ⟩ where O is a product of Pauli operators.
   The computation method is automatically selected based on the circuit's
   device configuration.

   Each Pauli operator is specified as ``(gate_matrix, [qubit_indices])``.

   :param pauli_ops: Variable number of Pauli operator tuples. Each tuple is ``(gate, qubits)`` where:
       
       - **gate**: Pauli gate matrix or gate object (X, Y, Z)
       - **qubits**: List of qubit indices
   
   :type pauli_ops: tuple
   
   :returns: Expectation value (real number for Hermitian operators)
   :rtype: float

   .. warning::
      **PERFORMANCE TIP** ⚡
      
      For multiple observables, avoid calling :meth:`expectation()` repeatedly as
      each call re-executes the circuit. Instead, use the Hamiltonian matrix
      approach for **10-15x speedup**:
      
      **❌ SLOW (避免)**:
      
      .. code-block:: python
      
         energy = 0.0
         for i in range(n):
             energy += circuit.expectation((gate_z(), [i]))  # N circuit executions!
      
      **✅ FAST (推荐)**:
      
      .. code-block:: python
      
         from tyxonq.libs.quantum_library.kernels.pauli import pauli_string_sum_dense
         
         H = pauli_string_sum_dense(pauli_terms, weights)
         psi = circuit.state()  # Execute circuit only once!
         energy = torch.real(torch.dot(torch.conj(psi), H @ psi))

   **Examples:**

   Single-qubit Pauli-X expectation ⟨X₀⟩:

   .. code-block:: python

      import tyxonq as tq
      
      c = tq.Circuit(2).h(0).cx(0, 1)
      exp_x = c.expectation((tq.gates.x(), [0]))

   Two-qubit Pauli product ⟨Z₀Z₁⟩:

   .. code-block:: python

      exp_zz = c.expectation((tq.gates.z(), [0]), (tq.gates.z(), [1]))

   Optimized multiple observables (RECOMMENDED):

   .. code-block:: python

      from tyxonq.libs.quantum_library.kernels.pauli import pauli_string_sum_dense
      import torch
      
      pauli_terms = [[3, 3, 0], [1, 0, 0]]  # ZZ, X, ...
      weights = [-1.0, -1.0]
      H = pauli_string_sum_dense(pauli_terms, weights)
      psi = c.state()
      energy = torch.real(torch.dot(torch.conj(psi), H @ psi))

   Works with MPS simulator:

   .. code-block:: python

      c = tq.Circuit(10)
      c.device(provider="simulator", device="matrix_product_state", max_bond=32)
      for i in range(10): 
          c.h(i)
      exp_x = c.expectation((tq.gates.x(), [0]))  # Uses MPS backend

   **Implementation Notes:**
   
   - Automatically uses appropriate simulator based on :meth:`device()` config
   - Supports statevector, MPS, and density matrix simulators
   - For statevector/MPS: computes exact expectation
   - For density matrix: supports mixed states and noise
   - Each call re-executes the circuit - use Hamiltonian approach for multiple observables

   **See Also:**
   
   - :doc:`/tutorials/expectation_values`: Comprehensive tutorial on expectation value computation
   - :doc:`/performance/optimization_tips`: Performance optimization guide

Noise and Decoherence
---------------------

kraus()
~~~~~~~

.. method:: Circuit.kraus(qubit, operators, status=None)

   Apply general Kraus channel (quantum noise/measurement) to a qubit.

   This method applies a completely positive trace-preserving (CPTP) map
   represented by Kraus operators {K₀, K₁, ..., Kₙ} satisfying ∑ᵢ K†ᵢKᵢ = I.

   **Kraus channels model:**
   
   - Quantum noise (decoherence, damping, dephasing)
   - Measurement-induced dynamics (MIPT, monitoring)
   - Open quantum systems evolution
   - Post-selection protocols

   **Physical interpretation:**
   
   - **Statevector**: Stochastic unraveling |ψ⟩ → Kᵢ|ψ⟩/||Kᵢ|ψ⟩|| (Monte Carlo)
   - **Density matrix**: Exact evolution ρ → ∑ᵢ KᵢρK†ᵢ

   :param qubit: Target qubit index (0-based)
   :type qubit: int
   
   :param operators: List of Kraus operators, each a 2×2 numpy array/tensor.
       Standard channels available in :mod:`tyxonq.libs.quantum_library.noise`:
       
       - ``depolarizing_channel(p)``
       - ``amplitude_damping_channel(gamma)``  # T₁ relaxation
       - ``phase_damping_channel(lambda)``     # T₂ dephasing
       - ``pauli_channel(px, py, pz)``
       - ``measurement_channel(p)``            # For MIPT
   
   :type operators: list of array-like
   
   :param status: Random variable in [0,1] for stochastic selection (statevector only).
                 If ``None``, uses uniform random sampling.
   :type status: float or None
   
   :returns: Self for method chaining
   :rtype: Circuit

   **Examples:**

   Apply amplitude damping (T₁ relaxation):

   .. code-block:: python

      from tyxonq.libs.quantum_library.noise import amplitude_damping_channel
      import tyxonq as tq
      
      c = tq.Circuit(2)
      c.h(0).cx(0, 1)
      kraus_ops = amplitude_damping_channel(gamma=0.1)
      c.kraus(0, kraus_ops)

   Measurement-induced phase transition (MIPT):

   .. code-block:: python

      from tyxonq.libs.quantum_library.noise import measurement_channel
      import numpy as np
      
      c = tq.Circuit(10)
      # ... apply random unitaries ...
      for i in range(10):
          c.kraus(i, measurement_channel(p=0.1), status=np.random.rand())

   Custom Kraus operators:

   .. code-block:: python

      import numpy as np
      
      K0 = np.array([[1, 0], [0, 0.9]])  # Custom channel
      K1 = np.array([[0, 0.1], [0, 0]])
      c.kraus(0, [K0, K1])

   Chain with other gates:

   .. code-block:: python

      c.h(0).kraus(0, amplitude_damping_channel(0.05)).cx(0, 1)

   **See Also:**
   
   - :doc:`/tutorials/advanced/kraus_channels`: Tutorial on Kraus channels
   - :mod:`tyxonq.libs.quantum_library.noise`: Standard noise channel library
   - :doc:`/devices/noise_simulation`: Noise simulation guide

with_noise()
~~~~~~~~~~~~

.. method:: Circuit.with_noise(noise_type, **noise_params)

   Configure noise model (chainable convenience wrapper).

   Automatically switches to density_matrix simulator and applies noise
   after every gate operation.

   :param noise_type: Type of noise channel:
       
       - ``"depolarizing"``: Depolarizing channel
       - ``"amplitude_damping"``: Amplitude damping (T₁ relaxation)
       - ``"phase_damping"``: Phase damping (T₂ dephasing)
       - ``"pauli"``: Asymmetric Pauli channel
   
   :type noise_type: str
   
   :param noise_params: Noise-specific parameters:
       
       - **depolarizing**: ``p`` (error rate)
       - **amplitude_damping**: ``gamma`` (T₁ decay rate)
       - **phase_damping**: ``lambda`` (T₂ dephasing rate)
       - **pauli**: ``px, py, pz`` (Pauli error rates)
   
   :returns: Self for method chaining
   :rtype: Circuit

   **Examples:**

   Depolarizing noise:

   .. code-block:: python

      import tyxonq as tq
      
      c = tq.Circuit(2).h(0).cx(0, 1)
      c.with_noise("depolarizing", p=0.05).run(shots=1024)

   Amplitude damping (T₁):

   .. code-block:: python

      c.with_noise("amplitude_damping", gamma=0.1).run(shots=1024)

   Phase damping (T₂):

   .. code-block:: python

      c.with_noise("phase_damping", lambda_=0.05).run(shots=1024)

   Pauli channel (asymmetric noise):

   .. code-block:: python

      c.with_noise("pauli", px=0.01, py=0.01, pz=0.05).run(shots=1024)

   Chain with other configuration:

   .. code-block:: python

      result = c.with_noise("depolarizing", p=0.05).device(shots=2048).run()

   **Notes:**
   
   - Automatically switches to ``density_matrix`` simulator
   - Noise is applied after every gate operation
   - For more fine-grained control, use :meth:`device()` directly

   **See Also:**
   
   - :meth:`kraus`: Fine-grained Kraus channel application
   - :doc:`/devices/noise_simulation`: Complete noise simulation guide

Measurements
------------

measure_z()
~~~~~~~~~~~

.. method:: Circuit.measure_z(q)

   Add Z-basis measurement instruction for qubit q.

   Measures the qubit in the computational basis {|0⟩, |1⟩}, collapsing
   the quantum state and producing a classical bit outcome.

   .. note::
      This adds a measurement instruction to the circuit but does not
      immediately execute it. The measurement occurs during circuit execution.

   :param q: Target qubit index (0-based) to measure
   :type q: int
   
   :returns: Self for method chaining
   :rtype: Circuit

   **Examples:**

   .. code-block:: python

      import tyxonq as tq
      
      c = tq.Circuit(2)
      c.h(0).cx(0, 1)  # Create Bell state
      c.measure_z(0).measure_z(1)  # Measure both qubits

   Measure all qubits in a loop:

   .. code-block:: python

      for i in range(c.num_qubits):
          c.measure_z(i)

   **See Also:**
   
   - :meth:`measure_reference`: Simulation-time measurement with immediate result
   - :meth:`mid_measurement`: Mid-circuit measurement with post-selection

measure_reference()
~~~~~~~~~~~~~~~~~~~

.. method:: Circuit.measure_reference(q, with_prob=False)

   Perform reference measurement (simulation-time measurement with result).

   This method immediately measures the qubit and returns the measurement
   outcome along with its probability. Unlike :meth:`measure_z()`, this is executed
   during circuit construction for simulation purposes.

   .. note::
      This is primarily for simulation/testing workflows, particularly
      useful for mid-circuit measurement scenarios where you need to condition
      subsequent gates on measurement outcomes.

   :param q: Target qubit index (0-based)
   :type q: int
   
   :param with_prob: If ``True``, return ``(outcome, probability)``.
                    If ``False``, return outcome only.
   :type with_prob: bool
   
   :returns: Measurement outcome:
       
       - If ``with_prob=False``: ``"0"`` or ``"1"``
       - If ``with_prob=True``: ``("0", probability)`` or ``("1", probability)``
   
   :rtype: str or tuple[str, float]

   **Examples:**

   Basic measurement:

   .. code-block:: python

      import tyxonq as tq
      
      c = tq.Circuit(2).h(0).cx(0, 1)
      outcome = c.measure_reference(0)  # "0" or "1"

   With probability:

   .. code-block:: python

      outcome, prob = c.measure_reference(0, with_prob=True)
      print(f"Measured {outcome} with probability {prob}")

   Use for conditional logic:

   .. code-block:: python

      if outcome == "0":
          c.x(1)  # Apply X if measured 0

   **See Also:**
   
   - :meth:`measure_z`: Standard measurement instruction
   - :doc:`/tutorials/advanced/conditional_circuits`: Conditional circuit tutorial

mid_measurement()
~~~~~~~~~~~~~~~~~

.. method:: Circuit.mid_measurement(q, keep=0)

   Perform mid-circuit measurement with post-selection.

   This method adds a projection operation that collapses the quantum state
   by measuring qubit q and post-selecting on the specified outcome.
   The state is renormalized after projection.

   .. note::
      This is a non-unitary operation that reduces the quantum state
      to a subspace. It's useful for:
      
      - Quantum error correction protocols
      - Adaptive quantum algorithms
      - Syndrome extraction circuits
      - Stabilizer simulation benchmarks

   :param q: Target qubit index (0-based) to measure and project
   :type q: int
   
   :param keep: Post-selected measurement outcome (0 or 1).
       
       - If ``keep=0``, project onto |0⟩ subspace for qubit q
       - If ``keep=1``, project onto |1⟩ subspace for qubit q
   
   :type keep: int
   
   :returns: Self for method chaining
   :rtype: Circuit

   **Examples:**

   Post-select on measuring 0:

   .. code-block:: python

      import tyxonq as tq
      
      c = tq.Circuit(2)
      c.h(0).cx(0, 1)
      c.mid_measurement(0, keep=0)  # Keep only |0⟩ component of qubit 0
      # State is now projected and renormalized

   Quantum error correction syndrome extraction:

   .. code-block:: python

      c = tq.Circuit(5)
      # ... encode + noise ...
      c.mid_measurement(3, keep=0)  # Ancilla qubit post-selection

   **See Also:**
   
   - :meth:`measure_reference`: Measurement with outcome sampling
   - :doc:`/tutorials/advanced/mid_circuit_measurement`: Mid-circuit measurement tutorial

Circuit Transformations
------------------------

inverse()
~~~~~~~~~

.. method:: Circuit.inverse(*, strict=False)

   Return a unitary inverse circuit for supported ops (h, cx, rz).

   Non-unitary ops like measure/reset/barrier are skipped unless ``strict=True``
   (in which case an error is raised). Unknown ops raise if ``strict=True``,
   otherwise they are skipped.

   **Supported gates:**
   
   - ``h``: H† = H (Hermitian, self-inverse)
   - ``cx``: CNOT† = CNOT (self-inverse)
   - ``rz(θ)``: Rz(θ)† = Rz(-θ)

   :param strict: If ``True``, raise error on non-invertible or unknown ops.
                 If ``False``, skip non-invertible ops silently.
   :type strict: bool
   
   :returns: Inverse circuit
   :rtype: Circuit
   
   :raises ValueError: If ``strict=True`` and non-unitary op encountered
   :raises NotImplementedError: If ``strict=True`` and unknown op encountered

   **Examples:**

   VQA adjoint:

   .. code-block:: python

      import tyxonq as tq
      
      c = tq.Circuit(2).h(0).rz(0, 0.5).cx(0, 1)
      c_inv = c.inverse()  # Automatically reverses and conjugates

   Strict mode (error on non-invertible ops):

   .. code-block:: python

      c_with_measure = tq.Circuit(2).h(0).measure_z(0)
      try:
          c_inv = c_with_measure.inverse(strict=True)  # ValueError!
      except ValueError:
          print("Cannot invert circuit with measurements")

   **See Also:**
   
   - :doc:`/tutorials/circuit_transformations`: Circuit transformation tutorial

Two-Qubit Interaction Gates
---------------------------

iswap()
~~~~~~~

.. method:: Circuit.iswap(q0, q1)

   Apply iSWAP gate between two qubits.

   The iSWAP gate exchanges quantum states and applies a relative phase:
   **iSWAP = exp(-iπ/4 · σ_x ⊗ σ_x)**

   **Matrix representation:**

   .. code-block:: text

      [[1,  0,  0,  0],
       [0,  0, 1j,  0],
       [0, 1j,  0,  0],
       [0,  0,  0,  1]]

   **Physical properties:**

   - Swaps quantum states: iSWAP|01⟩ = i|10⟩, iSWAP|10⟩ = i|01⟩
   - Adds π/2 relative phase to swapped basis states
   - Native gate on many superconducting platforms (Rigetti, IonQ)
   - Useful for exchanging and entangling qubits
   - Energy-preserving interaction (Heisenberg XX coupling)

   **Pulse Implementation:**

   - Decomposed to CX chain: CX(q0,q1) · CX(q1,q0) · CX(q0,q1)
   - Cross-resonance driven in pulse compiler
   - Supports three_level simulation with leakage modeling

   :param q0: First qubit index (0-based)
   :type q0: int

   :param q1: Second qubit index (0-based)
   :type q1: int

   :returns: Self for method chaining
   :rtype: Circuit

   **Examples:**

   Create iSWAP entanglement:

   .. code-block:: python

      import tyxonq as tq

      c = tq.Circuit(2)
      c.h(0).iswap(0, 1)  # Creates entangled state with phase

   Chain multiple iSWAPs:

   .. code-block:: python

      c = tq.Circuit(4)
      c.iswap(0, 1).iswap(2, 3)  # Two independent iSWAPs

   Mix with other gates:

   .. code-block:: python

      c = tq.Circuit(3)
      c.h(0).iswap(0, 1).cx(1, 2)  # Mix iSWAP and CNOT

   Use in VQE with Hubbard model:

   .. code-block:: python

      # iSWAP is natural for Hubbard/Fermi-Hubbard model simulation
      c = tq.Circuit(4)
      for i in range(3):
          c.iswap(i, i+1)  # Apply along chain

   **References:**

   - Shende & Markov, "Minimal universal two-qubit controlled-NOT-based circuits",
     *Physical Review A* **72**, 062305 (2005) [arXiv:quant-ph/0308033]
   - Rigetti, "A Practical Quantum Instruction Set Architecture" (2017)
     [arXiv:1903.02492]

   **See Also:**

   - :meth:`swap`: Standard SWAP gate (no relative phase)
   - :doc:`/tutorials/advanced/pulse_iswap_swap_decomposition`: Tutorial on gate decomposition
   - :doc:`/user_guide/pulse/index`: Pulse programming guide

swap()
~~~~~~

.. method:: Circuit.swap(q0, q1)

   Apply SWAP gate between two qubits.

   The SWAP gate exchanges quantum states without adding phase:

   **Matrix representation:**

   .. code-block:: text

      [[1, 0, 0, 0],
       [0, 0, 1, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 1]]

   **Physical properties:**

   - Pure state exchange: SWAP|01⟩ = |10⟩, SWAP|10⟩ = |01⟩
   - No relative phase (unlike iSWAP)
   - Equivalent to 3 CNOT gates: CX(q0,q1) · CX(q1,q0) · CX(q0,q1)
   - Useful for qubit relabeling and layout optimization
   - Commonly used in quantum algorithms to reorder qubits

   **Pulse Implementation:**

   - Identical CX chain decomposition to iSWAP
   - Software distinguishes physical difference (phase)
   - Cross-resonance driven pulse compilation
   - Supports three_level leakage simulation

   :param q0: First qubit index (0-based)
   :type q0: int

   :param q1: Second qubit index (0-based)
   :type q1: int

   :returns: Self for method chaining
   :rtype: Circuit

   **Examples:**

   Swap adjacent qubits:

   .. code-block:: python

      import tyxonq as tq

      c = tq.Circuit(3)
      c.h(0).cx(0, 1).swap(0, 2)  # Rearrange qubit order

   Use in layout optimization:

   .. code-block:: python

      c = tq.Circuit(4)
      c.cx(0, 1).swap(1, 3).cx(1, 2)  # Map logical circuit to physical hardware

   Verify SWAP property:

   .. code-block:: python

      # Initial: q0=1, q1=0 (|10⟩)
      c = tq.Circuit(2)
      c.x(0)  # Set q0 to |1⟩
      c.swap(0, 1)  # After: q0=0, q1=1 (|01⟩)
      c.measure_z(0).measure_z(1)

   Measure before and after:

   .. code-block:: python

      import tyxonq as tq

      c = tq.Circuit(2)
      c.h(0).h(1)  # Superposition
      c.swap(0, 1)  # Exchange
      c.measure_z(0).measure_z(1)  # Verify qubits swapped

   **Mathematical Properties:**

   - SWAP² = I (applying twice gives identity)
   - SWAP commutes with some gates but not all
   - Determinant = 1 (real unitary matrix)

   **See Also:**

   - :meth:`iswap`: iSWAP gate (with relative phase)
   - :doc:`/tutorials/advanced/pulse_iswap_swap_decomposition`: Tutorial on gate decomposition
   - :doc:`/user_guide/pulse/index`: Pulse programming guide

Aliases
-------

The following are aliases for commonly used methods:

- ``MEASURE_Z(q)``: Alias for :meth:`measure_z`
- ``CNOT(c, t)``: Alias for :meth:`cx`
- ``cnot(c, t)``: Alias for :meth:`cx`
- ``ISWAP(q0, q1)``: Alias for :meth:`iswap`
- ``SWAP(q0, q1)``: Alias for :meth:`swap`

.. seealso::

   **User Guides:**
   
   - :doc:`/user_guide/core/circuit_api`: Core circuit API guide
   - :doc:`/user_guide/devices/simulators`: Simulator usage guide
   
   **Tutorials:**
   
   - :doc:`/tutorials/custom_gates`: Custom gate implementation
   - :doc:`/tutorials/quantum_state_manipulation`: State manipulation
   - :doc:`/tutorials/expectation_values`: Expectation value computation
   
   **API Reference:**
   
   - :class:`Circuit`: Main circuit class
   - :mod:`tyxonq.libs.quantum_library`: Quantum library kernels
