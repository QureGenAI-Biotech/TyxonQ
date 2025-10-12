===
FAQ
===

===========================
Frequently Asked Questions
===========================

Installation and Setup
======================

How do I install TyxonQ?
-------------------------

TyxonQ can be installed via pip:

.. code-block:: bash

   pip install tyxonq

For development installation:

.. code-block:: bash

   git clone https://github.com/QureGenAI-Biotech/TyxonQ.git
   cd TyxonQ
   pip install -e .

What are the dependencies?
---------------------------

**Required dependencies**:

- Python 3.8+
- NumPy
- SciPy
- NetworkX

**Optional dependencies**:

- **Qiskit**: For Qiskit compiler backend and transpilation
- **PyTorch**: For PyTorch numeric backend and GPU acceleration
- **CuPy/CuPyNumeric**: For GPU-accelerated tensor operations
- **PySCF**: For quantum chemistry calculations
- **Matplotlib**: For visualization

See ``pyproject.toml`` for complete dependency list.

How do I configure the numeric backend?
----------------------------------------

TyxonQ supports three numeric backends:

.. code-block:: python

   import tyxonq as tq
   
   # NumPy backend (default, CPU)
   tq.set_backend("numpy")  # or "cpu"
   
   # PyTorch backend (GPU support)
   tq.set_backend("pytorch")  # or "torch", "pt"
   
   # CuPyNumeric backend (distributed GPU)
   tq.set_backend("cupynumeric")  # or "gpu"

Backends are configured in ``src/tyxonq/config.py``.

Circuit Construction
====================

How do I create a quantum circuit?
-----------------------------------

Use the ``Circuit`` class:

.. code-block:: python

   import tyxonq as tq
   
   # Create 2-qubit circuit
   c = tq.Circuit(2)
   c.h(0)        # Hadamard on qubit 0
   c.cx(0, 1)    # CNOT from qubit 0 to 1
   c.measure_all()  # Measure all qubits

See :doc:`user_guide/core` for detailed circuit construction.

What gates are supported?
--------------------------

**Single-qubit gates**:

- Pauli: ``x()``, ``y()``, ``z()``
- Hadamard: ``h()``
- Rotations: ``rx()``, ``ry()``, ``rz()``
- Phase: ``s()``, ``sdg()``, ``t()``, ``tdg()``
- Others: ``sx()``, ``sxdg()``

**Two-qubit gates**:

- CNOT: ``cx()``
- Controlled-Y: ``cy()``
- Controlled-Z: ``cz()``
- SWAP: ``swap()``
- Controlled-Phase: ``cp()``

**Multi-qubit gates**:

- Toffoli: ``ccx()``
- Multi-controlled gates: ``mcx()``, ``mcy()``, ``mcz()``

See ``src/tyxonq/core/ir/circuit.py`` for complete gate set.

How do I use parameterized gates?
----------------------------------

.. code-block:: python

   from tyxonq import Circuit, Param
   
   # Create parameter
   theta = Param("theta")
   
   # Use in circuit
   c = Circuit(1)
   c.rx(theta, 0)
   c.ry(2 * theta, 0)

See :doc:`user_guide/core` for parameterized circuits.

Compilation
===========

How do I compile a circuit?
----------------------------

TyxonQ supports two compilation backends:

.. code-block:: python

   # Native compiler (default)
   compiled = c.compile()
   
   # Qiskit compiler (requires Qiskit)
   compiled = c.compile(engine="qiskit", optimization_level=3)

See :doc:`user_guide/compiler` for compilation details.

What optimization passes are available?
---------------------------------------

**Native compiler passes**:

- ``measurement_rewrite``: Optimize measurement operations
- ``shot_scheduler``: Schedule shot-based execution
- ``auto_measure``: Automatically add measurements

**Qiskit compiler**:

- Full Qiskit transpilation with optimization levels 0-3
- Hardware-aware compilation
- Gate basis transformation

See ``src/tyxonq/compiler/`` for implementation.

Why do I get "Missing Qiskit Dependency" error?
-------------------------------------------------

The native compiler tries to delegate to Qiskit for certain operations. Install Qiskit:

.. code-block:: bash

   pip install qiskit

Or use the native compiler without Qiskit features.

Execution
=========

How do I run a circuit on a simulator?
---------------------------------------

.. code-block:: python

   from tyxonq.runtime.simulator import run
   
   # Statevector simulation
   result = run(c, shots=1000)
   
   # With specific backend
   result = run(c, shots=1000, method="statevector")

See :doc:`user_guide/devices` for execution details.

How do I run on cloud hardware?
--------------------------------

.. code-block:: python

   from tyxonq.cloud import apis
   import getpass
   
   # Configure API key
   API_KEY = getpass.getpass("API Key: ")
   apis.set_token(API_KEY)
   apis.set_provider("tyxonq")
   
   # Submit to Homebrew_S2
   task = apis.submit_task(
       circuit=c,
       shots=100,
       device="homebrew_s2?o=7"
   )
   
   result = task.results()

See :doc:`cloud_services/getting_started` for cloud access.

What does "o=7" mean in device names?
---------------------------------------

Optimization flags for Homebrew_S2 (additive):

- ``o=1``: Qubit mapping
- ``o=2``: Gate decomposition
- ``o=4``: Initial mapping
- ``o=7``: All optimizations (1+2+4)

Example: ``device="homebrew_s2?o=3"`` means qubit mapping + gate decomposition.

See :doc:`cloud_services/device_management` for details.

Quantum Chemistry
=================

How do I run VQE for molecules?
--------------------------------

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   from tyxonq.applications.chem.molecule import h2
   
   # Create UCCSD instance
   uccsd = UCCSD(h2)
   
   # Run VQE
   energy = uccsd.kernel(runtime="numeric")
   print(f"Ground state energy: {energy} Ha")

See :doc:`quantum_chemistry/algorithms` for quantum chemistry algorithms.

What molecules are predefined?
-------------------------------

Predefined molecules in ``tyxonq.applications.chem.molecule``:

- ``h2``: Hydrogen molecule
- ``lih``: Lithium hydride
- ``h2o``: Water
- ``nh3``: Ammonia
- ``ch4``: Methane
- ``n2``: Nitrogen molecule

See :doc:`quantum_chemistry/molecule` for molecule definitions.

How do I use custom molecules?
-------------------------------

.. code-block:: python

   from tyxonq.applications.chem.molecule import Molecule
   
   # Define custom molecule
   my_mol = Molecule(
       atom="C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63",
       basis="sto-3g",
       charge=0,
       spin=0
   )

See :doc:`quantum_chemistry/molecule` for custom molecules.

Pulse-Level Programming
=======================

How do I use pulse-level control?
----------------------------------

.. code-block:: python

   from tyxonq import Circuit, Param, waveforms
   
   # Enable pulse mode
   qc = Circuit(1)
   qc.use_pulse()
   
   # Define pulse calibration
   param = Param("q[0]")
   builder = qc.calibrate("rabi_test", [param])
   builder.new_frame("drive_frame", param)
   builder.play("drive_frame", waveforms.CosineDrag(50, 0.2, 0.0, 0.0))
   builder.build()
   
   # Add calibration call
   qc.add_calibration('rabi_test', ['q[0]'])
   qc.measure_z(0)
   
   # Generate TQASM 0.2 code
   tqasm = qc.to_tqasm()

See :doc:`cloud_services/hardware_access` for pulse programming.

What waveforms are supported?
------------------------------

**Supported waveforms**:

- ``CosineDrag(duration, amp, sigma, beta)``
- ``Gaussian(duration, amp, sigma)``
- ``Sine(duration, amp, phase, freq, angle)``
- ``Flattop(duration, amp, sigma)``
- ``Constant(duration, amp)``
- ``GaussianSquare(duration, amp, sigma, width)``
- ``Cosine(duration, amp, freq, phase)``
- ``Drag(duration, amp, sigma, beta)``

See ``src/tyxonq/waveforms.py`` and ``docs/pulse_support_en.md`` for details.

Do I need to disable optimization for pulse circuits?
------------------------------------------------------

**Yes, absolutely critical!** Pulse-level circuits must disable optimization:

.. code-block:: python

   task = apis.submit_task(
       circuit=pulse_circuit,
       shots=100,
       device="homebrew_s2",  # No ?o= flag
       enable_qos_gate_decomposition=False,
       enable_qos_qubit_mapping=False
   )

Optimization will destroy pulse-level programming.

Troubleshooting
===============

Why is my circuit not running?
-------------------------------

**Common issues**:

1. **No measurements**: Add ``c.measure_all()`` before execution
2. **Missing backend**: Install required backend (Qiskit, PyTorch, etc.)
3. **Invalid gate basis**: Check if gates are supported by target device
4. **Token not set**: Configure API token for cloud execution

How do I enable debug logging?
-------------------------------

.. code-block:: python

   import logging
   
   logging.basicConfig(level=logging.DEBUG)

Where can I find examples?
---------------------------

Examples are in the ``examples/`` directory:

- ``circuit_chain_demo.py``: Circuit construction and execution
- ``cloud_api_task.py``: Cloud API usage
- ``pulse_demo.py``: Pulse-level programming
- VQE and quantum chemistry examples

See :doc:`examples/index` for documented examples.

How do I report bugs?
----------------------

Report issues on GitHub:

- **Repository**: https://github.com/QureGenAI-Biotech/TyxonQ
- **Issues**: https://github.com/QureGenAI-Biotech/TyxonQ/issues

Include:

- Python version
- TyxonQ version
- Minimal reproducible example
- Error traceback

Performance
===========

How can I speed up simulations?
--------------------------------

**Strategies**:

1. **Use GPU backend**: ``tq.set_backend("pytorch")`` or ``"cupynumeric"``
2. **Reduce shot count**: Use fewer shots for testing
3. **Optimize circuits**: Use ``optimization_level=3`` in compilation
4. **MPS backend**: For low-entanglement circuits

See :doc:`technical_references/performance` for optimization techniques.

Why is PyTorch backend slower than NumPy?
------------------------------------------

PyTorch has overhead for small circuits. GPU acceleration benefits appear for:

- Large circuits (>15 qubits)
- Many shots (>10000)
- Gradient computation
- Batched execution

For small circuits, NumPy is usually faster.

Further Help
============

Where can I learn more?
-----------------------

- **User Guide**: :doc:`user_guide/index`
- **Tutorials**: :doc:`tutorials/index`
- **API Reference**: :doc:`api/index`
- **Examples**: :doc:`examples/index`
- **Cloud Services**: :doc:`cloud_services/index`

How do I contact support?
--------------------------

For questions and support:

- **GitHub Discussions**: https://github.com/QureGenAI-Biotech/TyxonQ/discussions
- **Email**: Support information on https://www.tyxonq.com/

Is there a community?
---------------------

Join the TyxonQ community:

- **GitHub**: https://github.com/QureGenAI-Biotech/TyxonQ
- **Website**: https://www.tyxonq.com/