Framework Comparison
====================

======================================
Comparison with Other Quantum Frameworks
======================================

This section provides a detailed comparison of TyxonQ with other major quantum computing frameworks, highlighting TyxonQ's unique architectural innovations.

Framework Overview
==================

TyxonQ vs. Qiskit vs. PennyLane
-------------------------------

.. list-table:: Comprehensive Framework Comparison
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - Qiskit
     - PennyLane  
     - TyxonQ
   * - **IR Design**
     - Implicit transpiler IR
     - Transform-based
     - **Explicit Stable IR**
   * - **Result Semantics**
     - Provider-dependent formats
     - Expectation-first
     - **Counts-first unified**
   * - **Measurement Optimization**
     - Device-layer handling
     - QNode encapsulation
     - **Compiler-layer metadata**
   * - **Backend Abstraction**
     - Provider-specific interfaces
     - Interface adapters
     - **Single ArrayBackend protocol**
   * - **Chemistry Applications**
     - Qiskit Nature (separate package)
     - PennyLane QChem
     - **Native Quantum AIDD stack**
   * - **Dual-Path Support**
     - Separate Aer/Terra ecosystems
     - QNode unification
     - **Semantic-consistent dual-path**
   * - **Gradient Computation**
     - Parameter-shift (manual)
     - Automatic differentiation
     - **Both parameter-shift & autograd**
   * - **Hardware Access**
     - IBM Quantum Network
     - Multiple providers (plugins)
     - **TyxonQ native + adapters**
   * - **Compilation Model**
     - Transpiler passes
     - Transform decorators
     - **Pluggable compiler engines**
   * - **Postprocessing**
     - Provider-specific
     - Built into QNode
     - **Unified pluggable layer**

Detailed Comparisons
====================

1. Intermediate Representation (IR)
------------------------------------

Qiskit Approach
~~~~~~~~~~~~~~~

**Design**: Implicit IR through transpiler stages

- ``QuantumCircuit`` is both user-facing and internal representation
- Transpiler modifies circuit in-place through passes
- No explicit IR contract between compiler and device
- IR structure changes across Qiskit versions

**Example**:

.. code-block:: python

   # Qiskit - circuit is modified by transpiler
   from qiskit import QuantumCircuit, transpile
   
   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)
   
   # Transpilation modifies circuit structure
   transpiled = transpile(qc, backend=backend, optimization_level=3)
   # Internal representation is implicit

PennyLane Approach
~~~~~~~~~~~~~~~~~~

**Design**: Transform-based operation graph

- QNode captures operations as tape
- Transforms modify tape structure
- No stable IR, only operation sequences
- IR tied to execution model (gradients, batching)

**Example**:

.. code-block:: python

   # PennyLane - operations captured in QNode tape
   import pennylane as qml
   
   @qml.qnode(dev)
   def circuit(params):
       qml.Hadamard(0)
       qml.CNOT([0, 1])
       return qml.expval(qml.PauliZ(0))
   
   # Internal tape representation, no explicit IR

TyxonQ Approach
~~~~~~~~~~~~~~~

**Design**: **Explicit Stable IR as system contract**

- ``Circuit`` IR is minimal and stable across versions
- Only ``num_qubits`` + ``ops`` core fields
- IR is explicit data structure, not implicit state
- Metadata layer for compiler-generated information

**Example**:

.. code-block:: python

   # TyxonQ - explicit stable IR
   import tyxonq as tq
   
   c = tq.Circuit(2).h(0).cx(0, 1)
   
   # Explicit IR inspection
   print(c.num_qubits)  # 2
   print(c.ops)         # Operation sequence
   
   # Compilation generates metadata without modifying IR structure
   compiled = c.compile(passes=["measurement_rewrite"])
   print(compiled.metadata)  # Compiler-generated grouping info

**Advantages**:

- ✅ **System-wide contract**: All components agree on IR structure
- ✅ **Version stability**: IR definition doesn't change between releases
- ✅ **Inspectable**: Users can examine IR before execution
- ✅ **Extensible**: Metadata layer for optional information

2. Measurement Optimization
----------------------------

Qiskit Approach
~~~~~~~~~~~~~~~

**Location**: Scattered across device drivers and job managers

- Measurement grouping handled at runtime by provider
- No compiler-level optimization
- Grouping strategy depends on backend implementation
- Unpredictable shot allocation

**Issues**:

- ❌ Vendor lock-in: Different providers use different strategies
- ❌ Non-reproducible: Same circuit may get different grouping
- ❌ Not inspectable: Grouping decisions hidden in provider code

PennyLane Approach
~~~~~~~~~~~~~~~~~~

**Location**: Encapsulated in QNode execution

- Automatic Pauli grouping for gradient computation
- Tied to QNode's execution model
- Limited user control over grouping strategy

**Benefits**:

- ✅ Automatic optimization
- ❌ Limited transparency
- ❌ Tied to QNode abstraction

TyxonQ Approach
~~~~~~~~~~~~~~~

**Location**: **Compiler-layer metadata generation**

**Design**:

- Measurement grouping is a **compiler pass**
- Generates grouping metadata attached to IR
- Shot scheduling based on variance weights
- Deterministic and reproducible

**Example**:

.. code-block:: python

   import tyxonq as tq
   
   c = tq.Circuit(4).h(0).cx(0,1).cx(1,2).cx(2,3)
   
   # Measurement grouping as compiler pass
   compiled = c.compile(passes=["measurement_rewrite", "shot_scheduler"])
   
   # Inspect grouping metadata before execution
   print(compiled.metadata["measurement_groups"])
   # Output: Grouped Pauli terms with basis rotations
   
   print(compiled.metadata["shot_plan"])  
   # Output: Deterministic shot allocation per group

**Advantages**:

- ✅ **Deterministic**: Same circuit → same grouping
- ✅ **Inspectable**: Users can see grouping before execution
- ✅ **Portable**: Same grouping metadata works for all devices
- ✅ **Reproducible**: Enables result verification

3. Dual-Path Execution Model
-----------------------------

Qiskit Approach
~~~~~~~~~~~~~~~

**Design**: Separate Aer (simulator) and Terra (hardware) ecosystems

- Aer simulators and real hardware use different code paths
- Different result formats between Aer and IBMQ
- Inconsistent noise models
- Migration requires code changes

**Example**:

.. code-block:: python

   # Qiskit - different backends, different semantics
   from qiskit import Aer, IBMQ
   
   # Simulator execution
   sim_backend = Aer.get_backend('qasm_simulator')
   sim_result = execute(qc, sim_backend).result()
   
   # Hardware execution (different interface)
   IBMQ.load_account()
   hw_backend = IBMQ.get_backend('ibm_quebec')
   hw_result = execute(qc, hw_backend).result()
   # Result formats may differ!

PennyLane Approach
~~~~~~~~~~~~~~~~~~

**Design**: QNode abstraction unifies device and simulator

- Same QNode code runs on simulators and hardware
- Device-specific plugins handle backend differences
- Automatic differentiation works across devices

**Benefits**:

- ✅ Unified interface
- ❌ QNode abstraction hides execution details
- ❌ Limited control over device vs. numeric paths

TyxonQ Approach
~~~~~~~~~~~~~~~

**Design**: **Semantic-consistent dual-path execution**

**Device Path**: Hardware-realistic execution

- Measurement grouping + shot scheduling
- Counts-based postprocessing
- Noise models and error mitigation

**Numeric Path**: Exact numeric simulation

- Direct statevector/MPS computation
- Exact expectation values
- PyTorch autograd support

**Key Innovation**: **Same algorithm API, explicit path selection**

**Example**:

.. code-block:: python

   from tyxonq.applications.chem import UCCSD
   import tyxonq as tq
   
   # Same algorithm object
   uccsd = UCCSD(n_qubits=4, hamiltonian=H)
   
   # Device path - hardware-realistic
   energy_device = uccsd.kernel(
       runtime="device",
       shots=4096,
       provider="ibm",
       device="ibm_quebec"
   )
   
   # Numeric path - exact computation  
   energy_numeric = uccsd.kernel(
       runtime="numeric"
   )
   
   # Validate consistency
   assert abs(energy_device - energy_numeric) < threshold

**Advantages**:

- ✅ **Semantic consistency**: Same Hamiltonian, same measurement grouping
- ✅ **Explicit control**: User chooses path via ``runtime`` parameter
- ✅ **Mutual validation**: Device results validated against numeric baseline
- ✅ **Seamless switching**: Change runtime without code changes

4. Numeric Backend Abstraction
-------------------------------

Qiskit Approach
~~~~~~~~~~~~~~~

**Design**: Provider-specific implementations

- Different simulators (Aer, BasicAer) use different backends
- No unified numeric interface
- Limited ML framework integration
- Gradient computation requires manual implementation

**Issues**:

- ❌ Code duplication across simulators
- ❌ Hard to add new backends (e.g., CuPy for GPU)
- ❌ ML integration requires wrappers

PennyLane Approach
~~~~~~~~~~~~~~~~~~

**Design**: Interface adapters for different backends

- Separate plugins for NumPy, TensorFlow, JAX, PyTorch
- Each plugin reimplements operations
- Automatic differentiation via framework integration

**Benefits**:

- ✅ Supports multiple ML frameworks
- ❌ Code duplication across plugins
- ❌ Maintenance overhead

TyxonQ Approach
~~~~~~~~~~~~~~~

**Design**: **Single ArrayBackend protocol**

**Unified Interface**: All backends implement same protocol

.. code-block:: python

   # Single ArrayBackend protocol
   from tyxonq.numerics.api import ArrayBackend
   import tyxonq as tq
   
   # Seamless backend switching
   tq.set_backend("numpy")     # Development
   tq.set_backend("pytorch")   # ML integration + autograd
   tq.set_backend("cupynumeric")  # GPU acceleration
   
   # Same code works across all backends
   nb = tq.get_backend()
   
   # Standard array operations
   psi = nb.zeros((2**n_qubits,), dtype=nb.complex128)
   H = nb.eye(2**n_qubits, dtype=nb.complex128)
   energy = nb.real(nb.conj(psi).T @ H @ psi)
   
   # PyTorch backend: automatic differentiation
   if backend_name == "pytorch":
       energy.backward()  # Gradient computation

**Advantages**:

- ✅ **Write once, run everywhere**: Same code across backends
- ✅ **Easy to extend**: Add new backend by implementing protocol
- ✅ **Native gradients**: PyTorch autograd without wrappers
- ✅ **Performance**: GPU acceleration via CuPyNumeric

5. Quantum Chemistry Applications
----------------------------------

Qiskit Nature
~~~~~~~~~~~~~

**Design**: Separate package (Qiskit Nature)

- Requires additional installation
- Less integrated with core Qiskit
- Focused on algorithm research

**Features**:

- VQE, QAOA implementations
- PySCF integration
- Multiple ansatz types

PennyLane QChem
~~~~~~~~~~~~~~~

**Design**: Built-in quantum chemistry module

- Integrated with PennyLane core
- Automatic differentiation for gradients
- Focus on machine learning integration

**Features**:

- Molecular Hamiltonians
- VQE with autograd
- Limited dual-path support

TyxonQ Quantum AIDD
~~~~~~~~~~~~~~~~~~~

**Design**: **Native first-class application stack**

**Inspiration**: Complete rewrite of TenCirChem algorithms

.. code-block:: python

   # From: src/tyxonq/applications/chem/__init__.py
   # ReWrite TenCirChem with TyxonQ

**Key Features**:

1. **PySCF-level UX**: Familiar molecular specification
2. **Hardware-realistic**: Dual-path execution model
3. **Complete algorithm suite**: HEA, UCC/UCCSD, k-UpCCGSD, pUCCD
4. **Cloud/local hybrid**: Offload heavy PySCF kernels to cloud

**Example**:

.. code-block:: python

   from tyxonq.applications.chem import UCCSD, HEA
   from tyxonq.applications.chem import molecule
   
   # Preset molecules or custom PySCF Mole
   mol = molecule.h2o
   
   # Unified algorithm API
   uccsd = UCCSD(mol)
   hea = HEA(mol, layers=3)
   
   # Device path - hardware execution
   e_device = uccsd.kernel(runtime="device", shots=4096)
   
   # Numeric path - exact computation with autograd
   e_numeric = uccsd.kernel(runtime="numeric")
   
   # Cloud offloading for heavy classical kernels
   e_cloud = uccsd.kernel(
       runtime="device",
       classical_provider="cloud",
       classical_device="gpu_node"
   )

**Advantages**:

- ✅ **Native integration**: Not a separate package
- ✅ **Dual-path consistency**: Device and numeric paths validated
- ✅ **Production-ready**: Cloud offloading for scalability
- ✅ **Drug discovery focus**: AI-driven drug discovery workflows

6. Compilation and Optimization
--------------------------------

Qiskit Transpiler
~~~~~~~~~~~~~~~~~

**Design**: Pass-based transpiler

- Extensive pass library
- Configurable optimization levels
- Hardware-specific compilation

**Strengths**:

- ✅ Mature and well-tested
- ✅ Rich optimization passes
- ❌ Modifies circuit in-place
- ❌ No stable IR between passes

PennyLane Transforms
~~~~~~~~~~~~~~~~~~~~

**Design**: Decorator-based transforms

- Functional approach to circuit modification
- Automatic differentiation integration
- Limited to QNode context

**Strengths**:

- ✅ Composable transforms
- ❌ Tied to QNode abstraction
- ❌ Limited hardware-specific optimization

TyxonQ Pluggable Compilers
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Design**: **Multiple compiler engines with stable IR**

**Native Compiler**:

- Measurement rewrite and grouping
- Light-cone simplification  
- Shot scheduling
- Preserves IR structure, generates metadata

**Qiskit Adapter**:

- Use Qiskit transpiler as compilation backend
- Convert TyxonQ IR ↔ Qiskit QuantumCircuit
- Hardware-specific optimizations

**Example**:

.. code-block:: python

   import tyxonq as tq
   
   c = tq.Circuit(4).h(0).cx(0,1).cx(1,2).cx(2,3)
   
   # Native compiler
   compiled_native = c.compile(
       engine="tyxonq",
       passes=["measurement_rewrite", "shot_scheduler"]
   )
   
   # Qiskit transpiler
   compiled_qiskit = c.compile(
       engine="qiskit",
       output="qasm",
       optimization_level=3
   )

**Advantages**:

- ✅ **Pluggable**: Choose compiler based on needs
- ✅ **Interoperable**: Use Qiskit's optimizations when needed
- ✅ **Stable**: IR preserved across compilation
- ✅ **Extensible**: Easy to add new compiler engines

7. Postprocessing and Error Mitigation
---------------------------------------

Qiskit Approach
~~~~~~~~~~~~~~~

**Location**: Provider-dependent implementations

- Readout mitigation in Aer
- Hardware-specific mitigation in IBMQ provider
- No unified postprocessing layer

**Issues**:

- ❌ Vendor lock-in
- ❌ Inconsistent APIs
- ❌ Hard to compare mitigation strategies

PennyLane Approach
~~~~~~~~~~~~~~~~~~

**Location**: Built into device/QNode execution

- Automatic postprocessing for expectation values
- Limited mitigation support
- Tied to QNode abstraction

TyxonQ Approach
~~~~~~~~~~~~~~~

**Design**: **Unified pluggable postprocessing layer**

**Features**:

- **Counts-first semantics**: All devices return uniform counts format
- **Pluggable strategies**: Readout correction, ZNE, custom methods
- **Metadata-driven**: Smart processing based on compiler metadata
- **Cross-vendor**: Same postprocessing for all devices

**Example**:

.. code-block:: python

   import tyxonq as tq
   
   c = tq.Circuit(2).h(0).cx(0,1)
   
   # Different postprocessing strategies
   result_raw = c.device(provider="ibm", device="ibm_quebec", shots=4096) \
                 .postprocessing(method=None) \
                 .run()
   
   result_readout = c.device(provider="ibm", device="ibm_quebec", shots=4096) \
                     .postprocessing(method="readout_mitigation") \
                     .run()
   
   result_zne = c.device(provider="ibm", device="ibm_quebec", shots=4096) \
                 .postprocessing(method="zero_noise_extrapolation") \
                 .run()

**Advantages**:

- ✅ **Unified API**: Same interface for all mitigation methods
- ✅ **Composable**: Chain multiple mitigation strategies
- ✅ **Portable**: Works across all device providers
- ✅ **Extensible**: Easy to add custom mitigation methods

Summary: TyxonQ's Unique Advantages
====================================

Architectural Innovations
--------------------------

1. **Stable IR as System Contract**
   
   - Qiskit/PennyLane: Implicit or transform-based IR
   - TyxonQ: **Explicit minimal IR with metadata layer**

2. **Compiler-Driven Measurement Optimization**
   
   - Qiskit/PennyLane: Device-layer or runtime handling
   - TyxonQ: **Compiler generates grouping metadata**

3. **Semantic-Consistent Dual-Path**
   
   - Qiskit: Separate Aer/Terra ecosystems
   - PennyLane: QNode abstraction
   - TyxonQ: **Explicit device/numeric paths with same semantics**

4. **Single ArrayBackend Protocol**
   
   - Qiskit: Provider-specific implementations
   - PennyLane: Multiple plugins
   - TyxonQ: **Unified protocol for NumPy/PyTorch/CuPyNumeric**

5. **Counts-First Postprocessing**
   
   - Qiskit/PennyLane: Provider-dependent
   - TyxonQ: **Unified pluggable layer**

Quantum Chemistry Excellence
-----------------------------

- **TenCirChem-inspired**: Complete rewrite of proven algorithms
- **PySCF integration**: Seamless classical chemistry workflows
- **Dual-path validation**: Device results validated against numeric baselines
- **Cloud/local hybrid**: Offload heavy kernels while keeping VQE local
- **Drug discovery focus**: Quantum AIDD application stack

When to Choose TyxonQ
=====================

**Choose TyxonQ if you need**:

✅ **Hardware-realistic research**: Dual-path execution for validation

✅ **Quantum chemistry applications**: Drug discovery, molecular simulation

✅ **Cross-vendor portability**: Write once, run on any provider

✅ **Reproducible experiments**: Deterministic compilation and postprocessing

✅ **ML framework integration**: Native PyTorch autograd support

✅ **Production deployment**: Cloud/local hybrid, stable APIs

**Choose Qiskit if you need**:

- Mature ecosystem with extensive documentation
- Direct access to IBM Quantum Network
- Rich library of optimization passes
- Established community and third-party tools

**Choose PennyLane if you need**:

- Automatic differentiation focus
- Strong QML (Quantum Machine Learning) support
- Simple QNode abstraction
- Integration with ML research workflows

Conclusion
==========

TyxonQ's architectural innovations address fundamental challenges in quantum software engineering:

- **Ecosystem fragmentation** → Stable IR and unified abstractions
- **Vendor lock-in** → Cross-vendor portability  
- **Research vs. hardware gap** → Dual-path execution model
- **Reproducibility issues** → Deterministic compilation and postprocessing

By learning from established frameworks (Qiskit, PennyLane) and building upon proven algorithms (TensorCircuit, TenCirChem), TyxonQ provides a next-generation platform that balances:

- **Research flexibility** (numeric path with autograd)
- **Hardware realism** (device path with noise and mitigation)
- **Engineering usability** (stable APIs, clear execution flow)
- **Domain specialization** (Quantum AIDD for drug discovery)

For detailed implementation examples and migration guides, see the main TyxonQ documentation.

Comparison with Other Frameworks documentation will be added soon.
