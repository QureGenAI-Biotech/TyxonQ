Architecture Design
===================

====================
Architecture Design
====================

Core Design Principles
======================

1. **Separation of Concerns**: Strict layering (Applications → Core → Execution → Libraries)
2. **Stable Contracts**: Minimal IR, ArrayBackend protocol, Device protocol
3. **Explicit Over Implicit**: Chain API makes execution flow transparent
4. **Hardware Realism**: Measurement-first design, counts-first semantics

Layered Architecture
====================

**Applications Layer**
  - Quantum AIDD (Drug Discovery)
  - Optimization Algorithms (VQE, QAOA)
  - Machine Learning Integration

**Core Framework**
  - Stable IR (``num_qubits`` + ``ops`` + ``metadata``)
  - Compiler Pipeline (measurement optimization, shot scheduling)
  - Device Abstraction (unified interface)
  - Numeric Backend (ArrayBackend protocol)
  - Postprocessing (counts-first, pluggable mitigation)

**Execution Layer**
  - Simulators (Statevector, MPS, Density Matrix)
  - Real Hardware (IBM, TyxonQ QPU)
  - Cloud Classical Kernels (PySCF HF/MP2/CCSD)

**Libraries**
  - Circuit Library (templates: VQE, QAOA, UCC)
  - Quantum Library (numeric kernels)
  - Hamiltonian Encoding (OpenFermion bridge)

Core Components
===============

1. Stable IR
------------

**Location**: ``src/tyxonq/core/ir/circuit.py``

**Design**: Minimal and stable

.. code-block:: python

   @dataclass
   class Circuit:
       num_qubits: int              # Number of qubits
       ops: List[Any]               # Operation sequence  
       metadata: Dict[str, Any]     # Compiler metadata

**Benefits**: Version stability, inspectable, extensible metadata layer

2. Compiler Architecture
------------------------

**Location**: ``src/tyxonq/compiler/``

**Pluggable Design**: Multiple engines (native, Qiskit adapter)

**Key Passes**:

- **Measurement Rewrite**: Groups commuting Pauli terms
- **Shot Scheduler**: Variance-weighted shot allocation
- **Light-cone Simplification**: Prunes unnecessary gates

3. Device Abstraction
---------------------

**Location**: ``src/tyxonq/devices/``

**Unified Protocol**:

.. code-block:: python

   class Device(Protocol):
       def run(self, circuit: Circuit, shots: int) -> Dict: ...
       @property
       def capabilities(self) -> Dict: ...

**Simulators**: Statevector (O(2^n)), Density Matrix (O(4^n)), MPS (O(poly(n)))

**Hardware**: IBM Quantum, TyxonQ native QPU

4. Numeric Backend
------------------

**Location**: ``src/tyxonq/numerics/``

**ArrayBackend Protocol**: Unified interface for NumPy/PyTorch/CuPyNumeric

**Key Operations**: ``zeros``, ``matmul``, ``einsum``, ``kron``, ``conj``, ``real``

**Benefits**: Write once, run everywhere; native PyTorch autograd; GPU acceleration

5. Postprocessing
-----------------

**Location**: ``src/tyxonq/postprocessing/``

**Unified Interface**: ``apply_postprocessing(result, method, **options)``

**Methods**:

- **Counts → Expectations**: Pauli expectation from counts
- **Readout Mitigation**: Calibration matrix correction
- **Zero-Noise Extrapolation**: Noise amplification + extrapolation

Dual-Path Execution
===================

Device Path
-----------

**For**: Hardware-realistic execution

**Flow**: Circuit → Compile (grouping) → Device → Counts → Postprocess → Energy

**Features**: Measurement grouping, shot scheduling, noise models, error mitigation

Numeric Path
------------

**For**: Fast iteration, gradient computation

**Flow**: Circuit → Direct statevector → Exact expectation → Energy

**Features**: Exact simulation, PyTorch autograd, CI-vector space

**Key Innovation**: Same algorithm API, explicit runtime selection

.. code-block:: python

   # Same algorithm, different paths
   uccsd = UCCSD(molecule)
   
   e_device = uccsd.kernel(runtime="device", shots=4096)
   e_numeric = uccsd.kernel(runtime="numeric")  # Exact + autograd

Quantum Chemistry Stack
=======================

**Inspiration**: Complete rewrite of TenCirChem algorithms

**Location**: ``src/tyxonq/applications/chem/``

**Components**:

1. **Algorithms** (``algorithms/``): HEA, UCC/UCCSD, k-UpCCGSD, pUCCD
2. **Runtimes** (``runtimes/``): Device runtime, numeric runtime  
3. **Chem Libs** (``chem_libs/``): Circuit library, quantum library, Hamiltonian builders

**Integration**: PySCF (molecular input, HF, integrals), OpenFermion (fermion-to-qubit)

**Cloud Offloading**: Heavy PySCF kernels (HF/MP2/CCSD) to cloud, VQE local

Source Code Organization
========================

.. code-block:: text

   src/tyxonq/
   ├── core/                    # IR and semantics
   │   ├── ir/circuit.py        # Circuit IR + chain API
   │   └── operations/          # Gate operations
   ├── compiler/                # Compilation pipeline
   │   ├── compile_engine/
   │   │   ├── native/          # TyxonQ compiler
   │   │   └── qiskit/          # Qiskit adapter
   │   └── stages/              # Compiler passes
   ├── devices/                 # Device abstraction
   │   ├── simulators/          # Local simulators
   │   └── hardware/            # Hardware drivers
   ├── numerics/                # Numeric backends
   │   ├── api.py               # ArrayBackend protocol
   │   └── backends/            # NumPy/PyTorch/CuPy
   ├── postprocessing/          # Unified postprocessing
   ├── applications/chem/       # Quantum chemistry
   │   ├── algorithms/          # VQE algorithms
   │   ├── runtimes/            # Dual runtimes
   │   └── chem_libs/           # Chemistry libraries
   └── libs/                    # General libraries
       ├── circuits_library/    # Circuit templates
       ├── quantum_library/     # Quantum kernels
       └── hamiltonian_encoding/# OpenFermion I/O

Design Patterns
===============

1. Chain API Pattern
--------------------

**Purpose**: Explicit execution flow

**Pattern**:

.. code-block:: python

   result = (
       circuit
       .compile(passes=[...])
       .device(provider=..., device=..., shots=...)
       .postprocessing(method=...)
       .run()
   )

**Benefits**: Transparent, configurable, lazy evaluation

2. Protocol-Based Abstraction
------------------------------

**Purpose**: Pluggable components without inheritance

**Pattern**:

.. code-block:: python

   class Component(Protocol):
       def interface_method(self, ...) -> ...: ...

**Benefits**: Loose coupling, easy testing, type safety

3. Metadata-Driven Processing
------------------------------

**Purpose**: Compiler-generated info guides execution

**Pattern**: Compiler attaches metadata → Device uses metadata → Postprocessing reads metadata

**Benefits**: Deterministic, inspectable, portable

4. Dual-Path Polymorphism
--------------------------

**Purpose**: Same algorithm, different execution strategies

**Pattern**: Algorithm selects runtime based on ``runtime`` parameter

**Benefits**: Validation, flexibility, seamless switching

For detailed implementation, see TYXONQ_TECHNICAL_WHITEPAPER.md and source code.

Architecture Design documentation will be added soon.
