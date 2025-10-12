Architecture Overview
=====================

TyxonQ is a modular quantum computing framework built around the core workflow: **问题 → 哈密顿量 → 电路 → 编译 → 执行 → 后处理**. This document provides an overview of TyxonQ's actual architecture based on the technical whitepaper and codebase.

.. contents:: Table of Contents
   :local:
   :depth: 3

Core Workflow
=============

TyxonQ implements the complete quantum computing pipeline:

.. code-block:: text

    问题定义 (Problem Definition)
           ↓
    哈密顿量构建 (Hamiltonian Construction)  
           ↓
    量子电路构建 (Quantum Circuit Building)
           ↓
    编译优化 (Compilation & Optimization)
           ↓
    设备执行 (Device Execution)
           ↓
    后处理 (Postprocessing)

System Architecture
==================

TyxonQ adopts a layered, modular architecture:

.. code-block:: text

    ┌───────────────────────────────────────────────────────────┐
    │                Applications Layer                           │
    │    Quantum Chemistry • Optimization • Machine Learning      │
    └───────────────────────────────────────────────────────────┘
                                 │
    ┌───────────────────────────────────────────────────────────┐
    │                 Core Framework                               │
    │    Stable IR • Compiler • Device Abstraction • Numerics    │
    └───────────────────────────────────────────────────────────┘
                                 │
    ┌───────────────────────────────────────────────────────────┐
    │               Execution Engines                              │
    │    Statevector • Density Matrix • MPS • Hardware           │
    └───────────────────────────────────────────────────────────┘

Key Architectural Innovations
============================

1. **Stable IR as System Contract**
2. **Compiler-Driven Measurement Optimization** 
3. **Dual-Path Execution Model**
4. **Counts-First Semantics**
5. **Single ArrayBackend Protocol**

Stable IR (Intermediate Representation)
======================================

TyxonQ's core IR is minimal and stable:

.. code-block:: python

    from tyxonq.core.ir.circuit import Circuit
    
    # Minimal design: num_qubits + ops sequence
    circuit = Circuit(num_qubits=2)
    circuit.h(0)      # ops: [('h', 0)]
    circuit.cx(0, 1)  # ops: [('h', 0), ('cx', 0, 1)]
    
    # Core structure:
    # - circuit.num_qubits: int
    # - circuit.ops: List[Any] - operation sequence
    # - circuit.metadata: Dict - compiler metadata

Chain API Design
===============

TyxonQ's signature chain API makes execution flow explicit:

.. code-block:: python

    import tyxonq as tq
    
    # Set numeric backend
    tq.set_backend("numpy")  # or "pytorch", "cupynumeric"
    
    # Build circuit
    circuit = tq.Circuit(2).h(0).cx(0, 1)
    
    # Chain execution: compile → device → run
    result = (
        circuit
        .compile(passes=["measurement_rewrite", "shot_scheduler"])
        .device(provider="simulator", device="statevector", shots=1024)
        .run()
    )

Compiler Pipeline
================

Compiler transforms circuits with measurement optimization:

.. code-block:: python

    # Native compiler pipeline
    final_pipeline = [
        "rewrite/auto_measure",      # Add measurements
        "rewrite/gates_transform",   # Gate transformations  
        "measurement_rewrite",       # Group measurements
        "shot_scheduler"              # Optimize shot allocation
    ]

**Key Features:**
- Measurement grouping metadata elevation
- Deterministic shot scheduling
- Multiple compiler engines (native, qiskit)

Device Abstraction
==================

Unified device interface:

.. code-block:: python

    # Simulator devices
    statevector_result = (
        circuit
        .device(provider="simulator", device="statevector", shots=1024)
        .run()
    )
    
    density_matrix_result = (
        circuit
        .device(provider="simulator", device="density_matrix", shots=1024) 
        .run()
    )
    
    mps_result = (
        circuit
        .device(provider="simulator", device="matrix_product_state", shots=1024)
        .run()
    )
    
    # Hardware devices
    hardware_result = (
        circuit.compile(output="qasm")
        .device(provider="tyxonq", device="homebrew_s2", shots=1024)
        .run()
    )

Simulation Engines
=================

Statevector Engine
-----------------

.. code-block:: python

    # Located: src/tyxonq/devices/simulators/statevector/engine.py
    class StatevectorEngine:
        name = "statevector"
        capabilities = {"supports_shots": True}
        
        def __init__(self, backend_name=None):
            self.backend = get_backend(backend_name)
        
        def run(self, circuit, shots=None, **kwargs):
            # Statevector simulation
            # Memory: O(2^n)
            pass

Density Matrix Engine
--------------------

.. code-block:: python

    # Located: src/tyxonq/devices/simulators/density_matrix/engine.py
    class DensityMatrixEngine:
        name = "density_matrix"
        capabilities = {"supports_shots": True}
        
        def run(self, circuit, shots=None, **kwargs):
            # Density matrix simulation with noise support
            # Memory: O(4^n)
            pass

Matrix Product State Engine
--------------------------

.. code-block:: python

    # Located: src/tyxonq/devices/simulators/matrix_product_state/engine.py  
    class MatrixProductStateEngine:
        name = "matrix_product_state"
        capabilities = {"supports_shots": True}
        
        def run(self, circuit, shots=None, **kwargs):
            # MPS simulation for low entanglement
            # Memory: O(poly(n))
            pass

Numeric Backend System
=====================

ArrayBackend Protocol:

.. code-block:: python

    # Located: src/tyxonq/numerics/api.py
    from tyxonq.numerics.api import get_backend
    import tyxonq as tq
    
    # Supported backends
    tq.set_backend("numpy")        # Default, CPU
    tq.set_backend("pytorch")      # GPU + autodiff
    tq.set_backend("cupynumeric")  # GPU accelerated
    
    # Get backend for operations
    backend = get_backend()
    array = backend.zeros((4, 4), dtype=backend.complex64)

Dual-Path Execution Model
========================

Device Path
----------

Optimized for hardware and shot-based simulation:

.. code-block:: python

    # Device path - counts-based results
    result = (
        circuit
        .device(provider="simulator", device="statevector", shots=4096)
        .run()
    )
    
    # Process counts
    counts = result[0]["result"] if isinstance(result, list) else result.get("result", {})
    
    # Postprocessing
    from tyxonq.postprocessing import metrics
    expectation = metrics.expectation(counts, z=[0, 1])

Numeric Path
-----------

Optimized for exact computation:

.. code-block:: python

    # Numeric path - exact computation
    from tyxonq.applications.chem.algorithms.uccsd import UCCSD
    from tyxonq.applications.chem import molecule
    
    uccsd = UCCSD(molecule.h2)
    
    # Exact energy
    energy = uccsd.energy(params, runtime="numeric")
    
    # With gradients
    tq.set_backend("pytorch")
    energy, grad = uccsd.energy_and_grad(params, runtime="numeric")

Quantum Chemistry Stack
======================

Location: `src/tyxonq/applications/chem/`

.. code-block:: python

    from tyxonq.applications.chem import HEA, UCC, UCCSD
    from tyxonq.applications.chem.molecule import h2, h4
    
    # Hardware-Efficient Ansatz
    hea = HEA(n_qubits=4, layers=2, hamiltonian=h2.hamiltonian)
    
    # Unitary Coupled Cluster
    ucc = UCC(molecule=h2)
    
    # UCCSD Algorithm
    uccsd = UCCSD(molecule=h2)
    # Device execution  
    energy_device = uccsd.kernel(shots=2048, provider="simulator", device="statevector")
    # Numeric execution
    energy_numeric = uccsd.energy(params, runtime="numeric")

Runtimes
-------

.. code-block:: python

    # Located: src/tyxonq/applications/chem/runtimes/
    # - hea_device_runtime.py
    # - hea_numeric_runtime.py  
    # - ucc_device_runtime.py
    
    class HEADeviceRuntime:
        # Device path for HEA
        pass
        
    class HEANumericRuntime:
        # Numeric path for HEA
        pass

Postprocessing
=============

Counts-first semantics with unified postprocessing:

.. code-block:: python

    # Located: src/tyxonq/postprocessing/
    from tyxonq.postprocessing import metrics
    from tyxonq.postprocessing.error_mitigation import apply_zne, apply_dd, apply_rc
    
    # Basic expectation values
    ez = metrics.expectation(counts, z=[0, 1])
    ex = metrics.expectation(counts, x=[0, 1])
    
    # Error mitigation
    mitigated_result = apply_zne(circuit, executor, num_to_average=3)

Libraries
========

Circuit Library
--------------

Location: `src/tyxonq/libs/circuits_library/`

.. code-block:: python

    # Reusable circuit templates
    from tyxonq.libs.circuits_library.qiskit_real_amplitudes import build_circuit_from_template

Quantum Library
--------------

Location: `src/tyxonq/libs/quantum_library/`

.. code-block:: python

    # Numeric kernels for gates
    from tyxonq.libs.quantum_library.kernels.gates import (
        gate_h, gate_rz, gate_rx, gate_cx_4x4
    )
    
    # Density matrix operations
    from tyxonq.libs.quantum_library.kernels.density_matrix import (
        init_density, apply_1q_density, apply_2q_density
    )

Hamiltonian Encoding
-------------------

Location: `src/tyxonq/libs/hamiltonian_encoding/`

.. code-block:: python

    # OpenFermion integration
    from openfermion import QubitOperator

Cloud Integration
================

Location: `src/tyxonq/cloud/`

TyxonQ Hardware Driver:

.. code-block:: python

    # Located: src/tyxonq/devices/hardware/tyxonq/driver.py
    # Cloud API integration for TyxonQ hardware
    
    def submit_task(device, source, shots=1024, **kwargs):
        # Submit to TyxonQ cloud API
        pass
    
    def get_task_details(task, wait=True):
        # Poll for results
        pass

Directory Structure
==================

Actual source code organization:

.. code-block:: text

    src/tyxonq/
    ├── core/                       # Stable IR and operations
    │   ├── ir/
    │   │   ├── circuit.py          # Circuit IR with chain API
    │   │   └── pulse.py            # Pulse-level IR
    ├── compiler/                   # Compilation pipeline
    │   ├── api.py
    │   └── compile_engine/
    │       ├── native/             # Native compiler
    │       └── qiskit/             # Qiskit adapter
    ├── devices/                    # Device abstraction
    │   ├── base.py                 # Device protocol
    │   ├── simulators/
    │   │   ├── statevector/
    │   │   ├── density_matrix/
    │   │   └── matrix_product_state/
    │   └── hardware/
    │       └── tyxonq/driver.py    # TyxonQ hardware
    ├── numerics/                   # ArrayBackend protocol
    │   ├── api.py
    │   └── backends/
    │       ├── numpy_backend.py
    │       ├── pytorch_backend.py
    │       └── cupynumeric_backend.py
    ├── postprocessing/             # Unified postprocessing
    │   ├── metrics.py
    │   └── error_mitigation.py
    ├── applications/               # Domain applications
    │   └── chem/                   # Quantum chemistry
    │       ├── algorithms/
    │       ├── runtimes/
    │       └── molecule.py
    └── libs/                       # Library components
        ├── circuits_library/
        ├── quantum_library/
        └── hamiltonian_encoding/

Performance Characteristics
=========================

.. code-block:: text

    Engine              | Memory      | Best Use Case
    --------------------|-------------|---------------------------
    Statevector         | O(2^n)     | Pure states, < 20 qubits
    Density Matrix      | O(4^n)     | Noise modeling, < 15 qubits  
    Matrix Product State| O(poly(n)) | Low entanglement, large n

Configuration
============

.. code-block:: python

    # Global configuration
    import tyxonq as tq
    
    # Backend configuration
    tq.set_backend("numpy")  # or "pytorch", "cupynumeric"
    
    # Device defaults
    tq.device(provider="simulator", device="statevector", shots=2048)
    tq.compile(compile_engine="native")
    
    # Use defaults
    result = tq.Circuit(2).h(0).cx(0, 1).run()

.. note::
   This architecture overview is based on TyxonQ's actual implementation
   as documented in the technical whitepaper and source code.

.. seealso::
   
   - :doc:`contributing` - Development guidelines
   - :doc:`extending_tyxonq` - Extension guide  
   - :doc:`custom_devices` - Device development
   - :doc:`testing_guidelines` - Testing practices