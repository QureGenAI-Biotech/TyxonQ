# API Reference

<cite>
**Referenced Files in This Document**   
- [src/tyxonq/core/__init__.py](file://src/tyxonq/core/__init__.py)
- [src/tyxonq/compiler/__init__.py](file://src/tyxonq/compiler/__init__.py)
- [src/tyxonq/compiler/api.py](file://src/tyxonq/compiler/api.py)
- [src/tyxonq/devices/__init__.py](file://src/tyxonq/devices/__init__.py)
- [src/tyxonq/numerics/__init__.py](file://src/tyxonq/numerics/__init__.py)
- [src/tyxonq/numerics/api.py](file://src/tyxonq/numerics/api.py)
- [src/tyxonq/postprocessing/__init__.py](file://src/tyxonq/postprocessing/__init__.py)
- [src/tyxonq/cloud/__init__.py](file://src/tyxonq/cloud/__init__.py)
- [src/tyxonq/cloud/api.py](file://src/tyxonq/cloud/api.py)
- [src/tyxonq/applications/__init__.py](file://src/tyxonq/applications/__init__.py)
</cite>

## Table of Contents
1. [Core](#core)
2. [Compiler](#compiler)
3. [Devices](#devices)
4. [Numerics](#numerics)
5. [Postprocessing](#postprocessing)
6. [Cloud](#cloud)
7. [Applications](#applications)
8. [Libraries](#libraries)
9. [Visualization](#visualization)

## Core

The Core module provides fundamental data structures for quantum circuits and Hamiltonians. It serves as the foundation for all quantum program representations in TyxonQ.

### Circuit
- **Class**: `Circuit(num_qubits: int)`
- **Purpose**: Represents a quantum circuit with a specified number of qubits.
- **Key Methods**:
  - `h(qubit)`: Apply Hadamard gate.
  - `cx(control, target)`: Apply CNOT gate.
  - `rx(qubit, theta)`: Apply X-rotation gate.
  - `measure_z(qubit)`: Measure qubit in Z-basis.
  - `to_json_str(indent=None)`: Serialize circuit to JSON string.
  - `from_json_str(json_str)`: Deserialize circuit from JSON string.

### Hamiltonian
- **Class**: `Hamiltonian`
- **Purpose**: Represents a quantum Hamiltonian operator for energy calculations.
- **Usage**: Used in variational algorithms like VQE and QAOA to define the problem Hamiltonian.

**Section sources**
- [src/tyxonq/core/__init__.py](file://src/tyxonq/core/__init__.py#L0-L11)

## Compiler

The Compiler module provides a unified interface for quantum circuit compilation, supporting multiple compilation backends and transformation passes.

### CompileResult
- **Type**: `TypedDict`
- **Fields**:
  - `circuit`: The compiled `Circuit` object.
  - `metadata`: Dictionary containing compilation metadata.

### Pass
- **Protocol**: Defines a compilation pass interface.
- **Method**:
  - `execute_plan(circuit: Circuit, **opts) -> Circuit`: Transforms the input circuit.

### compile()
- **Function**: `compile(circuit, *, compile_engine="default", output="ir", compile_plan=None, device_rule=None, options=None)`
- **Parameters**:
  - `circuit`: Input IR circuit to compile.
  - `compile_engine`: Compilation backend ("tyxonq", "qiskit", "native", or "default").
  - `output`: Output format ("ir", "qasm2", "qiskit").
  - `compile_plan`: Optional list of compilation stages.
  - `device_rule`: Target device constraints.
  - `options`: Engine-specific compilation options.
- **Returns**: `CompileResult` containing the compiled circuit and metadata.
- **Exceptions**: Falls back to native compiler if specified engine is unavailable.

**Section sources**
- [src/tyxonq/compiler/__init__.py](file://src/tyxonq/compiler/__init__.py#L0-L7)
- [src/tyxonq/compiler/api.py](file://src/tyxonq/compiler/api.py#L0-L65)

## Devices

The Devices module manages execution targets for quantum circuits, including simulators and hardware devices.

### Device
- **Class**: Base class for all quantum devices.
- **Responsibilities**: Circuit execution, result handling, and device-specific configuration.

### DeviceRule
- **Class**: Defines constraints and capabilities of a target device (e.g., connectivity, gate set).

### RunResult
- **Class**: Encapsulates execution results from a quantum device.
- **Contains**: Measurement outcomes, metadata, and execution statistics.

**Section sources**
- [src/tyxonq/devices/__init__.py](file://src/tyxonq/devices/__init__.py#L0-L7)

## Numerics

The Numerics module provides a unified interface for array and tensor operations across different computational backends.

### ArrayBackend
- **Protocol**: Defines the interface for numerical backends.
- **Key Attributes**:
  - `name`: Backend identifier.
  - Dtype constants: `complex64`, `complex128`, `float32`, `float64`, `int32`, `int64`, `bool`, `int`.
- **Core Methods**:
  - `array(data, dtype)`: Create array.
  - `matmul(a, b)`: Matrix multiplication.
  - `einsum(subscripts, *operands)`: Einstein summation.
  - `requires_grad(x, flag)`: Enable/disable gradient tracking.
  - `vmap(fn)`: Vectorize function.
  - `jit(fn)`: Just-in-time compile function.
  - `value_and_grad(fn, argnums=0)`: Compute value and gradients.

### VectorizationPolicy
- **Type**: `Literal["auto", "force", "off"]`
- **Usage**: Controls vectorization behavior in numerical computations.

### get_backend(name)
- **Function**: Returns an `ArrayBackend` instance by name.
- **Supported Backends**: "numpy", "pytorch", "cupynumeric".
- **Returns**: Backend instance or raises `RuntimeError` if unavailable.

### NumericBackend (Class Proxy)
- **Class**: Provides class-level access to the current backend.
- **Methods**:
  - All array operations accessible as class methods (e.g., `NumericBackend.array(data)`).
  - Dtype access via class properties (e.g., `NumericBackend.complex64`).

**Section sources**
- [src/tyxonq/numerics/__init__.py](file://src/tyxonq/numerics/__init__.py#L0-L197)
- [src/tyxonq/numerics/api.py](file://src/tyxonq/numerics/api.py#L0-L194)

## Postprocessing

The Postprocessing module provides tools for analyzing and transforming quantum computation results.

### apply_postprocessing()
- **Function**: `apply_postprocessing(result, options=None)`
- **Parameters**:
  - `result`: Dictionary containing execution results (counts, statevector, etc.).
  - `options`: Dictionary with `method` key specifying the postprocessing technique.
- **Supported Methods**:
  - `"readout_mitigation"`: Applies readout error mitigation using calibration data.
    - Options: `cals` (calibration data), `mitigation` (method: "inverse", etc.).
  - `"expval_pauli_term"`: Computes expectation value of a Pauli term from counts.
    - Options: `idxs` (qubit indices).
  - `"expval_pauli_terms"`: Computes expectations for multiple Pauli terms.
    - Options: `terms` (list of terms).
  - `"expval_pauli_sum"`: Computes expectation of a full Hamiltonian from grouped terms.
    - Options: `items`, `identity_const`, optional `readout_cals`.
- **Returns**: Dictionary with `method` and `result` fields, containing processed output or error information.

**Section sources**
- [src/tyxonq/postprocessing/__init__.py](file://src/tyxonq/postprocessing/__init__.py#L0-L135)

## Cloud

The Cloud module provides a unified interface for interacting with remote quantum computing services.

### set_token()
- **Function**: `set_token(token, *, provider=None, device=None)`
- **Purpose**: Sets authentication token for cloud providers.
- **Returns**: Dictionary of configured tokens.

### set_default()
- **Function**: `set_default(*, provider=None, device=None)`
- **Purpose**: Sets default provider and device for subsequent operations.

### device()
- **Function**: `device(name=None, *, provider=None, id=None, shots=None)`
- **Purpose**: Retrieves device descriptor.
- **Returns**: Dictionary with device properties.

### list_devices()
- **Function**: `list_devices(*, provider=None, token=None, **kws)`
- **Purpose**: Lists available quantum devices.
- **Returns**: List of device names.

### submit_task()
- **Function**: `submit_task(*, provider, device, circuit, source, shots, token, auto_compile, **opts)`
- **Purpose**: Submits a quantum task for execution.
- **Parameters**: Supports both circuit objects and source code strings.

### get_task_details()
- **Function**: `get_task_details(task, *, token=None, wait=False, poll_interval=2.0, timeout=60.0)`
- **Purpose**: Retrieves status and results of a submitted task.
- **Wait Option**: Polls until completion if `wait=True`.

### run()
- **Function**: `run(*, provider, device, circuit, source, shots, **opts)`
- **Purpose**: Convenience method that submits and retrieves results in one call.

### result()
- **Function**: `result(task, *, token=None, prettify=False)`
- **Purpose**: Extracts formatted results from a completed task.

### cancel()
- **Function**: `cancel(task, *, token=None)`
- **Purpose**: Cancels a running or queued task.

**Section sources**
- [src/tyxonq/cloud/__init__.py](file://src/tyxonq/cloud/__init__.py#L0-L8)
- [src/tyxonq/cloud/api.py](file://src/tyxonq/cloud/api.py#L0-L123)

## Applications

The Applications module organizes high-level quantum algorithms and domain-specific workflows.

### chem
- **Submodule**: `tyxonq.applications.chem`
- **Purpose**: Contains quantum chemistry algorithms and utilities.
- **Key Components**:
  - Molecular Hamiltonian construction.
  - Variational algorithms (VQE, UCCSD).
  - Classical-quantum hybrid solvers.
  - Runtime environments for chemistry simulations.

**Section sources**
- [src/tyxonq/applications/__init__.py](file://src/tyxonq/applications/__init__.py#L0-L15)

## Libraries

The Libraries module provides reusable components and algorithmic building blocks.

### circuits_library
- **Purpose**: Collection of predefined quantum circuits and templates.
- **Contents**: QAOA circuits, state preparation routines, variational forms.

### hamiltonian_encoding
- **Purpose**: Tools for mapping fermionic operators to qubit Hamiltonians.
- **Methods**: Jordan-Wigner, Parity, Bravyi-Kitaev transformations.

### optimizer
- **Purpose**: Optimization utilities for variational quantum algorithms.
- **Features**: Integration with external optimizers, SOAP optimizer.

### quantum_library
- **Purpose**: Low-level quantum operations and kernels.
- **Components**: Gate implementations, dynamics simulation, measurement processing.

## Visualization

The Visualization module provides tools for rendering quantum circuits and data.

### circuit_to_dot()
- **Function**: `circuit_to_dot(circuit)`
- **Purpose**: Converts a quantum circuit to DOT graph format.
- **Output**: String representation suitable for Graphviz rendering.
- **Use Case**: Visualizing circuit structure and gate connectivity.

**Section sources**
- [src/tyxonq/visualization/__init__.py](file://src/tyxonq/visualization/__init__.py#L0-L10)