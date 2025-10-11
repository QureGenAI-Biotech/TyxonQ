# Devices API

<cite>
**Referenced Files in This Document**   
- [base.py](file://src/tyxonq/devices/base.py)
- [session.py](file://src/tyxonq/devices/session.py)
- [simulators/driver.py](file://src/tyxonq/devices/simulators/driver.py)
- [simulators/statevector/engine.py](file://src/tyxonq/devices/simulators/statevector/engine.py)
- [simulators/matrix_product_state/engine.py](file://src/tyxonq/devices/simulators/matrix_product_state/engine.py)
- [simulators/density_matrix/engine.py](file://src/tyxonq/devices/simulators/density_matrix/engine.py)
- [simulators/noise/channels.py](file://src/tyxonq/devices/simulators/noise/channels.py)
- [hardware/tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py)
- [hardware/ibm/driver.py](file://src/tyxonq/devices/hardware/ibm/driver.py)
- [hardware/config.py](file://src/tyxonq/devices/hardware/config.py)
- [examples/noise_controls_demo.py](file://examples/noise_controls_demo.py)
- [examples/cloud_api_devices.py](file://examples/cloud_api_devices.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Core Abstraction Classes](#core-abstraction-classes)
3. [Simulator Implementations](#simulator-implementations)
4. [Hardware Drivers](#hardware-drivers)
5. [Session Management and Execution Workflow](#session-management-and-execution-workflow)
6. [Configuration Options](#configuration-options)
7. [Integration with Compiler, Numerics, and Postprocessing](#integration-with-compiler-numerics-and-postprocessing)
8. [Examples](#examples)

## Introduction
The Devices module in TyxonQ provides a unified abstraction layer for quantum device execution, enabling seamless interaction between quantum circuits and various backends including simulators and physical hardware. This API documentation details the foundational classes, simulator engines, hardware drivers, session management, configuration options, and integration points that form the core of the device execution system.

## Core Abstraction Classes

The Devices module defines three core classes that form the foundation of the unified device abstraction: `Device`, `DeviceRule`, and `RunResult`. These classes provide a consistent interface for interacting with different quantum backends.

### Device Class
The `Device` class is a protocol that defines the interface for all quantum execution devices. It specifies the methods and properties that any device must implement to be compatible with the TyxonQ framework.

**Section sources**
- [base.py](file://src/tyxonq/devices/base.py#L67-L78)

### DeviceRule Class
The `DeviceRule` class is a TypedDict that describes the capabilities of a device. It includes optional fields that allow devices to declare their supported features while maintaining forward compatibility.

**Section sources**
- [base.py](file://src/tyxonq/devices/base.py#L41-L52)

### RunResult Class
The `RunResult` class is a TypedDict that defines the structure of results returned by device execution. It uses optional keys to allow devices to report varying levels of detail while preserving a common contract for downstream processing.

**Section sources**
- [base.py](file://src/tyxonq/devices/base.py#L55-L64)

## Simulator Implementations

TyxonQ provides multiple simulator implementations that vary in their computational approach and resource requirements. These simulators enable quantum circuit execution with different trade-offs between accuracy, memory usage, and simulation speed.

### Statevector Simulator
The statevector simulator represents the quantum state as a dense statevector of size 2^n. It provides exact simulation of pure quantum states with O(2^n) memory complexity. This simulator supports standard quantum gates and can compute expectation values analytically when shots=0.

```mermaid
classDiagram
class StatevectorEngine {
+str name
+Dict[str, Any] capabilities
+ArrayBackend backend
+__init__(backend_name : str | None)
+run(circuit : Circuit, shots : int | None, **kwargs : Any) Dict[str, Any]
+expval(circuit : Circuit, obs : Any, **kwargs : Any) float
+state(circuit : Circuit) np.ndarray
+probability(circuit : Circuit) np.ndarray
+amplitude(circuit : Circuit, bitstring : str) complex
+perfect_sampling(circuit : Circuit, rng : np.random.Generator | None) tuple[str, float]
}
```

**Diagram sources**
- [simulators/statevector/engine.py](file://src/tyxonq/devices/simulators/statevector/engine.py#L1-L264)

### Matrix Product State (MPS) Simulator
The Matrix Product State simulator represents the quantum state using a tensor network decomposition. This approach enables simulation of larger systems when entanglement is limited, with memory and time complexity scaling with bond dimension rather than 2^n. The simulator supports bond dimension truncation to control computational resources.

```mermaid
classDiagram
class MatrixProductStateEngine {
+str name
+Dict[str, Any] capabilities
+ArrayBackend backend
+int | None max_bond
+__init__(backend : ArrayBackend | None, backend_name : str | None, max_bond : int | None)
+run(circuit : Circuit, shots : int | None, **kwargs : Any) Dict[str, Any]
+expval(circuit : Circuit, obs : Any, **kwargs : Any) float
}
```

**Diagram sources**
- [simulators/matrix_product_state/engine.py](file://src/tyxonq/devices/simulators/matrix_product_state/engine.py#L1-L212)

### Density Matrix Simulator
The density matrix simulator represents the quantum state as a mixed state using a 2^n × 2^n density matrix. This implementation is particularly suited for noise studies as it natively supports Kraus channel application. With O(4^n) memory complexity, it is more resource-intensive than statevector simulation but provides a complete description of mixed quantum states.

```mermaid
classDiagram
class DensityMatrixEngine {
+str name
+Dict[str, Any] capabilities
+ArrayBackend backend
+__init__(backend_name : str | None)
+run(circuit : Circuit, shots : int | None, **kwargs : Any) Dict[str, Any]
+expval(circuit : Circuit, obs : Any, **kwargs : Any) float
}
```

**Diagram sources**
- [simulators/density_matrix/engine.py](file://src/tyxonq/devices/simulators/density_matrix/engine.py#L1-L208)

### Noise Channels
The noise module provides implementations of common quantum noise channels that can be applied to simulations. These channels are implemented as Kraus operators and can be applied to density matrices to model various types of quantum noise.

```mermaid
classDiagram
class NoiseChannels {
+depolarizing(p : float) List[np.ndarray]
+amplitude_damping(gamma : float) List[np.ndarray]
+phase_damping(lmbda : float) List[np.ndarray]
+pauli_channel(px : float, py : float, pz : float) List[np.ndarray]
+apply_to_density_matrix(rho : np.ndarray, kraus : List[np.ndarray], wire : int, num_qubits : int | None) np.ndarray
}
```

**Diagram sources**
- [simulators/noise/channels.py](file://src/tyxonq/devices/simulators/noise/channels.py#L1-L64)

## Hardware Drivers

TyxonQ provides hardware drivers for interfacing with physical quantum processors. These drivers handle the communication protocol, task submission, and result retrieval for different quantum computing platforms.

### TyxonQ Hardware Driver
The TyxonQ hardware driver enables communication with TyxonQ quantum processors through a cloud API. It handles task submission, polling for results, and error handling for remote execution on TyxonQ hardware.

```mermaid
classDiagram
class TyxonQTask {
+str id
+str device
+str status
+Any task_info
+bool async_result
+Any result_metadata
+__init__(id : str, device : str, status : str, task_info : None, async_result : bool)
+get_result(token : str | None, wait : bool, poll_interval : float, timeout : float) Dict[str, Any]
}
```

**Diagram sources**
- [hardware/tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L1-L192)

### IBM Quantum Driver
The IBM Quantum driver provides a skeleton interface for connecting to IBM Quantum systems. This driver is designed to be extended with Qiskit adapters to enable submission of circuits to IBM's quantum processors.

```mermaid
classDiagram
class IBM_Driver {
+list_devices(token : str | None, **kws : Any) List[str]
+submit_task(device : str, token : str | None, circuit : Any | Sequence[Any] | None, source : str | Sequence[str] | None, shots : int | Sequence[int], **opts : Any) List[Any]
+get_task_details(task : Any, token : str | None, prettify : bool) Dict[str, Any]
}
```

**Diagram sources**
- [hardware/ibm/driver.py](file://src/tyxonq/devices/hardware/ibm/driver.py#L1-L39)

## Session Management and Execution Workflow

The Devices module provides a comprehensive session management system that handles the execution workflow from circuit submission to result retrieval. This system abstracts the differences between local simulation and remote hardware execution.

### Device Task Management
The `DeviceTask` class provides a unified wrapper for both synchronous and asynchronous execution results. This abstraction allows the same interface to be used for both local simulators (immediate results) and remote hardware (polling for completion).

```mermaid
classDiagram
class DeviceTask {
+str provider
+str device
+Any handle
+bool async_result
+__init__(provider : str, device : str, handle : Any, async_result : bool)
+get_result(wait : bool, poll_interval : float, timeout : float) Dict[str, Any]
+cancel() Any
}
```

**Diagram sources**
- [base.py](file://src/tyxonq/devices/base.py#L6-L39)

### Execution Workflow
The execution workflow in TyxonQ follows a consistent pattern regardless of the target device. The process begins with device selection and configuration, followed by circuit compilation (if necessary), task submission, and finally result retrieval.

```mermaid
sequenceDiagram
participant User as "User Application"
participant Device as "Device Interface"
participant Driver as "Device Driver"
participant Hardware as "Quantum Hardware/Cloud"
User->>Device : run(provider, device, circuit, shots)
Device->>Device : resolve_driver(provider, device)
alt Local Simulator
Device->>Driver : run(device, token, circuit, shots)
Driver->>Driver : Select appropriate engine
Driver->>Driver : Execute simulation
Driver-->>Device : Return SimTask
else Remote Hardware
Device->>Driver : submit_task(device, token, source, shots)
Driver->>Hardware : HTTP POST to API endpoint
Hardware-->>Driver : Return task ID
Driver-->>Device : Return TyxonQTask
end
Device->>Device : Wrap in DeviceTask
Device-->>User : Return DeviceTask list
User->>DeviceTask : get_result(wait=True)
alt Hardware Task
loop Poll until completion
DeviceTask->>Driver : get_task_details(handle)
Driver->>Hardware : Query task status
Hardware-->>Driver : Return status and results
Driver-->>DeviceTask : Return result if complete
end
end
DeviceTask-->>User : Return final results
```

**Diagram sources**
- [base.py](file://src/tyxonq/devices/base.py#L131-L290)
- [simulators/driver.py](file://src/tyxonq/devices/simulators/driver.py#L1-L141)
- [hardware/tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L1-L192)

### Segmented Shot Planning
The session module provides functionality for executing segmented shot plans, where a single circuit is executed in multiple segments with different shot counts. This enables advanced sampling strategies and error mitigation techniques.

```mermaid
flowchart TD
Start([Start device_job_plan]) --> ValidateInput["Validate input parameters"]
ValidateInput --> ExtractCircuit["Extract circuit from plan"]
ExtractCircuit --> InitializeVars["Initialize aggregation variables"]
InitializeVars --> LoopSegments["For each segment in plan"]
LoopSegments --> ExecuteSegment["Execute segment with specified shots"]
ExecuteSegment --> AggregateResults["Aggregate expectations and metadata"]
AggregateResults --> CheckMoreSegments{"More segments?"}
CheckMoreSegments --> |Yes| LoopSegments
CheckMoreSegments --> |No| ReturnResults["Return aggregated results"]
ReturnResults --> End([End])
```

**Diagram sources**
- [session.py](file://src/tyxonq/devices/session.py#L9-L47)

## Configuration Options

The Devices module provides extensive configuration options for device selection, noise modeling, and execution parameters. These options can be set globally or overridden for individual executions.

### Device Selection
Device selection in TyxonQ can be specified through provider and device identifiers. The system supports multiple providers including 'simulator', 'tyxonq', and 'ibm', with various device options within each provider.

```mermaid
flowchart TD
Start([Device Selection]) --> CheckName{"Name provided?"}
CheckName --> |Yes| ParseName["Parse name for provider/device"]
ParseName --> SetProvider["Set provider from name"]
SetProvider --> SetDevice["Set device from name"]
CheckName --> |No| UseDefaults["Use default provider/device"]
UseDefaults --> ResolveDriver["Resolve appropriate driver"]
ResolveDriver --> ReturnDriver["Return driver instance"]
```

**Diagram sources**
- [base.py](file://src/tyxonq/devices/base.py#L80-L114)

### Noise Modeling
TyxonQ provides a global noise configuration system that enables noise modeling in simulations. The noise system supports various noise types including depolarizing, amplitude damping, phase damping, and Pauli channels.

```mermaid
classDiagram
class NoiseConfiguration {
+bool _NOISE_ENABLED
+Dict[str, Any] | None _NOISE_CONFIG
+Dict[str, Any] _DEFAULT_NOISE
+enable_noise(enabled : bool, config : Dict[str, Any] | None) Dict[str, Any]
+is_noise_enabled() bool
+get_noise_config() Dict[str, Any]
}
```

**Diagram sources**
- [base.py](file://src/tyxonq/devices/base.py#L149-L154)
- [examples/noise_controls_demo.py](file://examples/noise_controls_demo.py#L1-L46)

### Execution Parameters
The execution system supports various parameters that control the behavior of circuit execution, including shot count, backend selection, and numerical precision.

**Section sources**
- [base.py](file://src/tyxonq/devices/base.py#L131-L290)

## Integration with Compiler, Numerics, and Postprocessing

The Devices module integrates seamlessly with other components of the TyxonQ framework, including the compiler, numerics, and postprocessing systems.

### Compiler Integration
The device layer works closely with the compiler to ensure circuits are properly compiled before execution on hardware. For local simulators, circuits can be executed directly, while hardware execution requires pre-compilation to a supported format.

```mermaid
graph TD
Circuit[Circuit] --> Compiler[Compiler]
Compiler --> |Compiled Circuit| Device[Device Layer]
Device --> |Source Code| Hardware[Hardware Driver]
Device --> |Circuit| Simulator[Simulator Engine]
```

**Section sources**
- [base.py](file://src/tyxonq/devices/base.py#L131-L290)

### Numerics Integration
Simulator engines integrate with the numerics system to support different computational backends including NumPy, PyTorch, and CuPy. This allows simulations to leverage GPU acceleration when available.

```mermaid
graph TD
Engine[Simulator Engine] --> Numerics[Numerics Backend]
Numerics --> NumPy[NumPy Backend]
Numerics --> PyTorch[PyTorch Backend]
Numerics --> CuPyNumeric[CuPyNumeric Backend]
```

**Section sources**
- [simulators/statevector/engine.py](file://src/tyxonq/devices/simulators/statevector/engine.py#L1-L264)
- [simulators/matrix_product_state/engine.py](file://src/tyxonq/devices/simulators/matrix_product_state/engine.py#L1-L212)
- [simulators/density_matrix/engine.py](file://src/tyxonq/devices/simulators/density_matrix/engine.py#L1-L208)

### Postprocessing Integration
The device execution results are structured to integrate seamlessly with the postprocessing system. The `RunResult` format includes fields for samples, expectations, and metadata that can be directly consumed by postprocessing functions.

```mermaid
graph TD
Device[Device Layer] --> |RunResult| Postprocessing[Postprocessing System]
Postprocessing --> ErrorMitigation[Error Mitigation]
Postprocessing --> Readout[Readout Correction]
Postprocessing --> ClassicalShadows[Classical Shadows]
Postprocessing --> Metrics[Metrics Calculation]
```

**Section sources**
- [base.py](file://src/tyxonq/devices/base.py#L55-L64)

## Examples

The following examples demonstrate common usage patterns for the Devices module.

### Device Configuration and Execution
This example shows how to configure and execute circuits on different devices:

```mermaid
flowchart TD
Start([Start]) --> ConfigureDevice["Configure device with provider and device"]
ConfigureDevice --> SetToken["Set authentication token"]
SetToken --> ListDevices["List available devices"]
ListDevices --> SelectDevice["Select target device"]
SelectDevice --> BuildCircuit["Build quantum circuit"]
BuildCircuit --> ExecuteCircuit["Execute circuit with specified shots"]
ExecuteCircuit --> RetrieveResults["Retrieve and process results"]
RetrieveResults --> End([End])
```

**Section sources**
- [examples/cloud_api_devices.py](file://examples/cloud_api_devices.py#L1-L28)

### Noise Modeling in Simulation
This example demonstrates how to enable and configure noise modeling in simulations:

```mermaid
flowchart TD
Start([Start]) --> CheckNoiseStatus["Check current noise status"]
CheckNoiseStatus --> EnableNoise["Enable noise with configuration"]
EnableNoise --> BuildCircuit["Build test circuit"]
BuildCircuit --> RunClean["Run without noise"]
BuildCircuit --> RunNoisy["Run with noise"]
RunClean --> CompareResults["Compare clean and noisy results"]
RunNoisy --> CompareResults
CompareResults --> End([End])
```

**Section sources**
- [examples/noise_controls_demo.py](file://examples/noise_controls_demo.py#L1-L46)