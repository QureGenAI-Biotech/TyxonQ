# Device Abstraction Layer

<cite>
**Referenced Files in This Document**   
- [base.py](file://src/tyxonq/devices/base.py)
- [session.py](file://src/tyxonq/devices/session.py)
- [driver.py](file://src/tyxonq/devices/simulators/driver.py)
- [statevector/engine.py](file://src/tyxonq/devices/simulators/statevector/engine.py)
- [matrix_product_state/engine.py](file://src/tyxonq/devices/simulators/matrix_product_state/engine.py)
- [density_matrix/engine.py](file://src/tyxonq/devices/simulators/density_matrix/engine.py)
- [tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py)
- [ibm/driver.py](file://src/tyxonq/devices/hardware/ibm/driver.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Unified Interface Design](#unified-interface-design)
3. [Base Device and Session Classes](#base-device-and-session-classes)
4. [Simulator Implementations](#simulator-implementations)
5. [Hardware Drivers](#hardware-drivers)
6. [Device Constraints and Compiler Integration](#device-constraints-and-compiler-integration)
7. [Execution Result Normalization](#execution-result-normalization)
8. [Configuration and Execution Examples](#configuration-and-execution-examples)
9. [Common Issues and Backend Selection Guidance](#common-issues-and-backend-selection-guidance)
10. [Conclusion](#conclusion)

## Introduction
The Device Abstraction Layer in TyxonQ provides a unified interface for executing quantum circuits across various backends, including simulators and real quantum hardware. This architecture enables seamless switching between different execution environments while maintaining consistent APIs for circuit execution, measurement, and result processing. The layer abstracts device-specific details, allowing users to focus on quantum algorithm development without being constrained by backend-specific implementations.

## Unified Interface Design
The Device Abstraction Layer employs a protocol-based design to define a consistent interface for all quantum devices. This design enables polymorphic behavior where simulators and hardware drivers implement the same contract, allowing higher-level components to interact with devices without knowing their specific implementation details. The unified interface supports both synchronous and asynchronous execution models, accommodating the different response characteristics of simulators and remote quantum processors.

**Section sources**
- [base.py](file://src/tyxonq/devices/base.py#L67-L78)

## Base Device and Session Classes
The core of the Device Abstraction Layer consists of the `Device` protocol and `Session` classes that define the execution contract. The `Device` protocol specifies the fundamental operations that all quantum devices must support, including circuit execution and expectation value computation. The `Session` class manages execution plans and provides utilities for aggregating results from multiple execution segments.

```mermaid
classDiagram
class Device {
+string name
+DeviceRule device_rule
+run(circuit : Circuit, shots : int | None, **kwargs) RunResult
+expval(circuit : Circuit, obs : Observable, **kwargs) float
}
class DeviceRule {
+set[string] native_gates
+int max_qubits
+Any connectivity
+bool supports_shots
+bool supports_batch
}
class RunResult {
+Any samples
+Dict[string, float] expectations
+Dict[string, Any] metadata
}
class DeviceTask {
+string provider
+string device
+Any handle
+bool async_result
+get_result(wait : bool, poll_interval : float, timeout : float) Dict[string, Any]
+cancel() Any
}
Device <|-- StatevectorEngine
Device <|-- MatrixProductStateEngine
Device <|-- DensityMatrixEngine
Device <|-- TyxonQHardwareDriver
Device <|-- IBMHardwareDriver
```

**Diagram sources**
- [base.py](file://src/tyxonq/devices/base.py#L67-L78)

**Section sources**
- [base.py](file://src/tyxonq/devices/base.py#L67-L78)
- [session.py](file://src/tyxonq/devices/session.py#L0-L50)

## Simulator Implementations
The Device Abstraction Layer includes multiple simulator engines, each optimized for different use cases and scalability requirements. These simulators implement the same device interface but employ different quantum state representations and computational approaches.

### Statevector Simulator
The statevector simulator represents the quantum state as a dense complex vector of size 2^n, where n is the number of qubits. This engine provides exact simulation of pure quantum states and supports both sampling and analytic expectation value computation.

```mermaid
classDiagram
class StatevectorEngine {
+string name
+Dict[string, Any] capabilities
+__init__(backend_name : string | None)
+run(circuit : Circuit, shots : int | None, **kwargs) Dict[string, Any]
+expval(circuit : Circuit, obs : Any, **kwargs) float
+state(circuit : Circuit) ndarray
+probability(circuit : Circuit) ndarray
+amplitude(circuit : Circuit, bitstring : string) complex
+perfect_sampling(circuit : Circuit, rng : Generator | None) tuple[string, float]
}
StatevectorEngine --> Device : "implements"
```

**Diagram sources**
- [statevector/engine.py](file://src/tyxonq/devices/simulators/statevector/engine.py#L20-L264)

### Matrix Product State (MPS) Simulator
The MPS simulator represents the quantum state using a tensor network decomposition, which can efficiently simulate systems with limited entanglement. This engine scales with the bond dimension rather than the full Hilbert space size, enabling simulation of larger qubit systems when entanglement is constrained.

```mermaid
classDiagram
class MatrixProductStateEngine {
+string name
+Dict[string, Any] capabilities
+ArrayBackend backend
+int | None max_bond
+__init__(backend : ArrayBackend | None, backend_name : string | None, max_bond : int | None)
+run(circuit : Circuit, shots : int | None, **kwargs) Dict[string, Any]
+expval(circuit : Circuit, obs : Any, **kwargs) float
}
MatrixProductStateEngine --> Device : "implements"
```

**Diagram sources**
- [matrix_product_state/engine.py](file://src/tyxonq/devices/simulators/matrix_product_state/engine.py#L20-L212)

### Density Matrix Simulator
The density matrix simulator represents the quantum state as a 2^n × 2^n density matrix, enabling simulation of mixed states and explicit noise modeling. This engine supports native application of Kraus operators for various noise channels, making it ideal for studying decoherence and error mitigation techniques.

```mermaid
classDiagram
class DensityMatrixEngine {
+string name
+Dict[string, Any] capabilities
+ArrayBackend backend
+__init__(backend_name : string | None)
+run(circuit : Circuit, shots : int | None, **kwargs) Dict[string, Any]
+expval(circuit : Circuit, obs : Any, **kwargs) float
}
DensityMatrixEngine --> Device : "implements"
```

**Diagram sources**
- [density_matrix/engine.py](file://src/tyxonq/devices/simulators/density_matrix/engine.py#L20-L208)

## Hardware Drivers
The Device Abstraction Layer includes drivers for interfacing with real quantum hardware, providing a consistent interface for remote execution on quantum processors.

### TyxonQ Hardware Driver
The TyxonQ hardware driver manages connections to TyxonQ quantum processors, handling task submission, monitoring, and result retrieval. It implements the standard device interface while managing the asynchronous nature of remote quantum computing.

```mermaid
classDiagram
class TyxonQTask {
+string id
+string device
+string status
+Any task_info
+bool async_result
+get_result(token : string | None, wait : bool, poll_interval : float, timeout : float) Dict[string, Any]
}
class TyxonQHardwareDriver {
+list_devices(token : string | None, **kws) List[string]
+submit_task(device : string, token : string | None, source : string | Sequence[string], shots : int | Sequence[int], lang : string, **kws) List[TyxonQTask]
+get_task_details(task : TyxonQTask, token : string | None) Dict[string, Any]
+remove_task(task : TyxonQTask, token : string | None) Dict[string, Any]
}
TyxonQHardwareDriver --> Device : "implements"
```

**Diagram sources**
- [tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L20-L192)

### IBM Quantum Driver
The IBM Quantum driver provides a skeleton implementation for interfacing with IBM's quantum systems. This driver follows the same interface pattern as other hardware drivers, ensuring consistency across different quantum computing platforms.

```mermaid
classDiagram
class IBMHardwareDriver {
+list_devices(token : string | None, **kws) List[string]
+submit_task(device : string, token : string | None, circuit : Any | Sequence[Any], source : string | Sequence[string], shots : int | Sequence[int], **opts) List[Any]
+get_task_details(task : Any, token : string | None, prettify : bool) Dict[string, Any]
}
IBMHardwareDriver --> Device : "implements"
```

**Diagram sources**
- [ibm/driver.py](file://src/tyxonq/devices/hardware/ibm/driver.py#L20-L39)

## Device Constraints and Compiler Integration
The Device Abstraction Layer communicates device-specific constraints to the compiler through the `DeviceRule` structure, which describes capabilities such as native gate sets, qubit connectivity, and maximum qubit count. This information enables the compiler to perform appropriate circuit optimizations and transpilation to match the target device's physical characteristics.

```mermaid
flowchart TD
A[Circuit] --> B{Device Rule}
B --> C[native_gates]
B --> D[max_qubits]
B --> E[connectivity]
B --> F[supports_shots]
B --> G[supports_batch]
C --> H[Gate Decomposition]
D --> I[Qubit Mapping]
E --> J[Swap Insertion]
F --> K[Shots Handling]
G --> L[Batch Optimization]
H --> M[Optimized Circuit]
I --> M
J --> M
K --> M
L --> M
M --> N[Execution]
```

**Diagram sources**
- [base.py](file://src/tyxonq/devices/base.py#L80-L88)

**Section sources**
- [base.py](file://src/tyxonq/devices/base.py#L80-L88)

## Execution Result Normalization
The Device Abstraction Layer normalizes execution results across different backends to provide a consistent output format. All devices return results in a standardized structure that includes samples, expectations, and metadata, regardless of whether the execution was performed on a simulator or real hardware.

```mermaid
classDiagram
class RunResult {
+Any samples
+Dict[string, float] expectations
+Dict[string, Any] metadata
}
class DeviceTask {
+get_result(wait : bool, poll_interval : float, timeout : float) Dict[string, Any]
}
class ResultNormalizer {
+normalize_result(raw_result : Any) RunResult
+unify_metadata(raw_metadata : Any) Dict[string, Any]
+convert_counts_format(counts : Any) Dict[string, int]
}
DeviceTask --> RunResult : "returns"
ResultNormalizer --> RunResult : "produces"
```

**Diagram sources**
- [base.py](file://src/tyxonq/devices/base.py#L89-L97)
- [driver.py](file://src/tyxonq/devices/simulators/driver.py#L20-L141)

## Configuration and Execution Examples
The Device Abstraction Layer provides straightforward APIs for configuring and running circuits on different devices. Users can specify the target device by name and execute circuits with minimal code changes.

```mermaid
sequenceDiagram
participant User as "User Code"
participant Device as "Device Interface"
participant Driver as "Backend Driver"
participant Hardware as "Quantum Hardware"
User->>Device : run(circuit, device="statevector", shots=1000)
Device->>Driver : resolve_driver("simulator", "statevector")
Driver->>Driver : run(circuit, shots=1000)
Driver-->>Device : SimTask with results
Device-->>User : RunResult with counts
User->>Device : run(circuit, device="tyxonq : : processor_1", shots=1024)
Device->>Driver : resolve_driver("tyxonq", "processor_1")
Driver->>Hardware : submit_task(source, shots=1024)
Hardware-->>Driver : Task ID and status
Driver-->>Device : TyxonQTask
Device-->>User : DeviceTask for asynchronous polling
```

**Diagram sources**
- [base.py](file://src/tyxonq/devices/base.py#L100-L200)
- [driver.py](file://src/tyxonq/devices/simulators/driver.py#L20-L141)
- [tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L20-L192)

## Common Issues and Backend Selection Guidance
The Device Abstraction Layer addresses common issues encountered when working with quantum devices, providing guidance for selecting appropriate backends based on problem characteristics.

### Device Availability
Remote quantum hardware may be unavailable due to maintenance, calibration, or high demand. The abstraction layer provides utilities to check device availability and list accessible devices.

### Calibration Errors
Real quantum devices require regular calibration, and performance can vary over time. The layer provides access to device properties and calibration data to inform execution decisions.

### Simulation Memory Limits
Different simulators have varying memory requirements and scalability characteristics:
- **Statevector**: Suitable for up to ~30 qubits on typical workstations
- **MPS**: Can handle 50+ qubits for circuits with limited entanglement
- **Density Matrix**: Limited to ~15 qubits due to O(4^n) memory scaling

```mermaid
flowchart TD
A[Problem Size] --> B{Number of Qubits}
B --> |< 15| C[Density Matrix<br/>For noise studies]
B --> |15-30| D[Statevector<br/>For exact simulation]
B --> |30-50+| E[MPS<br/>For limited entanglement]
B --> |> 50| F[Hardware<br/>Or specialized simulators]
G[Circuit Depth] --> H{Entanglement Level}
H --> |Low| I[MPS preferred]
H --> |High| J[Statevector or hardware]
K[Noise Requirements] --> L{Need explicit noise modeling?}
L --> |Yes| M[Density Matrix]
L --> |No| N[Statevector or MPS]
```

**Diagram sources**
- [statevector/engine.py](file://src/tyxonq/devices/simulators/statevector/engine.py#L20-L264)
- [matrix_product_state/engine.py](file://src/tyxonq/devices/simulators/matrix_product_state/engine.py#L20-L212)
- [density_matrix/engine.py](file://src/tyxonq/devices/simulators/density_matrix/engine.py#L20-L208)

**Section sources**
- [statevector/engine.py](file://src/tyxonq/devices/simulators/statevector/engine.py#L20-L264)
- [matrix_product_state/engine.py](file://src/tyxonq/devices/simulators/matrix_product_state/engine.py#L20-L212)
- [density_matrix/engine.py](file://src/tyxonq/devices/simulators/density_matrix/engine.py#L20-L208)

## Conclusion
The Device Abstraction Layer in TyxonQ provides a comprehensive and flexible framework for executing quantum circuits across diverse backends. By defining a unified interface through the `Device` protocol and implementing specialized engines for different simulation approaches and hardware platforms, the layer enables seamless switching between execution environments. The architecture supports both high-performance simulation for algorithm development and testing, as well as integration with real quantum hardware for production workloads. Through consistent result normalization and comprehensive device capability reporting, the layer simplifies the development of quantum applications that can adapt to different computational resources and requirements.