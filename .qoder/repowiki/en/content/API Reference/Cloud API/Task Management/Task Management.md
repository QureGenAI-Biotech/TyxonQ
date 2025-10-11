# Task Management

<cite>
**Referenced Files in This Document**   
- [cloud_api_task.py](file://examples/cloud_api_task.py)
- [api.py](file://src/tyxonq/cloud/api.py)
- [driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Task Submission](#task-submission)
3. [Task Details Retrieval](#task-details-retrieval)
4. [Task Listing](#task-listing)
5. [Workflow Examples](#workflow-examples)
6. [Error Handling](#error-handling)

## Introduction
The TyxonQ Cloud API provides a comprehensive interface for managing quantum computing tasks through three primary endpoints: task submission, task details retrieval, and task listing. This documentation details the functionality, parameters, response formats, and usage patterns for these endpoints, with emphasis on the device optimization matrix, circuit execution parameters, and SDK abstractions that simplify interaction with the quantum computing backend.

**Section sources**
- [api.py](file://src/tyxonq/cloud/api.py#L1-L123)
- [driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L0-L192)

## Task Submission
The task submission endpoint `/api/v1/tasks/submit_task` accepts quantum circuits in OpenQASM format and schedules them for execution on specified quantum devices. The endpoint is accessible through the `submit_task` function in the cloud API module.

### Parameters
The task submission accepts the following parameters:

- **provider**: Specifies the quantum computing provider (default: "tyxonq")
- **device**: Target quantum device with optional optimization flags (e.g., "homebrew_s2?o=3")
- **source**: Quantum circuit in OpenQASM format
- **shots**: Number of circuit executions (default: 1024)
- **lang**: Language specification (default: "OPENQASM")
- **version**: API version (default: "1")

### Device Optimization Matrix
The device selection supports optimization flags through the query parameter `o` in the device string. These flags control various aspects of circuit compilation and execution:

- **o=1**: Basic optimization level
- **o=2**: Enhanced gate fusion
- **o=3**: Advanced circuit simplification
- **o=4**: Noise-aware compilation
- **o=7**: Full optimization with error mitigation

Combinations of these flags can be specified (e.g., "homebrew_s2?o=3,4") to enable multiple optimization strategies simultaneously. The optimization level affects circuit depth reduction, gate count minimization, and noise resilience.

### Metadata and Priority
Additional metadata can be included in the submission:
- **prior**: Priority level for task scheduling
- **remarks**: Descriptive notes about the task purpose

### Response Format
Successful submissions return a task object with the following structure:
- **id**: Unique task identifier
- **status**: Current task state ("submitted", "processing", etc.)
- **device**: Target device with optimization parameters
- **shots**: Number of requested shots

Error responses include an error field with diagnostic information and may include device status for troubleshooting.

**Section sources**
- [driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L79-L124)
- [api.py](file://src/tyxonq/cloud/api.py#L41-L62)
- [cloud_api_task.py](file://examples/cloud_api_task.py#L0-L64)

## Task Details Retrieval
The task details endpoint `/api/v1/tasks/detail` retrieves comprehensive information about a submitted task, including execution status, quantum circuit metrics, and measurement results.

### Request Parameters
- **task_id**: Unique identifier of the task to query

### Response Structure
The response includes detailed execution information:

- **Execution Timestamps**:
  - **pending**: Time when task entered queue
  - **scheduled**: Time when task was assigned to device
  - **completed**: Time when execution finished
  - **runAt**: Time when circuit execution began
  - **runDur**: Total execution duration in microseconds

- **Quantum Circuit Metrics**:
  - **qubits**: Number of qubits used
  - **depth**: Circuit depth
  - **shots**: Number of circuit executions
  - **md5**: Circuit fingerprint for reproducibility

- **Device Execution Metrics**:
  - **atChip**: Time when chip operations began
  - **durChip**: Chip operation duration in microseconds

- **Measurement Results**:
  - **result**: Dictionary of measurement outcomes with counts (e.g., {"00": 33, "11": 61})

The response is normalized to a unified format with 'result' containing the measurement counts and 'result_meta' containing additional metadata.

**Section sources**
- [driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L128-L182)
- [api.py](file://src/tyxonq/cloud/api.py#L65-L70)

## Task Listing
The task listing functionality allows retrieval of tasks filtered by device and task type. While the specific endpoint `/api/v1/tasks/api_key/list` is not directly implemented in the provided code, the API supports device listing through the `list_devices` function.

### Device Listing
The `list_devices` function returns available quantum devices with their capabilities:
- **id**: Device identifier
- **qubits**: Number of available qubits
- **connectivity**: Qubit connectivity map
- **status**: Current device availability

Filtering can be performed by specifying the provider parameter to restrict results to specific quantum computing platforms.

**Section sources**
- [driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L80-L88)
- [api.py](file://src/tyxonq/cloud/api.py#L80-L88)

## Workflow Examples
The examples demonstrate complete workflows for quantum task management using both direct API calls and SDK abstractions.

### Direct API Workflow
```python
# Submit task with OpenQASM source
res = tq.api.submit_task(
    provider="tyxonq", 
    device="homebrew_s2?o=3", 
    source=bell_qasm(), 
    shots=100
)

# Retrieve task details
details = tq.api.get_task_details(res)
```

### SDK Abstraction Workflow
The SDK provides higher-level abstractions that encapsulate the API endpoints:

- **submit_task**: Abstracts the task submission endpoint
- **get_task_details**: Wraps the task details retrieval
- **result**: Convenience function for obtaining measurement results

```python
# Chain-style workflow using Circuit abstraction
c = tq.Circuit(2)
c.h(0).cx(0, 1).measure_z(0).measure_z(1)
res = c.compile().device(provider="tyxonq", device="homebrew_s2", shots=100).run()
```

The SDK methods handle authentication, compilation, and result normalization automatically.

**Section sources**
- [cloud_api_task.py](file://examples/cloud_api_task.py#L0-L64)
- [api.py](file://src/tyxonq/cloud/api.py#L96-L100)

## Error Handling
The API implements comprehensive error handling for task management operations.

### Submission Errors
Failed submissions return structured error responses:
- **success**: Boolean indicating operation success
- **error**: Error message describing the failure reason

The system attempts to provide diagnostic information by fetching device properties when submission fails.

### Authentication and Connectivity
Authentication is handled through bearer tokens provided via the Authorization header. The system first checks in-memory tokens, then environment variables (TYXONQ_API_KEY) as fallback.

### Timeout and Polling
The `get_task_details` function supports optional polling with configurable parameters:
- **wait**: Whether to poll until task completion
- **poll_interval**: Interval between polling attempts (default: 2.0 seconds)
- **timeout**: Maximum wait time (default: 60.0 seconds)

**Section sources**
- [driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L100-L124)
- [api.py](file://src/tyxonq/cloud/api.py#L65-L70)