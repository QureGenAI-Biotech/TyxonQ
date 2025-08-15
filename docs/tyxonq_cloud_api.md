# TyxonQ Cloud API Documentation

## Overview

TyxonQ Cloud API provides HTTP endpoints for quantum computing task management, device information retrieval, and task submission. The API uses JSON for data exchange.

Example code：[examples/cloud_api_task.py](../examples/cloud_api_task.py) , [examples/cloud_api_device.py](../examples/cloud_api_device.py)

## Base Configuration

- **Base URL**: `https://api.tyxonq.com/qau-cloud/tyxonq/`
- **API Version**: `v1`
- **Authentication**: Bearer token in Authorization header
- **Content-Type**: `application/json`

## Authentication

All API requests require authentication using a Bearer token:

```http
Authorization: Bearer YOUR_TOKEN
```

## API Endpoints

### 1. Device Management

#### 1.1 List Available Devices

**Endpoint**: `POST /api/v1/devices/list`

**Request Body**:
```json
{
  // Optional filter parameters
}
```

**Response**:
```json
{
  "devices": [
    {
      "id": "homebrew_s2",
      "qubits": 13,
      "T1": 84.44615173339844,
        "T2": 45.41538619995117,
        "Err": {
            "SQ": 0.0007843076923076923,
            "CZ": 0.009009666666666666,
            "Readout": {
                "F0": 0.016538461538461537,
                "F1": 0.04118461538461538
            }
        },
      "state": "running"
    }
  ]
}
```

**Example Usage**:

Example program file: examples/cloud_api_devices.py

```python
import requests
import json
import getpass

token = getpass.getpass("Enter your token: ")

url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/devices/list"
headers = {"Authorization": "Bearer " + token}
response = requests.post(url, json={}, headers=headers)
response_json = response.json()

if 'success' in response_json and response_json['success']:
    if 'devices' in response_json:
        print(json.dumps(response_json['devices'], indent=4))
    else:
        print("No devices found")
else:
    print("Error:")
    print(response_json['detail'])

```

### 2. Task Management

#### 2.1 Submit Task

**Endpoint**: `POST /api/v1/tasks/submit_task`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `device` | string | Yes | - | Target quantum device ID with optional optimization flags |
| `shots` | int \| array[int] | No | 1024 | Number of measurement shots to execute |
| `source` | string \| array[string] | Yes* | - | Circuit representation in OpenQASM format |
| `lang` | string | No | "OPENQASM" | Language of the circuit source code |
| `version` | string | No | "1" | Task submission protocol version |
| `prior` | int | No | 1 | Task priority in the execution queue (1-10) |
| `remarks` | string | No | - | Optional description or notes for the task |
| `group` | string | No | - | Optional group identifier for organizing tasks |

*Note: Either `source` or `circuit` parameter must be provided.


**Request Body**:
```json
{
  "device": "device_id?o=3",
  "shots": 1024,
  "source": "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];",
  "version": "1",
  "lang": "OPENQASM",
  "prior": 1,
  "remarks": "Optional task description",
  "group": "Optional group identifier"
}
```

**Device Options**:

The device parameter supports optimization flags that can be combined using addition:

| Option | Value | Description |
|--------|-------|-------------|
| `enable_qos_qubit_mapping` | 1 | Enable automatic qubit mapping to optimize circuit execution |
| `enable_qos_gate_decomposition` | 2 | Decompose gates into native gate set supported by the device |
| `enable_qos_initial_mapping` | 4 | Perform initial qubit placement optimization |

**Common Device Parameter Examples**:

| Device Parameter | Optimization Level | Description |
|-----------------|-------------------|-------------|
| `device_id` | No optimization | Basic execution without any optimizations |
| `device_id?o=0` | No optimization | Explicitly disable all optimizations |
| `device_id?o=1` | Qubit mapping only | Enable automatic qubit mapping |
| `device_id?o=2` | Gate decomposition only | Decompose gates to native set |
| `device_id?o=3` | Qubit mapping + Gate decomposition | Enable both qubit mapping and gate decomposition (1+2=3) |
| `device_id?o=4` | Initial mapping only | Enable initial qubit placement |
| `device_id?o=7` | All optimizations | Enable all optimization strategies (1+2+4=7) |
| `device_id?o=3&dry` | Dry run | Compile circuit but don't execute (for testing) |

**Response**:
```json
{
    "id": "<JOB_ID>",
    "job_name": "<JOB_NAME>",
    "status": "<STATUS>",
    "user_id": "<USER_ID>",
    "success": true,
    "error": null
}
```

**Example Usage**:

**Single Task Submission**:
```python
def create_task():
    url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/submit_task"
    headers = {"Authorization": "Bearer " + token}

    data = {
    "device": "homebrew_s2?o=3",
    "shots": 100,
    "source": """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];""",
    "version": "1",
    "lang": "OPENQASM",
    "prior": 1,
    "remarks": "Bell state preparation"
    }
    
    response = requests.post(url, json=data, headers=headers)
    response_json = response.json()
    return response_json
```

**Parameter Validation**:

| Parameter | Validation Rules |
|-----------|-----------------|
| `device` | Must be a valid device ID. Optimization flags must be valid combinations |
| `shots` | Must be positive integer. Range: 1-1000000 |
| `source` | Must be valid OpenQASM 2.0 syntax when lang="OPENQASM" |
| `prior` | Must be integer between 1-10 (1=highest priority) |
| `version` | Currently only supports "1" |
| `lang` | Currently only supports "OPENQASM" |


#### 2.2 Get Task Details

**Endpoint**: `POST /api/v1/tasks/detail`

**Request Body**:
```json
{
  "task_id": "task_uuid"
}
```

**Response**:
```json
{
    "success": true,
    "task": {
        "id": "<JOB_ID>",
        "queue": "quregenai.lab",
        "device": "homebrew_s2?o=3",
        "qubits": 2,
        "depth": 3,
        "state": "completed",
        "shots": 100,
        "at": 1754275505649825,
        "ts": {
            "completed": 1754275505649825,
            "pending": 1754275502265270,
            "scheduled": 1754275502260031
        },
        "md5": "f31a82f44a53bc8fa6e08ef0c6a34d53",
        "runAt": 1754275488761744,
        "runDur": 2532053,
        "atChip": 1754275446369691,
        "durChip": 120185,
        "result": {
            "00": 33,
            "01": 2,
            "10": 4,
            "11": 61
        },
        "task_type": "quantum_api"
    }
}
```

**Task States**:
- `pending`: Task is waiting in queue
- `running`: Task is currently executing
- `completed`: Task completed successfully
- `failed`: Task failed with error

**Example Usage**:
```python
url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/detail"
data = {"task_id": "task_uuid"}
response = requests.post(url, json=data, headers=headers)
task_details = response.json()["task"]
```

#### 2.3 List Tasks

**Endpoint**: `POST /api/v1/tasks/api_key/list`

**Request Body**:
```json
{
  "device": "device_id",
  "task_type": "quantum_api"
  // Optional filter parameters
}
```

**Response**:
```json
{
  "tasks": [
    {
      "task_id": "task_id",
      "task_type": "quantum_api",
      "status": "completed",
      "parameters": "",
      "result": "",
      "job_name": "task_uuid",
      "device": "device_id",
      "created_at": "",
      "updated_at": "",
      "completed_at": ""
    }
  ]
}
```

**Example Usage**:
```python
url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/api_key/list"
data = {
    "device": "quantum_processor",
    "task_type": "quantum_api"
}
response = requests.post(url, json=data, headers=headers)
tasks = response.json()["tasks"]
```


## Circuit Language Support

### OpenQASM 2.0

The API supports OpenQASM 2.0 format for circuit definition:

```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
```

### Supported Gates

- Single-qubit gates: `h`, `x`, `y`, `z`, `s`, `t`, `rz`
- Two-qubit gates: `cx`, `cz`
- Measurement: `measure`

## Best Practices

1. **Token Management**: Store your API token securely and rotate it regularly.
2. **Error Handling**: Always check for error responses and handle them appropriately.
3. **Task Monitoring**: Use the task details endpoint to monitor task progress.
4. **Optimization**: Choose appropriate optimization levels based on your circuit requirements.

## SDK Integration

The TyxonQ Python SDK provides high-level abstractions for these API endpoints:

```python
import tyxonq

# Set token
tyxonq.set_token("YOUR_TOKEN")

# List devices
devices = tyxonq.list_devices()

# Submit task
task = tyxonq.submit_task(
    device="quantum_processor",
    circuit=my_circuit,
    shots=1024
)

# Get results
results = task.results(blocked=True)
```

## Support

For API support and questions, please refer to the TyxonQ documentation or contact the support team. 