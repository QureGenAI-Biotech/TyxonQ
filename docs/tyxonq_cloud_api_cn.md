# TyxonQ Cloud API 文档

## 概述

TyxonQ Cloud API 提供了用于量子计算任务管理、设备信息获取和任务提交的 HTTP 接口。API 使用 JSON 格式进行数据交换。

示例代码：[examples/cloud_api_task.py](../examples/cloud_api_task.py) , [examples/cloud_api_device.py](../examples/cloud_api_device.py)

## 基本设置

- **基础 URL**: `https://api.tyxonq.com/qau-cloud/tyxonq/`
- **API 版本**: `v1`
- **认证方式**: 在请求头中使用 Bearer Token
- **内容类型**: `application/json`

## 身份认证

所有 API 请求都需要使用 Bearer Token 进行身份验证：

```http
Authorization: Bearer YOUR_TOKEN
```

## API 接口列表

### 1. 设备管理

#### 1.1 获取可用设备列表

**接口**: `POST /api/v1/devices/list`

**请求体**:
```json
{
  // Optional filter parameters
}
```

**响应体**:
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

**示例用法**:

示例程序文件: examples/cloud_api_devices.py

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

### 2. 任务管理

#### 2.1 提交任务

**接口**: `POST /api/v1/tasks/submit_task`

**参数说明**:

| 参数名     | 类型                | 是否必填        | 默认值      | 描述                |
| --------- | ------------------- | --------------  | ---------- | ----------------- |
| `device`  | string              | 是              | -          | 目标量子设备 ID，可附带优化flag |
| `shots`   | int \| array[int]   | 否              | 1024       | 执行的测量次数              |
| `source`  | string \| array[string] | 是*             | -          | OpenQASM 格式的电路代码 |
| `lang`    | string              | 否              | "OPENQASM" | 电路代码所使用的语言          |
| `version` | string              | 否              | "1"        | 任务提交协议版本             |
| `prior`   | int                 | 否              | 1          | 执行队列中的任务优先级（1~10）|
| `remarks` | string              | 否              | -          | 可选任务备注或描述         |
| `group`   | string              | 否              | -          | 可选任务分组标识           |

*注意：必须提供 `source` 或 `circuit` 中的一个参数。


**请求体示例**:
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

**设备优化选项说明**:

设备参数支持使用加法组合的优化flag：

| 选项名                           | 数值 | 描述                |
| ------------------------------- | -- | ----------------- |
| `enable_qos_qubit_mapping`      | 1  | 启用自动量子比特映射以优化电路执行 |
| `enable_qos_gate_decomposition` | 2  | 将门操作分解为设备支持的本地门集  |
| `enable_qos_initial_mapping`    | 4  | 执行初始量子比特的优化布局     |

**常见设备参数示例**:

| 设备参数             | 优化等级       | 描述                                 |
|----------------------|----------------|--------------------------------------|
| `device_id`          | 无优化         | 基本执行，不启用任何优化             |
| `device_id?o=0`      | 无优化         | 明确禁用所有优化                     |
| `device_id?o=1`      | 仅比特映射     | 启用自动量子比特映射                 |
| `device_id?o=2`      | 仅门分解       | 将门操作分解为设备支持的本地门集     |
| `device_id?o=3`      | 映射 + 分解    | 启用比特映射与门分解（1+2=3）        |
| `device_id?o=4`      | 仅初始映射     | 启用初始量子比特布局优化             |
| `device_id?o=7`      | 所有优化       | 启用所有优化策略（1+2+4=7）          |
| `device_id?o=3&dry`  | Dry run 测试   | 编译电路但不执行（用于测试用途）     |

**响应示例**:
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

**示例用法**:

**单任务提交**:
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

**参数校验**:

| 参数名       | 校验要求                                          |
| --------- | --------------------------------------------- |
| `device`  | 必须为有效设备 ID，优化标志必须合法组合                         |
| `shots`   | 必须为正整数，范围：1\~1000000                          |
| `source`  | 当 `lang` 为 "OPENQASM" 时，必须为合法 OpenQASM 2.0 语法 |
| `prior`   | 必须为 1\~10 的整数（1 表示最高优先级）                      |
| `version` | 当前仅支持 "1"                                     |
| `lang`    | 当前仅支持 "OPENQASM"                              |


#### 2.2 获取任务详情

**接口**: `POST /api/v1/tasks/detail`

**请求体**:
```json
{
  "task_id": "task_uuid"
}
```

**响应体**:
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

**任务状态说明**:
- `pending`: 任务正在排队中
- `running`: 任务正在执行中
- `completed`: 任务成功完成
- `failed`: 任务执行失败

**示例用法**:
```python
url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/detail"
data = {"task_id": "task_uuid"}
response = requests.post(url, json=data, headers=headers)
task_details = response.json()["task"]
```

#### 2.3 获取任务列表

**接口**: `POST /api/v1/tasks/api_key/list`

**请求体**:
```json
{
  "device": "device_id",
  "task_type": "quantum_api"
  // Optional filter parameters
}
```

**响应体**:
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

**示例用法**:
```python
url = "https://api.tyxonq.com/qau-cloud/tyxonq/api/v1/tasks/api_key/list"
data = {
    "device": "quantum_processor",
    "task_type": "quantum_api"
}
response = requests.post(url, json=data, headers=headers)
tasks = response.json()["tasks"]
```


## 电路语言支持

### OpenQASM 2.0

该 API 支持使用 OpenQASM 2.0 格式定义量子电路：

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

### 支持的门操作

- 单量子比特门: `h`, `x`, `y`, `z`, `s`, `t`, `rz`
- 双量子比特门: `cx`, `cz`
- 测量指令: `measure`

## 推荐做法

1. **Token 管理**：请安全存储您的 API token，并定期轮换。
2. **错误处理**：始终检查 API 响应中的错误信息并做出恰当处理。
3. **任务监控**：使用获取任务详情接口以监控任务进度。
4. **优化选项**：根据您的电路需求选择合适的优化等级。

## SDK 集成

TyxonQ Python SDK 提供了对这些 API 接口的高级抽象：

```python
import tyxonq

# 设置 token
tyxonq.set_token("YOUR_TOKEN")

# 获取设备列表
devices = tyxonq.list_devices()

# 提交任务
task = tyxonq.submit_task(
    device="quantum_processor",
    circuit=my_circuit,
    shots=1024
)

# 获取结果
results = task.results(blocked=True)
```

## 技术支持

如需获取 API 支持或有相关问题，请参考 TyxonQ 官方文档或联系支持团队。