# 云API集成

<cite>
**本文档引用的文件**   
- [src/tyxonq/cloud/api.py](file://src/tyxonq/cloud/api.py) - *在最近提交中更新*
- [src/tyxonq/devices/hardware/tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py) - *在最近提交中更新*
- [src/tyxonq/devices/base.py](file://src/tyxonq/devices/base.py) - *在最近提交中更新*
- [src/tyxonq/core/ir/circuit.py](file://src/tyxonq/core/ir/circuit.py) - *在最近提交中更新*
- [examples/cloud_api_task.py](file://examples/cloud_api_task.py) - *重构完成*
- [examples/cloud_api_devices.py](file://examples/cloud_api_devices.py) - *重构完成*
</cite>

## 更新摘要
**已做更改**   
- 更新了API端点部分，以反映`submit_task`和`run`函数的最新参数
- 重构了代码示例部分，以匹配重构后的示例代码
- 更新了认证机制部分，以准确描述令牌设置流程
- 修订了请求与响应格式部分，以匹配实际的API负载结构
- 更新了所有受影响部分的源代码引用

## 目录
1. [简介](#简介)
2. [API端点](#api端点)
3. [认证机制](#认证机制)
4. [请求与响应格式](#请求与响应格式)
5. [代码示例](#代码示例)
6. [速率限制与错误处理](#速率限制与错误处理)
7. [最佳实践](#最佳实践)
8. [量子API网关](#量子api网关)

## 简介

TyxonQ量子云服务提供了一套RESTful接口，允许用户通过云API与量子处理器进行交互。该API作为量子API网关，将用户请求路由到后端的真实量子处理器，如Homebrew_S2。本文档详细说明了如何使用`cloud/api.py`中提供的API端点，包括任务提交、状态查询、结果获取和设备列表查询等功能。

**Section sources**
- [src/tyxonq/cloud/api.py](file://src/tyxonq/cloud/api.py#L1-L158)

## API端点

云API提供了多个端点，用于与TyxonQ量子云服务进行交互。主要端点包括：

- `submit_task`: 提交量子任务
- `get_task_details`: 查询任务状态
- `result`: 获取任务结果
- `list_devices`: 查询设备列表
- `run`: 执行量子电路（主要入口点）
- `cancel`: 取消任务

这些端点通过`cloud/api.py`文件中的函数实现，提供了统一的接口来与不同的量子处理器进行交互。

```mermaid
graph TD
A[用户] --> B[云API]
B --> C[量子处理器]
C --> D[结果]
D --> B
B --> A
```

**Diagram sources**
- [src/tyxonq/cloud/api.py](file://src/tyxonq/cloud/api.py#L1-L158)

**Section sources**
- [src/tyxonq/cloud/api.py](file://src/tyxonq/cloud/api.py#L1-L158)

## 认证机制

云API使用API密钥进行认证。用户需要通过`set_token`函数设置API密钥，该密钥将用于后续的所有API调用。API密钥可以通过环境变量`TYXONQ_API_KEY`或直接在代码中设置。

```python
tq.set_token(token, provider="tyxonq", device="homebrew_s2")
```

`set_token`函数将令牌存储在配置中，供所有后续的API调用使用。令牌验证在`src/tyxonq/devices/hardware/config.py`中处理。

**Section sources**
- [src/tyxonq/devices/hardware/config.py](file://src/tyxonq/devices/hardware/config.py#L8-L67)

## 请求与响应格式

云API的请求和响应格式遵循RESTful规范。请求通常包含任务的量子电路、测量次数等信息，响应则包含任务的状态、结果等。

### 请求格式

`submit_task`和`run`函数的请求包含以下字段：

- `provider`: 云提供商名称（"tyxonq"、"simulator"等）
- `device`: 设备名称（"homebrew_s2"、"statevector"等）
- `circuit`: 要执行的单个电路对象或电路对象列表
- `source`: 预编译的源代码（OpenQASM、TQASM）或源代码列表
- `shots`: 执行的测量次数
- `token`: 可选的身份验证令牌
- `auto_compile`: 是否自动编译电路

### 响应格式

响应通常包含以下字段：

- `id`: 任务ID
- `status`: 任务状态
- `result`: 任务结果
- `metadata`: 任务元数据

```mermaid
erDiagram
TASK {
string id PK
string status
json result
json metadata
}
```

**Diagram sources**
- [src/tyxonq/devices/hardware/tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L1-L192)

**Section sources**
- [src/tyxonq/devices/hardware/tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L1-L192)

## 代码示例

以下代码示例展示了如何使用云API执行量子任务，从任务构建、提交到结果解析的全过程。

### 任务提交

```python
res_legacy = tq.api.submit_task(provider="tyxonq", device="homebrew_s2", source=bell_qasm(), shots=100)
```

### 状态查询

```python
details = tq.api.get_task_details(t)
```

### 结果获取

```python
result = tq.api.result(task)
```

### 设备列表查询

```python
devs = tq.api.list_devices(provider="tyxonq")
```

### 链式调用示例

```python
c = tq.Circuit(2)
c.h(0).cx(0, 1).measure_z(0).measure_z(1)
res_chain = c.compile().device(provider="tyxonq", device="homebrew_s2", shots=100).postprocessing().run(wait_async_result=True)
```

**Section sources**
- [examples/cloud_api_task.py](file://examples/cloud_api_task.py#L1-L64)
- [examples/cloud_api_devices.py](file://examples/cloud_api_devices.py#L1-L28)

## 速率限制与错误处理

云API对请求频率进行了限制，以防止滥用。当请求频率超过限制时，API将返回错误。用户应实现适当的错误处理机制，以应对这些情况。

### 错误处理

```python
try:
    details = tq.api.get_task_details(t)
except Exception as e:
    print("legacy detail error:", e)
```

当提交任务失败时，API会尝试获取设备属性以进行诊断，并在错误信息中包含这些信息。

**Section sources**
- [src/tyxonq/devices/hardware/tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L1-L192)

## 最佳实践

使用云API时，应遵循以下最佳实践：

- 始终设置API密钥
- 使用适当的测量次数
- 处理可能的错误
- 遵守速率限制
- 使用`run`函数作为主要入口点
- 考虑使用链式调用来简化工作流

**Section sources**
- [src/tyxonq/cloud/api.py](file://src/tyxonq/cloud/api.py#L1-L158)

## 量子API网关

云API作为量子API网关，将用户请求路由到后端的真实量子处理器。这种设计使得用户可以透明地与不同的量子处理器进行交互，而无需关心底层的实现细节。

```mermaid
graph TD
A[用户] --> B[云API]
B --> C[Homebrew_S2]
B --> D[其他处理器]
C --> E[结果]
D --> E
E --> B
B --> A
```

**Diagram sources**
- [src/tyxonq/cloud/api.py](file://src/tyxonq/cloud/api.py#L1-L158)
- [src/tyxonq/devices/hardware/tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L1-L192)

**Section sources**
- [src/tyxonq/cloud/api.py](file://src/tyxonq/cloud/api.py#L1-L158)
- [src/tyxonq/devices/hardware/tyxonq/driver.py](file://src/tyxonq/devices/hardware/tyxonq/driver.py#L1-L192)