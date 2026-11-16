---
kind: logging_system
name: TyxonQ 日志系统：基于 Python 标准库 logging 的模块级 logger
category: logging_system
scope:
    - '**'
source_files:
    - src/tyxonq/applications/chem/__init__.py
    - src/tyxonq/compiler/pulse_compiler/defcal_library.py
    - src/tyxonq/devices/hardware/guodun/driver.py
    - src/tyxonq/devices/hardware/lqcloud/driver.py
    - src/tyxonq/devices/hardware/qcos/driver.py
    - src/tyxonq/devices/hardware/quafu/driver.py
    - src/tyxonq/devices/hardware/topology.py
    - src/tyxonq/applications/chem/algorithms/vqe/kupccgsd.py
    - src/tyxonq/applications/chem/algorithms/vqe/ucc_base.py
    - src/tyxonq/applications/chem/dynamics/models/pyrazine.py
---

## 1. 使用的框架与工具

TyxonQ 没有引入第三方日志框架（如 `loguru`、`structlog`、`sentry-sdk`），而是直接使用 **Python 标准库 `logging`**。代码中未发现任何自定义 Handler/Formatter/Filter，也没有集中式的 `basicConfig` 或根 logger 配置；日志输出由调用方（用户脚本、测试、示例）自行决定。

在依赖的外部库层面，`src/tyxonq/applications/chem/__init__.py` 通过设置环境变量 `RENO_LOG_LEVEL=100` 来抑制第三方库 `reno` 的日志输出，这是仓库内唯一对第三方日志进行显式调用的地方。

## 2. 关键文件

- `src/tyxonq/applications/chem/__init__.py`：唯一对 root logger (`"tyxonq"`) 和命名空间 logger (`"tyxonq.chem"`) 设置 level 的地方。
- `src/tyxonq/compiler/pulse_compiler/defcal_library.py`：脉冲校准库，使用最多 logger 调用（debug/info/warning/error）。
- `src/tyxonq/devices/hardware/guodun/driver.py`：国盾硬件驱动，包含设备发现、token 解析、拓扑校验等路径的 warning/error 日志。
- `src/tyxonq/devices/hardware/lqcloud/driver.py`、`src/tyxonq/devices/hardware/qcos/driver.py`、`src/tyxonq/devices/hardware/quafu/driver.py`：其他硬件驱动的日志入口。
- `src/tyxonq/devices/hardware/topology.py`：拓扑缓存相关日志。
- `src/tyxonq/applications/chem/algorithms/vqe/kupccgsd.py`、`ucc_base.py`：VQE 算法优化流程的 info/warning 日志。
- `src/tyxonq/applications/chem/dynamics/models/pyrazine.py`：动力学模型构建中的 debug 日志。

## 3. 架构与约定

### 3.1 Logger 获取方式
每个需要日志的模块采用统一模式：
```python
import logging
logger = logging.getLogger(__name__)
```
即使用 Python 模块名作为 logger 名称，形成层级化的 logger 树（例如 `tyxonq.devices.hardware.guodun.driver`、`tyxonq.compiler.pulse_compiler.defcal_library`）。

### 3.2 日志级别策略
- 业务错误/参数校验失败 → `logger.error(...)`（集中在 `defcal_library.py` 的校准导入校验路径）。
- 可恢复异常/降级提示 → `logger.warning(...)`（如 Guodun `list_devices` 捕获异常后 warning、校准文件版本兼容性警告、优化失败提示）。
- 流程跟踪/状态汇报 → `logger.info(...)`（如 VQE try 次数、能量值、校准导出/导入计数、进入 analytic numeric engine path）。
- 详细调试信息 → `logger.debug(...)`（如 pyrazine 项展开、defcal_library 添加/移除校准条目）。

### 3.3 全局级别设置
`applications/chem/__init__.py` 在包初始化时做了两段设置：
1. 将 root logger `"tyxonq"` 设为 `FATAL`（仅致命消息）。
2. 将 `"tyxonq.chem"` 设为 `WARNING`。
随后 `del logger` 清理局部变量。该文件是仓库中**唯一**调用 `setLevel` 的位置，因此默认情况下只有 WARNING 及以上级别的日志会传播到 root handler（如果存在的话）。

### 3.4 结构化字段
日志消息全部使用 f-string 或 `%` 格式化拼接为纯文本字符串，**没有使用结构化字段**（无 JSON 序列化、无额外键值对）。例如：
- `f"Added calibration: {gate} on qubits {qubits}"`
- `f"k-UpCCGSD try {i} energy {float(r.fun)}"`
- `"Guodun list_devices: %s", exc`

### 3.5 Sink / 输出目标
仓库内**未定义任何 StreamHandler/FileHandler/ConsoleHandler**，也没有 `basicConfig`。日志输出完全依赖 Python logging 的默认行为（当 root logger 有 handler 时输出到 stderr），实际输出由应用层（用户脚本、pytest、CI）控制。

## 4. 约定与约束

- **约定（描述性）**：所有 TyxonQ 内部模块如需记录日志，均通过 `logging.getLogger(__name__)` 获取模块级 logger，并使用 `debug/info/warning/error` 四个级别表达不同严重程度的事件。
- **约定（描述性）**：日志消息以人类可读的中文/英文混合字符串形式输出，不携带结构化字段。
- **约定（描述性）**：外部依赖 `reno` 的日志通过设置 `os.environ["RENO_LOG_LEVEL"] = "100"` 被静默抑制。
- **约束（事实）**：仓库中没有自定义 logging 组件（无自定义 Handler/Formatter/Filter）、没有集中式 `basicConfig`、没有在 `pyproject.toml` 中声明任何日志框架依赖。
- **约束（事实）**：`applications/chem/__init__.py` 是唯一设置 logger level 的文件，root logger 默认 `FATAL`，`tyxonq.chem` 命名空间默认 `WARNING`。
- **约束（事实）**：所有 logger 调用均来自 Python 标准库 `logging`，未见第三方日志 SDK 的 import。