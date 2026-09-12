---
kind: logging_system
name: 基于 Python 标准库 logging 的模块级日志系统
category: logging_system
scope:
    - '**'
source_files:
    - src/tyxonq/applications/chem/__init__.py
    - src/tyxonq/devices/hardware/topology.py
    - src/tyxonq/devices/hardware/guodun/driver.py
    - src/tyxonq/devices/hardware/lqcloud/driver.py
    - src/tyxonq/devices/hardware/qcos/driver.py
    - src/tyxonq/devices/hardware/quafu/driver.py
    - src/tyxonq/compiler/pulse_compiler/defcal_library.py
---

## 1. 使用的系统与框架

仓库没有引入第三方日志框架（如 `loguru`、`structlog`、`sentry-sdk`），全部使用 Python 标准库 `logging`。每个需要输出的模块通过 `logger = logging.getLogger(__name__)` 获取一个以模块全限定名为名的 logger，并在该模块内直接调用 `logger.debug/info/warning/error`。

## 2. 关键文件与位置

- 顶层初始化：`src/tyxonq/applications/chem/__init__.py` 在包导入时创建并配置根命名空间下的 logger：
  - `logging.getLogger("tyxonq")` 设置为 `FATAL`；
  - `logging.getLogger("tyxonq.chem")` 设置为 `WARNING`；
  - 随后 `del logger`，避免全局变量泄漏。
  - 同时设置 `os.environ["RENO_LOG_LEVEL"] = "100"` 以抑制第三方依赖（Reno）的日志。
- 硬件驱动层（设备执行入口）：
  - `src/tyxonq/devices/hardware/guodun/driver.py`：`logger = logging.getLogger(__name__)`，用于 `list_devices` 失败等场景的 `logger.warning`。
  - `src/tyxonq/devices/hardware/lqcloud/driver.py`：同上，`list_devices` 失败时 `logger.warning`。
  - `src/tyxonq/devices/hardware/qcos/driver.py`：轮询失败时 `logger.warning`。
  - `src/tyxonq/devices/hardware/quafu/driver.py`：`list_devices` 异常时 `logger.warning`。
  - `src/tyxonq/devices/hardware/topology.py`：拓扑缓存读写失败、抓取失败、删除缓存失败等路径使用 `logger.warning` 记录。
- 编译器/脉冲层：
  - `src/tyxonq/compiler/pulse_compiler/defcal_library.py`：添加/移除校准、验证、导出/导入 JSON 时使用 `debug/info/warning/error` 输出结构化信息（gate、qubits、hardware、timestamp 等）。
- 算法层：
  - `src/tyxonq/applications/chem/algorithms/vqe/kupccgsd.py`、`ucc_base.py`、`dynamics/models/pyrazine.py`：均按模块名创建 logger，但实际调用点较少。

## 3. 架构与约定

- **命名空间划分**：logger 名称即模块名（`__name__`），因此 `devices/hardware/guodun/driver.py` 的 logger 为 `tyxonq.devices.hardware.guodun.driver`，`applications/chem/__init__.py` 显式操作 `tyxonq` 和 `tyxonq.chem` 两个命名空间。
- **级别策略**：默认不配置任何 Handler，因此所有 logger 的 effective level 由 root logger 决定；`applications/chem/__init__.py` 将 `tyxonq` 设为 `FATAL`、`tyxonq.chem` 设为 `WARNING`，其余模块继承 root 的默认 `WARNING`。这意味着默认运行时只有 warning 及以上可见，除非用户在应用启动时自行配置 handler。
- **无集中 sink**：仓库未定义统一的 `StreamHandler`/`FileHandler`/`RotatingFileHandler`，也没有配置文件（如 `logging.conf`）。日志输出完全交由调用方或外部进程（如 pytest、IDE、容器 stdout/stderr）处理。
- **结构化字段**：除 `defcal_library.py` 在 `debug/info` 中通过 f-string 携带 gate/qubits/hardware/timestamp 等上下文外，其他日志均为纯文本消息，没有统一的结构化字段（JSON、trace_id、span_id 等均不存在）。
- **错误与告警区分**：驱动层对可恢复问题（网络超时、SDK 缺失、token 缺失、拓扑抓取失败）使用 `logger.warning` + 返回空结果/空列表；不可恢复问题（参数校验失败、平台返回结构异常）则抛出 `RuntimeError`/`ValueError`/`TypeError`，不依赖日志作为控制流。
- **第三方日志抑制**：通过环境变量 `RENO_LOG_LEVEL=100` 关闭 TenCirChem/Reno 子系统的日志，体现“仅 TyxonQ 自身模块输出日志”的约束。

## 4. 约定与约束

- **每个模块独立持有 logger**：所有使用日志的模块都遵循 `import logging; logger = logging.getLogger(__name__)` 的模式，未见共享单例 logger 或自定义 Logger 子类。
- **默认静默**：包导入不会自动注册任何 handler，因此默认安装后用户看不到任何日志——这是有意设计，便于测试与生产环境各自配置。
- **警告优先于异常**：对于可选能力（如拓扑发现、设备列表查询、磁盘缓存写入），失败时记录 `warning` 并降级返回空值，而不是中断流程；这保证了编译/提交链路不因辅助功能失败而崩溃。
- **Token/密钥相关错误走异常**：缺少 token 时直接 `raise RuntimeError(...)` 并附带帮助信息（如 `TYXONQ_GUODUN_TOKEN`、`TYXONQ_LQCLOUD_API_KEY`、`TYXONQ_QUAFU_TOKEN`），不在日志中兜底。
- **无日志级别 API**：仓库没有暴露 `set_log_level()`、`enable_debug()` 等公共函数，日志级别只能通过标准 `logging` API 在应用入口处配置。
- **测试友好**：由于没有全局 handler，pytest 运行时无额外 I/O 开销；`clear_topology_cache()` 等函数也配合测试保证确定性。

总体而言，这是一个极简的、基于标准库 `logging` 的模块级日志方案：无集中配置、无结构化格式、无异步 sink，依赖调用方在应用启动时按需配置 handler，并通过命名空间粒度控制输出级别。