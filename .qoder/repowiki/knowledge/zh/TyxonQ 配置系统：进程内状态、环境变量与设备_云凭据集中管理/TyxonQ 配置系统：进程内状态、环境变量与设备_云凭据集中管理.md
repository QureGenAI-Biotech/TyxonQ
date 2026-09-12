---
kind: configuration_system
name: TyxonQ 配置系统：进程内状态、环境变量与设备/云凭据集中管理
category: configuration_system
scope:
    - '**'
source_files:
    - src/tyxonq/config.py
    - src/tyxonq/devices/hardware/config.py
    - src/tyxonq/devices/base.py
    - src/tyxonq/cloud/api.py
    - src/tyxonq/applications/chem/classical_chem_cloud/config.py
---

## 1. 整体方案

TyxonQ 没有引入外部配置框架（如 pydantic-settings、dynaconf、python-dotenv），而是采用轻量级 Python 模块 + 进程内全局字典 + 环境变量的组合来实现运行时配置。配置被拆分为三个层次：

- 包级常量与类型归一化：`src/tyxonq/config.py` 提供 `PACKAGE_NAME`、默认 dtype（complex64/float32）、支持的数值后端列表 `SUPPORTED_BACKENDS`，以及后端名归一化器 `normalize_backend_name`（支持 cpu→numpy、gpu→cupynumeric、torch/pt→pytorch 等别名）和向量化策略校验 `is_valid_vectorization_policy`。
- 设备/云平台凭据与默认值：`src/tyxonq/devices/hardware/config.py` 维护进程内 `_TOKENS`、`_DEFAULTS`、`ENDPOINTS`，并通过 `set_token` / `get_token` / `set_default` / `get_default_provider` / `get_default_device` 暴露 API。
- 应用层配置对象：`src/tyxonq/applications/chem/classical_chem_cloud/config.py` 中的 `CloudClassicalConfig` 以类实例形式组织云端经典计算 provider 的 endpoint、engine、default_config 等。

所有配置加载均无第三方依赖，保证可被核心模块在启动早期安全导入。

## 2. 关键文件与职责

- `src/tyxonq/config.py`：包级常量、dtype 默认值、后端名归一化、向量化策略校验。
- `src/tyxonq/devices/hardware/config.py`：设备/云平台 token 存储、默认 provider/device、各 provider 端点（base_url/api_version）。
- `src/tyxonq/devices/base.py`：全局噪声开关 `_NOISE_ENABLED`、噪声配置 `_NOISE_CONFIG`、统一 `run()` 入口中注入 noise 参数并过滤模拟器专属选项。
- `src/tyxonq/cloud/api.py`：面向用户的云 API 门面，转发 `set_token` / `set_default` / `device` / `submit_task` / `run` 到底层 devices.base。
- `src/tyxonq/applications/chem/classical_chem_cloud/config.py`：云端经典计算 provider 配置（endpoint/engine/default_config）及客户端工厂。
- `src/tyxonq/__init__.py`：将 `enable_noise` / `is_noise_enabled` / `get_noise_config` 暴露为顶层 API。

## 3. 架构与设计约定

### 3.1 凭据与默认值分层
进程内存中的 `_TOKENS` 字典按 `provider::device` 与 `provider::` 双键存储，优先匹配设备级 token，再回退到 provider 级 token。默认 provider 为 `tyxonq`，默认 device 为 `tyxonq::simulator:mps`，可通过 `set_default(provider=..., device=...)` 覆盖。端点通过 `ENDPOINTS` 字典集中声明；TyxonQ 云的 `base_url` 与 `api_version` 可从环境变量 `TYXONQ_BASE_URL`、`TYXONQ_API_VERSION` 覆盖。

### 3.2 环境变量优先级
`get_token` 的解析顺序为：显式传入 token → 进程内存 token（设备级 → provider 级） → 环境变量 `TYXONQ_API_KEY`。当 provider 不是 `tyxonq` 时，`env_fallback=False`，禁止误用 TyxonQ 云的全局密钥，从而在多 provider 场景下实现凭据隔离。

### 3.3 噪声配置作为运行时开关
噪声启用状态与配置保存在模块级全局变量 `_NOISE_ENABLED`、`_NOISE_CONFIG`，默认值为 `{"type": "depolarizing", "p": 0.0}`。`devices.base.run` 在执行前通过 `_inject_noise` 把全局噪声设置注入到模拟器的 kwargs 中，并在提交到真机前剔除模拟器专属键 `noise`、`use_noise`（定义于 `_SIMULATOR_ONLY_RUN_OPTS`），使调用方无需区分 provider。

### 3.4 数值后端名称归一化
`config.normalize_backend_name` 将用户输入的后端名标准化为 numpy / cupynumeric / pytorch 之一，屏蔽 cpu、gpu、torch、pt、numpy(cpu)、cupynumeric(gpu) 等别名，配合 `SUPPORTED_BACKENDS` 做白名单校验。

### 3.5 云端经典计算配置
`CloudClassicalConfig` 以类实例持有 providers 映射，当前仅内置 tyxonq provider（endpoint 指向本地 http://127.0.0.1:8009，engine=tyxonq_classical，默认 timeout=7200）。`create_classical_client` 是简单工厂，未知 provider 会抛 ValueError。

## 4. 约定与约束

- 无配置文件加载器：仓库未使用 .yaml、.toml、.env、pyproject.toml 等外部配置文件来驱动运行时行为（pyproject.toml 仅用于打包元数据）。所有运行期配置通过 Python 代码或环境变量设置。
- token 不落地磁盘：虽然定义了 `_AUTH_FILE = os.path.expanduser("~/.tyxonq.auth.json")`，但当前 get_token / set_token 并未读写该文件，仅使用进程内存与 TYXONQ_API_KEY 环境变量。
- 多 provider 凭据隔离：非 tyxonq provider 调用 get_token(env_fallback=True) 不会回退到 TYXONQ_API_KEY，避免跨 provider 泄漏密钥。
- 噪声配置仅限模拟器：run 中对硬件 provider 明确丢弃 noise / use_noise，因为“真机噪声由硬件自身决定”，这是强制约束而非可选行为。
- 配置不可变常量：config.py 中的 PACKAGE_NAME、DEFAULT_COMPLEX_DTYPE_STR、DEFAULT_REAL_DTYPE_STR、SUPPORTED_BACKENDS 为模块级只读常量，不应被修改。
- 配置导入无副作用：文档注释强调 config.py “intentionally lightweight and free of heavy dependencies so that it can be imported early by subsystems (core, numerics, devices) without side effects”。

## 5. 适用性说明

本仓库的配置系统属于轻量级进程内状态 + 环境变量模式，适用于 TyxonQ 这类量子计算框架对“设备凭据、默认执行目标、数值后端、噪声开关”等运行时参数的快速访问需求，但不包含 YAML/TOML 文件解析、配置热重载、Schema 校验等高级特性。