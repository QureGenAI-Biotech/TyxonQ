---
kind: configuration_system
name: TyxonQ 配置系统：进程内常量 + 环境变量 + 内存令牌缓存
category: configuration_system
scope:
    - '**'
source_files:
    - src/tyxonq/config.py
    - src/tyxonq/devices/hardware/config.py
    - pyproject.toml
    - examples/pulse_cloud_submission_e2e.py
    - examples/run_circuit_on_quafu.py
    - examples/run_circuit_on_guodun.py
    - examples/run_circuit_on_lqcloud.py
    - examples/qcos_cloud_task.py
    - src/tyxonq/applications/chem/__init__.py
---

## 1. 采用的方式

TyxonQ 没有引入统一的配置框架（如 pydantic-settings、dynaconf），而是采用**轻量级进程内配置 + 环境变量注入**的组合模式，依赖 `python-dotenv`（见 `pyproject.toml` 的 dependencies）但核心代码中并未调用 `load_dotenv`。配置来源分三层：

1. **包级常量与默认值**：集中在 `src/tyxonq/config.py`，提供 `PACKAGE_NAME`、`DEFAULT_COMPLEX_DTYPE_STR`、`DEFAULT_REAL_DTYPE_STR`、`SUPPORTED_BACKENDS` 等不可变常量。
2. **设备/云平台凭据与端点**：集中在 `src/tyxonq/devices/hardware/config.py`，维护进程内 `_TOKENS` 字典、`_DEFAULTS` 字典以及 `ENDPOINTS` 映射。
3. **运行期环境变量**：通过 `os.getenv` / `os.environ` 在示例脚本和子模块中直接读取，用于 API Key、GPU 可见性、线程数、外部服务地址等。

## 2. 关键文件

- `src/tyxonq/config.py` — 全局常量、后端名归一化 (`normalize_backend_name`)、向量化策略校验 (`is_valid_vectorization_policy`)、默认 dtype 返回。
- `src/tyxonq/devices/hardware/config.py` — 设备 token 管理、默认 provider/device、云端 ENDPOINTS。
- `pyproject.toml` — 声明 `python-dotenv>=1.1.1` 为运行时依赖；同时承载构建/pytest 配置。
- `examples/*.py` — 各硬件驱动示例通过 `os.getenv("TYXONQ_*")` 读取密钥。
- `src/tyxonq/applications/chem/__init__.py` — 设置 `RENO_LOG_LEVEL` 等第三方库环境变量。

## 3. 架构与约定

### 3.1 进程内单例式状态
`devices/hardware/config.py` 使用模块级可变字典 `_TOKENS` 和 `_DEFAULTS` 作为隐式全局状态：

```python
_TOKENS: Dict[str, str] = {}
_DEFAULTS: Dict[str, str] = {"provider": "tyxonq", "device": "tyxonq::simulator:mps"}
_AUTH_FILE = os.path.join(os.path.expanduser("~"), ".tyxonq.auth.json")
```

`set_token(token, *, provider=None, device=None)` 会同时写入 `prov::device` 和 `prov::` 两个键，实现**设备级覆盖 provider 级**的查找顺序；`get_token` 先查 `_TOKENS`，再按 `env_fallback=True` 回退到 `TYXONQ_API_KEY`。

### 3.2 后端名称归一化
`config.normalize_backend_name` 将用户传入的后端别名映射为规范名：

| 输入 | 输出 |
|---|---|
| `cpu`, `numpy`, `numpy(cpu)` | `numpy` |
| `gpu`, `cupynumeric`, `cupynumeric(gpu)` | `cupynumeric` |
| `torch`, `pt`, `pytorch` | `pytorch` |

该函数配合 `SUPPORTED_BACKENDS = ("numpy", "pytorch", "cupynumeric")` 共同约束数值后端选择。

### 3.3 云端端点与环境变量
`ENDPOINTS` 对每个 provider 暴露 `base_url` / `api_version`，其中 `tyxonq` 端点的 base_url 和 api_version 分别由 `TYXONQ_BASE_URL`、`TYXONQ_API_VERSION` 环境变量覆盖，默认值为 `https://api.tyxonq.com/qau-cloud/tyxonq/` 和 `v1`。

### 3.4 环境变量命名约定
仓库中使用的 key 遵循 `TYXONQ_*` 前缀或平台特定前缀：

- `TYXONQ_API_KEY` — TyxonQ 云 API 密钥（被 `devices/hardware/config.get_token` 及多个示例读取）。
- `TYXONQ_BASE_URL` / `TYXONQ_API_VERSION` — 覆盖默认云端端点。
- `TYXONQ_QUAFU_TOKEN` / `QPU_API_TOKEN` — Quafu 硬件驱动密钥（`examples/run_circuit_on_quafu.py`）。
- `TYXONQ_GUODUN_TOKEN` — 国盾量子驱动密钥（`examples/run_circuit_on_guodun.py`）。
- `TYXONQ_LQCLOUD_API_KEY` / `LQCLOUD_API_KEY` — 庐州云密钥（`examples/run_circuit_on_lqcloud.py`）。
- `QCOS_ACCESS_KEY` / `QCOS_SECRET_KEY` — QCos 密钥（`examples/qcos_cloud_task.py`）。
- `CUDA_VISIBLE_DEVICES` / `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` — 底层计算资源控制（多处示例与 `applications/chem/algorithms/sqd/gpu/batch.py`）。
- `RENO_LOG_LEVEL` — 在 `applications/chem/__init__.py` 启动时强制设为 `100`。
- `CHEM_PROFILE_SHOTS` — `scripts/profile_chem.py` 中用量子化学性能分析时的 shot 数。
- `TYXONQ_A100_HOST_GPU_ID` — SQD GPU 验证路径中使用。

### 3.5 类型化字符串子类型
`config.py` 定义 `BackendName(str)` 与 `VectorizationPolicy(str)` 两个自定义 `str` 子类，并通过重写 `__instancecheck__` 使 `isinstance(x, BackendName)` 等价于 `isinstance(x, str)`——这是一种**仅用于可读性的运行时标记**，不做强类型约束。

## 4. 约定与约束

- **无集中配置文件加载器**：仓库中没有 `settings.py`、`app_config.py` 之类的统一入口；配置散落在 `config.py`、`devices/hardware/config.py` 与各示例脚本中。
- **token 优先级**：`get_token` 的实现顺序是 **进程内 `_TOKENS[prov::device]` → `_TOKENS[prov::]` → `TYXONQ_API_KEY` 环境变量**，且可通过 `env_fallback=False` 关闭环境变量回退。
- **默认 provider/device**：`set_default(provider=..., device=...)` 修改模块级 `_DEFAULTS`，`get_default_provider()` / `get_default_device()` 读取；未显式设置时默认为 `tyxonq` 与 `tyxonq::simulator:mps`。
- **后端白名单**：`SUPPORTED_BACKENDS` 元组是数值后端的唯一合法集合，新增后端需同时更新此处与 `normalize_backend_name`。
- **向量化策略白名单**：`is_valid_vectorization_policy` 仅接受 `{"auto", "force", "off"}` 三个值。
- **dotenv 已声明但未使用**：`pyproject.toml` 依赖了 `python-dotenv`，但当前源码未调用 `load_dotenv`；`.env` 文件不会被自动加载。
- **示例中的 config.json**：`examples/SKQD/workflow.py` 读写 `config.json` 属于示例数据，不是框架级配置机制。
- **第三方库环境变量覆盖**：`applications/chem/__init__.py` 在导入时强制设置 `RENO_LOG_LEVEL="100"`，体现“框架启动时主动改写下游库环境变量”的模式。

## 5. 总结

TyxonQ 的配置系统以**极轻量的进程内字典 + 环境变量直读**为主，没有引入独立的配置 DSL 或序列化格式。其设计目标是让核心模块（core、numerics、devices）在导入时零副作用地获取常量与默认值，而凭据与端点则通过 `devices/hardware/config.py` 提供的 `set_token` / `get_token` API 与一组 `TYXONQ_*` 环境变量完成装配。这种模式的优点是简单、可测试性强，代价是缺乏 schema 校验与集中文档化的配置项清单。