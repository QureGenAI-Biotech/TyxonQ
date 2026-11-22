---
kind: error_handling
name: TyxonQ 错误处理体系：轻量级异常基类与领域细分
category: error_handling
scope:
    - '**'
source_files:
    - src/tyxonq/core/errors.py
    - tests_core_module/test_core_types.py
---

## 1. 采用的系统/方法

TyxonQ 采用 Python 原生异常机制，没有引入第三方错误库（如 `pydantic`、`exceptiongroup`、结构化日志等）。核心定义集中在单一文件 `src/tyxonq/core/errors.py`，提供：

- `TyxonQError(Exception)` — 框架自定义异常的根类。
- `CompilationError(TyxonQError)` — 编译阶段失败或产出无效输出时抛出。
- `DeviceExecutionError(TyxonQError)` — 设备执行失败（超时、连接问题等）时抛出。
- `VectorizationFallbackWarning(Warning)` — 向量化被禁用或回退到 eager 模式时的警告。

除此之外，代码中广泛使用 Python 内置异常：`ValueError`、`TypeError`、`RuntimeError`、`AssertionError`、`NotImplementedError`，用于参数校验、数值异常和未实现路径。未见 `try/except` 捕获这些异常后转换为 TyxonQError 的统一中间层；也未见全局的 panic/recover 式策略或中间件式的错误拦截器。

## 2. 关键文件与包

- `src/tyxonq/core/errors.py` — 唯一集中定义的 TyxonQ 自定义异常类型。
- `tests_core_module/test_core_types.py` — 对 `TyxonQError` / `CompilationError` / `DeviceExecutionError` 的导入与断言测试，是除定义文件外唯一显式引用该模块的地方。
- 业务代码（如 `applications/chem/algorithms/ewf/bath.py`、`cluster.py` 等）直接使用 `ValueError` / `TypeError` / `RuntimeError` 进行参数与数值校验。

## 3. 架构与约定

- **分层**：`TyxonQError` 作为顶层抽象，`CompilationError`、`DeviceExecutionError` 分别对应编译器流水线与设备执行两个高层子系统。
- **传播方式**：目前仅通过直接 `raise` 向上冒泡，没有统一的 error wrapper、error code 枚举或错误上下文对象。
- **警告**：`VectorizationFallbackWarning` 继承自 `Warning`，遵循 Python 警告语义，可通过 `warnings.filterwarnings` 控制。
- **未覆盖区域**：网络 I/O、云平台 API、硬件驱动层的错误在仓库证据中未发现统一封装；示例脚本（如 `examples/cloud_api_*.py`）通常让底层异常直接上抛。

## 4. 约定与约束

### 观察到的约定（描述性）

- 框架级异常以 `TyxonQError` 为基类，子类命名后缀为 `Error`。
- 参数合法性检查普遍使用 `ValueError` / `TypeError`，数值计算异常使用 `RuntimeError`。
- 自定义异常目前仅在测试中被显式 import 和使用，核心业务模块尚未大量替换为 `CompilationError` / `DeviceExecutionError`。

### 可验证的规则（有明确来源）

- **规则 1**：`TyxonQError` 继承自 `Exception`，`CompilationError` 与 `DeviceExecutionError` 继承自 `TyxonQError`。
  - 依据：`src/tyxonq/core/errors.py` 第 1–10 行的类定义。
- **规则 2**：`VectorizationFallbackWarning` 继承自 `Warning`，属于 Python 警告而非异常。
  - 依据：`src/tyxonq/core/errors.py` 第 13–14 行。
- **规则 3**：测试套件通过 `pytest.raises` 断言 `CompilationError` 与 `DeviceExecutionError` 的行为，构成对这两个异常类型的契约测试。
  - 依据：`tests_core_module/test_core_types.py` 第 15 行导入及第 48、52 行 raise 断言。

### 当前缺口

- 自定义异常未在核心业务模块（compiler、devices、applications 等）中广泛使用，实际错误传播仍以 Python 内置异常为主。
- 没有统一的错误码、错误消息格式化或错误分类工具。
- 未见针对 `TyxonQError` 子类的统一捕获与转换逻辑（例如将第三方库异常包装为 `DeviceExecutionError`），因此上层调用方需要自行区分内置异常与 TyxonQ 自定义异常。
