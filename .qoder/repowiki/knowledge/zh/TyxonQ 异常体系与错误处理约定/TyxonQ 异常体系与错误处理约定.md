---
kind: error_handling
name: TyxonQ 异常体系与错误处理约定
category: error_handling
scope:
    - '**'
source_files:
    - src/tyxonq/core/errors.py
    - src/tyxonq/devices/base.py
    - src/tyxonq/compiler/api.py
    - src/tyxonq/__init__.py
    - tests_core_module/test_core_types.py
    - src/tyxonq/applications/chem/algorithms/lucj/double_factorization.py
    - src/tyxonq/applications/chem/algorithms/lucj/circuit_builder.py
---

## 1. 使用的系统/方法

TyxonQ 采用 Python 原生异常体系，并在 `src/tyxonq/core/errors.py` 中定义了一个统一的异常层次结构：
- `TyxonQError(Exception)`：框架自定义异常的根类。
- `CompilationError(TyxonQError)`：编译阶段失败时抛出（如编译计划构建失败）。
- `DeviceExecutionError(TyxonQError)`：设备执行阶段失败时抛出（如超时、连接问题）。
- `VectorizationFallbackWarning(Warning)`：向量化被禁用或回退到 eager 模式时的警告。

该模块是仓库内唯一集中定义的 TyxonQ 专属异常类型；业务代码并未广泛使用这些类型，而是主要依赖标准异常。

## 2. 关键文件与位置

- `src/tyxonq/core/errors.py`：定义 `TyxonQError`、`CompilationError`、`DeviceExecutionError`、`VectorizationFallbackWarning`。
- `tests_core_module/test_core_types.py`：唯一显式断言异常继承层级的测试，通过 `raise CompilationError(...)` / `raise DeviceExecutionError(...)` 并 `except TyxonQError` 验证多态捕获。
- `src/tyxonq/devices/base.py`：统一设备协议 (`Device` Protocol)、`DeviceTask`、`RunResult`、`DeviceRule` 等类型化契约，作为模拟器与硬件驱动的统一抽象。
- `src/tyxonq/compiler/api.py`：统一编译入口 `compile()`，返回 `CompileResult` / `PulseCompileResult` TypedDict，并通过平台分派调用各编译器。
- `src/tyxonq/__init__.py`：包级懒加载与可选依赖导入，大量使用 `try/except Exception` 包裹可选模块导入，保证包可被无可选依赖环境导入。
- `src/tyxonq/applications/chem/algorithms/lucj/*.py`：化学算法子模块中使用 `ValueError`、`ImportError` 进行参数校验与可选依赖缺失提示。

## 3. 架构与约定

### 3.1 分层异常设计
框架在核心层定义了 `TyxonQError` 及其子类，意图让上层（编译器、设备层、应用层）按领域区分异常来源。但实际代码中，除测试外，业务逻辑几乎全部使用标准异常（`ValueError`、`TypeError`、`ImportError`、`AttributeError`），自定义异常尚未被广泛消费——这使当前异常体系更像“预留扩展点”而非已全面落地的规范。

### 3.2 设备执行层的错误传播
`devices.base.Device` 以 Protocol 形式定义 `run` / `expval` 接口，所有模拟器与硬件驱动实现该协议。`DeviceTask.get_result()` 对结果 schema 做归一化处理（补齐 `result`、`metadata`、`expectations`、`probabilities` 等键），将不同驱动的原始 payload 转换为下游后处理可用的统一结构。设备执行过程中的运行时错误（超时、连接失败等）应通过 `DeviceExecutionError` 表达，但目前未见具体驱动主动抛出的证据。

### 3.3 编译器的错误路径
`compiler.api.compile()` 是门电路与脉冲电路的统一编译入口，内部根据 `platform`、`output`、是否含脉冲操作等条件自动选择编译通道（gate compiler / pulse compiler / 平台方言）。文档注释中声明可能抛出 `ValueError`（不支持的 provider）、`ImportError`（驱动模块不可用），但源码正文未直接 raise；实际错误由底层步骤或平台适配器抛出。`CompileResult` / `PulseCompileResult` 作为 TypedDict 明确输出结构，便于调用方安全消费。

### 3.4 包导入期的健壮性
`src/tyxonq/__init__.py` 对大量可选依赖（如 PySCF、OpenMM、i-PI、Quafu SDK 等）使用 `try/except ImportError` / `except Exception` 包裹，确保即使缺少可选依赖也能成功 import 包主体；仅在真正调用相关功能时才触发缺失错误。这是框架“按需加载”的错误策略。

### 3.5 业务层参数校验
化学算法子模块（如 LUCJ）普遍使用 `raise ValueError("...")` 对输入形状、取值范围、必填字段做前置校验，例如 `t2_amplitudes must be real-valued`、`max_terms must be at least 1`、`Missing LUCJ parameter ...` 等。这类错误属于用户输入错误，不包装为 TyxonQ 自定义异常。

## 4. 约定与约束

- **异常层次**：框架定义了 `TyxonQError` → `CompilationError` / `DeviceExecutionError` 的两层分类，但仅测试覆盖其继承关系；业务代码未强制使用该层次。
- **标准异常优先**：参数校验、类型错误、缺失依赖等场景统一使用 Python 标准异常（`ValueError`、`TypeError`、`ImportError`、`AttributeError`），未引入第三方异常库。
- **无全局中间件**：仓库中没有类似 Web 框架的异常中间件或全局异常处理器；错误通过函数返回值 + 异常向上冒泡的方式传播。
- **无 panic/recover**：Python 语义下不使用 `panic`；未见 `try/finally` 形式的资源清理兜底，资源释放依赖上下文管理器或析构。
- **可选依赖容错**：包导入期对所有可选依赖使用宽泛 `except Exception` 捕获，避免安装时失败；运行时再暴露真实错误。
- **设备结果规范化**：`DeviceTask.get_result()` 强制补齐结果字典中的关键字段，保证下游后处理不因驱动差异而崩溃，这是一种“防御性错误规避”策略。
- **测试约束**：`test_core_types.py` 是唯一断言异常类型的测试，要求 `CompilationError` / `DeviceExecutionError` 能被 `TyxonQError` 捕获，构成对异常层次的回归约束。

综上，TyxonQ 的错误处理以“标准异常为主 + 自定义异常骨架预留”的模式运行：核心层提供清晰的异常分类命名空间，但实际业务仍广泛依赖 Python 内置异常；设备层通过 Protocol 和结果归一化降低跨驱动错误处理的复杂度；包导入期通过宽泛 try/except 保证最小可用面。