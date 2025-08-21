## TyxonQ 重构迁移方案（提案稿 · 团队评审）

本方案面向将 TyxonQ 重构为工程化、鲁棒、横向可扩展的量子计算框架。聚焦抽象清晰、插件化扩展、与实际硬件/云后端兼容，同时保留对数值计算与自动微分生态（NumPy、PyTorch、cuNumeric）的良好支持。

### 目标与设计原则
- **工程化与稳定 API**：核心模块公开稳定接口，内部细节可演进。
- **分层清晰**：应用 → 编译 → 执行 → 后处理，职责单一、耦合最小。
- **可插拔扩展**：设备、编译器、后处理均通过插件注册与能力声明。
- **数值后端统一**：统一 `numerics/` 抽象，内置支持 NumPy、PyTorch、cuNumeric。
- **安全的矢量化策略**：vmap/JIT 遇风险自动回退为安全 eager 路径，记录可观测信息。

---

## 新的整体目录结构（拟）

```
src/tyxonq/
  core/
    ir/
      circuit.py
      hamiltonian.py
      pulse.py                 # 可选：后续需要再落地
    operations/
    measurements/
    types.py
    errors.py

  applications/
    api.py                    # Application/ProblemDefinition 协议
    domains/
      ai/
      finance/
      physics/

  compiler/
    api.py                    # Compiler、Pass、Pipeline 抽象
    passes/
      layout/
      decompose/
      optimize/
      scheduling/
      rewrite/                # 变换类操作归于编译重写，无 Tape
    gradients/
      parameter_shift.py
      adjoint.py
      finite_diff.py
    dialects/
    backends/
      qiskit/
        __init__.py
        compiler.py

  devices/
    base.py                   # Device 抽象/能力声明
    simulators/
      statevector/
      densitymatrix/
      mps/
    hardware/
      ibm/
      braket/
    session.py
    executor.py

  numerics/
    api.py                    # ArrayBackend 协议与工厂
    backends/
      numpy_backend.py
      torch_backend.py
      cunumeric_backend.py
    checks.py                 # 前置静态/运行期检查
    linalg.py
    random.py
    autodiff/
      bridge.py               # 与编译梯度方法的薄桥

  postprocessing/             # 原 results/ 更名
    mitigation/
    readout/
    metrics.py
    io.py

  plugins/
    registry.py

  compat/                     # 旧 API 的瘦适配与弃用提示

  utils/
  config/
```

说明：
- 取消单独 `transforms/` 与 Tape 抽象；变换行为通过 `compiler/passes/rewrite/` 与 `compiler/gradients/` 落地。
- 移除 `templates/` 与 `workflows/`；示例放在 `examples/`，教程放在 `docs/tutorials/`。

---

## 旧 → 新 模块映射清单（第一批）

> 注：以下为建议映射，实际迁移中会根据代码粒度进一步细化与合并。

| 旧位置 | 新位置 | 备注 |
|---|---|---|
| `src/tyxonq/cons.py` | `src/tyxonq/core/types.py` | 常量/类型合并到 core；命名规范化 |
| `src/tyxonq/utils.py` | `src/tyxonq/utils/`（多文件拆分） | 日志、并行、随机数、缓存、配置等拆分 |
| `src/tyxonq/gates.py` | `src/tyxonq/core/operations/` | 门与通道归到 operations；与 measurements 区分 |
| `src/tyxonq/channels.py` | `src/tyxonq/core/operations/` | 通道操作合并管理 |
| `src/tyxonq/quantum.py` | `src/tyxonq/core/ir/` | 电路/算符表示拆分至 circuit/hamiltonian |
| `src/tyxonq/circuit.py` | `src/tyxonq/core/ir/circuit.py` | 作为 IR 的核心对象 |
| `src/tyxonq/densitymatrix.py` | `src/tyxonq/devices/simulators/densitymatrix/` | 执行层模拟器实现 |
| `src/tyxonq/mpscircuit.py` | `src/tyxonq/devices/simulators/mps/` | MPS 模拟器内核与包装 |
| `src/tyxonq/mps_base.py` | `src/tyxonq/devices/simulators/mps/` | 与上同 |
| `src/tyxonq/noisemodel.py` | `src/tyxonq/postprocessing/mitigation/` | 噪声/误差缓解归后处理；必要时设备校准配合 |
| `src/tyxonq/results/` | `src/tyxonq/postprocessing/` | 目录整体更名（保持语义） |
| `src/tyxonq/results/readout_mitigation.py` | `src/tyxonq/postprocessing/readout/` | 读出校正 |
| `src/tyxonq/results/qem/` | `src/tyxonq/postprocessing/mitigation/` | 误差缓解方法 |
| `src/tyxonq/compiler/qiskit_compiler.py` | `src/tyxonq/compiler/backends/qiskit/compiler.py` | 后端化与方言/Pass 解耦 |
| `src/tyxonq/backends/*` | `src/tyxonq/numerics/backends/*` + `src/tyxonq/devices/simulators/*` | 前端数值与执行后端解耦 |
| `src/tyxonq/torchnn.py` | `examples/` 或 `applications/` 相关 | 训练流程示例化/领域化 |
| `src/tyxonq/templates/*` | 移除（合并到 applications 或 examples） | 以应用为准，不再单列模板 |

兼容策略见下文 `compat/`。

---

## 最小接口草案（仅类型与 docstring）

> 以下为关键协议与数据结构草案，实际实现可按需扩展。统一使用类型标注与精简 docstring，便于 IDE 与文档生成。

```python
# src/tyxonq/numerics/api.py
from __future__ import annotations
from typing import Protocol, Any, Literal, Callable, Tuple

VectorizationPolicy = Literal["auto", "force", "off"]

class ArrayBackend(Protocol):
    """统一的数组/张量后端协议。

    实现需提供：
    - 创建/转换：array, asarray, to_numpy
    - 基本算术/广播/索引
    - 线性代数：matmul, einsum, svd, qr, eigh 等（按需）
    - 随机：rng(seed), randn/normal
    - 自动微分桥（可选）：requires_grad, detach
    - 设备/数据类型管理：device, dtype
    """

    name: str

    def array(self, data: Any, dtype: Any | None = None) -> Any: ...
    def asarray(self, data: Any) -> Any: ...
    def to_numpy(self, data: Any) -> "np.ndarray": ...

    def matmul(self, a: Any, b: Any) -> Any: ...
    def einsum(self, subscripts: str, *operands: Any) -> Any: ...

    def rng(self, seed: int | None = None) -> Any: ...
    def normal(self, rng: Any, shape: Tuple[int, ...], dtype: Any | None = None) -> Any: ...

    def requires_grad(self, x: Any, flag: bool = True) -> Any: ...
    def detach(self, x: Any) -> Any: ...

def vectorize_or_fallback(
    fn: Callable[..., Any],
    backend: ArrayBackend,
    policy: VectorizationPolicy = "auto",
    *,
    enable_checks: bool = True,
) -> Callable[..., Any]:
    """返回包装后的函数：
    - 在可行时使用后端的向量化/vmap 机制执行；
    - 检测到不安全模式或运行时警告时回退为 eager；
    - 记录回退原因到日志/元数据。
    """
    ...
```

```python
# src/tyxonq/devices/base.py
from __future__ import annotations
from typing import Protocol, Any, TypedDict, Dict

class DeviceCapabilities(TypedDict, total=False):
    native_gates: set[str]
    max_qubits: int
    connectivity: Any
    supports_shots: bool
    supports_batch: bool

class RunResult(TypedDict, total=False):
    samples: Any
    expectations: Dict[str, float]
    metadata: Dict[str, Any]

class Device(Protocol):
    """设备抽象。负责电路执行、测量与采样。"""

    name: str
    capabilities: DeviceCapabilities

    def run(self, circuit: "Circuit", shots: int | None = None, **kwargs) -> RunResult: ...
    def expval(self, circuit: "Circuit", obs: "Observable", **kwargs) -> float: ...
```

```python
# src/tyxonq/compiler/api.py
from __future__ import annotations
from typing import Protocol, Any, Dict, TypedDict

class CompileRequest(TypedDict):
    circuit: "Circuit"
    target: "DeviceCapabilities"
    options: Dict[str, Any]

class CompileResult(TypedDict):
    circuit: "Circuit"
    metadata: Dict[str, Any]

class Pass(Protocol):
    """编译 Pass：输入电路与目标能力，输出等价/优化后的电路。"""
    def run(self, circuit: "Circuit", caps: "DeviceCapabilities", **opts) -> "Circuit": ...

class Compiler(Protocol):
    """编译器：将 IR 转换为面向目标设备/后端的电路。"""
    def compile(self, request: CompileRequest) -> CompileResult: ...
```

```python
# src/tyxonq/core/ir/circuit.py（草案）
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List

@dataclass
class Circuit:
    """量子电路 IR 的最小骨架。"""
    num_qubits: int
    ops: List[Any]

@dataclass
class Hamiltonian:
    """哈密顿量 IR（可表示 Pauli 和、稀疏/稠密矩阵等）。"""
    terms: Any
```

```python
# src/tyxonq/plugins/registry.py（草案）
from __future__ import annotations
from typing import Any, Dict, Type

def discover(group: str) -> Dict[str, Type[Any]]:
    """通过 entry points 发现扩展组件，返回 name→class 的映射。"""
    ...

def get_device(name: str):
    """按名称加载设备实例，支持懒加载与缓存。"""
    ...

def get_compiler(name: str):
    """按名称加载编译器实例，进行能力校验与冲突检测。"""
    ...
```

---

## 与 PennyLane 的对比与对齐关系

- **设备与执行**：
  - PennyLane：`devices/` + `QNode/Tape` 执行范式。
  - TyxonQ（本方案）：保留 `devices/`，但不引入 Tape；通过 `compiler/passes` 的重写与 `devices/session` 执行，简化心智模型。

- **运算与观测**：
  - PennyLane：`ops/`、`measurements/` 分层清晰。
  - TyxonQ：`core/operations/`、`core/measurements/` 对齐，保持 IR 稳定。

- **变换与梯度**：
  - PennyLane：大量 `transforms/` 与 `gradients/` 基于 Tape。
  - TyxonQ：将变换纳入编译 Pass（`rewrite/`），梯度方法放 `compiler/gradients/`，与 `numerics.autodiff` 薄桥对接。

- **接口/数值后端**：
  - PennyLane：多接口（torch/jax/tf）整合在 QNode 层。
  - TyxonQ：集中在 `numerics/`，内置仅 NumPy/PyTorch/cuNumeric，后续扩展走插件。

- **插件化**：
  - PennyLane：`entry points` 扩展设备与变换。
  - TyxonQ：提供 `plugins/registry.py` 与 entry points 组：`tyxonq.devices`、`tyxonq.compilers`、`tyxonq.postprocessing`、`tyxonq.applications`。

---

## vmap / 向量化策略与回退考量

目标：在不牺牲正确性与稳定性的前提下获得批量/并行收益。

### 风险点（以 PyTorch 为例，亦适用于其他后端）
- in-place 操作导致别名/梯度破坏（`x += ...`）。
- 隐式 `.clone()` 依赖与 `torch.func.vmap` 的 alias 警告。
- 数据依赖控制流使得 vmap/JIT 不可静态化。
- dtype/shape 多态导致编译/缓存抖动。

### 前置检查与围栏
- `numerics/checks.py`：
  - 静态/半静态检查常见危险模式（可通过装饰器对函数 AST/bytecode 进行轻量扫描，或在运行时对输入特征进行验证）。
  - 运行时“警告转异常”策略：在矢量化执行上下文中捕获特定类别警告并转化为可捕获异常。

### 策略选择与自动回退
- 策略：`VectorizationPolicy = {auto, force, off}`
  - `auto`：优先 vmap/编译，发生风险/警告→回退 eager。
  - `force`：强制 vmap/编译（仅用于基准/研究，不建议默认）。
  - `off`：始终 eager。

```python
# 伪代码
def vectorize_or_fallback(fn, backend, policy="auto", enable_checks=True):
    def wrapped(*args, **kwargs):
        if policy == "off":
            return fn(*args, **kwargs)
        if enable_checks and not checks.safe_for_vectorization(fn, args, kwargs, backend):
            return fn(*args, **kwargs)  # 直接回退
        try:
            with checks.warn_as_error(["AliasWarning", "CloneRequiredWarning"]):
                vfn = backend.vmap(fn)  # 若后端支持，否则使用广播批次
                return vfn(*args, **kwargs)
        except Exception as ex:
            logger.info("vectorization fallback to eager: %s", ex)
            return fn(*args, **kwargs)
    return wrapped
```

### 可观测性与配置
- 在 `RunResult.metadata` 与框架日志中记录：是否矢量化、回退原因、次数与耗时占比。
- `tyxonq.toml` 配置：

```toml
[numerics]
backend = "torch"        # numpy | torch | cunumeric
vectorization_policy = "auto"  # auto | force | off
warn_on_fallback = true
```

---

## 插件与注册（plugins/registry）

### 入口点建议
```
[project.entry-points."tyxonq.devices"]
ibm = "tyxonq_ext_ibm.device:IBMDevice"

[project.entry-points."tyxonq.compilers"]
qiskit = "tyxonq.compiler.backends.qiskit.compiler:QiskitCompiler"

[project.entry-points."tyxonq.postprocessing"]
m3 = "tyxonq_ext_m3.readout:M3Mitigator"
```

### 运行时行为
- 懒加载、名称冲突检测、版本与能力校验（如门集、shots 支持）。
- 缓存发现结果，允许覆盖/屏蔽内置组件（按配置优先级）。

---

## 兼容层（compat/）策略

- 目的：平滑迁移旧 API，避免一次性破坏用户代码。
- 方式：
  - 提供旧模块名到新实现的轻量转发；
  - 进入点抛出 `DeprecationWarning`，附带迁移建议；
  - 在 1–2 个小版本窗口后移除；
  - 维护“旧→新”映射清单，CI 增加兼容契约测试。

---

## 渐进式迁移计划（建议）

1. 设计冻结（1 周）
   - 确认 `core/ir`、`devices/base`、`compiler/api`、`numerics/api` 最小接口。
2. 骨架搭建（1–2 周）
   - 新目录就绪；实现 numerics 三后端与 `vectorize_or_fallback`；落 `plugins/registry`；起 `compat/`。
3. 编译与设备（2 周）
   - 迁移 `qiskit_compiler`；模拟器按目录拆分；打通最小 VQE 路径（applications→compiler→device→postprocessing）。
4. 梯度与优化（1–2 周）
   - `compiler/gradients` 参数移位/伴随；与 numerics.autodiff 薄桥验证。
5. 后处理与文档（1 周）
   - `results/`→`postprocessing/` 重命名与整合；示例与教程完善；性能/回退日志验证。
6. 清理与稳定（1 周）
   - 去冗余；补充测试与基准；冻结公共 API，发布迁移指南。

---

## 评审关注点（建议）
- 命名与目录是否清晰传达职责？
- `numerics/` 的范围是否合适（只含三后端）？
- 无 Tape 的变换路径是否足够覆盖现有/预期需求？
- 向量化回退策略的默认行为与可观测性是否合理？
- 插件注册与兼容层的维护成本可接受吗？

---

本文档为评审草案。若方向达成一致，可据此生成 ADR 与接口骨架文件并开始迁移实施。


