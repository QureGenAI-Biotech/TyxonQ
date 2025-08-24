## TyxonQ 重构迁移方案（提案稿 · 团队评审）

本方案面向将 TyxonQ 重构为工程化、鲁棒、横向可扩展的量子计算框架。聚焦抽象清晰、插件化扩展、与实际硬件/云后端兼容，同时保留对数值计算与自动微分生态（NumPy、PyTorch、cupynumeric）的良好支持。

### 目标与设计原则
- **工程化与稳定 API**：核心模块公开稳定接口，内部细节可演进。
- **分层清晰**：应用 → 编译 → 执行 → 后处理，职责单一、耦合最小。
- **可插拔扩展**：设备、编译器、后处理均通过插件注册与能力声明。
- **数值后端统一**：统一 `numerics/` 抽象，内置支持 NumPy、PyTorch、cuNumeric。
- **安全的矢量化策略**：vmap/JIT 遇风险自动回退为安全 eager 路径，记录可观测信息。

---

## 执行方案总览（可落地）

- **目标（Outcome）**：
  - 稳定的公共 API（`core`/`compiler`/`devices`/`numerics`）；
  - 化学主线从分子到哈密顿量、编译、执行、后处理的最小可用链路；
  - 默认安全的向量化策略（auto 回退），并在元数据中可观测；
  - 测试驱动与 CI 门禁，任何增量在全绿后合入。
- **方法（How）**：
  - 渐进式分阶段；每阶段都有可验收产出与 DoD；
  - TDD 优先，契约/单元/集成/性能分层；
  - ADR 记录关键设计决策；
  - 严格的模块边界与命名规范；
  - 插件机制可选，兼容层移除（破坏性迁移）。
- **度量（Metrics）**：
  - 覆盖率≥80%（核心模块）；
  - 向量化回退比例与主要原因；
  - 关键 E2E 路径性能基线与波动阈值；
  - 示例与文档可运行性（CI 验证）。

---

## 范围与边界（Scope）

- **在范围内**：
  - 目录重构与模块抽象；
  - `app/chem` PySCF 风格接口；
  - `compiler` 阶段化与目标适配；
  - `devices` 模拟器/硬件抽象与会话；
  - `numerics` 三后端与向量化策略；
  - `postprocessing` 误差缓解与读出校正；
  - TDD 与 CI 门禁落地。
- **不在范围内（本期）**：
  - JAX/TF 接口（以后通过插件或扩展）；
  - Tape 抽象；
  - 全量硬件厂商支持（先以 qiskit/IBMQ 作为样板）。

---

## 新的整体目录结构（拟）

```

### 示例六：集成 OpenFermion（化学到 Hamiltonian）

```python
from tyxonq.integrations.openfermion import jordan_wigner_to_qubit_hamiltonian
from openfermion import MolecularData

mol = MolecularData(geometry=[('H',(0,0,0)),('H',(0,0,0.74))], basis='sto-3g', multiplicity=1, charge=0)
# 假设已通过外部驱动获得 one-/two-body integrals
ham = jordan_wigner_to_qubit_hamiltonian(mol)

# ham: tyxonq.core.ir.Hamiltonian，可直接进入 compiler → devices
```

### 示例七：集成 OpenQAOA（QASM/Qiskit → IR Circuit）

```python
from tyxonq.integrations.openqaoa import to_ir_circuit

# openqaoa 可生成 qiskit.QuantumCircuit 或 QASM 字符串
qc = build_openqaoa_qc(...)  # 来自 openqaoa 的构建
circ = to_ir_circuit(qc)     # 或传入 qasm 字符串

compiled = compiler.compile({"circuit": circ, "target": dev.capabilities, "options": {}})
res = dev.run(compiled["circuit"], shots=10000)
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

  app/
    chem/
      __init__.py
      gto.py                  # 分子与几何/基组（PySCF 风格 Mole）
      scf.py                  # 自洽场计算（RHF/UHF 等）
      integrals.py            # 一/二电子积分
      active_space.py         # 活性空间定义
      fermion_to_qubit.py     # 费米子→量子比特哈密顿量映射（JW/Parity/BK）

  compiler/
    api.py                    # Compiler、Stage、Pipeline 抽象 统一
    pipeline.py               # 可组合流水线定义与构建器
    native_complier.py        # 自带编译器
    stages/
      layout/
      decompose/
      optimize/
      scheduling/
      rewrite/                # 变换类操作归于编译重写，无 Tape
      measurement/            # 新增：期望/分组/概率等测量级重写
      shot_scheduler.py       # 新增：shot vectors 调度与合并
    gradients/
      parameter_shift.py
      adjoint.py
      finite_diff.py
    providers/                # 统一编译器提供方
      qiskit/
        __init__.py           # 暴露 Qiskit 编译器（IR→QuantumCircuit/QASM2）
        qiskit_compiler.py    # 具体实现；文件名避免与目录“compiler”混淆
        dialect.py            # 可选：Qiskit 方言/兼容层（门/指令映射）

  devices/
    base.py                   # Device 抽象/能力声明
    simulators/
      statevector/            # 纯态（原 wavefunction）
      density_matrix/         # 密度矩阵法
      matrix_product_state/       # 压缩态模拟器（面向大尺度/低秩态态的高效模拟）
        backends/
          backend_base.py     # 压缩态后端协议
          pytorch_backend.py  # 基于 PyTorch 的实现
          numpy_backend.py    # 基于 NumPy 的实现
          cupynumeric_backend.py # 基于 cupynumeric 的实现（可选 cuTensorNet 加速）
      vendor/
        cuquantum/            # NVIDIA cuQuantum 供应商模拟器封装（custatevec/cutensornet）
    hardware/
      ibm/
      braket/
    session.py
    executor.py

  numerics/
    api.py                    # ArrayBackend 协议与工厂（已用于模拟器内核）
    backends/
      numpy_backend.py
      pytorch_backend.py
      cupynumeric_backend.py
    vectorization_checks.py                 # 前置静态/运行期检查
    linalg.py
    random.py
    autodiff/
      bridge.py               # 与编译梯度方法的薄桥
    accelerators/             # 说明：专属数值库的可选绑定（仅数值算子层）
      cutensornet.py          # NVIDIA cuTensorNet（张量收缩加速）
      custatevec.py           # NVIDIA cuStateVec（波函数加速）
      custom_ops.py           # 自定义核/优化路径

  postprocessing/             # 原 results/ 更名（io/metrics/readout/qem 已迁移）
    mitigation/
    readout/
    metrics.py
    io.py

  plugins/
    registry.py
    openfermion.py            # OpenFermion 适配（FO/QO → Hamiltonian）
    openqaoa.py               # OpenQAOA 适配（QASM/Qiskit → Circuit）

  utils/
  config/
```

说明：
- 取消单独 `transforms/` 与 Tape 抽象；变换行为通过 `compiler/stages/rewrite/` 与 `compiler/gradients/` 落地。
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
| `src/tyxonq/densitymatrix.py` | `src/tyxonq/devices/simulators/density_matrix/engine.py` | 执行层模拟器实现（已达语义对齐，可删 legacy） |
| `src/tyxonq/mpscircuit.py` | `src/tyxonq/devices/simulators/mps/` | MPS 模拟器内核与包装 |
| `src/tyxonq/mps_base.py` | `src/tyxonq/devices/simulators/mps/` | 与上同 |
| `src/tyxonq/noisemodel.py` | `src/tyxonq/postprocessing/mitigation/` | 噪声/误差缓解归后处理；必要时设备校准配合 |
| `src/tyxonq/results/` | `src/tyxonq/postprocessing/` | 目录整体更名（保持语义） |
| `src/tyxonq/results/readout_mitigation.py` | `src/tyxonq/postprocessing/readout/` | 读出校正 |
| `src/tyxonq/results/qem/` | `src/tyxonq/postprocessing/mitigation/` | 误差缓解方法 |
| `src/tyxonq/compiler/qiskit_compiler.py` | `src/tyxonq/compiler/providers/qiskit/{qiskit_compiler.py,dialect.py}` | 目标适配放入 provider；方言/映射在 `dialect.py` |
| `src/tyxonq/backends/*` | `src/tyxonq/numerics/backends/*` + `src/tyxonq/devices/simulators/*` | 前端数值与执行后端解耦 |
| `src/tyxonq/torchnn.py` | `examples/` 或 `app/chem/` 相关 | 训练流程示例化/领域化 |
| `src/tyxonq/templates/*` | 移除（合并到 applications 或 examples） | 以应用为准，不再单列模板 |

（已移除兼容层，本次为破坏性迁移，参见迁移计划与映射清单。）

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
  - TyxonQ（本方案）：保留 `devices/`，但不引入 Tape；通过 `compiler/stages` 的重写与 `devices/session` 执行，简化心智模型。

- **运算与观测**：
  - PennyLane：`ops/`、`measurements/` 分层清晰。
  - TyxonQ：`core/operations/`、`core/measurements/` 对齐，保持 IR 稳定。

- **变换与梯度**：
  - PennyLane：大量 `transforms/` 与 `gradients/` 基于 Tape。
  - TyxonQ：将变换纳入编译 Stage（`stages/rewrite/`），梯度方法放 `compiler/gradients/`，与 `numerics.autodiff` 薄桥对接。

- **接口/数值后端**：
  - PennyLane：多接口（torch/jax/tf）整合在 QNode 层。
  - TyxonQ：集中在 `numerics/`，内置仅 NumPy/PyTorch/cuNumeric，后续扩展走插件。

- **插件化**：
  - PennyLane：`entry points` 扩展设备与变换。
  - TyxonQ：提供 `plugins/registry.py` 与 entry points 组：`tyxonq.devices`、`tyxonq.compilers`、`tyxonq.postprocessing`、`tyxonq.applications`。

### 明确映射（Tape → TyxonQ 方案）
- Tape 记录 qfunc → `core.ir.Circuit` + 可选 `CircuitBuilder`（轻量录制器，后续补强）。
- Tape expand → `compiler/stages/decompose` 与 `compiler/stages/rewrite`。
- 测量级变换（聚合/分组/重写） → `compiler/stages/rewrite/measurement.py`（新增）。
- 设备执行 → `compiler/targets/*` 产物 + `devices/*` 执行。
- 梯度变换 → `compiler/gradients/*`（parameter-shift、adjoint、finite-diff）。
- 批量/shot vectors → `numerics.vectorize_or_fallback` + `devices/session` + `stages/scheduling/shot_scheduler.py`（新增）。

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

## 无 Tape 方案的覆盖性与补强建议

本方案不引入 Tape，而以 IR + 编译阶段化（stages + pipeline）实现等价能力：

### 已覆盖能力
- 操作/测量展开与重写：`stages/decompose`、`stages/rewrite`（含 measurement 重写）。
- 目标方言映射与能力协商：`compiler/targets/*` + `DeviceCapabilities`。
- 梯度与可微：`compiler/gradients/*` + `numerics/autodiff` 薄桥。
- 批量/向量化与 shot vectors：`numerics.vectorize_or_fallback` + `devices/session` + Shot 调度（见下）。
- 可组合变换：`compiler/pipeline.py` 有序 stage 组合，可插拔扩展。

### 补强项（建议落地）
- `core/operations` 增加梯度元数据：是否可移位、生成元、shift 系数等。
- 新增 `stages/rewrite/measurement.py`：统一期望/方差/概率/观测量分组等测量级重写。
- 新增 `stages/scheduling/shot_scheduler.py`：支持 shot 向量（分段采样）与合并策略。
- 提供 `core/ir.CircuitBuilder`（轻量录制器，可选）：便于装饰器式体验，将用户函数生成的 IR 交由 pipeline 处理。

上述补强可使能力对齐 PennyLane Tape 的主要使用场景，同时保持工程实现简洁、面向硬件优化更直接。

---

## 示例

### 示例一：从 qfunc/CircuitBuilder 到 pipeline 的最小用例

```python
# 伪代码，仅示意接口风格
from tyxonq.core.ir import Circuit
from tyxonq.core.ir.builder import CircuitBuilder  # 轻量录制器（建议新增）
from tyxonq.compiler.pipeline import build_pipeline
from tyxonq.compiler.api import Compiler
from tyxonq.devices import get_device

def qfunc(theta):
    with CircuitBuilder(num_qubits=2) as cb:
        cb.rx(0, theta)
        cb.cx(0, 1)
        cb.measure_z(1)
    return cb.circuit()

circ: Circuit = qfunc(0.3)

pipeline = build_pipeline([
    "decompose",
    "rewrite/measurement",
    "layout",
    "scheduling",
])

compiler: Compiler = ...  # 选择 `compiler/targets/qiskit` 等
compiled = compiler.compile({
    "circuit": circ,
    "target": get_device("ibm").capabilities,
    "options": {"opt_level": 2}
})

res = get_device("ibm").run(compiled["circuit"], shots=4000)
print(res["expectations"])  # 或 samples/metadata
```

### 示例二：shot vectors 经 shot_scheduler 的执行路径

```python
# 伪代码，仅示意接口风格
from tyxonq.compiler.stages.scheduling import shot_scheduler

shot_plan = [100, 1000, 5000]  # shot vectors
compiled = compiler.compile({"circuit": circ, "target": dev.capabilities, "options": {}})

scheduled = shot_scheduler.schedule(compiled["circuit"], shot_plan)
# scheduled 可能包含：多分段电路或同一电路的分段执行计划

agg = devices.session.execute_plan(dev, scheduled)
# session 聚合每段结果，同时记录 metadata：每段耗时、误差条、合并方法

print(agg.expectations, agg.metadata["per_segment"])
```

---

### 示例三：PySCF 风格的化学接口用法

```python
from tyxonq.app.chem import Mole, RHF, Integrals, ActiveSpace
from tyxonq.app.chem.fermion_to_qubit import to_qubit_hamiltonian

mol = Mole(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", charge=0, spin=0, unit="Angstrom").build()
mf = RHF(mol).run()
ints = Integrals(mol, mf)
aspace = ActiveSpace(nelec=2, norb=2)

ham = to_qubit_hamiltonian(ints, aspace, mapping="jw")
# 直接使用 ham 进入编译与执行流程，无需 Problem 封装
```

---

## 插件与注册（plugins/registry）

### 为什么需要？（可选）
- 默认情况下，TyxonQ 通过配置文件直接选择内置的 `compiler.targets.*` 和 `devices.*` 实现，已能满足典型需求。
- 当需要第三方扩展包“无修改核心代码”接入时，可选地启用 `plugins/registry`：
  - 允许外部 pip 包注册新设备/编译目标/后处理模块；
  - 核心仅做“名称→类”的查找与加载，不改变既有编译与执行流程。

### 简化使用方式
- 默认关闭插件发现，仅使用配置映射：
  - `config/loader.py` 解析 `tyxonq.toml` 中的 `device = "ibm"`、`compiler = "qiskit"` 等配置，映射到内置模块。
- 如需启用第三方扩展：
  - 在 `config` 中声明 `plugins.enabled = true` 与 `plugins.groups = ["tyxonq.devices", ...]`；
  - `plugins/registry.py` 负责按名称解析并实例化（可基于 entry points 或显式模块路径）。

注：若团队当前阶段无需第三方扩展，可暂不实现 `plugins/registry`，待需求明确再补充。

---

## 兼容层（compat/）策略（本次移除）

- 本次为大版本破坏性迁移，不再提供兼容层。
- 提供完整的“旧→新”映射清单与示例迁移指南，协助用户手动迁移。

---

## 渐进式迁移计划（建议）

1. 设计冻结（1 周）
   - 确认 `core/ir`、`devices/base`、`compiler/api`、`compiler/pipeline`、`numerics/api` 最小接口。
   - 产出：ADR-001（分层与命名）、ADR-002（接口骨架），样板测试套件；评审通过。
2. 骨架搭建（1–2 周）
   - 新目录就绪；实现 numerics 三后端与 `vectorize_or_fallback`；完成 `compiler/stages` 与 `pipeline` 骨架；新增 `stages/rewrite/measurement` 与 `stages/scheduling/shot_scheduler` 骨架；（插件系统可后置，不是必须）。
   - 产出：核心模块单元/契约测试全绿；样例示例编译通过；文档初稿。
3. 编译与设备（2 周）
   - 迁移 `qiskit_compiler` 至 `compiler/targets/qiskit`；模拟器按目录拆分；打通最小 VQE 路径（app→compiler→device→postprocessing）。
   - 产出：端到端 E2E 用例与性能基线；回退比例报告。
4. 梯度与优化（1–2 周）
   - `compiler/gradients` 参数移位/伴随；与 numerics.autodiff 薄桥验证；在 `core/operations` 标注梯度元数据（可移位/生成元/shift 规则）。
   - 向量化策略落地：
     - 默认 `vectorization_policy = "auto"`；实现 `numerics/checks.safe_for_vectorization` 与 warn→error 围栏；
     - `devices/session` 与 `RunResult.metadata` 记录矢量化/回退原因与成本；
     - 为核心实现编写 vmap-safe 规范并通过测试验证。
   - 产出：梯度契约测试、测量重写与 shot 调度测试绿。
5. TN/压缩态与供应商模拟器（1–2 周）
   - `simulators/compressed_state/backends` 完成协议与 pytorch/numpy/cupynumeric 三后端；
   - `simulators/vendor/cuquantum` 封装 custatevec/cutensornet，打通与 numerics.accelerators 探测；
   - 产出：跨后端一致 API、性能基准与自动加速验证测试。
6. 后处理与文档（1 周）
   - `results/`→`postprocessing/` 重命名与整合；示例与教程完善；性能/回退日志验证。
   - 产出：postprocessing 模块测试绿；教程与示例 CI 可运行。
7. 清理与稳定（1 周）
   - 去冗余；补充测试与基准；冻结公共 API，发布迁移指南。
   - 产出：迁移指南与旧→新映射最终版；版本标签与发布说明。

---

## 测试驱动开发（TDD）与 CI 门禁

### 总体原则
- 每个模块/子模块的重构必须伴随新增或更新的测试用例；
- 引入“最小可用版本”增量推进：每个阶段完成后，必须在 CI 中全绿后才能进入下阶段；
- 对核心路径（编译→设备→后处理）添加端到端（E2E）用例与性能基线。

### 测试层次
- 单元测试：`core/`、`numerics/`、`compiler/stages/*`、`devices/*`、`postprocessing/*`；
- 契约测试：
  - `Device` 契约：shots、batch、期望/采样一致性；
  - `Compiler` 契约：等价性测试（电路语义不变）、目标能力遵从；
  - `Stage` 契约：输入输出 IR 正确性与稳定属性（大小、测量集合等）；
- 集成测试：`app/chem` 典型问题到结果的完整链路；
- 性能/回归：为关键 pipeline 建立阈值（运行时、内存、回退比例）。

### 接受标准（DoD）
- 每个模块/子模块：
  - API 与文档注释完整；
  - 单元与契约测试通过，覆盖率达标；
  - 若影响 E2E 路径，需更新集成测试与基线；
  - 性能与回退比例不超过阈值；
  - 变更记录（CHANGELOG/ADR）更新。

### CI 门禁与报告
- PR 必须：单元/契约/集成测试通过；
- 覆盖率门槛：核心模块 ≥ 80%；
- 性能回归：关键基准不得超过阈值（配置化）；
- 向量化回退报告：CI 输出回退比率与主要原因统计。

---

## 模块职责与边界

为保证可维护性与清晰的协作边界，定义各模块目的、核心功能与不做的事情：

### core/
- 目的：提供稳定的中间表示（IR）、基本运算抽象与错误类型。
- 功能：
  - `ir/`：`Circuit`、`Hamiltonian` 基本结构；
  - `operations/`：量子门/通道元信息（可分解性、梯度元数据等）；
  - `measurements/`：测量与观测量定义；
  - `types.py`、`errors.py`：公共类型与异常；
- 不做：设备执行细节、目标方言映射、数值后端实现。

### app/
- 目的：领域问题到 IR 的构建（以化学为先）。
- 功能：
  - `chem/`：PySCF 风格组件（`Mole`/`RHF`/`Integrals`/`ActiveSpace`/`to_qubit_hamiltonian` 等）；
  - 入口方式：用户从 `app.chem` 直接导入类与函数进行组装；
- 不做：编译/设备的具体实现与优化；

### numerics/
- 目的：统一数组/张量后端抽象，提供可控的向量化与检查。
- 功能：
  - `api.py`：`ArrayBackend` 协议与选择；
  - `backends/`：numpy/pytorch/cupynumeric 实现；
  - `checks.py`：vmap 安全检查与 warn→error 围栏；
  - `autodiff/bridge.py`：与编译梯度方法的薄桥；
  - `accelerators/`：特定数值库加速（如 cuTensorNet/custatevec、自定义核），遵循“仅数值层增强、与设备层解耦”的原则；
- 不做：量子电路语义、设备交互。

### compiler/
- 目的：将 IR 转换为目标可执行形式，承载所有结构性变换与优化。
- 功能：
  - `api.py`：`Compiler`、`Stage`、`Pipeline` 抽象；
  - `pipeline.py`：可组合流水线；
  - `stages/`：布局、分解、优化、调度、测量重写、shot 调度；
  - `gradients/`：parameter-shift、adjoint、finite-diff；
  - `targets/`：目标方言/约束/专属 stage；
- 不做：直接设备执行（交由 devices）、数值运算实现（交由 numerics）。

#### 编译流水线示例（可配置）
```toml
[compiler.pipeline]
stages = [
  "decompose",
  "rewrite/measurement",
  "layout",
  "scheduling",
  "scheduling/shot_scheduler"
]
```

### 示例四：跨后端的压缩态后端选择（保持核心特性）

```python
from tyxonq.devices.simulators.compressed_state import CSDevice
from tyxonq.numerics import get_backend

# 选择数值后端（pytorch/numpy/cupynumeric）
backend = get_backend("pytorch")

# 创建 TN 模拟设备，选择后端
dev = CSDevice(num_qubits=32, backend="pytorch_backend", max_bond=128, dtype="float32")

# 如果是 NVIDIA 环境，自动探测并启用 cuTensorNet 加速
dev.enable_accelerator("cutensornet")  # 若不可用则无操作

res = dev.run(compiled_circuit, shots=None)
```

### 示例五：替换 tensornetwork（遗留）为 PyTorch 压缩态后端

```python
# 旧：from tensornetwork import Node, contractor
# 新：使用 devices.simulators.tensor_network.backends.pytorch_backend 的高层接口
from tyxonq.devices.simulators.compressed_state.backends import pytorch_backend as cs

psi = cs.init_product_state(num_qubits=20, dtype="float32")
psi = cs.apply_two_qubit_gate(psi, gate="cx", q0=0, q1=1)
exp = cs.expectation(psi, observable="Z", qubit=0)
```

### devices/
- 目的：承载仿真与硬件执行，统一作业调度与会话管理。
- 功能：
  - `base.py`：`Device` 抽象与能力声明；
  - `simulators/`、`hardware/`：具体执行器；
  - `session.py`、`executor.py`：批量、异步、重试与结果聚合；
  - `simulators/compressed_state/backends/*`：压缩态后端（pytorch/numpy/cupynumeric），可替换/可扩展；
  - `simulators/vendor/cuquantum/*`：供应商模拟器（custatevec/cutensornet）封装；
- 不做：编译/变换（输入应为已编译 IR）。

#### 会话与执行策略
- `session` 负责：批量/异步提交、重试、合并、记录 metadata（向量化/回退/每段耗时）。
- 支持 shot vectors 与分段执行计划；

### devices/hardware 与 cloud API 一体化（新增）

为满足“保留 set_token + 指定 provider/device 即可上真机”的体验并便于多供应商接入（如 TyxonQ 自研、IBM Quantum），我们将硬件适配与云 API 门面一体化：

- 目录设计（与模拟器平行）
  - `cloud/api.py`（对外门面，仅路由）
    - `set_token(token, provider)`、`set_default(provider, device, **opts)`
    - `device("tyxonq.homebrew_s2", shots=...)` 语法糖
    - `list_devices/submit_task/get_task_details`
  - `devices/hardware/`
    - `config.py`：统一配置中心（默认 provider/device、通用运行选项、token/endpoint 注册与校验，支持环境变量覆盖）
    - `session.py`：作业生命周期（JobHandle、轮询/重试、结果规约、错误类别统一）
    - `tyxonq/driver.py`：TyxonQ 云后端适配（HTTP 提交/查询），IR→TQASM/QASM 打包与解析
    - `ibm/driver.py`：IBM Quantum 适配，复用 `compiler/providers/qiskit/dialect.py` 做 IR↔Qiskit 转换与作业提交

- 职责边界
  - `cloud/api.py`：仅做外观层与参数解析；不含业务逻辑
  - `devices/hardware/config.py`：集中读写与缓存默认项与 token；provider 子配置仅声明 schema/校验
  - `devices/hardware/session.py`：统一 Job 接口与轮询策略、错误规约、结果标准化（counts + metadata）
  - `devices/hardware/<vendor>/driver.py`：供应商特定实现，完成“IR→方言→提交→结果解析”的闭环

- 设计要点
  - 不再单独维护 `cloud.registry` 文件；provider 映射采用轻量字典，驻留在 `cloud/api.py` 或 `devices/hardware/__init__.py`
  - 设备命名统一为 `"{provider}.{device_name}"`，同时支持显式 `provider=..., device=...`
  - 结果规约与后处理对齐 `postprocessing/io.py` 与 `postprocessing/metrics.py`

- 最小落地清单（DoD）
  1. `cloud/api.py`：`set_token/set_default/device/list_devices/submit_task/get_task_details`
  2. `devices/hardware/config.py`：默认项、token、endpoint 的集中管理（含环境变量覆盖）
  3. `devices/hardware/session.py`：`JobHandle`、`status`、指数退避轮询、错误规约、`RunResult` 标准结构
  4. `devices/hardware/tyxonq/driver.py`：迁移现有 TyxonQ HTTP 调用流到统一接口
  5. `devices/hardware/ibm/driver.py`：基于 qiskit 方言的最小可用提交流程
  6. 兼容层：`cloud/apis.py` 保留旧函数名薄封装（deprecated），内部路由到 `cloud/api.py`
  7. 冒烟测试：stubbed providers 下 `run/submit/result` 的最小用例与 counts 规约

- 使用示例（对齐现 README 流程）
  - TyxonQ：
    - `tq.api.set_token("<TQ_API_KEY>", provider="tyxonq")`
    - `dev = tq.api.device("tyxonq.homebrew_s2", shots=100)`
    - `tasks = tq.api.submit_task(provider="tyxonq", device=dev["device"], circuit=circuit, shots=100)`
  - Simulator（本地）：
    - `dev = tq.api.device(provider="simulator", id="matrix_product_state")`
    - `tasks = tq.api.submit_task(provider="simulator", device=dev["device"], circuit=c, shots=1000)`

- 迁移提示
  - 旧 `cloud/apis.py` 与现 TyxonQ 真机例子维持可用（经薄封装）；新项目推荐使用 `cloud/apis.py`
  - provider 扩展遵循 `devices/hardware/<vendor>/driver.py` 模式增量接入

#### 数值后端（ArrayBackend）切换计划（新增）

目标：在不破坏既有端到端通路的前提下，将模拟器与编译器逐步迁移到统一的 `numerics/ArrayBackend`（numpy/pytorch/cupynumeric），获得矢量化与加速能力。

阶段性路线：
- 第 1 阶段（已开始）
  - 在模拟器引擎构造器中注入后端句柄：`self.backend = get_backend(backend_name)`；
  - 元数据回传所用后端名称，保证可观测；
  - 保持 numpy 运算路径，端到端测试绿（稳定契约）。

- 第 2 阶段（进行中）
  - 将波函数引擎内部状态从 ndarray 过渡为后端 array；
  - 单比特门、双比特门的应用改用后端的 `matmul/einsum`；
  - 在可用时启用 `vectorize_or_fallback` 做批量/shot 维度的安全矢量化；
  - 为 pytorch（可选安装）增加跳过逻辑的测试，验证一致性。

- 第 3 阶段（计划）
  - density_matrix 引擎语义迁移：基于后端线代原语实现 rho 的演化与观测；
  - compressed_state 定义 backends 协议与最小实现，串联 numerics 后端选择；
  - 在编译器侧将多参数 shift/测量分组作为批量维度，落地矢量化执行与回退路径。

兼容性与风险控制：
- 通过测试切分保障每一步落地不破坏端到端绿；
- 对可选依赖（pytorch/cupynumeric）使用条件跳过与清晰的报错信息；
- 保留 eager 路径作为回退，记录回退原因与成本（后续在 metadata 中增强）。

#### 真实硬件支持与接口不变更承诺
- 接口不变更（签名级别）：对外 `Device` 的核心方法保持不变，特别是：
  - `run(circuit, shots: int | None = None, **kwargs) -> RunResult`
  - `expval(circuit, obs, **kwargs) -> float`
  - 若已有异步接口（如 `submit`/`get_job`/`cancel`），其签名保持不变；
- 目录迁移不影响方法签名：硬件适配器迁移至 `devices/hardware/<vendor>/`，保留同名类与方法；
- `DeviceCapabilities` 用于能力协商，新增字段仅为可选（total=False），不影响既有调用；
- 与 `cloud/` 现有模块的关系：
  - 保持 `cloud/*` 作为低层 API 调用；
  - `devices/hardware/*` 封装 `cloud/*`，对外仍以 `Device` 抽象提供相同接口；
- 配置与凭证：
  - 通过 `config/loader.py` 读取 `[[devices.hardware]]` 配置（名称、区域、tokens、超时/队列策略等），不在代码中硬编码；
  - 支持环境变量覆盖（如 `TYXONQ_IBM_TOKEN`）；
- 验收与测试：
  - 提供真实或沙箱硬件的集成测试（可打上 `hardware` 标记，在受控 CI 或手动流水线执行）；
  - 契约测试确保 shots、批量、期望/采样一致性；
  - 失败容错：网络/配额/队列错误具备重试与明确错误类型；

### postprocessing/
- 目的：误差缓解、读出校正、指标与结果 IO。
- 功能：`mitigation/`、`readout/`、`metrics.py`、`io.py`；
- 不做：编译与设备执行；

### plugins/（可选）
- 目的：第三方扩展接入；默认关闭，仅当需要时开启。

### config/ 与 utils/
- 目的：配置解析/合并与通用工具（日志、并行、缓存、随机数）。


---

## 评审关注点（建议）
- 命名与目录是否清晰传达职责？
- `numerics/` 的范围是否合适（只含三后端）？
- 无 Tape 的变换路径是否足够覆盖现有/预期需求？
- 向量化回退策略的默认行为与可观测性是否合理？
- 插件注册与兼容层的维护成本可接受吗？

---

## 分支与评审策略

- 分支命名：`feat/refactor-<module>`；阶段性里程碑分支 `milestone/<n>`；
- PR 规范：
  - 关联 ADR/Issue；
  - 包含测试与文档更新；
  - 说明对性能与回退比例的影响；
- 评审清单：API 变更、边界遵守、测试覆盖、性能影响、文档更新。

## 风险与缓解

- 风险：向量化导致隐性内存放大 → 缓解：默认 auto 回退与检查、CI 报告；
- 风险：模块边界侵蚀 → 缓解：职责清单与契约测试；
- 风险：目标平台差异导致编译失败 → 缓解：`targets` 与通用 `stages` 分层，能力协商；
- 风险：性能回退 → 缓解：基线与阈值、阶段性优化任务单。

## 启动清单（Kickoff Checklist）

- [ ] 冻结 ADR-001/002 并评审通过
- [ ] 初始化目录与空实现骨架（含 docstring）
- [ ] 配置 CI（测试、覆盖率、性能、回退报告）
- [ ] 建立首批测试（契约 + E2E 化学主线）
- [ ] 里程碑 1 开发开始

---

本文档为评审草案。若方向达成一致，可据此生成 ADR 与接口骨架文件并开始迁移实施。

---

## 深度思考：向量化回退策略的默认行为与可观测性

### 立场 A（工程优先，默认启用回退）
- 现实问题：当前代码在 vmap/JIT 下产生大量警告（如建议使用 `detach().clone()`），在批量维度下会几何级放大内存与时间成本，甚至破坏梯度正确性。
- 策略：默认 `vectorization_policy = "auto"`。
  - 运行前检查（`numerics/checks.safe_for_vectorization`）+ 运行时“警告转异常”围栏。
  - 命中风险则回退 eager，并记录 metadata（原因、次数、耗时占比）。
- 优点：生产可用、稳定性强，避免隐性 O(N·shots) 的代价失控。
- 代价：少数可安全矢量化但被保守规则拦截的场景会损失性能，可通过白名单/禁用检查解决。

### 立场 B（性能优先，默认强制矢量化）
- 策略：默认 `vectorization_policy = "force"`，仅当后端显式报错时才回退。
- 优点：最大限度吃满批量并行性能。
- 代价：对用户代码质量要求高，警告被忽略时更易出现内存暴涨或梯度错误；不适合当前代码基的稳定性现状。

### 折中方案（推荐）
- 框架默认 `auto`；核心模块内部实现“vmap-safe 保证”：
  - 在 `core`/`compiler`/`devices` 自有实现中，禁止隐式 `torch.tensor(t)` 之类构造；统一使用 `detach().clone()` 或后端安全构造；避免 in-place；对别名进行显式控制。
  - 在 `numerics` 提供“张量别名/clone 检查”工具，CI 基准覆盖。
- 对用户代码：
  - 提供快速诊断开关：`TYXONQ_NUMERICS_DEBUG=1` 输出命中规则、建议修复点。
  - 提供白名单：对特定函数/路径标记“已审计安全”以跳过部分检查。

结论：在当前阶段采用工程优先（默认 auto 回退）是合理的；同时我们通过内部 vmap-safe 编码规范与工具，逐步降低回退频率，达到性能与安全的平衡。

---

### 附录 A：领域入口与协议位置

- 协议位置：`core/types.py` 定义 `Problem` 与必要类型，作为公共契约；
- 领域入口：PySCF 风格；从 `app.chem` 导入 `Mole`、`RHF`、`Integrals`、`ActiveSpace`、`to_qubit_hamiltonian`、`to_problem` 等进行组装；
- 无需 `prepare` 函数与 `contracts.py` 文件。

### 附录 B：`compiler/targets` 的职责边界

- 职责：
  - 目标平台（如 Qiskit/IBMQ、Braket）的方言与适配。
  - 在通用 `stages` 与 `pipeline` 基础上，提供目标专属的映射、约束与额外 stage。
- 不做：
  - 通用优化/布局/分解逻辑（应放 `compiler/stages`）。
  - 与 `devices/` 的直接耦合（通过编译产物与能力描述对接）。

### 附录 C: `core和compiler`的关系

#### 简明回答
- 核心关系：**core 提供“稳定的数据模型与语义（IR/operations/measurements）”，compiler 提供“对这些模型的变换与落地（passes/pipeline/providers）”**。前者是“语言”，后者是“编译器与调度器”。

#### 各层职责与依赖方向
- **core/**
  - **ir/**: `Circuit`, `Hamiltonian`, `Pulse` 等“统一中间表示”。所有上层（apps）产出、下游（compiler/devices/postprocessing）消费的唯一载体。
  - **operations/**: `GateSpec` 注册表与梯度元数据（如 parameter-shift 支持）。被 compiler 的分解/重写/梯度阶段、以及模拟器共同使用，保证“门的语义”一致。
  - **measurements/**: 期望值/采样等度量的定义。compiler 的 measurement-rewrite 会基于它生成分组与 `basis_map`，放入 `Circuit.metadata`，供调度/设备复用设置。
  - 方向性：core 对外零依赖；compiler/providers/devices 只读 core，不反向依赖。

- **compiler/**
  - **stages/**: 各种“语义保持/优化/布局/调度/梯度”等 Pass，输入/输出都是 core 的 `Circuit`（必要信息写入 `Circuit.metadata`）。
  - **pipeline.py**: 只是“组合器”，顺序执行 passes，不替代 core。它让“如何编译”与“编什么”解耦。
  - **providers/**: 面向目标生态的“最终产物适配器”
    - `tyxonq/`：原生路径，当前默认把分组与 shot 计划写回 IR/metadata，便于直接交给 `devices/session` 执行。
    - `qiskit/`：将 core IR 降到 `QuantumCircuit` 或 `qasm2`，并保留逻辑-物理映射元数据。

- **devices/**
  - 消费 compiler 写入的 `Circuit.metadata`（如 measurement_groups，shot segments），按计划执行并返回结果；不关心编译细节。

- **postprocessing/**
  - 在结果面做指标与误差缓解等，与 core/IR 定义的测量语义对齐。

#### 为什么 core 必不可少
- **IR 是稳定契约**：apps、compiler、devices、postprocessing 之间通过 IR 解耦。pipeline 只是“怎么变换 IR”的组织者。
- **语义一致性**：`operations` 的门定义与梯度信息，在分解、重写、梯度生成（parameter-shift）与模拟器里“同源”，避免语义漂移。
- **测量优化闭环**：`measurements` 定义→编译阶段产出 `basis_map`/分组→`scheduling` 生成 shot plan→`devices` 复用设置执行→`postprocessing` 基于同一语义解释结果。
- **向下延展**：`pulse` IR 为未来硬件/方言下沉准备承载层，而非直接绑死到某个 provider。

#### 典型流转（最小闭环）
- 应用层用 `CircuitBuilder` 产出 `Circuit`（core/ir）。
- `compiler.api.compile(provider='tyxonq', output='tyxonq')`：
  - pipeline 运行 `rewrite/measurement` → `scheduling/shot_scheduler`，把分组/shot 计划写入 `Circuit.metadata`。
  - 原生 provider 返回“可执行 IR + 元数据”。
- `devices.session.execute_plan` 执行 shot segments，聚合结果。
- `postprocessing` 做指标与误差缓解（与 core/measurements 语义对齐）。

一句话：core 定义“我们说什么”；compiler/pipeline/providers 定义“我们怎么把这件事做成、做高效、做对硬件”；devices 执行；postprocessing 总结。core 不会被 pipeline 取代，pipeline 只是 orchestration。

---

## 最终落地更新（2025-08-24）

本次迭代完成了“cloud API 一体化 + 模拟器/硬件统一驱动 + 应用与数值后端梳理”的主要落地，关键结果如下：

- Cloud API 门面统一：`src/tyxonq/cloud/api.py`
  - 提供 `set_token/set_default/device/list_devices/submit_task/get_task_details/run/result/cancel` 一致入口。
  - 路由至硬件驱动或模拟器驱动（见下）。
- 硬件驱动与会话：`src/tyxonq/devices/hardware/`
  - `config.py` 统一管理 token、默认 provider/device 与 endpoints（允许环境变量覆盖）。
  - `session.py` 统一 `JobHandle`、轮询/重试策略与结果规约。
  - `tyxonq/driver.py`：TyxonQ 云平台 HTTP 驱动完成迁移；`ibm/driver.py` 骨架到位。
- 模拟器提供方（如同真机一致）：`src/tyxonq/devices/simulators/driver.py`
  - Cloud API 将 `provider="simulator"` 路由到本驱动。
  - 在驱动内完成 OpenQASM→IR 转换，然后调用相应引擎（MPS/Statevector/DensityMatrix）。
  - 默认设备已设置为 MPS 引擎（推荐）。
- 目录清理与重命名：
  - 旧 `cloud/abstraction.py`、`cloud/wrapper.py`、`cloud/utils.py` 删除。
  - `templates/` 重命名为 `circuits_library/`，保留 `qaoa_ising_ir`，删除低价值实现。
  - `utils.py` 精简，仅保留高价值方法：`arg_alias/return_partial/append/benchmark/is_sequence/is_number/gpu_memory_share`。
  - 旧 `src/tyxonq/backends/*` 全部删除，统一至 `src/tyxonq/numerics/backends/*` 与 `numerics/api.py` 的 ArrayBackend 协议。
- 可视化：新增 `visualization/dot.py`（基于 IR）；旧 `vis.py` 删除。
- 实验功能：将 QNG 以无依赖的数值实现落地于 `compiler/stages/gradients/qng.py`；旧 `experimental.py` 删除。
- 化学应用迁移：
  - 将 `origin/chem` 分支的 `src/tyxonq/chem` 迁移为 `src/tyxonq/applications/chem`（含 `static/`、`dynamic/`、`utils/`）。
  - 顶层包不新增 `tyxonq/chem/` 目录，转而在 `tyxonq/__init__.py` 中通过 `sys.modules` 提供别名：`tyxonq.chem` → `tyxonq.applications.chem`，同时映射常用子模块路径（例如 `tyxonq.chem.static.uccsd`）。
- 冒烟测试：新增 `tests/test_cloud_api_smoke.py`，覆盖模拟器 provider 的 list/submit/run/result/cancel。

以上变更均已在当前代码树落地并通过冒烟测试验证。

---

## 化学应用（applications/chem）迁移与兼容策略（新增）

- 源代码迁移：将 `origin/chem` 分支下的 `src/tyxonq/chem/*` 完整迁移至 `src/tyxonq/applications/chem/*`，保留 `static/`、`dynamic/`、`utils/` 子模块。
- 兼容导入：不新增独立的 `tyxonq/chem` 目录；在 `tyxonq/__init__.py` 中注册别名：
  - `tyxonq.chem` → `tyxonq.applications.chem`
  - 同时映射 `constants/molecule/dynamic/static/utils` 等常见子模块，保证 `import tyxonq.chem.static.uccsd` 这类历史用法继续可用。
- 后续规划：逐步将 `static/` 中的 VQE 类算子（UCC/HEA 等）向 IR 构建与编译阶段收敛；将 `utils/backend.py` 中与数值后端相关的逻辑迁往 `numerics/` 或以 ArrayBackend 方式接入。

