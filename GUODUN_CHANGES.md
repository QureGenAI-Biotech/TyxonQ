# TyxonQ 国盾量子云接入说明

TyxonQ 新增 `guodun` provider，通过官方 `cqlib` SDK 将 TyxonQ 线路编译为
QCIS，并连接国盾平台完成任务提交、查询、取消和结果归一化。

本接入只覆盖国盾官方支持的门和原生门脉冲；FSim 属于独立实验标定，
不在 `guodun` provider 的正式能力范围内。

---

## 新增文件

### `src/tyxonq/compiler/compile_engine/guodun/compiler.py`

门线路编译器，负责：

| 功能 | 说明 |
|---|---|
| 显式测量检查 | 必须调用 `add_measure(...)` 或 `measure_z(...)` |
| Qiskit 降门 | 将 TyxonQ 门降到国盾门级 QCIS |
| 物理映射 | `physical_qubits[i]` 表示逻辑比特 `i` 对应的物理比特 |
| QCIS 校验 | 使用 `cqlib==1.3.11` 的 `Circuit.load()` 做离线解析 |
| 元数据 | 返回门统计和逻辑到物理映射 |

最终门级 QCIS 只包含 `X/H/RZ/CZ/M`。`iswap/rxx/ryy/rzz` 等 TyxonQ
已有门会先经 Qiskit 降门，不直接作为平台指令提交。

### `src/tyxonq/compiler/compile_engine/guodun/pulse.py`

国盾官方原生门脉冲的生成与安全校验：

- 原生门模板：`X2P/X2M/Y2P/Y2M/CZ`。
- pulse-QCIS：`PXY/PZ/PZ0/G/I/B/M/RZ`。
- 原生门参数来自提交时刻的 `GetPulse` 当前标定。
- 校验频率、相位、时长、波形参数以及每个数值采样点。
- 不提供通用 TyxonQ `PulseProgram` 到国盾 QCIS 的猜测性转换。

### `src/tyxonq/devices/hardware/guodun/driver.py`

国盾平台驱动，内部使用 `cqlib.GuoDunPlatform`。

| 组件 | 说明 |
|---|---|
| `_resolve_token()` | 按固定优先级取得国盾密钥 |
| `_create_platform()` | 关闭 SDK 隐式重试后执行一次显式登录 |
| `run()` | 校验当前配置后调用一次 `submit_job()` |
| `get_task_details()` | 只查询已有 query ID，并归一化状态和 counts |
| `remove_task()` | 取消已有任务，不创建新任务 |
| `list_devices()` | 从 SDK 公共查询结果动态提取设备代码 |
| `open_pulse_context()` | 登录 `gd_qc1` 并读取当前脉冲标定，不提交任务 |

提交固定使用 `is_verify=True`。驱动不会自动重提；批量线路必须使用相同
shots。`gd_sim1` 和 `gd_qc1` 提交前会校验禁用比特、禁用耦合器和 CZ 边。

### `examples/run_circuit_on_guodun.py`

国盾门线路最小示例：

- 默认只在本地生成 QCIS，不登录平台。
- `--circuit bell` 自动使用两比特默认映射。
- `--circuit x` 自动使用单比特默认映射。
- 只有显式添加 `--run-online` 才提交。
- 提交成功后打印 query ID 并立即停止，不自动查询或创建第二个任务。

### 测试文件

- `tests_core_module/test_guodun_compiler.py`
- `tests_core_module/test_guodun_driver.py`
- `tests_core_module/test_guodun_example.py`
- `tests_core_module/test_guodun_pulse.py`

覆盖编译、物理映射、拓扑、密钥优先级、单次提交、结果解析、取消、
官方原生门脉冲、逐点安全检查和 example 行为。

---

## 修改文件

### `src/tyxonq/compiler/api.py`

注册 `compile_engine="guodun"`，输出格式为 `qcis`。

### `src/tyxonq/core/ir/circuit.py`

当 `provider="guodun"` 时自动使用国盾编译器，并把 `physical_qubits`
传入编译阶段。

### `src/tyxonq/devices/base.py`

- 在 `resolve_driver()` 中注册 `guodun`。
- 正确消费显式 `token=`。
- 非 TyxonQ provider 不再回退到全局 `TYXONQ_API_KEY`。
- 轮询遇到 `failed/error/cancelled` 时立即返回。

### `pyproject.toml`

新增可选依赖：

```toml
[project.optional-dependencies]
guodun = [
    "cqlib==1.3.11",
]
```

`cqlib` 采用懒加载，未使用国盾 provider 时不会影响其他 provider。

### 文档

- `docs/guodun_provider.md`：安装、密钥、门线路、脉冲和安全边界。
- `README.md`：新增国盾 provider 入口。
- `CHANGELOG.md`：记录国盾 provider 新增能力。

---

## 架构

门线路：

```text
TyxonQ Circuit
     |  Qiskit 降门 + 显式物理映射
     v
X/H/RZ/CZ/M QCIS
     |  cqlib Circuit.load() 离线解析
     v
GuoDunPlatform.submit_job(is_verify=True)
     |
     v
query ID -> query_experiment() -> TyxonQ 统一结果
```

官方原生门脉冲：

```text
GuoDunPlatform + 当前设备配置
     |  GetPulse 读取当前标定
     v
官方原生门 pulse-QCIS
     |  语法、拓扑和逐采样安全检查
     v
单次 submit_job(is_verify=True)
```

---

## 安装

推荐在 TyxonQ 所在 Python 环境安装：

```bash
pip install 'tyxonq[guodun]'
```

当前兼容基线固定为 `cqlib==1.3.11`。如果 `cqlib` 安装在另一个 Conda
环境中，开发阶段可通过 `PYTHONPATH` 指向其 `site-packages`；正式环境建议
把 TyxonQ 和 `cqlib` 安装在同一环境。

---

## 密钥

国盾密钥按以下顺序解析：

1. 调用时显式 `token=...`。
2. `tq.set_token(..., provider="guodun")`。
3. 环境变量 `TYXONQ_GUODUN_TOKEN`。

不回退到 `TYXONQ_API_KEY`，也不应把真实 JWT 写入源码或 Git。

终端安全输入方式：

```bash
read -s "TYXONQ_GUODUN_TOKEN?请输入国盾 JWT: "
export TYXONQ_GUODUN_TOKEN
echo
```

---

## 使用方式

本地生成 QCIS，不连接平台：

```bash
python examples/run_circuit_on_guodun.py \
  --device gd_qc1 \
  --circuit x \
  --physical-qubits 0 \
  --shots 100
```

确认 QCIS 后，显式提交一个真机任务：

```bash
python examples/run_circuit_on_guodun.py \
  --run-online \
  --device gd_qc1 \
  --circuit x \
  --physical-qubits 0 \
  --shots 100
```

排队或校准期间只重复查询原 query ID，不能重复执行提交命令。

---

## 结果和状态

- `runStatus=2` 映射为 `completed`。
- `runStatus=3` 映射为 `failed`。
- 其他状态作为非终态保留。
- counts 按 `resultStatus[1:]` 统计，不颠倒 bitstring。
- 物理测量顺序、`probability` 和完整平台响应保存在 `result_meta`。

---

## 验证情况

离线和 mock 验证：

- 国盾及相关 provider 回归测试：`80 passed`。
- 新增国盾源码、测试和 example 通过 Ruff。
- wheel 构建成功，元数据包含 `cqlib==1.3.11` 可选依赖。
- 未发现 JWT、私钥、平台 PAT、真实 query ID 或硬编码密码。

真机验证：

- 设备：`gd_qc1`。
- 线路：`X Q0`、`M Q0`。
- shots：100。
- 平台返回 `runStatus=2`，TyxonQ 归一化状态为 `completed`。
- counts 为 `1: 95`、`0: 5`。
- 提交和查询严格分离，没有创建第二个任务。

该结果只证明最小门线路的真机调用链已经打通，不能作为正式门保真度结论。
官方原生门脉冲目前通过离线和 mock 安全测试，尚未宣称真机端到端验收完成。

---

## 范围边界

- 只接入国盾官方支持的门和原生门脉冲。
- 不新增 TyxonQ `Circuit.fsim()` API。
- 不接入实验中的 FSim 标定线路。
- 不自动添加测量、重提任务、扩大 shots 或并发提交。
- 脉冲真机提交前必须重新读取当前标定，并通过全部逐点安全检查。
