# 国盾接入使用指引（适配 cqlib 1.3.11）

TyxonQ 的 `guodun` provider 通过官方 `cqlib` SDK 提交 QCIS 门线路和
pulse-QCIS。当前兼容基线为 `cqlib==1.3.11`。

## 安装

```bash
pip install 'tyxonq[guodun]'
```

`cqlib` 只在使用国盾编译器或驱动时懒加载，不影响其他 provider。

## 配置密钥

密钥按以下顺序解析：

1. 调用时显式传入 `token=...`
2. `tq.set_token(..., provider="guodun")`
3. 环境变量 `TYXONQ_GUODUN_TOKEN`

```python
import tyxonq as tq

tq.set_token("你的 SDK 密钥", provider="guodun", device="gd_qc1")
```

不要把真实密钥写入源码或提交到 Git。

## 门线路

国盾编译器将 TyxonQ 线路降到平台使用的 `X/H/RZ/CZ/M` QCIS。线路必须
显式测量；`physical_qubits[i]` 表示逻辑比特 `i` 对应的物理比特。

```python
import tyxonq as tq

circuit = tq.Circuit(2)
circuit.h(0).cx(0, 1).add_measure(0, 1)

results = circuit.run(
    provider="guodun",
    device="gd_qc1",
    shots=100,
    physical_qubits=[0, 1],
    wait_async_result=True,
)

print(results[0]["result"])
```

## 5 比特完整示例

仓库中的 `examples/run_guodun_variational.py` 构造了一条非平凡线路：5 个
逻辑比特、两层参数旋转和 8 个相邻 `RZZ` 纠缠，最后显式测量全部比特。
国盾编译器会将它统一降到官方门级 QCIS：`X/H/RZ/CZ/M`。

先根据国盾控制台或当前设备配置选择一条连续可用的 5 比特路径。下面的
`0,6,12,7,1` 是 2026-08-07 实际运行对应的映射，仅用于说明；设备重新标定后
必须重新确认。

只做本地编译，不登录、不提交：

```bash
PYTHONPATH=src python examples/run_guodun_variational.py \
  --physical-qubits 0,6,12,7,1
```

终端中安全读取密钥，再提交一个任务并等待结果：

```bash
read -s "TYXONQ_GUODUN_TOKEN?请输入国盾 SDK key: "
export TYXONQ_GUODUN_TOKEN

PYTHONPATH=src python examples/run_guodun_variational.py \
  --device gd_qc1 \
  --physical-qubits 0,6,12,7,1 \
  --shots 256 \
  --run-online
```

在线运行仍然通过 TyxonQ 的统一接口完成：

```python
results = circuit.device(
    provider="guodun",
    device="gd_qc1",
).run(
    shots=256,
    physical_qubits=[0, 6, 12, 7, 1],
    wait_async_result=True,
)
```

驱动会在提交前重新下载当前配置，校验禁用比特、禁用耦合器和每一条 `CZ`
边。任一检查失败都会停止，不会自动重提。

### 实际运行结果

以下文字记录来自一次 `gd_qc1` 真机运行。动态实验名和
query ID 已隐藏，SDK key 未写入输出：

```text
逻辑到物理映射: {0: 0, 1: 6, 2: 12, 3: 7, 4: 1}
QCIS 指令统计: {'H': 57, 'RZ': 33, 'CZ': 16, 'M': 5}
提交一个任务：device=gd_qc1, shots=256, exp_name=<自动生成>
query ID: <已隐藏>
任务状态: completed
counts: {'10000': 10, '01100': 2, '11001': 7, '11101': 16, '00101': 23, '10101': 27, '01001': 6, '10111': 16, '10100': 20, '11100': 13, '00110': 7, '00111': 15, '11110': 4, '00100': 12, '10010': 4, '10110': 8, '11111': 7, '11000': 6, '10001': 7, '00001': 5, '10011': 5, '00010': 3, '00000': 4, '01101': 9, '01111': 5, '01010': 3, '11011': 4, '11010': 3, '01110': 1, '01011': 3, '00011': 1}
总 shots: 256
```

该线路编译为 111 行 QCIS，完整结果包含 31 种 bitstring，所有 counts 之和
严格等于 256。

运行完成后，主要关注三个字段：

- `任务状态`：`completed` 表示平台正常完成。
- `counts`：键是按平台测量顺序返回的 bitstring，值是出现次数。
- `总 shots`：应当等于提交时设置的 shots；不相等时不要继续使用该结果。

仅本地查看 QCIS，不连接平台：

```bash
python examples/run_circuit_on_guodun.py \
  --circuit bell \
  --physical-qubits 0,1
```

## 官方原生门脉冲

原生门脉冲参数必须来自提交时刻的 `GetPulse` 标定。当前接口支持官方
模板 `X2P/X2M/Y2P/Y2M/CZ`，并对 `PXY/PZ/PZ0/G/I/B/M/RZ` QCIS
执行语法和逐采样安全检查。

```python
from tyxonq.compiler.compile_engine.guodun import compile_native_gate_pulse
from tyxonq.devices.hardware.guodun.driver import open_pulse_context

context = open_pulse_context("gd_qc1")
compiled = compile_native_gate_pulse(
    context.get_pulse,
    "X2P",
    "Q0",
    measure_qubits=["Q0"],
)
print(compiled["qcis"])
```

`open_pulse_context()` 会登录并下载当前标定，但不会提交实验或创建波形。
真正提交 pulse-QCIS 时，驱动会再次读取当前标定并逐采样校验；任一检查失败
都会在 `submit_job()` 之前停止。

## 波形与任务安全

- `create_waveform()` 只创建一个波形任务。
- `get_waveform()` 只查询已有 waveform ID。
- 实验查询和取消只操作已有 query ID。
- 驱动不自动重提、并发提交或扩大 shots。
- `runStatus=2` 表示完成，`runStatus=3` 表示失败。
- counts 按平台 `resultStatus` 和物理测量顺序解析，不反转 bitstring。

## 范围边界

- 只接入官方支持的门、pulse-QCIS 指令和原生门模板。
- 不提供通用 `PulseProgram` 到 QCIS 的猜测性波形转换。
- FSim 属于独立实验标定，不是 `guodun` provider 的正式能力。
- pulse-QCIS 当前只允许提交到 `gd_qc1`。
