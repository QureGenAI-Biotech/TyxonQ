# 国盾量子云 Provider

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
