# LQCloud 接入使用指南

TyxonQ 的 `lqcloud` provider 通过官方 `lqcloud` SDK 提交门线路，并统一完成设备发现、任务查询、取消和 counts 结果解析。当前实现只包含运行门线路所需的最小功能。

## 1. 安装

```bash
pip install -e '.[lqcloud]'
```

当前兼容基线固定为 `lqcloud==0.4.2`。SDK 仅在使用 `provider="lqcloud"` 时延迟导入，不影响其他 provider。

## 2. 配置 API key

不要把真实 API key 写入源码或提交到 Git。可以选择以下任一方式：

```python
import tyxonq as tq

tq.set_token("你的 LQCloud API key", provider="lqcloud")
```

使用时请将占位文字替换为真实 API key，不要原样复制。

或在终端设置环境变量：

```bash
export TYXONQ_LQCLOUD_API_KEY='你的 LQCloud API key'
# 官方 SDK 的环境变量 LQCLOUD_API_KEY 也可以使用
```

优先级依次为：运行时显式 `api_key=` / `token=`、`tq.set_token(...)`、`TYXONQ_LQCLOUD_API_KEY`、`LQCLOUD_API_KEY`。

## 3. 查询设备

设备列表来自平台实时接口，不在 TyxonQ 中保存静态名称：

```python
import tyxonq as tq

print(tq.api.list_devices(provider="lqcloud"))
# ['simulator::...', 'lqcloud::<当前账号可见设备>', ...]
```

没有配置 API key 或平台暂时不可达时，硬件部分返回空列表。

## 4. 运行门线路

完整的线路构造和提交代码见
[`examples/run_circuit_on_lqcloud.py`](../examples/run_circuit_on_lqcloud.py)。

`physical_qubits[i]` 表示逻辑比特 `i` 映射到列表中第 `i` 个物理比特。该参数会传给官方 SDK 的 `initial_layout`；如果省略，则由 LQCloud 使用默认布局。

线路必须显式调用 `add_measure(...)` 或 `measure_z(...)`。TyxonQ 不会自动添加测量，但会在末尾测量块前补充 LQCloud 调度所需的 barrier。

## 5. 支持范围

当前支持以下 TyxonQ 门：

```text
h, x, y, z, s, sdg, t, tdg,
rx, ry, rz,
cx, cy, cz, swap, iswap,
reset, barrier, measure
```

当前不支持 `rxx`、`ryy`、`rzz`、任意矩阵门、噪声操作、脉冲、waveform、脚本任务、批量线路、动态解耦和读出修正。遇到不支持的操作会在本地报错，不会提交任务。

## 6. 结果和状态

`Circuit.run()` 返回 TyxonQ 统一结果。LQCloud 的 job 元数据保存在第二层 `result_meta`：

```python
{
    "result": {"0000": 17, "1110": 18, ...},
    "uni_status": "completed",
    "error": "",
    "result_meta": {
        "result": {...},
        "uni_status": "completed",
        "error": "",
        "result_meta": {
            "job_id": "task_...",
            "device": "MQ02",
            "shots": 64,
            "probability": {...},
            "result_format": "counts",
            "raw": {...},
        },
    },
}
```

例如读取 job ID：

```python
job_id = results[0]["result_meta"]["result_meta"]["job_id"]
```

状态统一为 `queued`、`running`、`completed`、`failed`、`cancelled`、`error` 或 `unknown`。查询只访问原 Job，不会重新提交；进入终态后会停止轮询。

## 7. 使用 example 运行真机

文件 `examples/run_circuit_on_lqcloud.py` 提供四比特真机示例，包含参数旋转、两比特纠缠门、物理比特映射和测量。默认只做离线转换检查。

### 7.1 先做离线检查

```bash
PYTHONPATH=src python examples/run_circuit_on_lqcloud.py \
  --physical-qubits 0,1,2,3
```

### 7.2 提交一个真机任务

先按第 2、3 节配置 API key 并查询当前可用设备。设备状态和拓扑可能变化，提交前应确认 `--physical-qubits` 符合目标设备的当前拓扑。只有显式增加 `--run-online` 才会提交：

```bash
PYTHONPATH=src python examples/run_circuit_on_lqcloud.py \
  --device MQ02 \
  --physical-qubits 0,1,2,3 \
  --shots 64 \
  --run-online
```

如 `MQ02` 当前不可用，请将它替换为账号可见的其他设备名称。

### 7.3 已验证的真机结果

以下为一次 `MQ02` 真机运行结果：

```text
逻辑到物理映射: [0, 1, 2, 3]
LQCloud 指令统计: {'h': 2, 'ry': 1, 'rz': 1, 'cx': 2, 'cz': 1, 'barrier': 1, 'measure': 4}
即将提交一个任务: device=MQ02, shots=64
任务状态: completed
counts: {'1111': 10, '1110': 12, '0001': 16, '1000': 2, '1011': 4, '0000': 15, '1100': 2, '1010': 1, '0011': 2}
总 shots: 64
```

counts 合计为 64，与请求的 shots 一致。

## 8. 运行说明

- 驱动每次只调用一次官方 `backend.run()`，TyxonQ 不自动重提。
- example 默认离线；不带 `--run-online` 时不会创建 provider 或访问网络。
