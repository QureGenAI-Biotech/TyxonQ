# TyxonQ RiverONE QML 适配器说明

TyxonQ 目前可以读取 RiverONE 训练得到的单条 `TQVQC` 参数，将给定振幅输入转换为等价的量子线路，并生成 X、Y、Z 三种测量基对应的 OpenQASM 2。生成结果既可在 TyxonQ 本地模拟器中执行，也可复用现有 `provider="tyxonq"` 提交到 `homebrew_s2` 或 `homebrew_s3`。

本文说明的是当前 RiverONE 适配功能本身，不是 PR 修改记录。

---

## 功能范围

当前适配器覆盖 RiverONE 原始 `paramgen/vqc_models.py` 中一条 `TQVQC` 的量子部分：

1. 从 checkpoint 读取指定 VQC 的 RX、RY、RZ 训练参数。
2. 对外部输入的振幅向量补零并归一化。
3. 复现 RiverONE 的 amplitude encoding 和每两层一次的数据重新上传语义。
4. 添加每层的 RX/RY/RZ 与环形 CNOT。
5. 分别生成 X、Y、Z 测量线路。
6. 将线路分解为服务器可接收的 OpenQASM 2。
7. 通过示例在本地模拟，或作为一个三线路批量任务提交到 TyxonQ 真机。

当前适配器不包含 RiverONE 的经典 `WeightEncoder`、`BlockHyper`、完整 `VQCWeightGenerator`、训练和蒸馏流程。因此它是 RiverONE 单条量子 VQC 的执行适配层，不等同于完整 RiverONE 网络推理。

---

## 主要文件

| 文件 | 作用 |
|---|---|
| `src/tyxonq/applications/qml/riverone.py` | checkpoint 读取、输入校验、线路构造和 QASM2 生成 |
| `src/tyxonq/applications/qml/__init__.py` | 导出公开 RiverONE QML 接口 |
| `examples/riverone_qml.py` | 本地模拟与 TyxonQ 真机批量提交示例 |
| `tests_core_module/test_riverone_qml_adapter.py` | checkpoint、re-upload、X/Y/Z 语义和 8 比特 QASM 测试 |
| `tests_examples/test_riverone_qml_example.py` | 示例参数、本地执行和真机提交行为测试 |

公开接口只有以下三个：

```python
from tyxonq.applications.qml import (
    RiverONEVQCSpec,
    load_riverone_vqc,
    riverone_to_qasm2,
)
```

---

## 整体流程

```text
RiverONE checkpoint (.pt)
        |
        | load_riverone_vqc(..., vqc_index=...)
        v
RiverONEVQCSpec
  - 量子比特数
  - 变分层数
  - 每层、每个比特的 RX/RY/RZ 角度
        |
        | riverone_to_qasm2(spec, amplitudes)
        v
振幅编码 + 最后一次 re-upload 之后的变分层
        |
        +-------- X 换基 --------> X 基 QASM2
        +-------- Y 换基 --------> Y 基 QASM2
        +-------- Z 直读 --------> Z 基 QASM2
                                      |
                 +--------------------+-------------------+
                 |                                        |
                 v                                        v
       TyxonQ 本地 simulator                  TyxonQ 现有真机 provider
                                              homebrew_s2 / homebrew_s3
```

---

## Checkpoint 读取规则

`load_riverone_vqc()` 使用 CPU 读取 PyTorch checkpoint。checkpoint 顶层必须是字典，并包含字典字段 `state`：

```python
{
    "state": {
        "vqcs.0.variational.l0_w0.rx.params": ...,
        "vqcs.0.variational.l0_w0.ry.params": ...,
        "vqcs.0.variational.l0_w0.rz.params": ...,
        # 其余 layer、wire 和 gate 参数
    }
}
```

参数键的格式为：

```text
vqcs.<vqc_index>.variational.l<layer>_w<wire>.<rx|ry|rz>.params
```

适配器会根据键名推断量子比特数和层数，并检查每一层、每一个比特是否都有完整的 RX、RY、RZ 标量参数。缺少参数、参数不是标量、索引不存在或数值包含 `NaN/Inf` 时会直接报错。

读取 checkpoint 需要安装 PyTorch，但不需要安装或导入 TorchQuantum。

---

## 振幅输入与状态制备

`riverone_to_qasm2()` 接收一维实数或复数振幅：

```python
qasm_by_basis = riverone_to_qasm2(spec, amplitudes)
```

输入规则如下：

| 项目 | 规则 |
|---|---|
| 维度 | 必须是一维数组 |
| 元素数量 | 不能为空，且不能超过 `2 ** n_qubits` |
| 数值 | 必须全部有限，不能是全零向量 |
| 长度不足 | 在末尾补零到 `2 ** n_qubits` |
| 归一化 | 自动除以二范数 |
| 比特序 | 从 RiverONE 的 q0-first 状态轴转换为 Qiskit 小端序 |

随后由 Qiskit `StatePreparation` 构造状态制备线路，并进一步分解为基础门。RiverONE 原始 `WeightEncoder` 产生实数振幅；适配器也允许直接传入复数振幅，但这属于通用接口能力。

---

## 变分层与数据重新上传

每个变分层按以下顺序执行：

1. 对每个量子比特依次执行 RX、RY、RZ。
2. 执行相邻 CNOT：`0 -> 1 -> ... -> n-1`。
3. 执行末尾回到起点的 CNOT：`n-1 -> 0`。

RiverONE 每两层重新上传一次相同的振幅状态。重新上传会覆盖此前的完整量子态，因此最后一次状态覆盖之前的门不会影响最终测量。适配器只导出最后一次 re-upload 之后仍然有效的变分层。

以 RiverONE 常用的 8 比特、6 层配置为例：

- checkpoint 中仍会读取全部 6 层，共 `6 × 8 × 3 = 144` 个旋转角；
- 最后一次 re-upload 发生在第 4 层之前；
- 转译前的高层线路保留第 4、5 层，共 48 个参数旋转门和 16 个环形 CNOT，再加上状态制备、换基与测量线路。

这里的层编号从 0 开始。Qiskit 转译后，RY 和状态制备还会继续分解，因此最终 QASM 中的实际基础门数量会更大。

---

## X/Y/Z 测量

`riverone_to_qasm2()` 返回一个固定包含三个键的字典：

```python
{
    "X": "OPENQASM 2.0; ...",
    "Y": "OPENQASM 2.0; ...",
    "Z": "OPENQASM 2.0; ...",
}
```

三份线路的状态制备和变分层相同，仅测量前处理不同：

| 测量基 | 测量前操作 |
|---|---|
| X | 每个比特执行 H |
| Y | 每个比特依次执行 Sdg、H |
| Z | 不换基，直接测量 |

所有线路最后仍使用计算基测量。转译后的允许操作为 `cx`、`h`、`rz`、`rx`、`cz`、`measure` 和 `barrier`；如果还有未分解操作，适配器会拒绝输出。

对于 8 比特 VQC，本地示例会从三组 counts 计算 24 个单比特期望值，即每个比特各一个 X、Y、Z 期望值。这与原始 `TQVQC` 的量子输出维度一致，但不包含后续 `BlockHyper` 生成的经典网络权重。

---

## 安装

在 TyxonQ 仓库根目录使用正确的 Python 环境安装当前源码：

```bash
python -m pip install -e .
```

当前项目要求 Python `>=3.10,<3.13`，RiverONE 适配涉及的主要依赖为 NumPy、PyTorch 和 Qiskit。

建议先确认解释器确实来自安装 TyxonQ 的同一个环境：

```bash
which python
python -c "import sys, tyxonq; print(sys.executable); print(tyxonq.__file__)"
```

---

## 直接调用公开接口

```python
import numpy as np

from tyxonq.applications.qml import (
    load_riverone_vqc,
    riverone_to_qasm2,
)

# 读取 checkpoint 中第 0 条 TQVQC。
spec = load_riverone_vqc("path/to/checkpoint.pt", vqc_index=0)

# 输入必须是一维振幅；这里也可以使用 np.load(...) 读取 .npy。
amplitudes = np.arange(1, (1 << spec.n_qubits) + 1, dtype=float)

# 返回 X、Y、Z 三份完整 OpenQASM 2 字符串。
qasm_by_basis = riverone_to_qasm2(spec, amplitudes)
print(qasm_by_basis["Z"])
```

该接口只负责加载和生成 QASM，不会自动提交任务。

---

## 运行本地示例

最小命令如下：

```bash
python examples/riverone_qml.py \
  --checkpoint path/to/checkpoint.pt \
  --vqc-index 0 \
  --device simulator \
  --shots 1024
```

如需使用自己的振幅，先保存为一维 `.npy` 文件：

```bash
python examples/riverone_qml.py \
  --checkpoint path/to/checkpoint.pt \
  --vqc-index 0 \
  --amplitudes path/to/amplitudes.npy \
  --device simulator \
  --shots 1024
```

未提供 `--amplitudes` 时，示例会使用确定性的 `[1, 2, ..., 2^n]` 演示向量。这个默认向量只用于检查流程，不是 RiverONE `WeightEncoder` 的真实输出。

本地模式会分别提交 X、Y、Z 三个模拟任务，取得 counts 后打印每个线编号对应的期望值。当前 TyxonQ 模拟器返回 q0-first bitstring，例如两比特线路中只有 `q[0] = 1` 时返回 `"10"`，示例会据此将第一位解释为 `q0`。

---

## 提交到 TyxonQ 真机

先设置 API key：

```bash
export TYXONQ_API_KEY="your_api_key"
```

然后选择已有设备：

```bash
python examples/riverone_qml.py \
  --checkpoint path/to/checkpoint.pt \
  --vqc-index 0 \
  --amplitudes path/to/amplitudes.npy \
  --device homebrew_s2 \
  --shots 1024
```

也可以把设备改为 `homebrew_s3`。真机模式会把 X、Y、Z 三份 QASM 作为一个批量任务提交，并打印 `task_id`。示例不会在提交后自动轮询结果。

> 选择 `homebrew_s2` 或 `homebrew_s3` 会产生真实的远程任务；仅生成 QASM 或检查输入时，请直接调用公开接口，不要运行真机命令。

---

## 当前能力边界

1. **不是完整 RiverONE 推理。** 当前只适配一条 `TQVQC` 的量子线路；经典输入编码和输出权重生成需要由 RiverONE 侧另行完成。
2. **不负责训练。** checkpoint 必须已经包含训练好的 `vqcs.<index>.variational...` 参数。
3. **振幅由调用方提供。** 示例默认向量仅用于演示，真实工作流应传入与 RiverONE 经典编码阶段一致的振幅。
4. **状态制备开销较大。** 振幅维度随量子比特数按 `2 ** n_qubits` 增长，分解后的 QASM 可能很长，不能把 8 比特示例的规模直接外推到更大量子比特。
5. **不处理底层硬件标定。** 适配器生成逻辑 QASM，不负责脉冲、IQ、物理比特映射或设备校准。
6. **真机示例只提交不轮询。** 返回结果需要使用 TyxonQ 现有任务查询接口继续获取。

---

## 当前验证结果

在当前 TyxonQ checkout 和 `tyxonq-main` Python 环境中已完成以下离线验证：

- RiverONE adapter、example 和 TyxonQ driver 相关测试：`21 passed`；
- 相关文件 Ruff 检查：通过；
- 8 比特、6 层 X/Y/Z QASM2：可由 Qiskit 重新解析；
- X/Y/Z 测量语义：通过 statevector 对照；
- 本地 simulator 的 q0-first counts 位序：通过两比特最小线路实测。

以上验证没有提交任何真机任务，也不能替代特定设备的队列、映射和硬件标定检查。
