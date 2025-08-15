# TyxonQ Pulse 接口使用文档

## 目录

- [概述](#概述)
- [TQASM 0.2 语法规范](#tqasm-02-语法规范)
- [核心组件](#核心组件)
- [波形参数总览表](#波形参数总览表)
- [波形参数详细说明](#波形参数详细说明)
- [使用方法](#使用方法)
- [TQASM 输出格式](#tqasm-输出格式)
- [高级功能](#高级功能)
- [最佳实践](#最佳实践)
- [故障排除](#故障排除)
- [实际应用示例](#实际应用示例)
- [总结](#总结)

---

## 概述

TyxonQ 提供了强大的脉冲级控制接口，允许用户直接操作量子比特的脉冲信号，实现精确的量子控制。通过 Pulse 接口，您可以：

- 定义自定义的脉冲波形
- 创建量子比特校准程序
- 实现高级量子控制算法
- 生成 TQASM 0.2 格式的脉冲级电路

### 支持的波形类型

目前支持以下四种主要波形类型：
- **cosine_drag** - 余弦DRAG波形，用于抑制泄漏态跃迁
- **flattop** - 平顶波形，适用于量子态制备
- **gaussian** - 高斯波形，提供平滑的脉冲过渡
- **sine** - 正弦波形，用于周期性振荡实验

详细的参数定义和数学表达式请参考下文的波形参数详细说明部分。

## TQASM 0.2 语法规范

### 语法定义

TQASM 0.2 使用巴科斯-诺尔范式(BNF)定义语法结构：

```
<pulse> ::= <defcal>

<defcal> ::= "defcal" <id> <idlist> { <calgrammar> }

<calgrammar> ::= <calstatement>
               | <calgrammar> <calstatement>

<calstatement> ::= <framedecl>
                | <waveformplay>

<framedecl> ::= "frame" <id> "=" "newframe" (<idlist>);

<waveformplay> ::= "play" (<id>, <waveform>);

<waveform> ::= <id> (<explist>)
```

### 关键字说明

| 关键字 | 功能描述 | 语法格式 |
|--------|----------|----------|
| `defcal` | 定义自定义参数化波形量子门 | `defcal <门名称> <参数列表> { <校准语句> }` |
| `frame` | 声明一个变量为帧类型 | `frame <帧名称> = newframe(<量子比特>);` |
| `newframe` | 在目标量子比特上创建新帧，用于承载波形 | `newframe(<量子比特标识>)` |
| `play` | 在指定帧上播放波形 | `play(<帧名称>, <波形函数>(<参数>));` |

### 支持的波形类型

目前支持的波形函数包括：
- `cosine_drag(duration, amp, phase, alpha)` - 余弦DRAG波形
- `flattop(duration, amp, width)` - 平顶波形
- `gaussian(duration, amp, sigma, angle)` - 高斯波形
- `sin(duration, amp, phase, freq, angle)` - 正弦波形

### 完整示例

下面是一个完整的TQASM 0.2代码示例，展示了如何定义和使用参数化波形：

```tqasm
TQASM 0.2;
QREG q[1];

defcal hello_world a {
  frame drive_frame = newframe(a);
  play(drive_frame, cosine_drag(50, 0.2, 0.0, 0.0));
}

hello_world q[0];
```

### 代码解析

1. **TQASM 0.2;** - 声明使用TQASM 0.2版本
2. **QREG q[1];** - 定义1个量子比特的寄存器
3. **defcal hello_world a { ... }** - 定义名为"hello_world"的校准程序，参数为"a"
4. **frame drive_frame = newframe(a);** - 在量子比特"a"上创建名为"drive_frame"的帧
5. **play(drive_frame, cosine_drag(50, 0.2, 0.0, 0.0));** - 在帧上播放余弦DRAG波形
6. **hello_world q[0];** - 调用校准程序，作用于量子比特q[0]

### 波形参数说明

`cosine_drag(50, 0.2, 0.0, 0.0)` 中的参数含义：
- `50` - 脉冲持续时间（采样周期）
- `0.2` - 波形振幅
- `0.0` - 相位角（弧度）
- `0.0` - DRAG系数

## 波形参数总览表

下表提供了所有支持波形的快速参考，包括参数格式和主要应用场景：

| 序号 | 波形类型 | 波形参数 | 主要用途 |
|------|----------|----------|----------|
| 1 | `cosine_drag` | `CosineDrag(duration, amp, phase, alpha)` | 抑制泄漏态跃迁的精确控制 |
| 2 | `flattop` | `Flattop(duration, amp, width)` | 平顶脉冲，量子态制备 |
| 3 | `gaussian` | `Gaussian(duration, amp, sigma, angle)` | 高斯脉冲，平滑过渡 |
| 4 | `sin` | `Sin(duration, amp, phase, freq, angle)` | 正弦脉冲，周期性振荡 |
| 5 | `drag` | `Drag(duration, amp, sigma, beta)` | DRAG协议，超导量子比特控制 |
| 6 | `constant` | `Constant(duration, amp)` | 常数脉冲，直流偏置 |
| 7 | `gaussian_square` | `GaussianSquare(duration, amp, sigma, width)` | 高斯边缘方波 |
| 8 | `cosine` | `Cosine(duration, amp, freq, phase)` | 余弦脉冲 |

## 波形参数详细说明

本节详细介绍每种波形的参数定义、数学表达式和物理意义。每种波形都有其特定的应用场景和参数约束。

---

### 1. CosineDrag 波形参数

| 参数 | 类型 | 说明 | 约束条件 |
|------|------|------|----------|
| `amp` | real value | 波形振幅 | \|amp\| ≤ 2 |
| `duration` | int | 脉冲长度（采样周期为单位） | 0 < duration < 10000 |
| `phase` | real value | 相位角（弧度） | 无特殊限制 |
| `alpha` | real value | DRAG系数 | \|alpha\| ≤ 10 |

**数学表达式**: 
- `g(x) = (Amp / 2) × e^(i × phase) × [cos((2πx / duration) - π) + 1]`
- `output(x) = g(x) + i × alpha × g'(x)`
- 定义域: `x ∈ [0, duration)`

**参数说明**: 
- `amp`: 波形振幅，控制波形的强度
- `duration`: 脉冲持续时间，以采样周期为单位
- `phase`: 相位角，控制波形的相位偏移
- `alpha`: DRAG系数，用于抑制泄漏态跃迁

### 2. Flattop 波形参数

| 参数 | 类型 | 说明 | 约束条件 |
|------|------|------|----------|
| `amp` | real value | 波形振幅 | amp ≤ 2 |
| `width` | real value | 高斯分量的半高全宽(FWHM) | width ≤ 100 |
| `duration` | int | 脉冲长度（采样周期为单位） | duration ≤ 100,000 |

**数学表达式**: 
- `w = width` (高斯分量的半高全宽)
- `σ = w / √(4 log 2)` (标准差)
- `A = amp` (振幅)
- `T = duration` (持续时间)
- `output(x) = (A / 2) × [erf((w + T - x) / σ) - erf((w - x) / σ)]`
- 定义域: `x ∈ [0, T + 2w)`

**参数说明**: 
- `amp`: 波形振幅，控制波形的整体强度
- `width`: 高斯分量的半高全宽，控制高斯边缘的宽度
- `duration`: 脉冲持续时间，控制平顶部分的长度

### 3. Gaussian 波形参数

| 参数 | 类型 | 说明 | 约束条件 |
|------|------|------|----------|
| `amp` | real value | 波形振幅 | \|amp\| ≤ 2 |
| `duration` | int | 脉冲长度（采样周期为单位） | 0 < duration < 10000 |
| `sigma` | real value | 高斯波形的标准差 | 无特殊限制 |
| `angle` | real value | 复数相位因子的角度（弧度） | 无特殊限制 |

**数学表达式**: 
- `f'(x) = exp(- (1/2) × ((x - duration/2)² / sigma²))`
- `f(x) = A × f'(x)` 当 `0 ≤ x < duration`
- `A = amp × exp(i × angle)`

**参数说明**: 
- `amp`: 波形振幅，控制波形的强度
- `duration`: 脉冲持续时间，以采样周期为单位
- `sigma`: 高斯分布的标准差，控制波形的宽度
- `angle`: 复数相位因子，控制波形的相位

### 4. Sine 波形参数

| 参数 | 类型 | 说明 | 约束条件 |
|------|------|------|----------|
| `amp` | real value | 正弦波振幅，波形范围为 [-amp, amp] | \|amp\| ≤ 2 |
| `phase` | real value | 正弦波的相位（弧度） | 无特殊限制 |
| `freq` | real value | 正弦波频率（采样周期的倒数） | 无特殊限制 |
| `angle` | real value | 复数相位因子的角度（弧度） | 无特殊限制 |
| `duration` | int | 脉冲长度（采样周期为单位） | 0 < duration < 10000 |

**数学表达式**: 
- `f(x) = A sin(2π × freq × x + phase)` 当 `0 ≤ x < duration`
- `A = amp × exp(i × angle)`

**参数说明**: 
- `amp`: 正弦波振幅，控制波形的强度，波形范围为 [-amp, amp]
- `phase`: 正弦波的相位，控制波形的相位偏移
- `freq`: 正弦波频率，以采样周期的倒数为单位
- `angle`: 复数相位因子，控制波形的复数相位
- `duration`: 脉冲持续时间，以采样周期为单位

### 5. 其他波形参数

#### GaussianSquare 波形
| 参数 | 类型 | 说明 | 约束条件 |
|------|------|------|----------|
| `amp` | real value | 波形振幅 | \|amp\| ≤ 2 |
| `duration` | int | 脉冲长度（采样周期为单位） | 0 < duration < 10000 |
| `sigma` | real value | 高斯分量的标准差 | 无特殊限制 |
| `width` | real value | 方波部分的宽度 | width ≤ duration |

#### Drag 波形
| 参数 | 类型 | 说明 | 约束条件 |
|------|------|------|----------|
| `amp` | real value | 波形振幅 | \|amp\| ≤ 2 |
| `duration` | int | 脉冲长度（采样周期为单位） | 0 < duration < 10000 |
| `sigma` | real value | 高斯分量的标准差 | 无特殊限制 |
| `beta` | real value | DRAG参数，用于抑制泄漏态跃迁 | 无特殊限制 |

**数学表达式**: 
- `f(x) = A × exp(-(x - duration/2)² / (2 × sigma²))` 当 `0 ≤ x < duration`
- `A = amp × exp(i × angle)` (如果支持angle参数)

#### Constant 波形
| 参数 | 类型 | 说明 | 约束条件 |
|------|------|------|----------|
| `amp` | real value | 常数振幅 | \|amp\| ≤ 2 |
| `duration` | int | 脉冲长度（采样周期为单位） | 0 < duration < 10000 |

**数学表达式**: 
- `f(x) = amp` 当 `0 ≤ x < duration`
- 定义域: `x ∈ [0, duration)`

**参数说明**: 
- `amp`: 常数振幅，在整个持续时间内保持恒定
- `duration`: 脉冲持续时间，以采样周期为单位

## 参数使用注意事项

1. **时间单位**: 所有时间相关参数（duration, width等）都以采样周期为单位
2. **振幅限制**: 大多数波形的振幅都限制在 [-2, 2] 范围内
3. **参数类型**: 支持数值和参数化两种输入方式
4. **约束检查**: 系统会自动检查参数是否满足约束条件
5. **物理意义**: 参数值应根据具体的量子硬件特性进行调整
6. **复数相位**: 部分波形支持复数相位因子（angle参数），用于精细相位控制
7. **定义域**: 注意每种波形的定义域范围，确保参数设置合理

## 波形选择指南

### 根据应用场景选择波形

- **CosineDrag**: 适用于需要抑制泄漏态跃迁的精确控制，如单比特门操作
- **Flattop**: 适用于需要平顶脉冲的应用，如量子态制备
- **Gaussian**: 适用于需要平滑过渡的脉冲，如绝热演化
- **Sine**: 适用于需要周期性振荡的应用，如Rabi振荡实验
- **Drag**: 适用于超导量子比特的精确控制
- **Constant**: 适用于简单的常数脉冲，如直流偏置
- **GaussianSquare**: 适用于需要高斯边缘的方波脉冲 






## 核心组件

TyxonQ Pulse 接口的核心组件包括波形类型、参数化支持和校准构建器。这些组件协同工作，为用户提供完整的脉冲控制能力。

---

### 1. 波形类型 (Waveforms)

TyxonQ 支持多种预定义的脉冲波形类型，每种波形都有特定的参数：

#### 高斯波形 (Gaussian)
```python
from tyxonq import waveforms

# 创建高斯波形：振幅、持续时间、标准差
gaussian_wf = waveforms.Gaussian(amp=0.5, duration=100, sigma=20)
```

#### 高斯方波 (GaussianSquare)
```python
# 创建高斯方波：振幅、持续时间、标准差、宽度
gaussian_square_wf = waveforms.GaussianSquare(amp=0.5, duration=100, sigma=20, width=60)
```

#### DRAG 波形 (Drag)
```python
# 创建 DRAG 波形：振幅、持续时间、标准差、β参数
drag_wf = waveforms.Drag(amp=0.5, duration=100, sigma=20, beta=0.5)
```

#### 常数波形 (Constant)
```python
# 创建常数波形：振幅、持续时间
constant_wf = waveforms.Constant(amp=0.5, duration=100)
```

#### 正弦波形 (Sine)
```python
# 创建正弦波形：振幅、频率、持续时间
sine_wf = waveforms.Sine(amp=0.5, frequency=0.1, duration=100)
```

#### 余弦波形 (Cosine)
```python
# 创建余弦波形：振幅、频率、持续时间
cosine_wf = waveforms.Cosine(amp=0.5, frequency=0.1, duration=100)
```

#### 余弦 DRAG 波形 (CosineDrag)
```python
# 创建余弦 DRAG 波形：振幅、持续时间、相位、α参数
cosine_drag_wf = waveforms.CosineDrag(amp=0.5, duration=100, phase=0.0, alpha=0.2)
```

#### 平顶波形 (Flattop)
```python
# 创建平顶波形：振幅、宽度、持续时间
flattop_wf = waveforms.Flattop(amp=0.5, width=60, duration=100)
```

### 2. 参数化支持 (Param)

所有波形都支持参数化，可以使用 `Param` 类创建动态参数：

```python
from tyxonq import Param

# 创建参数化波形
param_t = Param("t")
parametric_wf = waveforms.CosineDrag(param_t, 0.2, 0.0, 0.0)
```

### 3. 校准构建器 (DefcalBuilder)

`DefcalBuilder` 是创建量子比特校准程序的核心工具：

```python
from tyxonq import Circuit, Param

# 创建电路并启用脉冲模式
qc = Circuit(1)
qc.use_pulse()

# 创建参数
param0 = Param("a")

# 开始构建校准程序
builder = qc.calibrate("calibration_name", [param0])

# 定义帧
builder.new_frame("drive_frame", param0)

# 播放波形
builder.play("drive_frame", waveforms.CosineDrag(param0, 0.2, 0.0, 0.0))

# 构建校准程序
builder.build()
```

## 使用方法

TyxonQ Pulse 接口提供了直观易用的API，让用户能够轻松创建复杂的脉冲控制程序。

---

### 基本工作流程

1. **启用脉冲模式**
```python
qc = Circuit(n_qubits)
qc.use_pulse()
```

2. **定义校准程序**
```python
# 使用 DefcalBuilder 构建校准程序
builder = qc.calibrate("cal_name", [param1, param2])
builder.new_frame("frame_name", qubit_param)
builder.play("frame_name", waveform)
builder.build()
```

3. **调用校准程序**
```python
# 在电路中调用校准程序
qc.add_calibration('cal_name', ['q[0]'])
```

4. **生成 TQASM 代码**
```python
tqasm_code = qc.to_tqasm()
```

### 完整示例

#### 示例 1：简单的 Rabi 振荡实验

```python
import sys
import os
sys.path.insert(0, "..")

from tyxonq import Circuit, Param, waveforms
from tyxonq.cloud import apis

def create_rabi_circuit(t):
    """创建 Rabi 振荡实验电路"""
    qc = Circuit(1)
    qc.use_pulse()
    
    # 创建参数
    param_t = Param("t")
    
    # 构建校准程序
    builder = qc.calibrate("rabi_experiment", [param_t])
    builder.new_frame("drive_frame", param_t)
    builder.play("drive_frame", waveforms.CosineDrag(param_t, 0.2, 0.0, 0.0))
    builder.build()
    
    # 调用校准程序
    qc.add_calibration('rabi_experiment', ['q[0]'])
    
    return qc

# 创建不同时间参数的电路
for t in [10, 30, 50, 70, 90]:
    qc = create_rabi_circuit(t)
    print(f"TQASM for t={t}:")
    print(qc.to_tqasm())
    print("-" * 50)
```

## TQASM 输出格式

生成的 TQASM 代码遵循 TQASM 0.2 标准：

```tqasm
TQASM 0.2;
QREG q[1];

defcal rabi_experiment a {
  frame drive_frame = newframe(a);
  play(drive_frame, cosine_drag(a, 0.2, 0.0, 0.0));
}

rabi_experiment q[0];
```

## 高级功能

### 1. 时间控制

可以为波形添加开始时间参数：

```python
builder.play("frame_name", waveform, start_time=50)
```

### 2. 复杂校准程序

可以构建包含多个指令的复杂校准程序：

```python
builder = qc.calibrate("complex_cal", [param])
builder.new_frame("frame1", param)
builder.play("frame1", waveform1)
builder.new_frame("frame2", param)
builder.play("frame2", waveform2)
builder.build()
```

### 3. 与云 API 集成

```python
from tyxonq.cloud import apis

# 设置认证
apis.set_token("your_token")
apis.set_provider("tyxonq")

# 提交脉冲电路任务
task = apis.submit_task(
    circuit=qc,
    shots=1000,
    device="homebrew_s2",
    enable_qos_gate_decomposition=False,
    enable_qos_qubit_mapping=False,
)

# 获取结果
result = task.results()
```

## 最佳实践

1. **参数命名**: 使用有意义的参数名称，便于理解和调试
2. **波形选择**: 根据物理需求选择合适的波形类型
3. **时间单位**: 注意时间参数的单位（通常是纳秒）
4. **错误处理**: 在提交到硬件前验证 TQASM 代码的正确性
5. **文档化**: 为复杂的校准程序添加注释和说明

## 故障排除

### 常见问题

1. **波形类型不支持**: 确保使用预定义的波形类型
2. **参数类型错误**: 检查参数是否为 `Param` 类型或数值
3. **帧未定义**: 确保在播放波形前定义了帧
4. **TQASM 生成失败**: 检查校准程序的构建顺序

## 实际应用示例

### 示例：精确的 Rabi 振荡实验

基于数学定义的精确参数设置：

```python
from tyxonq import Circuit, Param, waveforms

def create_precise_rabi_circuit(t_duration, amplitude, frequency):
    """
    创建精确的 Rabi 振荡实验电路
    
    参数:
    - t_duration: 脉冲持续时间（采样周期）
    - amplitude: 正弦波振幅 (|amp| ≤ 2)
    - frequency: 正弦波频率 (采样周期的倒数)
    """
    qc = Circuit(1)
    qc.use_pulse()
    
    # 创建参数化波形
    param_t = Param("t")
    
    sine_wave = waveforms.Sine(
        duration=t_duration,      # 持续时间
        amp=amplitude,            # 振幅
        frequency=frequency,           # 频率
    )

    
    # 构建校准程序
    builder = qc.calibrate("precise_rabi", [param_t])
    builder.new_frame("drive_frame", param_t)
    builder.play("drive_frame", sine_wave)
    builder.build()
    
    # 调用校准程序
    qc.add_calibration('precise_rabi', ['q[0]'])
    
    return qc

# 创建不同参数的电路进行参数扫描
frequencies = [0.01, 0.02, 0.05, 0.1]  # 不同频率
amplitudes = [0.5, 1.0, 1.5]            # 不同振幅

for freq in frequencies:
    for amp in amplitudes:
        qc = create_precise_rabi_circuit(
            t_duration=100,    # 100个采样周期
            amplitude=amp,      # 振幅
            frequency=freq      # 频率
        )
        print(f"Frequency: {freq}, Amplitude: {amp}")
        print(qc.to_tqasm())
        print("-" * 50)
```

### 示例：DRAG 脉冲优化

```python
def create_optimized_drag_pulse():
    """创建优化的 DRAG 脉冲"""
    qc = Circuit(1)
    qc.use_pulse()
    
    param_qubit = Param("q")
    
    # 使用 DRAG 协议抑制泄漏态跃迁
    # f(x) = A × exp(-(x - duration/2)² / (2 × sigma²))
    # A = amp × exp(i × angle)
    drag_wave = waveforms.Drag(
        duration=100,    
        amp=1.0,         # 振幅
        sigma=20,        # 高斯标准差
        beta=0.5         # DRAG参数，抑制泄漏态
    )
    
    builder = qc.calibrate("optimized_drag", [param_qubit])
    builder.new_frame("drive_frame", param_qubit)
    builder.play("drive_frame", drag_wave)
    builder.build()
    
    qc.add_calibration('optimized_drag', ['q[0]'])
    return qc
```

## 总结

TyxonQ 的 Pulse 接口提供了强大的微波脉冲控制能力。通过合理使用波形类型、参数化和校准构建器，您可以实现各种复杂的量子控制实验。建议从简单的示例开始，逐步构建更复杂的应用。

---

### 关键要点

1. **数学精确性**: 每种波形都有精确的数学定义，确保物理实现的准确性
2. **参数约束**: 严格遵守参数约束条件，避免硬件错误
3. **应用导向**: 根据具体的量子控制需求选择合适的波形类型
4. **实验设计**: 利用参数化功能进行系统性的参数扫描和优化
