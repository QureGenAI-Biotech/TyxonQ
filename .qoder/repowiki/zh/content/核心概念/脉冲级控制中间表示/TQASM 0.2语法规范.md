# TQASM 0.2语法规范

<cite>
**本文档中引用的文件**  
- [pulse.py](file://src/tyxonq/core/ir/pulse.py)
- [circuit.py](file://src/tyxonq/core/ir/circuit.py)
- [waveforms.py](file://src/tyxonq/waveforms.py)
- [pulse_support_cn.md](file://docs/pulse_support_cn.md)
- [pulse_demo.py](file://examples/pulse_demo.py)
</cite>

## 目录

1. [引言](#引言)
2. [TQASM 0.2语法结构](#tqasm-02语法结构)
3. [核心关键字语法规则](#核心关键字语法规则)
4. [校准程序模块化定义](#校准程序模块化定义)
5. [脉冲指令文本表示格式](#脉冲指令文本表示格式)
6. [TQASM输出示例与PulseSchedule映射](#tqasm输出示例与pulseschedule映射)
7. [跨平台交换与重放支持](#跨平台交换与重放支持)
8. [结论](#结论)

## 引言

TQASM 0.2是TyxonQ框架中用于描述量子脉冲级控制的文本表示语言。该语言旨在提供一种标准化、可读性强且可跨平台交换的脉冲程序描述方式，支持对量子硬件进行精确的低级控制。通过defcal、frame、play等核心关键字，用户可以定义参数化的校准程序，构建复杂的脉冲序列，并确保在不同设备和后端之间的一致性执行。

**Section sources**
- [pulse_support_cn.md](file://docs/pulse_support_cn.md#L0-L619)

## TQASM 0.2语法结构

TQASM 0.2采用类QASM的语法结构，结合巴科斯-诺尔范式(BNF)定义其语法规则。该语言以声明式方式描述脉冲程序，支持参数化定义和模块化调用。

```tqasm
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

该语法结构定义了从顶层脉冲程序到具体波形播放指令的层次化组织方式，确保了语言的可扩展性和结构性。

**Diagram sources**
- [pulse_support_cn.md](file://docs/pulse_support_cn.md#L0-L619)

**Section sources**
- [pulse_support_cn.md](file://docs/pulse_support_cn.md#L0-L619)

## 核心关键字语法规则

### defcal关键字

`defcal`用于定义参数化的校准程序，是TQASM 0.2中实现模块化的核心机制。

**语法规则**:
```
defcal <校准名称> <参数列表> { <校准语句> }
```

该关键字允许用户创建可重用的脉冲序列模板，通过参数化实现灵活的脉冲控制。

### frame与newframe关键字

`frame`和`newframe`用于声明和创建脉冲帧，为波形播放提供时序和通道上下文。

**语法规则**:
```
frame <帧名称> = newframe(<量子比特标识>);
```

帧(frame)代表了在特定量子比特上承载波形的逻辑通道，是脉冲时序控制的基础单元。

### play关键字

`play`用于在指定帧上播放波形，是执行实际脉冲操作的关键指令。

**语法规则**:
```
play(<帧名称>, <波形函数>(<参数>));
```

该指令将波形与帧关联，实现对量子硬件的精确控制。

### shift_phase与set_frequency关键字

虽然在当前文档中未直接体现，但`shift_phase`和`set_frequency`作为高级脉冲控制指令，用于动态调整脉冲相位和频率，支持复杂的量子控制序列。

**Section sources**
- [pulse_support_cn.md](file://docs/pulse_support_cn.md#L0-L619)

## 校准程序模块化定义

### 命名与参数声明

校准程序通过`defcal`关键字进行命名，并接受参数列表作为输入。参数可以是数值或符号参数(Param)，支持动态配置。

```python
builder = qc.calibrate("calibration_name", [param0])
```

这种设计实现了校准程序的参数化和可重用性，提高了脉冲程序的灵活性和可维护性。

### 调用机制

校准程序定义后，可通过其名称在电路中调用，作用于指定的量子比特。

```python
qc.add_calibration('hello_world', ['q[0]'])
```

调用机制实现了校准程序与具体量子硬件的解耦，支持跨不同量子设备的程序复用。

### 构建流程

校准程序的构建遵循特定的流程：创建电路→启用脉冲模式→定义参数→构建校准程序→定义帧→播放波形→构建完成。

```python
qc = Circuit(1)
qc.use_pulse()
param0 = Param("a")
builder = qc.calibrate("calibration_name", [param0])
builder.new_frame("drive_frame", param0)
builder.play("drive_frame", waveforms.CosineDrag(param0, 0.2, 0.0, 0.0))
builder.build()
```

**Section sources**
- [pulse_support_cn.md](file://docs/pulse_support_cn.md#L350-L376)
- [pulse_demo.py](file://examples/pulse_demo.py#L0-L81)

## 脉冲指令文本表示格式

### 通道寻址

通道通过字符串标识符进行寻址，如"d0"、"u1"等，直接对应硬件通道。

```python
channel: str
```

这种寻址方式确保了脉冲指令与物理硬件的精确对应。

### 时间戳

时间戳以采样周期为单位，使用整数表示，保持了与后端无关的特性。

```python
start: int
duration: int
```

时间单位的标准化确保了脉冲程序在不同采样率设备间的可移植性。

### 波形引用

波形通过预定义的波形类进行引用，如`CosineDrag`、`Flattop`等，每个波形类封装了特定的数学表达式和物理意义。

```python
waveform: List[Any]
```

波形引用的标准化确保了脉冲形状的精确再现。

### 元数据嵌入

元数据以字典形式嵌入，包含波形形状、振幅、标准差等描述信息。

```python
metadata: Dict[str, Any]
```

元数据的灵活设计支持扩展的脉冲特性描述和后端特定的优化提示。

**Section sources**
- [pulse.py](file://src/tyxonq/core/ir/pulse.py#L7-L27)

## TQASM输出示例与PulseSchedule映射

### 完整TQASM示例

```tqasm
TQASM 0.2;
QREG q[1];

defcal hello_world a {
  frame drive_frame = newframe(a);
  play(drive_frame, cosine_drag(50, 0.2, 0.0, 0.0));
}

hello_world q[0];
```

该示例展示了TQASM 0.2的基本结构，包括版本声明、量子寄存器定义、校准程序定义和调用。

### PulseSchedule对象映射

TQASM程序被映射为`PulseSchedule`对象，包含采样率、指令列表和全局参数。

```python
class PulseSchedule:
    sampling_rate_hz: float
    instructions: List[PulseInstruction]
    globals: Dict[str, Any]
```

每个`PulseInstruction`对应TQASM中的一个play指令，包含通道、时间戳、波形和元数据。

```python
class PulseInstruction:
    channel: str
    start: int
    duration: int
    waveform: List[Any]
    metadata: Dict[str, Any]
```

这种映射关系确保了文本表示与程序对象的一致性，支持双向转换和验证。

**Section sources**
- [pulse_support_cn.md](file://docs/pulse_support_cn.md#L0-L619)
- [pulse.py](file://src/tyxonq/core/ir/pulse.py#L31-L63)

## 跨平台交换与重放支持

### 后端无关性设计

TQASM 0.2采用采样周期作为时间单位，避免了对特定采样率的依赖。`PulseSchedule`中的`sampling_rate_hz`字段提供了时间转换的基础，确保了在不同硬件平台间的可移植性。

### 统一的波形表示

通过`waveforms`模块提供统一的波形类，如`Gaussian`、`Drag`、`Sine`等，每个类实现了`qasm_name()`和`to_args()`方法，确保了波形参数的标准化序列化。

```python
@dataclass
class Gaussian:
    amp: ParamType
    duration: int
    sigma: ParamType
    def qasm_name(self) -> str:
        return "gaussian"
    def to_args(self) -> List[ParamType]:
        return [self.amp, self.duration, self.sigma]
```

这种设计确保了波形在不同后端的一致性解释和执行。

### 参数化支持

通过`Param`类支持符号参数，允许在运行时动态绑定参数值，增强了脉冲程序的灵活性和适应性。

### 标准化接口

`Circuit`类提供了`to_tqasm()`方法，将电路对象转换为TQASM文本，实现了程序的标准化输出和交换。

```python
def to_tqasm(self):
    # 实现TQASM文本生成逻辑
    pass
```

**Section sources**
- [waveforms.py](file://src/tyxonq/waveforms.py#L0-L98)
- [circuit.py](file://src/tyxonq/core/ir/circuit.py#L0-L779)

## 结论

TQASM 0.2通过精心设计的语法结构和语义规则，为量子脉冲级控制提供了一种标准化、模块化且可跨平台交换的描述语言。其核心关键字`defcal`、`frame`、`play`等实现了校准程序的参数化定义和模块化调用，而`PulseSchedule`和`PulseInstruction`对象则提供了程序的内部表示和执行基础。通过统一的波形表示、标准化的参数化支持和后端无关的时间单位，TQASM 0.2确保了脉冲程序在不同设备和后端之间的一致性执行，为量子计算的精确控制和可重复研究提供了坚实的基础。