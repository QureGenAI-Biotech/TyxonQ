# QASM3 完整工作流

<cite>
**本文档中引用的文件**
- [README.md](file://README.md)
- [src/tyxonq/__init__.py](file://src/tyxonq/__init__.py)
- [src/tyxonq/compiler/api.py](file://src/tyxonq/compiler/api.py)
- [src/tyxonq/core/ir/circuit.py](file://src/tyxonq/core/ir/circuit.py)
- [src/tyxonq/devices/base.py](file://src/tyxonq/devices/base.py)
- [src/tyxonq/compiler/compile_engine/native/native_compiler.py](file://src/tyxonq/compiler/compile_engine/native/native_compiler.py)
- [src/tyxonq/compiler/compile_engine/qiskit/qiskit_compiler.py](file://src/tyxonq/compiler/compile_engine/qiskit/qiskit_compiler.py)
- [src/tyxonq/devices/simulators/driver.py](file://src/tyxonq/devices/simulators/driver.py)
- [src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py](file://src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py)
- [src/tyxonq/waveforms.py](file://src/tyxonq/waveforms.py)
- [examples/qasm3_complete_workflow.py](file://examples/qasm3_complete_workflow.py)
</cite>

## 目录
1. [简介](#简介)
2. [项目架构概览](#项目架构概览)
3. [QASM3 工作流程核心组件](#qasm3-工作流程核心组件)
4. [完整工作流程详解](#完整工作流程详解)
5. [编译引擎与输出格式](#编译引擎与输出格式)
6. [脉冲编程与 TQASM 导出](#脉冲编程与-tqasm-导出)
7. [QASM3 解析与导入](#qasm3-解析与导入)
8. [驱动层执行](#驱动层执行)
9. [最佳实践与优化](#最佳实践与优化)
10. [故障排除指南](#故障排除指南)

## 简介

TyxonQ 是一个模块化的全栈量子软件框架，提供了完整的 QASM3 工作流程支持。该框架实现了 OpenQASM 3.0 标准的全面兼容性，包括相位 2（门级）、相位 3（帧定义）和相位 4（defcal 校准）的所有功能。

### 核心特性

- **完整的 QASM3 支持**：Phase 2、Phase 3、Phase 4 功能全覆盖
- **双向互操作性**：Circuit IR ↔ QASM3 Round-trip 完美一致性
- **脉冲级控制**：支持 OpenPulse 语法和自定义波形
- **多编译器后端**：Native、Qiskit、Pulse 编译器
- **硬件就绪**：直接导出到 TQASM 0.2 格式用于硬件执行

## 项目架构概览

TyxonQ 的 QASM3 工作流程基于分层架构设计，确保了从电路构建到硬件执行的完整链路。

```mermaid
graph TB
subgraph "用户层"
A[Circuit API] --> B[脉冲编程 API]
B --> C[Cloud API]
end
subgraph "编译层"
D[Native Compiler] --> E[Qiskit Compiler]
E --> F[Pulse Compiler]
F --> G[QASM3 Exporter]
G --> H[TQASM Exporter]
end
subgraph "解析层"
I[QASM3 Importer] --> J[脉冲解析器]
J --> K[校准解析器]
end
subgraph "执行层"
L[模拟器驱动] --> M[硬件驱动]
M --> N[云服务]
end
A --> D
D --> I
I --> L
L --> C
```

**图表来源**
- [src/tyxonq/compiler/api.py](file://src/tyxonq/compiler/api.py#L1-L50)
- [src/tyxonq/devices/base.py](file://src/tyxonq/devices/base.py#L1-L50)

## QASM3 工作流程核心组件

### 1. 编译器 API

编译器 API 提供统一的入口点，支持多种输出格式和编译策略。

```mermaid
classDiagram
class CompileResult {
+circuit : Any
+compiled_source : Optional[str]
+metadata : Dict[str, Any]
}
class PulseCompileResult {
+pulse_program : Any
+compiled_pulse_schedule : Optional[str]
+metadata : Dict[str, Any]
}
class NativeCompiler {
+compile(circuit, options) CompileResult
+name : str
}
class QiskitCompiler {
+compile(circuit, options) CompileResult
+name : str
}
CompileResult --> NativeCompiler
CompileResult --> QiskitCompiler
PulseCompileResult --> NativeCompiler
```

**图表来源**
- [src/tyxonq/compiler/api.py](file://src/tyxonq/compiler/api.py#L15-L35)
- [src/tyxonq/compiler/compile_engine/native/native_compiler.py](file://src/tyxonq/compiler/compile_engine/native/native_compiler.py#L15-L25)

### 2. 电路 IR 层

电路中间表示（IR）是整个工作流程的核心数据结构。

```mermaid
classDiagram
class Circuit {
+num_qubits : int
+ops : List[Any]
+metadata : Dict[str, Any]
+instructions : List[Tuple[str, Tuple[int, ...]]]
+compile(output) CompileResult
+device(provider, device) Circuit
+postprocessing(method) Circuit
}
class PulseProgram {
+num_qubits : int
+ops : List[Any]
+metadata : Dict[str, Any]
+compile(output) PulseCompileResult
}
Circuit --> PulseProgram : "脉冲增强"
```

**图表来源**
- [src/tyxonq/core/ir/circuit.py](file://src/tyxonq/core/ir/circuit.py#L50-L100)

**章节来源**
- [src/tyxonq/compiler/api.py](file://src/tyxonq/compiler/api.py#L1-L100)
- [src/tyxonq/core/ir/circuit.py](file://src/tyxonq/core/ir/circuit.py#L50-L150)

## 完整工作流程详解

### Phase 2: 门级 QASM3 导入导出

这是最基础的功能，支持标准的量子门操作在 TyxonQ IR 和 QASM3 之间的转换。

#### 工作流程图

```mermaid
sequenceDiagram
participant User as 用户
participant Circuit as Circuit IR
participant Compiler as Native Compiler
participant QASM3 as QASM3 Exporter
participant Parser as QASM3 Importer
User->>Circuit : 创建量子电路
Circuit->>Compiler : compile(output="qasm3")
Compiler->>QASM3 : 导出 QASM3 代码
QASM3-->>Compiler : 返回 QASM3 字符串
Compiler-->>User : CompileResult
User->>Parser : qasm3_to_circuit(qasm3_code)
Parser->>Circuit : 解析并重建 Circuit
Circuit-->>User : 返回 Circuit IR
```

**图表来源**
- [examples/qasm3_complete_workflow.py](file://examples/qasm3_complete_workflow.py#L20-L80)
- [src/tyxonq/compiler/compile_engine/native/gate_qasm3_exporter.py](file://src/tyxonq/compiler/compile_engine/native/gate_qasm3_exporter.py#L17-L66)

#### 示例代码路径

1. **电路创建**：[`Circuit(2).h(0).cx(0, 1)`](file://examples/qasm3_complete_workflow.py#L30-L35)
2. **QASM3 导出**：[`compile(circuit, output="qasm3")`](file://examples/qasm3_complete_workflow.py#L40-L45)
3. **QASM3 解析**：[`qasm3_to_circuit(qasm3_code)`](file://examples/qasm3_complete_workflow.py#L50-L55)

### Phase 3: OpenPulse 帧定义

支持 OpenPulse 的 cal 块语法，包含端口和帧声明。

#### 帧定义语法

```mermaid
flowchart TD
A[QASM3 源码] --> B[解析 cal 块]
B --> C[提取端口声明]
B --> D[提取帧定义]
C --> E[存储到 metadata]
D --> E
E --> F[Frame 对象]
F --> G[频率和相位信息]
```

**图表来源**
- [src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py](file://src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py#L150-L200)

### Phase 4: defcal 校准定义

支持复杂的门校准定义，包含波形和脉冲指令序列。

#### defcal 定义结构

```mermaid
classDiagram
class DefcalDefinition {
+gate_name : str
+qubits : List[int]
+parameters : List[str]
+body : List[str]
}
class Waveform {
+waveform_type : str
+amplitude : float
+duration : int
+params : Dict[str, Any]
}
class Frame {
+name : str
+port : str
+frequency : float
+phase : float
}
DefcalDefinition --> Waveform : "包含"
DefcalDefinition --> Frame : "使用"
```

**图表来源**
- [src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py](file://src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py#L50-L80)

**章节来源**
- [examples/qasm3_complete_workflow.py](file://examples/qasm3_complete_workflow.py#L80-L150)
- [src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py](file://src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py#L1-L100)

## 编译引擎与输出格式

### 支持的输出格式

TyxonQ 编译器支持多种输出格式，满足不同的使用场景：

| 输出格式 | 描述 | 使用场景 |
|---------|------|----------|
| `ir` | TyxonQ 内部 IR | 开发和调试 |
| `qasm3` | OpenQASM 3.0 | 标准互操作性 |
| `qasm2` | OpenQASM 2.0 | 向后兼容 |
| `qiskit` | Qiskit QuantumCircuit | Qiskit 生态系统 |
| `tyxonq_homebrew_tqasm` | TyxonQ 特定 TQASM | 硬件就绪 |

### 编译流程

```mermaid
flowchart TD
A[输入电路] --> B{检测编译器类型}
B --> |native| C[Native Compiler]
B --> |qiskit| D[Qiskit Compiler]
B --> |pulse| E[Pulse Compiler]
C --> F{检测输出格式}
D --> F
E --> F
F --> |qasm3| G[GateQASM3Exporter]
F --> |tqasm| H[TQASMExporter]
F --> |qiskit| I[Qiskit Dialect]
F --> |ir| J[直接返回 IR]
G --> K[QASM3 字符串]
H --> L[TQASM 字符串]
I --> M[Qiskit 对象]
J --> N[原始 IR]
```

**图表来源**
- [src/tyxonq/compiler/compile_engine/native/native_compiler.py](file://src/tyxonq/compiler/compile_engine/native/native_compiler.py#L60-L110)

**章节来源**
- [src/tyxonq/compiler/compile_engine/native/native_compiler.py](file://src/tyxonq/compiler/compile_engine/native/native_compiler.py#L1-L112)
- [src/tyxonq/compiler/compile_engine/qiskit/qiskit_compiler.py](file://src/tyxonq/compiler/compile_engine/qiskit/qiskit_compiler.py#L1-L85)

## 脉冲编程与 TQASM 导出

### 波形类型支持

TyxonQ 支持多种脉冲波形类型，用于精确的量子控制：

```mermaid
classDiagram
class Waveform {
<<abstract>>
+qasm_name() str
+to_args() List[ParamType]
}
class Gaussian {
+amp : ParamType
+duration : int
+sigma : ParamType
+phase : ParamType
}
class Drag {
+amp : ParamType
+duration : int
+sigma : ParamType
+beta : ParamType
+phase : ParamType
}
class GaussianSquare {
+amp : ParamType
+duration : int
+sigma : ParamType
+width : ParamType
}
class Constant {
+amp : ParamType
+duration : int
}
Waveform <|-- Gaussian
Waveform <|-- Drag
Waveform <|-- GaussianSquare
Waveform <|-- Constant
```

**图表来源**
- [src/tyxonq/waveforms.py](file://src/tyxonq/waveforms.py#L10-L100)

### TQASM 导出流程

```mermaid
sequenceDiagram
participant Circuit as 脉冲电路
participant Exporter as TQASM Exporter
participant Waveform as 波形生成器
participant CalBlock as 校准块生成器
Circuit->>Exporter : export(pulse_circuit)
Exporter->>Exporter : 分析脉冲操作
Exporter->>CalBlock : 生成 cal 块
CalBlock-->>Exporter : 端口和帧声明
Exporter->>Waveform : 生成波形定义
Waveform-->>Exporter : 波形参数
Exporter->>Exporter : 生成 defcal 定义
Exporter-->>Circuit : TQASM 0.2 字符串
```

**图表来源**
- [src/tyxonq/compiler/pulse_compile_engine/native/tqasm_exporter.py](file://src/tyxonq/compiler/pulse_compile_engine/native/tqasm_exporter.py#L100-L200)

**章节来源**
- [src/tyxonq/waveforms.py](file://src/tyxonq/waveforms.py#L1-L157)
- [src/tyxonq/compiler/pulse_compile_engine/native/tqasm_exporter.py](file://src/tyxonq/compiler/pulse_compile_engine/native/tqasm_exporter.py#L1-L200)

## QASM3 解析与导入

### 解析器架构

QASM3 解析器采用分层解析策略，逐步提取不同层次的语法结构：

```mermaid
flowchart TD
A[QASM3 源码] --> B[词法分析器]
B --> C[语法分析器]
C --> D{解析阶段}
D --> |Phase 1| E[量子比特声明]
D --> |Phase 2| F[门操作]
D --> |Phase 3| G[cal 块]
D --> |Phase 4| H[defcal 定义]
E --> I[提取 qubit 数量]
F --> J[解析门序列]
G --> K[解析端口和帧]
H --> L[解析波形和指令]
I --> M[构建 Circuit IR]
J --> M
K --> M
L --> M
M --> N[添加元数据]
N --> O[返回完整 Circuit]
```

**图表来源**
- [src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py](file://src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py#L100-L200)

### 支持的语法结构

| 语法类别 | 支持程度 | 描述 |
|---------|---------|------|
| 量子比特声明 | ✅ 完全支持 | `qubit[n] q;` 和 `qreg q[n];` |
| 单量子门 | ✅ 完全支持 | `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz` |
| 双量子门 | ✅ 完全支持 | `cx`, `cy`, `cz`, `swap` |
| 测量操作 | ✅ 完全支持 | `measure q[i];` |
| cal 块 | ✅ Phase 3 | 端口和帧定义 |
| defcal 定义 | ✅ Phase 4 | 波形和脉冲指令 |

**章节来源**
- [src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py](file://src/tyxonq/compiler/pulse_compile_engine/native/qasm3_importer.py#L1-L396)

## 驱动层执行

### 自动检测机制

驱动层具备智能的 QASM3 版本检测能力，能够自动识别和处理不同格式的量子程序：

```mermaid
flowchart TD
A[输入源码] --> B{检测格式}
B --> |OPENQASM 3.0| C[标准 OpenQASM 3.0]
B --> |TQASM 0.2| D[TyxonQ 特定格式]
B --> |QASM 2.0| E[传统 QASM 2.0]
B --> |其他| F[未知格式]
C --> G[调用 qasm3_to_circuit]
D --> G
E --> H[调用 qasm_to_ir]
F --> I[抛出错误]
G --> J[返回 Circuit IR]
H --> J
I --> J
```

**图表来源**
- [src/tyxonq/devices/simulators/driver.py](file://src/tyxonq/devices/simulators/driver.py#L50-L100)

### 执行流程

```mermaid
sequenceDiagram
participant User as 用户
participant Driver as 模拟器驱动
participant Engine as 仿真引擎
participant Result as 结果处理器
User->>Driver : run(device, source, shots)
Driver->>Driver : _qasm_to_ir_if_needed()
Driver->>Engine : 选择合适的引擎
Engine->>Engine : 执行量子电路
Engine->>Result : 生成结果
Result-->>User : 返回执行结果
```

**图表来源**
- [src/tyxonq/devices/simulators/driver.py](file://src/tyxonq/devices/simulators/driver.py#L100-L167)

**章节来源**
- [src/tyxonq/devices/simulators/driver.py](file://src/tyxonq/devices/simulators/driver.py#L1-L168)

## 最佳实践与优化

### 1. 编译器选择策略

根据具体需求选择合适的编译器：

- **开发调试**：使用 `output="ir"` 获取原始 IR
- **标准互操作**：使用 `output="qasm3"` 保证兼容性
- **硬件就绪**：使用 `output="tyxonq_homebrew_tqasm"` 准备硬件执行
- **Qiskit 生态**：使用 `output="qiskit"` 集成现有工具链

### 2. 性能优化建议

- **批量处理**：对多个电路使用批量执行
- **缓存机制**：利用编译结果缓存避免重复编译
- **内存管理**：合理设置模拟器参数优化内存使用

### 3. 错误处理策略

```mermaid
flowchart TD
A[编译请求] --> B{检查输入}
B --> |有效| C[执行编译]
B --> |无效| D[抛出 ValueError]
C --> E{编译成功?}
E --> |成功| F[返回 CompileResult]
E --> |失败| G[捕获异常]
D --> H[错误日志]
G --> H
H --> I[用户反馈]
```

## 故障排除指南

### 常见问题及解决方案

| 问题类型 | 症状 | 解决方案 |
|---------|------|----------|
| 编译失败 | `RuntimeError: qiskit not available` | 安装 qiskit 库或使用 native 编译器 |
| 格式不匹配 | `ValueError: Cannot parse angle parameter` | 检查 QASM3 语法兼容性 |
| 硬件不支持 | `Unsupported provider` | 确认硬件提供商可用性 |
| 内存不足 | `MemoryError` | 减少量子比特数或优化电路 |

### 调试技巧

1. **分步验证**：逐个检查编译、解析、执行各环节
2. **格式检查**：验证 QASM3 语法的正确性
3. **元数据检查**：确认 Circuit.metadata 包含预期信息
4. **版本兼容**：确保使用的 QASM3 版本与硬件兼容

### 性能监控

```mermaid
graph LR
A[编译开始] --> B[记录时间戳]
B --> C[执行编译]
C --> D[记录结束时间]
D --> E[计算耗时]
E --> F[性能报告]
G[内存使用] --> H[监控峰值]
H --> I[内存报告]
F --> J[综合评估]
I --> J
```

通过遵循这些最佳实践和故障排除指南，用户可以充分利用 TyxonQ 的 QASM3 工作流程，实现高效的量子电路开发和硬件执行。