# Qiskit编译器

<cite>
**Referenced Files in This Document**   
- [qiskit_compiler.py](file://src/tyxonq/compiler/compile_engine/qiskit/qiskit_compiler.py)
- [dialect.py](file://src/tyxonq/compiler/compile_engine/qiskit/dialect.py)
- [api.py](file://src/tyxonq/compiler/api.py)
- [circuit.py](file://src/tyxonq/core/ir/circuit.py)
</cite>

## 目录
1. [简介](#简介)
2. [核心组件](#核心组件)
3. [架构概述](#架构概述)
4. [详细组件分析](#详细组件分析)
5. [依赖分析](#依赖分析)
6. [性能考量](#性能考量)
7. [故障排除指南](#故障排除指南)
8. [结论](#结论)

## 简介

TyxonQ Qiskit编译器（QiskitCompiler）是TyxonQ框架与外部量子硬件（特别是IBM Quantum设备）集成的关键桥梁。该编译器的核心职责是将TyxonQ内部的中间表示（IR）电路转换为Qiskit QuantumCircuit对象，从而利用Qiskit强大的transpile功能，实现针对特定量子硬件的门集映射、拓扑适配和深度优化。本文档将深入剖析其设计与实现，重点阐述其编译流程、关键功能和与原生编译器的差异。

## 核心组件

Qiskit编译器的核心功能由`QiskitCompiler`类实现，其`compile`方法是整个编译流程的入口。该方法通过`ir_to_qiskit`函数将TyxonQ的IR转换为Qiskit对象，并根据`transpile`参数决定是否触发Qiskit的优化流程。编译过程产生的`logical_physical_mapping`和`positional_logical_mapping`元数据对于结果解析至关重要。

**Section sources**
- [qiskit_compiler.py](file://src/tyxonq/compiler/compile_engine/qiskit/qiskit_compiler.py#L19-L76)

## 架构概述

Qiskit编译器作为TyxonQ编译引擎的一部分，位于核心IR与外部硬件之间。它通过`compiler/api.py`提供的统一入口被调用，将TyxonQ的`Circuit`对象作为输入，最终输出一个包含编译后电路和元数据的`CompileResult`。其主要依赖于`dialect.py`中定义的转换函数和辅助工具。

```mermaid
graph TB
subgraph "TyxonQ Core"
IR[Circuit IR]
end
subgraph "Compiler Layer"
API[compiler/api.py]
QiskitCompiler[qiskit_compiler.py]
Dialect[dialect.py]
end
subgraph "External"
Qiskit[Qiskit QuantumCircuit]
Hardware[Quantum Hardware]
end
IR --> API
API --> QiskitCompiler
QiskitCompiler --> Dialect
Dialect --> Qiskit
Qiskit --> Hardware
```

**Diagram sources**
- [qiskit_compiler.py](file://src/tyxonq/compiler/compile_engine/qiskit/qiskit_compiler.py#L19-L76)
- [dialect.py](file://src/tyxonq/compiler/compile_engine/qiskit/dialect.py#L26-L244)
- [api.py](file://src/tyxonq/compiler/api.py#L23-L62)

## 详细组件分析

### QiskitCompiler类分析

`QiskitCompiler`类是Qiskit编译功能的封装。其`compile`方法是核心，负责协调整个编译流程。

#### 编译流程分析
```mermaid
flowchart TD
Start([开始编译]) --> GetOptions["获取编译选项\noutput, add_measures,\ndo_transpile, norm_opts"]
GetOptions --> CheckQiskit{"Qiskit可用?"}
CheckQiskit --> |是| ImportQiskit["导入Qiskit模块\nQuantumCircuit,\ntranspile"]
CheckQiskit --> |否| SetNone["QuantumCircuit = None"]
ImportQiskit --> ConvertIR["调用 ir_to_qiskit\n将IR转换为Qiskit电路"]
SetNone --> ConvertIR
ConvertIR --> CheckTranspile{"do_transpile=True?"}
CheckTranspile --> |是| FilterOpts["过滤编译选项\ntp_opts = {k: v for k, v in norm_opts.items() if k not in ('output', 'transpile', ...)}"]
CheckTranspile --> |否| SetCompiledQC["compiled_qc = qc"]
FilterOpts --> Transpile["调用 qk_transpile(qc, **tp_opts)\n执行硬件适配优化"]
Transpile --> GetMappings["调用 _get_logical_physical_mapping_from_qiskit\n和 _get_positional_logical_mapping_from_qiskit\n获取映射元数据"]
SetCompiledQC --> GetMappings
GetMappings --> BuildMetadata["构建元数据字典\n包含 output, options,\nlogical_physical_mapping等"]
BuildMetadata --> CheckOutput{"output == 'qiskit'?"}
CheckOutput --> |是| ReturnQiskit["返回 {'circuit': compiled_qc, 'metadata': metadata}"]
CheckOutput --> |否| CheckQASM{"output in ('qasm', 'qasm2')?"}
CheckQASM --> |是| CallQasm2["调用 qasm2_dumps_compat(compiled_qc)\n生成QASM字符串"]
CheckQASM --> |否| CheckIR{"output == 'ir'?"}
CheckIR --> |是| ReturnIR["返回 {'circuit': circuit, 'metadata': metadata}"]
CheckIR --> |否| ReturnDefault["返回 {'circuit': compiled_qc, 'metadata': metadata}"]
CallQasm2 --> ReturnQASM["返回 {'circuit': qasm_string, 'metadata': metadata}"]
```

**Diagram sources**
- [qiskit_compiler.py](file://src/tyxonq/compiler/compile_engine/qiskit/qiskit_compiler.py#L19-L76)

**Section sources**
- [qiskit_compiler.py](file://src/tyxonq/compiler/compile_engine/qiskit/qiskit_compiler.py#L19-L76)

#### IR到Qiskit的转换
`ir_to_qiskit`函数是实现IR到Qiskit转换的关键。它遍历TyxonQ IR中的操作列表，并将其映射为对应的Qiskit门操作。

```mermaid
classDiagram
class ir_to_qiskit {
+circuit : Circuit
+add_measures : bool
+return : QuantumCircuit
+__init__(circuit : Circuit, add_measures : bool)
+convert_ops() : void
+add_measurements() : void
}
class Circuit {
+num_qubits : int
+ops : List[Tuple[str, ...]]
}
class QuantumCircuit {
+h(qubit)
+rx(theta, qubit)
+rz(theta, qubit)
+cx(control, target)
+measure_all()
+add_register(creg)
}
ir_to_qiskit --> Circuit : "输入"
ir_to_qiskit --> QuantumCircuit : "输出"
```

**Diagram sources**
- [dialect.py](file://src/tyxonq/compiler/compile_engine/qiskit/dialect.py#L200-L240)

#### 编译选项标准化
`normalize_transpile_options`函数负责过滤和标准化传递给Qiskit的编译选项，确保参数兼容性。

```mermaid
flowchart TD
Start([开始标准化]) --> InitNorm["初始化 norm 字典"]
InitNorm --> UpdateNorm["将 options 内容更新到 norm"]
UpdateNorm --> CheckOptLevel{"'opt_level' in norm 且 'optimization_level' not in norm?"}
CheckOptLevel --> |是| ConvertOptLevel["尝试将 norm['opt_level'] 转换为整数\n并赋值给 norm['optimization_level']\n然后删除 'opt_level'"]
CheckOptLevel --> |否| SkipOptLevel["跳过"]
ConvertOptLevel --> CheckBasisGates{"norm.get('basis_gates') 为空?"}
SkipOptLevel --> CheckBasisGates
CheckBasisGates --> |是| SetDefaultBasis["norm['basis_gates'] = DEFAULT_BASIS_GATES"]
CheckBasisGates --> |否| SkipBasisGates["跳过"]
SetDefaultBasis --> SetDefaultOpt["norm.setdefault('optimization_level', DEFAULT_OPT_LEVEL)"]
SkipBasisGates --> SetDefaultOpt
SetDefaultOpt --> ReturnNorm["返回 norm 字典"]
```

**Diagram sources**
- [dialect.py](file://src/tyxonq/compiler/compile_engine/qiskit/dialect.py#L26-L39)

### 元数据映射分析

编译后返回的`logical_physical_mapping`和`positional_logical_mapping`是解析硬件执行结果的关键。

#### 逻辑到物理量子比特映射
`_get_logical_physical_mapping_from_qiskit`函数通过比较编译前后的测量指令，建立逻辑量子比特到物理量子比特的映射关系。

```mermaid
sequenceDiagram
participant Compiler
participant QCAfter
participant QCBefore
Compiler->>QCAfter : 遍历编译后电路的指令
loop 每个指令
QCAfter-->>Compiler : 获取操作、量子比特、经典比特
alt 是测量操作
Compiler->>QCBefore : 在编译前电路中查找对应测量
QCBefore-->>Compiler : 找到匹配的测量指令
Compiler->>Compiler : 获取编译前逻辑量子比特索引
Compiler->>Compiler : 获取编译后物理量子比特索引
Compiler->>Compiler : 建立映射 logical_q -> physical_q
end
end
Compiler-->>Compiler : 返回映射字典
```

**Diagram sources**
- [dialect.py](file://src/tyxonq/compiler/compile_engine/qiskit/dialect.py#L95-L121)

#### 位置到逻辑量子比特映射
`_get_positional_logical_mapping_from_qiskit`函数根据测量指令在电路中的位置，建立测量结果位置到逻辑量子比特的映射。

```mermaid
flowchart TD
Start([开始]) --> Init["初始化 i=0, positional_logical_mapping={}\n遍历电路指令"]
Init --> Loop{"指令是测量?"}
Loop --> |是| GetIndex["获取测量的量子比特\npositional_logical_mapping[i] = qubit_index"]
GetIndex --> Inc["i += 1"]
Inc --> Loop
Loop --> |否| Next["下一个指令"]
Next --> Loop
Loop --> |遍历结束| Return["返回 positional_logical_mapping"]
```

**Diagram sources**
- [dialect.py](file://src/tyxonq/compiler/compile_engine/qiskit/dialect.py#L79-L92)

## 依赖分析

Qiskit编译器对Qiskit库有可选依赖。在`compile`方法中，通过`try...except`块尝试导入Qiskit模块。如果导入失败，会抛出`RuntimeError`异常，提示用户Qiskit不可用。

```mermaid
graph TD
QiskitCompiler --> TryImport{"尝试导入\nfrom qiskit import QuantumCircuit\nfrom qiskit.compiler import transpile"}
TryImport --> |成功| Proceed[继续编译流程]
TryImport --> |失败| RaiseError["抛出 RuntimeError\n'qiskit not available: {exc}'"]
```

**Diagram sources**
- [qiskit_compiler.py](file://src/tyxonq/compiler/compile_engine/qiskit/qiskit_compiler.py#L30-L37)

## 性能考量

Qiskit编译器的性能主要受Qiskit `transpile`函数的影响。`transpile`过程会根据指定的`optimization_level`进行不同程度的优化，级别越高，优化越彻底，但耗时也越长。用户可以通过`options`参数中的`optimization_level`或`opt_level`来控制优化级别。此外，`basis_gates`的设置也会影响门集映射的效率。

## 故障排除指南

### Qiskit依赖缺失
当系统中未安装Qiskit时，调用`QiskitCompiler.compile`会抛出`RuntimeError`。解决方法是安装Qiskit：
```bash
pip install qiskit
```

### 不支持的操作
`ir_to_qiskit`函数目前仅支持`h`, `rx`, `rz`, `cx`, `measure_z`等基本操作。如果IR中包含不支持的操作，会抛出`NotImplementedError`。需要确保电路中的操作在支持列表内。

### 测量指令问题
如果`add_measures`选项为`True`，但IR中没有`measure_z`操作，编译器会自动调用`qc.measure_all()`添加测量。如果用户希望精确控制测量位置，应在IR中显式添加`measure_z`操作。

**Section sources**
- [qiskit_compiler.py](file://src/tyxonq/compiler/compile_engine/qiskit/qiskit_compiler.py#L30-L37)
- [dialect.py](file://src/tyxonq/compiler/compile_engine/qiskit/dialect.py#L200-L240)

## 结论

TyxonQ Qiskit编译器是一个功能强大且设计精巧的桥梁，它成功地将TyxonQ的IR与Qiskit生态系统连接起来。通过`ir_to_qiskit`和`transpile`机制，它能够将高级量子电路编译并优化为可在真实硬件上执行的指令。其返回的`logical_physical_mapping`和`positional_logical_mapping`元数据为结果解析提供了必要的信息。与原生编译器相比，Qiskit编译器专注于硬件执行，输出Qiskit对象或QASM，而原生编译器则侧重于内部优化，输出TyxonQ IR。这种设计使得TyxonQ框架既能进行高效的内部优化，又能无缝对接外部量子硬件。