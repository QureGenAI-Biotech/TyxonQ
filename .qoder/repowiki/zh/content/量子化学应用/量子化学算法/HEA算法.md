# HEA算法

<cite>
**本文档引用的文件**   
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py) - *在最近的提交中进行了性能增强，并支持GPU服务器和修复了开壳层bug，新增HOMO-LUMO能隙计算功能*
- [hea_device_runtime.py](file://src/tyxonq/applications/chem/runtimes/hea_device_runtime.py) - *更新了性能*
- [hea_numeric_runtime.py](file://src/tyxonq/applications/chem/runtimes/hea_numeric_runtime.py) - *更新了性能*
- [blocks.py](file://src/tyxonq/libs/circuits_library/blocks.py) - *构建RY-only电路*
- [qiskit_real_amplitudes.py](file://src/tyxonq/libs/circuits_library/qiskit_real_amplitudes.py) - *Qiskit电路转换*
- [hamiltonian_builders.py](file://src/tyxonq/applications/chem/chem_libs/hamiltonians_chem_library/hamiltonian_builders.py) - *哈密顿量构建*
- [hamiltonian_grouping.py](file://src/tyxonq/compiler/utils/hamiltonian_grouping.py) - *哈密顿量分组*
- [demo_homo_lumo_gap.py](file://examples/demo_homo_lumo_gap.py) - *HOMO-LUMO能隙计算演示*
</cite>

## 更新摘要
**变更内容**   
- 新增HOMO-LUMO能隙计算功能，通过`get_homo_lumo_gap()`方法和`homo_lumo_gap`属性提供
- 更新了`kernel()`方法中关于shots=0时的默认行为，以避免采样噪声
- 增强了HEA算法在numeric和device运行时路径下的性能
- 更新了能量评估和梯度计算的实现细节
- 修正了RDM计算中对shots参数的使用
- 更新了与Qiskit电路兼容性转换的实现
- 新增对GPU服务器的支持
- 修复了开壳层分子计算中的bug

## 目录
1. [引言](#引言)
2. [核心电路设计](#核心电路设计)
3. [能量评估与梯度计算](#能量评估与梯度计算)
4. [参数化电路构建](#参数化电路构建)
5. [优化与收敛](#优化与收敛)
6. [密度矩阵计算](#密度矩阵计算)
7. [与Qiskit的兼容性](#与qiskit的兼容性)
8. [性能分析](#性能分析)
9. [常见问题与解决方案](#常见问题与解决方案)
10. [HOMO-LUMO能隙计算](#homo-lumo能隙计算)
11. [结论](#结论)

## 引言

硬件高效变分量子电路（Hardware-Efficient Ansatz, HEA）是一种专为当前含噪声中等规模量子（NISQ）设备设计的参数化量子电路。该算法通过RY-only结构和CNOT链构成的参数化电路，实现了在变分量子本征求解器（VQE）中的高效应用。HEA算法支持从哈密顿量项列表或分子积分构建电路，并在device与numeric两种运行时路径下进行能量评估与参数移位梯度计算。

**Section sources**
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L1-L50)

## 核心电路设计

HEA算法的核心电路设计采用交替的单比特旋转与纠缠层（CNOT链）构成参数化电路。具体结构如下：

- **初始层**：逐比特RY(θ0,i)旋转
- **每层l=1..L**：CNOT链(0→1→...→n-1) + 逐比特RY(θl,i)
- 层间插入barrier指令，便于可视化与编译边界控制

电路参数总数为(layers + 1) * n_qubits。该设计平衡了表达能力与硬件可行性，特别适合当前量子设备的拓扑约束。

**Section sources**
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L44-L123)
- [blocks.py](file://src/tyxonq/libs/circuits_library/blocks.py#L58-L81)

## 能量评估与梯度计算

HEA算法提供两种运行时路径进行能量评估：

### Device路径
基于计数的能量评估，内部对哈密顿量进行按基分组测量：
1. 对每个基组应用基变换（X→H，Y→S†H）
2. 执行Z基测量
3. 从计数中估计<H>

### Numeric路径
当runtime="numeric"或shots=0时，使用数值模拟器进行解析计算，避免采样噪声。

梯度计算采用参数移位法，对每个参数θ_k使用标准移位s=π/2：
g_k = 0.5 * (E(θ_k+s) - E(θ_k-s))

**Section sources**
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L147-L182)
- [hea_device_runtime.py](file://src/tyxonq/applications/chem/runtimes/hea_device_runtime.py#L20-L145)
- [hea_numeric_runtime.py](file://src/tyxonq/applications/chem/runtimes/hea_numeric_runtime.py#L14-L97)

## 参数化电路构建

HEA算法支持多种方式构建参数化电路：

### 从分子积分构建
使用`from_integral`接口：
1. 由(int1e, int2e)构造费米子算符H_f
2. 按映射（parity/JW/BK）将H_f → QubitOperator
3. 转为轻量哈密顿量列表并实例化HEA

### 从分子对象构建
使用`from_molecule`接口自动运行RHF得到积分与e_core，然后复用from_integral流程。

### 与Qiskit电路兼容
通过`from_qiskit_circuit`接口支持Qiskit RealAmplitudes电路的转换。

**Section sources**
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L296-L410)
- [hamiltonian_builders.py](file://src/tyxonq/applications/chem/chem_libs/hamiltonians_chem_library/hamiltonian_builders.py#L67-L104)

## 优化与收敛

`kernel()`方法执行L-BFGS-B优化，具有以下特性：
- 当runtime="device"且provider为simulator/local时，默认shots=0以避免采样噪声
- 真机运行时默认使用2048 shots
- 使用确定性非平凡初始猜测(init_guess)避免梯度平坦问题
- 支持GPU服务器加速计算
- 修复了开壳层分子的计算bug

优化器选项可通过scipy_minimize_options配置。

**Section sources**
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L210-L253)

## 密度矩阵计算

HEA算法支持计算自旋约化的一体与二体约化密度矩阵：

### 一体RDM (make_rdm1)
计算自旋约化的一体RDM（spin-traced 1RDM），需要在from_integral/from_molecule构建后使用。

### 二体RDM (make_rdm2)
计算自旋约化的二体RDM（spin-traced 2RDM），同样需要在from_integral/from_molecule构建后使用。

**Section sources**
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L519-L601)

## 与Qiskit的兼容性

HEA算法通过`from_qiskit_circuit`接口支持Qiskit RealAmplitudes电路的转换。该接口将Qiskit QuantumCircuit转换为内部参数化模板，实现与Qiskit生态的兼容。

**Section sources**
- [qiskit_real_amplitudes.py](file://src/tyxonq/libs/circuits_library/qiskit_real_amplitudes.py#L7-L73)

## 性能分析

在模拟器与真实硬件上的性能对比表明，HEA算法在VQE中表现出高效性。通过GPU服务器支持，显著加速了数值计算路径的执行。

## 常见问题与解决方案

- **优化收敛慢**：检查初始猜测是否合理，考虑调整优化器参数。
- **采样噪声干扰**：在优化过程中使用shots=0以避免采样噪声影响。

## HOMO-LUMO能隙计算

HEA算法新增了HOMO-LUMO能隙计算功能，通过`get_homo_lumo_gap()`方法和`homo_lumo_gap`属性提供。该功能委托给内部的UCC对象进行化学相关计算，利用Hartree-Fock计算结果进行HOMO-LUMO分析。

### 主要特性
- **自动确定**：自动根据分子系统（闭壳或开壳）确定HOMO和LUMO轨道
- **手动指定**：支持手动指定HOMO和LUMO轨道索引
- **单位转换**：可选择包含eV单位的转换结果
- **系统类型识别**：自动识别闭壳或开壳系统

### 使用方法
```python
from tyxonq.chem import HEA
from tyxonq.chem.molecule import h2

# 创建HEA实例
hea = HEA(molecule=h2, layers=1)

# 获取详细能隙信息
gap_info = hea.get_homo_lumo_gap()
print(f"HOMO-LUMO gap: {gap_info['gap']:.6f} Hartree")

# 包含eV转换
gap_info = hea.get_homo_lumo_gap(include_ev=True)
print(f"HOMO-LUMO gap: {gap_info['gap_ev']:.6f} eV")

# 使用属性快速访问
gap = hea.homo_lumo_gap
print(f"Gap: {gap:.6f} Hartree")
```

### 注意事项
- 仅适用于通过`from_molecule()`或直接分子输入构建的HEA实例
- 对于直接从积分构建的HEA，无法进行HOMO-LUMO能隙计算
- 需要分子构建时提供完整的化学元数据

**Section sources**
- [hea.py](file://src/tyxonq/applications/chem/algorithms/hea.py#L729-L815)
- [demo_homo_lumo_gap.py](file://examples/demo_homo_lumo_gap.py#L1-L201)

## 结论

HEA算法作为一种硬件高效的变分量子电路，为NISQ时代的量子化学计算提供了有效的解决方案。其灵活的构建方式和高效的优化路径使其成为VQE应用中的重要工具。新增的HOMO-LUMO能隙计算功能进一步增强了其在量子化学分析中的实用性。