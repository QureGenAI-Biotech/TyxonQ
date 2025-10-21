# TyxonQ Examples

本目录包含TyxonQ量子计算框架的示例代码，展示各种量子算法、优化方法和高级特性。

## 📚 目录

- [快速入门](#快速入门)
- [基础示例](#基础示例)
- [变分量子算法](#变分量子算法)
- [量子模拟](#量子模拟)
- [高级特性](#高级特性)
- [性能优化](#性能优化)
- [云计算接口](#云计算接口)
- [运行示例](#运行示例)

---

## 快速入门

### 最简单的示例

```bash
# 基础链式API
python examples/basic_chain_api.py

# 简单的QAOA算法
python examples/simple_qaoa.py
```

---

## 基础示例

### 1. Circuit链式API
**文件**: [`basic_chain_api.py`](basic_chain_api.py), [`circuit_chain_demo.py`](circuit_chain_demo.py)

展示TyxonQ的链式API设计，如何构建量子电路：

```python
import tyxonq as tq

# 创建电路并链式添加门
c = tq.Circuit(3)
c.h(0).cx(0, 1).cx(1, 2)

# 获取状态向量
state = c.state()
```

**关键特性**:
- 方法链式调用
- 多种单/双量子比特门
- 状态向量获取

---

### 2. 数值后端切换
**文件**: [`numeric_backend_switching.py`](numeric_backend_switching.py)

演示如何在NumPy、PyTorch、JAX等后端之间切换：

```python
import tyxonq as tq

# 使用NumPy后端
tq.set_backend("numpy")
c1 = tq.Circuit(10)
# ... 电路操作 ...

# 切换到PyTorch后端（支持GPU加速）
tq.set_backend("pytorch")
c2 = tq.Circuit(10)
# ... 电路操作 ...
```

**支持的后端**:
- `numpy` - CPU计算，适合小规模电路
- `pytorch` - 支持GPU加速和自动微分
- `jax` - JIT编译和函数式编程
- `cupy` - GPU加速的NumPy替代

---

### 3. 自动微分与梯度计算
**文件**: [`autograd_vs_counts.py`](autograd_vs_counts.py), [`gradient_benchmark.py`](gradient_benchmark.py)

对比不同梯度计算方法的性能：

- **自动微分（Autograd）**: 精确梯度，速度快
- **参数平移法（Parameter Shift）**: 量子硬件兼容
- **有限差分（Finite Difference）**: 数值梯度，通用但慢

```python
import tyxonq as tq
import torch

tq.set_backend("pytorch")

# 定义变分电路
def circuit_energy(theta):
    c = tq.Circuit(4)
    for i in range(4):
        c.rx(i, theta[i])
    c.cx(0, 1).cx(1, 2).cx(2, 3)
    # ... 计算期望值 ...
    return energy

# PyTorch自动微分
theta = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
energy = circuit_energy(theta)
energy.backward()
grad = theta.grad  # 梯度
```

---

## 变分量子算法

### 4. VQE (Variational Quantum Eigensolver)

**文件**: 
- [`vqe_simple_hamiltonian.py`](vqe_simple_hamiltonian.py) - 简单哈密顿量
- [`vqe_extra.py`](vqe_extra.py) - 完整VQE流程
- [`vqetfim_benchmark.py`](vqetfim_benchmark.py) - 横场Ising模型
- [`vqeh2o_benchmark.py`](vqeh2o_benchmark.py) - 水分子基态能量
- [`vqe_parallel_pmap.py`](vqe_parallel_pmap.py) - 并行优化
- [`vqe_shot_noise.py`](vqe_shot_noise.py) - 考虑采样噪声
- [`vqe_noisyopt.py`](vqe_noisyopt.py) - 噪声环境优化

**功能**:
- 求解分子/材料的基态能量
- 多种Ansatz: HEA, UCCSD等
- PyTorch/JAX优化器集成
- 并行参数搜索

**示例代码**:
```python
import tyxonq as tq
from tyxonq.libs.quantum_library.kernels.gates import gate_x, gate_z

tq.set_backend("numpy")

# 定义横场Ising模型哈密顿量: H = -Σ_i Z_i Z_{i+1} - h Σ_i X_i
n = 6
h_field = 1.0

def compute_energy(circuit):
    energy = 0.0
    # ZZ coupling
    for i in range(n-1):
        energy += circuit.expectation((gate_z(), [i]), (gate_z(), [i+1]))
    # Transverse field
    for i in range(n):
        energy -= h_field * circuit.expectation((gate_x(), [i]))
    return energy

# VQE优化...
```

---

### 5. QAOA (Quantum Approximate Optimization Algorithm)
**文件**: [`simple_qaoa.py`](simple_qaoa.py), [`cloud_api_task_qaoa.py`](cloud_api_task_qaoa.py)

求解组合优化问题（如MaxCut）：

```python
import tyxonq as tq

# MaxCut问题图
edges = [(0,1), (1,2), (2,3), (3,0)]

def qaoa_circuit(beta, gamma, p=2):
    c = tq.Circuit(4)
    # 初始化叠加态
    for i in range(4):
        c.h(i)
    
    # QAOA layers
    for layer in range(p):
        # Cost Hamiltonian
        for i, j in edges:
            c.cx(i, j).rz(j, 2*gamma[layer]).cx(i, j)
        # Mixer Hamiltonian  
        for i in range(4):
            c.rx(i, 2*beta[layer])
    
    return c

# 优化beta, gamma参数...
```

---

### 6. VQE内存扩展性分析
**文件**: [`vqe_memory_scaling_demo.py`](vqe_memory_scaling_demo.py)

展示VQE在不同规模下的内存使用情况和优化策略：

**关键特性**:
- 内存使用分析: 状态向量随量子比特数指数增长
- Pauli字符串表示: 高效的哈密顿量编码
- 梯度演示: PyTorch自动微分集成
- 性能对比: 不同后端的内存和速度权衡

**示例代码**:
```python
import tyxonq as tq
import torch

tq.set_backend("pytorch")

# 内存分析
for n_qubits in [4, 6, 8, 10, 12]:
    memory_per_state = (2 ** n_qubits) * 16  # bytes (complex128)
    print(f"{n_qubits} qubits: {memory_per_state / 1024**2:.2f} MB")

# Pauli字符串哈密顿量
pauli_terms = [
    [3, 3, 0, 0],  # ZZ on qubits 0,1
    [0, 3, 3, 0],  # ZZ on qubits 1,2
]
weights = [-1.0, -1.0]
```

---

### 7. VQE深度可调结构训练
**文件**: [`vqe_deep_structures_training.py`](vqe_deep_structures_training.py)

展示如何使用可调量子门结构进行深度变分学习：

**关键特性**:
- 可调gate: `unitary_kraus` 支持多种门类型的随机选择
- 深度电路: 演示如何避免梯度消失
- 结构学习: 优化电路结构和参数
- Heisenberg模型: 使用Trotter分解

**示例代码**:
```python
import tyxonq as tq
import torch

tq.set_backend("pytorch")

def build_circuit(params, structure_choices, n_qubits, depth):
    c = tq.Circuit(n_qubits)
    for layer in range(depth):
        for i in range(n_qubits):
            # 可调gate: 根据structure_choices选择Rx/Ry/Rz
            c.unitary_kraus(
                [tq.gates.rx(params[layer, i]),
                 tq.gates.ry(params[layer, i]),
                 tq.gates.rz(params[layer, i])],
                i,
                prob=[1/3, 1/3, 1/3],
                status=structure_choices[layer, i]
            )
        # 纠缠层
        for i in range(n_qubits-1):
            c.cx(i, i+1)
    return c
```

---

### 8. VQE 2D格点系统
**文件**: [`vqe_2d_lattice.py`](vqe_2d_lattice.py)

在2D正方格点上求解量子多体系统：

**关键特性**:
- Grid2D坐标系统: 管理2D拓扑结构
- 最近邻交互: 自动生成格点上的耦合
- SWAP网络: 处理非临近量子比特的纠缠
- 周期边界条件: 支持环面拓扑

**示例代码**:
```python
import tyxonq as tq
import torch

class Grid2D:
    """2D square lattice coordinate system."""
    def __init__(self, n_rows, n_cols, periodic=False):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_qubits = n_rows * n_cols
        self.periodic = periodic
    
    def nearest_neighbors(self):
        """Get all nearest-neighbor pairs."""
        pairs = []
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                idx = row * self.n_cols + col
                # Right neighbor
                if col < self.n_cols - 1 or self.periodic:
                    right = row * self.n_cols + (col + 1) % self.n_cols
                    pairs.append((idx, right))
                # Down neighbor
                if row < self.n_rows - 1 or self.periodic:
                    down = ((row + 1) % self.n_rows) * self.n_cols + col
                    pairs.append((idx, down))
        return pairs

# 2x2格点VQE
grid = Grid2D(2, 2, periodic=False)
for i, j in grid.nearest_neighbors():
    # 添加ZZ相互作用
    energy += circuit.expectation_ps(z=[i, j])
```

---

### 9. 噪声量子机器学习（Noisy QML）
**文件**: [`noisy_quantum_machine_learning.py`](noisy_quantum_machine_learning.py)

在NISQ时代噪声环境下训练量子机器学习模型：

**关键特性**:
- MNIST手写数字二分类（0 vs 1）
- 真实硬件噪声模拟（去极化噪声）
- PyTorch后端自动微分
- 参数化量子电路（PQC）作为特征提取器

**示例代码**:
```python
import tyxonq as tq
import torch

tq.set_backend("pytorch")

# 定义参数化量子电路
def build_pqc(x, params, noise_level=0.005):
    c = tq.Circuit(9)  # 9量子比特（3x3图像）
    
    # 数据编码层
    for i in range(9):
        c.rx(i, theta=x[i] * np.pi / 2)
    
    # 变分层
    for layer in range(4):
        # 纠缠门
        for i in range(8):
            c.cnot(i, i + 1)
        # 参数化旋转
        for i in range(9):
            c.rz(i, theta=params[layer, i, 0])
            c.rx(i, theta=params[layer, i, 1])
    
    # 应用噪声
    if noise_level > 0:
        c = c.with_noise("depolarizing", p=noise_level)
    
    # 测量期望值
    return c.expectation_ps(z=list(range(9)))

# 使用PyTorch优化器训练
params = torch.nn.Parameter(torch.randn(4, 9, 2))
optimizer = torch.optim.Adam([params], lr=0.01)

# 训练循环
for epoch in range(100):
    loss = binary_cross_entropy(build_pqc(x_train, params, 0.005), y_train)
    loss.backward()
    optimizer.step()
```

**预期结果**:
- 训练准确率: ~85-95%（有噪声）
- 训练准确率: ~95-99%（理想情况）
- 展示NISQ算法对噪声的鲁棒性

**依赖**:
- PyTorch
- torchvision（用于MNIST数据集）
- ~2GB RAM

---

### 10. MERA变分算法
**文件**: [`vqe_mera_mpo.py`](vqe_mera_mpo.py)

使用多尺度纠缠重整化Ansatz（MERA）求解强关联系统：

**关键特性**:
- MERA张量网络结构
- 分层纠缠模式
- MPO（矩阵乘积算子）期望值计算
- 适合量子多体系统

**示例代码**:
```python
import tyxonq as tq
import torch

tq.set_backend("pytorch")

def mera_layer(c, params, layer_idx, n_qubits):
    """Single MERA layer: disentanglers + isometries."""
    # Disentanglers (unitary gates)
    for i in range(0, n_qubits-1, 2):
        c.rxx(i, i+1, theta=params[layer_idx, i, 0])
        c.ryy(i, i+1, theta=params[layer_idx, i, 1])
        c.rzz(i, i+1, theta=params[layer_idx, i, 2])
    return c

# Build MERA ansatz
n_qubits = 8
n_layers = 3
params = torch.randn(n_layers, n_qubits, 3, requires_grad=True)

c = tq.Circuit(n_qubits)
for layer in range(n_layers):
    c = mera_layer(c, params, layer, n_qubits)

# Compute energy using direct expectation
energy = c.expectation((gate_z(), [0]), (gate_z(), [1]))
energy.backward()
```

**优势**:
- 比HEA更适合强关联系统
- 分层结构降低电路深度
- 高效编码长程纠缠

**运行**:
```bash
python examples/vqe_mera_mpo.py
```

---

### 11. Barren Plateau基准测试
**文件**: [`barren_plateau_benchmark.py`](barren_plateau_benchmark.py)

研究量子神经网络中的梯度消失现象（Barren Plateau）：

**核心创新**:
这个示例实现了**三种不同的梯度计算方法**，对比理想理论与硬件现实：

**方法A: 理想Autograd（PyTorch自动微分）**
- 假设直接访问量子态和精确导数
- ❌ **不可在真实量子硬件实现**
- 作为理论基准进行对比
- 最快速、最准确，无shot噪声

**方法B: Parameter Shift规则（硬件可实现，shot-based）**
- 使用parameter shift规则: ∂⟨H⟩/∂θ = [⟨H⟩(θ+π/4) - ⟨H⟩(θ-π/4)]/2
- 模拟有限测量shots（采样噪声）
- ✅ **完全可在真实量子设备上实现**
- TyxonQ链式API: `circuit.device(shots=1024).run()`

**方法C: Parameter Shift + 噪声（最真实场景）**
- 包含硬件噪声：去极化误差(p=0.001)
- 使用TyxonQ的`.with_noise()`链式API
- 模拟NISQ时代量子处理器的真实环境

**示例代码**:
```python
import tyxonq as tq
import torch
import numpy as np

tq.set_backend("pytorch")

# 方法A: 理想Autograd
def gradient_autograd(circuit_fn, params):
    theta = torch.tensor(params, requires_grad=True)
    energy = circuit_fn(theta)
    energy.backward()
    return theta.grad

# 方法B: Parameter Shift (shot-based)
def gradient_parameter_shift(circuit_fn, params, shots=1024):
    shift = np.pi / 4
    grads = []
    
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += shift
        
        # Shot-based measurement
        c_plus = circuit_fn(params_plus)
        c_plus.measure_z(0).measure_z(1)
        counts_plus = c_plus.device(shots=shots).run()
        
        # Compute expectation from counts
        exp_plus = compute_expectation_from_counts(counts_plus)
        
        # 同样处理 theta - shift
        # ...
        
        grad = (exp_plus - exp_minus) / 2
        grads.append(grad)
    
    return grads

# 方法C: Parameter Shift + Noise
def gradient_with_noise(circuit_fn, params, shots=1024):
    # 构建电路时添加噪声
    def noisy_circuit(theta):
        c = circuit_fn(theta)
        c = c.with_noise("depolarizing", p=0.001)  # 链式API
        return c
    
    # 使用parameter shift计算梯度
    return gradient_parameter_shift(noisy_circuit, params, shots)

# Barren Plateau分析
for n_qubits in [4, 6, 8, 10]:
    print(f"\n{n_qubits} qubits:")
    
    # 理论预测: σ² ~ O(1/2^n)
    theoretical_std = 1 / 2**(n_qubits/2)
    print(f"  Theoretical: {theoretical_std:.6f}")
    
    # 方法A: Ideal
    std_ideal = measure_gradient_variance_autograd(n_qubits)
    print(f"  Method A (Ideal):     {std_ideal:.6f}")
    
    # 方法B: Shot-based
    std_shots = measure_gradient_variance_parameter_shift(n_qubits, shots=1024)
    print(f"  Method B (Shots):     {std_shots:.6f}")
    
    # 方法C: Noise + Shots
    std_noisy = measure_gradient_variance_with_noise(n_qubits, shots=1024)
    print(f"  Method C (Realistic): {std_noisy:.6f}")
```

**输出示例**:
```
6 qubits:
  Theoretical:          0.125000
  Method A (Ideal):     0.112665  (9.9% error)
  Method B (Shots):     0.114486  (8.4% error) 
  Method C (Realistic): 0.140485  (12.4% error)
```

**关键洞察**:
1. **理想vs现实**: Autograd给出干净信号，但**不可在真机实现**
2. **Parameter Shift**: 实现硬件可实现的梯度估计
3. **Shot噪声**: 有限采样带来统计误差
4. **硬件噪声**: 进一步降低梯度质量
5. **指数缩放**: Barren plateau随系统规模指数级恶化

**缓解策略**:
- 使用局部代价函数（避免全局可观测量）
- 设计问题启发的ansatz（减少电路深度）
- 采用分层或预训练策略
- 增加关键梯度计算的shot预算
- 使用噪声感知训练算法

**TyxonQ特色展示**:
- 链式API: `.with_noise().device(shots=...).run()`
- 统一接口支持理想和真实模拟
- PyTorch后端集成自动微分
- 灵活的噪声模型

**运行**:
```bash
python examples/barren_plateau_benchmark.py
```

**依赖**:
- PyTorch
- PennyLane（可选，用于交叉验证）

**参考文献**:
- McClean et al. (2018). Nat. Commun. 9, 4812
- Cerezo et al. (2021). Nat. Commun. 12, 1791
- Schuld et al. (2019). Phys. Rev. A 99, 032331

---

## 量子模拟

### 12. 测量诱导相变（MIPT）
**文件**: [`measurement_induced_phase_transition.py`](measurement_induced_phase_transition.py)

研究量子系统中由测量引起的纠缠相变现象：

**关键特性**:
- 测量诱导动力学：随机幺正演化 + 投影测量
- 纠缠熵追踪：Half-chain entanglement entropy
- 相变分析：Volume-law vs Area-law
- Kraus算子实现：使用TyxonQ的`.kraus()` API

**物理背景**:
在量子多体系统中，竞争的幺正演化（产生纠缠）和投影测量（破坏纠缠）会导致相变：
- **低测量率** (p < pc): Volume-law phase, S ~ L
- **高测量率** (p > pc): Area-law phase, S ~ constant

**示例代码**:
```python
import tyxonq as tq
import numpy as np

tq.set_backend("pytorch")

def mipt_trajectory(n_qubits, depth, p_measure):
    """Single MIPT trajectory with measurement probability p."""
    c = tq.Circuit(n_qubits)
    
    # Initialize in product state |0...0⟩
    
    for layer in range(depth):
        # Unitary layer: random 2-qubit gates
        for i in range(n_qubits - 1):
            theta = np.random.uniform(0, 2*np.pi, 3)
            c.rxx(i, i+1, theta=theta[0])
            c.ryy(i, i+1, theta=theta[1])
            c.rzz(i, i+1, theta=theta[2])
        
        # Measurement layer: projective Z measurements
        for i in range(n_qubits):
            if np.random.rand() < p_measure:
                # Kraus operators for Z measurement
                c.kraus(
                    "measurement",  # Uses predefined Z measurement
                    [i],
                    status=np.random.choice([0, 1])  # Random outcome
                )
    
    # Compute entanglement entropy
    half = n_qubits // 2
    rho_A = c.reduced_density_matrix(list(range(half)))
    entropy = von_neumann_entropy(rho_A)
    
    return entropy

# Phase transition analysis
p_values = np.linspace(0, 1.0, 20)
entropies = []

for p in p_values:
    # Average over trajectories
    S_avg = np.mean([mipt_trajectory(12, 50, p) for _ in range(20)])
    entropies.append(S_avg)
    print(f"p={p:.2f}: S={S_avg:.3f}")

# Plot phase diagram
import matplotlib.pyplot as plt
plt.plot(p_values, entropies, 'o-')
plt.xlabel('Measurement probability p')
plt.ylabel('Entanglement entropy S')
plt.title('Measurement-Induced Phase Transition')
plt.axvline(p_critical, color='r', linestyle='--', label=f'pc ≈ {p_critical:.2f}')
plt.legend()
plt.show()
```

**运行**:
```bash
python examples/measurement_induced_phase_transition.py
```

**输出示例**:
```
Measurement-Induced Phase Transition (MIPT)
============================================
System: 12 qubits, 50 layers, 100 trajectories

p=0.00: S=6.234 (Volume-law)
p=0.10: S=5.987
p=0.20: S=5.123 (Critical region)
p=0.30: S=2.456
p=0.50: S=1.234 (Area-law)

Estimated critical point: pc ≈ 0.18
```

**关键观察**:
1. **Volume-law相** (p < 0.2): 纠缠熵随系统尺寸线性增长
2. **Area-law相** (p > 0.2): 纠缠熵饱和到常数
3. **临界点** (pc ≈ 0.18): 纠缠熵标度行为改变

**参考文献**:
- Skinner et al. (2019). Phys. Rev. X 9, 031009
- Li et al. (2018). Phys. Rev. B 98, 205136

---

### 13. 量子混沌分析
**文件**: [`quantum_chaos_analysis.py`](quantum_chaos_analysis.py)

分析量子电路的混沌特性，包括Frame Potential和Jacobian分析：

**关键特性**:
- Frame Potential: 量化电路对Haar随机的逼近程度
- Spectral Form Factor (SFF): 能谱统计分析
- Jacobian矩阵: 参数空间的几何结构
- 随机电路vs结构化电路对比

**示例代码**:
```python
import tyxonq as tq
import torch
import numpy as np

tq.set_backend("pytorch")

def compute_frame_potential(n_qubits, depth, n_samples=100):
    """Compute 2-design frame potential.
    
    FP(2) = ∫ |Tr(UU†VV†)|² dU dV
    
    For Haar random: FP(2) → 2
    """
    fp = 0.0
    
    for _ in range(n_samples):
        # Generate two random circuits
        params_u = torch.randn(depth, n_qubits, 3)
        params_v = torch.randn(depth, n_qubits, 3)
        
        U = random_circuit(n_qubits, params_u)
        V = random_circuit(n_qubits, params_v)
        
        # Compute overlap
        overlap = torch.abs(torch.trace(U @ U.T.conj() @ V @ V.T.conj()))**2
        fp += overlap.item()
    
    return fp / n_samples

# Analyze convergence to Haar randomness
for depth in [1, 2, 4, 8, 16]:
    fp = compute_frame_potential(n_qubits=6, depth=depth)
    haar_target = 2.0
    print(f"Depth={depth:2d}: FP={fp:.4f} (Haar={haar_target})")
```

**运行**:
```bash
python examples/quantum_chaos_analysis.py
```

---
**文件**: [`hamiltonian_time_evolution.py`](hamiltonian_time_evolution.py)

使用Trotter-Suzuki分解模拟Hamiltonian的时间演化：

**关键特性**:
- Trotter分解: exp(-iHt) ≈ [∏ⱼ exp(-iwⱼPⱼδt)]^n
- 支持所有Pauli模式: X, Y, Z, XX, YY, ZZ, XYZ...
- 精度分析: 对比Trotter近似与精确演化
- Heisenberg模型: H = J·(XX + YY + ZZ)

**示例代码**:
```python
import tyxonq as tq
import numpy as np
from tyxonq.libs.circuits_library.trotter_circuit import trotter_circuit

tq.set_backend("pytorch")

# 定义Heisenberg哈密顿量
def build_hamiltonian_pauli_strings(n_qubits, J=1.0):
    """Build Hamiltonian as list of Pauli strings.
    
    Example: 2-qubit Heisenberg model
    H = J·(XX + YY + ZZ)
    """
    if n_qubits == 2:
        pauli_terms = [
            [1, 1],  # XX (Pauli code: 0=I, 1=X, 2=Y, 3=Z)
            [2, 2],  # YY
            [3, 3],  # ZZ
        ]
        weights = [J, J, J]
    return pauli_terms, weights

# Trotter演化
n_qubits = 2
pauli_terms, weights = build_hamiltonian_pauli_strings(n_qubits, J=1.0)

def evolve_trotter(psi0, time, n_steps):
    """Evolve state using Trotter decomposition."""
    c = tq.Circuit(n_qubits, inputs=psi0)
    c = trotter_circuit(
        c,
        pauli_strings=pauli_terms,
        weights=weights,
        time=time,
        n_trotter_steps=n_steps,
        order=1
    )
    return c.state()

# 精确演化对比
from scipy.linalg import expm

def evolve_exact(psi0, time):
    """Exact evolution via matrix exponential."""
    H = build_full_hamiltonian(pauli_terms, weights)  # 4x4 matrix
    U = expm(-1j * H * time)
    return U @ psi0

# 比较Fidelity
psi_trotter = evolve_trotter(psi0, time=1.0, n_steps=10)
psi_exact = evolve_exact(psi0, time=1.0)

fidelity = np.abs(np.vdot(psi_exact, psi_trotter))**2
print(f"Fidelity: {fidelity:.9f}")  # 1.000000000 (完美匹配)
```

**算法详情**:
- 一阶Trotter: U(t) ≈ [∏ⱼ e^{-iθⱼPⱼ}]^{t/δt}
- 二阶Trotter: Suzuki对称分解，更高精度

**支持的Pauli模式**:
- 单比特: X→H-RZ-H, Y→S†-H-RZ-H-S, Z→RZ
- 两比特: XX, YY, ZZ via CNOT ladder
- 多比特: 任意Pauli字符串via基变换

**运行**:
```bash
python examples/hamiltonian_time_evolution.py
```

**输出示例**:
```
Trotter approximation vs Exact evolution:
Time=0.5, Steps=5:  Fidelity=0.99999987
Time=1.0, Steps=10: Fidelity=1.00000000
Time=2.0, Steps=20: Fidelity=1.00000000

Exact evolution time: 0.12 ms
Trotter (10 steps):   1.45 ms
Trotter (100 steps): 14.23 ms
```

---

### 11. 变分量子动力学（VQD）
**文件**: [`variational_quantum_dynamics_sbm.py`](variational_quantum_dynamics_sbm.py), [`variational_quantum_dynamics_tfim.py`](variational_quantum_dynamics_tfim.py)

使用变分算法模拟量子系统的时间演化：

**关键特性**:
- 基于DynamicsNumericRuntime的高级API
- Spin-Boson Model (SBM) 和 Transverse-Field Ising Model (TFIM)
- VQD和p-VQD算法
- 可观测量追踪（⟨Z⟩, ⟨X⟩）
- 能量守恒监控
- 精确解对比（Fidelity追踪）

**示例代码**:
```python
from renormalizer import Op
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library import sbm
from tyxonq.libs.hamiltonian_encoding.operator_encoding import (
    qubit_encode_op, qubit_encode_basis
)
from tyxonq.applications.chem.runtimes.dynamics_numeric import DynamicsNumericRuntime

# Build Spin-Boson Model
ham_terms = sbm.get_ham_terms(epsilon=0, delta=1.0, n_modes=1, 
                               omega_list=[1.0], g_list=[0.5])
basis = sbm.get_basis([1.0], [8])
ham_terms_spin, _ = qubit_encode_op(ham_terms, basis, "gray")
basis_spin = qubit_encode_basis(basis, "gray")

# Initialize dynamics runtime
dynamics = DynamicsNumericRuntime(
    ham_terms_spin, basis_spin,
    n_layers=3, eps=1e-5
)

# Add observables
dynamics.add_property_op("Z", Op("Z", "spin"))
dynamics.add_property_op("X", Op("X", "spin"))

# Time evolution
for step in range(50):
    props = dynamics.properties()
    print(f"⟨Z⟩ = {props['Z']:.4f}, ⟨X⟩ = {props['X']:.4f}")
    dynamics.step_vqd(0.1)  # VQD step
    # or: dynamics.step_pvqd(0.1)  # p-VQD step
```

**算法**:
- **VQD** (Variational Quantum Dynamics): McLachlan变分原理
- **p-VQD** (Projected VQD): 改进的长时间稳定性

**依赖**:
- renormalizer
- PyTorch backend

**参考**:
- PRL 125, 010501 (2020) - VQD算法
- src/tyxonq/applications/chem/runtimes/dynamics_numeric.py
- tests_mol_valid/test_dynamics.py

---

### 8. 量子自然梯度（QNG）
**文件**: [`quantum_natural_gradient_optimization.py`](quantum_natural_gradient_optimization.py)

使用量子Fisher信息矩阵加速优化：

**优势**:
- 比普通梯度下降快10-100倍
- 克服平坦景观（Barren Plateau）
- 适合深层量子电路

---

## 量子模拟

### 7. 哈密顿量时间演化
**文件**: [`hamiltonian_time_evolution.py`](hamiltonian_time_evolution.py), [`timeevolution_trotter.py`](timeevolution_trotter.py)

模拟量子系统的时间演化 ψ(t) = e^{-iHt} ψ(0)：

```python
import tyxonq as tq

# 构建哈密顿量
n = 6
# H = Σ_i X_i + Σ_i Z_i Z_{i+1}

# Trotter分解
def trotter_step(circuit, dt, trotter_steps):
    for step in range(trotter_steps):
        # 演化X项
        for i in range(n):
            circuit.rx(i, 2*dt/trotter_steps)
        # 演化ZZ项
        for i in range(n-1):
            circuit.rzz(i, i+1, 2*dt/trotter_steps)
```

---

### 8. MPS近似模拟
**文件**: [`mps_approximation_benchmark.py`](mps_approximation_benchmark.py)

使用矩阵乘积态(MPS)模拟大规模低纠缠系统：

**特点**:
- 10-15量子比特：MPS vs 精确模拟对比
- Bond维度 vs 精度权衡
- O(nχ³) 复杂度，χ为bond维度

**运行**:
```bash
python examples/mps_approximation_benchmark.py
```

**输出示例**:
```
Bond dimension:   20
  Exact energy:    1.52532598
  MPS energy:      1.52523811
  Relative error:  0.0058%
  Fidelity:        1.000000
```

---

### 9. Stabilizer模拟
**文件**: [`stabilizer_clifford_entropy.py`](stabilizer_clifford_entropy.py)

使用稳定子形式主义快速模拟Clifford电路：

**优势**:
- Clifford电路：O(n²) vs O(2^n)
- 内存节省100倍以上
- 适合量子纠错代码

**运行**:
```bash
python examples/stabilizer_clifford_entropy.py
```

**输出示例**:
```
Stim entropy:           2.7725887222
TyxonQ entropy:         2.7725887289
Absolute difference:    6.63e-09
Agreement (tol=1e-8):   ✓ PASS

Stim computation time:  47.07 ms
TyxonQ computation time:39.78 ms
Memory ratio:           114x
```

---

## 高级特性

### 10. 内存优化与梯度检查点
**文件**: [`memory_optimization_checkpointing.py`](memory_optimization_checkpointing.py)

展示如何在深度量子电路中使用梯度检查点减少内存消耗：

**关键特性**:
- 内存扩展性分析: O(depth × 2^n) vs O(√depth × 2^n)
- 梯度检查点概念解释
- PyTorch集成建议
- 深度电路优化演示

**运行**:
```bash
python examples/memory_optimization_checkpointing.py
```

**输出示例**:
```
=== 内存扩展性分析 ===
Layers=100: 1.60 MB (标准) → 0.16 MB (检查点) = 90% savings
```

---

### 11. 性能优化：分层状态计算
**文件**: [`performance_layerwise_optimization.py`](performance_layerwise_optimization.py)

针对深度噪声电路的JIT编译优化技术：

**问题场景**:
- 深度电路（>10层）+ 大量噪声通道
- JIT编译时间过长（数分钟）
- Monte Carlo轨迹模拟

**优化技巧**:
```python
# ❌ 标准方式：单一计算图（编译慢）
c = tq.Circuit(n)
for layer in range(100):  # 深度电路
    c.cnot(0, 1)
    c.kraus(0, noise_ops)

# ✅ 优化方式：分层状态计算（编译快）
c = tq.Circuit(n)
state = None
for layer in range(100):
    c = tq.Circuit(n, inputs=state) if state else c
    c.cnot(0, 1)
    c.kraus(0, noise_ops)
    state = c.state()  # 强制计算，打断计算图
```

**性能提升**:
- 编译时间：10-30x加速
- 运行时间：~1.2x变慢（可接受）
- 适用于NISQ算法模拟

**运行**:
```bash
python examples/performance_layerwise_optimization.py
```

**输出示例**:
```
Metric                         Standard        Optimized       Ratio
----------------------------------------------------------------------
Compilation time               12.345s         0.456s          27.1x
Runtime (avg)                  0.0123s         0.0145s         1.2x
```

**适用场景**:
- ✓ 深度 > 10层的噪声电路
- ✓ >50个Kraus操作
- ✓ 重复执行（MC轨迹）
- ✗ 浅层电路（< 5层）

---

### 11. 编译器优化
**文件**: [`circuit_compiler.py`](circuit_compiler.py), [`compiler_lightcone_optimization.py`](compiler_lightcone_optimization.py)

**优化技术**:
- **Light Cone优化**: 移除冗余门
- **门合并**: 连续旋转门合并
- **电路简化**: 降低深度

**性能提升**:
```
原始电路: 500门, 100层
优化后:   120门, 25层 (75%门减少)
```

---

### 11. 读出误差缓解（Readout Error Mitigation）
**文件**: [`readout_mitigation.py`](readout_mitigation.py)

校正量子硬件的读出误差：

```python
import tyxonq as tq

# 校准矩阵
calibration_matrix = [[0.95, 0.05],
                     [0.03, 0.97]]

# 应用缓解
raw_counts = {"00": 45, "01": 5, "10": 3, "11": 47}
mitigated_counts = tq.readout_mitigation(raw_counts, calibration_matrix)
```

---

### 12. 噪声模拟
**文件**: [`noise_controls_demo.py`](noise_controls_demo.py)

模拟真实量子硬件的噪声：

- 去极化噪声（Depolarizing）
- 相位衰减（Phase Damping）
- 振幅衰减（Amplitude Damping）

---

### 13. 采样与统计
**文件**: [`sample_benchmark.py`](sample_benchmark.py), [`sample_value_gradient.py`](sample_value_gradient.py)

模拟测量采样和统计噪声：

```python
# 采样100次
samples = circuit.sample(shots=100)
# 结果: {"000": 48, "111": 52}
```

---

## 性能优化

### 14. 混合量子-经典训练
**文件**: [`hybrid_quantum_classical_training.py`](hybrid_quantum_classical_training.py)

结合PyTorch深度学习框架：

**特性**:
- GPU加速量子电路计算
- 与PyTorch nn.Module集成
- 端到端可微分

**示例**:
```python
import torch
import tyxonq as tq

tq.set_backend("pytorch")

class QuantumLayer(torch.nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.randn(n_qubits))
    
    def forward(self, x):
        c = tq.Circuit(len(self.theta))
        for i, angle in enumerate(self.theta):
            c.rx(i, angle)
        # ... 计算期望值 ...
        return expectation

# 与经典层组合
model = torch.nn.Sequential(
    torch.nn.Linear(10, 4),
    QuantumLayer(4),
    torch.nn.Linear(1, 2)
)
```

---

### 15. Jacobian计算
**文件**: [`jacobian_cal.py`](jacobian_cal.py), [`parameter_shift.py`](parameter_shift.py)

高效计算变分电路的Jacobian矩阵：

- 用于量子自然梯度
- 灵敏度分析
- 参数平移规则

---

### 16. Incremental Two-Qubit Gates
**文件**: [`incremental_twoqubit.py`](incremental_twoqubit.py)

优化两量子比特门的实现。

---

## 云计算接口

### 17. 云平台任务提交
**文件**: 
- [`cloud_api_devices.py`](cloud_api_devices.py) - 查询设备
- [`cloud_api_task.py`](cloud_api_task.py) - 提交任务
- [`cloud_api_task_qaoa.py`](cloud_api_task_qaoa.py) - QAOA云计算
- [`cloud_classical_methods_demo.py`](cloud_classical_methods_demo.py)
- [`cloud_uccsd_hea_demo.py`](cloud_uccsd_hea_demo.py)

---

## 其他示例

### 18. 分子化学
**文件**: [`demo_hea_homo_lumo_gap.py`](demo_hea_homo_lumo_gap.py), [`demo_homo_lumo_gap.py`](demo_homo_lumo_gap.py), [`hchainhamiltonian.py`](hchainhamiltonian.py)

计算分子的HOMO-LUMO能隙。

---

### 19. 哈密顿量构建
**文件**: [`hamiltonian_building.py`](hamiltonian_building.py)

构建复杂的量子哈密顿量。

---

### 20. 脉冲控制
**文件**: [`pulse_demo.py`](pulse_demo.py), [`pulse_demo_scan.py`](pulse_demo_scan.py)

低层脉冲级别的量子控制。

---

### 21. JSON输入输出
**文件**: [`jsonio.py`](jsonio.py)

电路的序列化和反序列化。

---

## 运行示例

### 环境准备

```bash
# 安装TyxonQ（开发模式）
pip install -e .

# 安装额外依赖（可选）
pip install torch numpy matplotlib scipy

# 对于stabilizer示例，需要安装stim
pip install stim
```

---

### 快速测试

运行所有示例的测试脚本：

```bash
# 基础示例
python examples/basic_chain_api.py
python examples/numeric_backend_switching.py

# VQE示例
python examples/vqe_simple_hamiltonian.py
python examples/simple_qaoa.py
python examples/vqe_mera_mpo.py
python examples/barren_plateau_benchmark.py

# 高级模拟器
python examples/mps_approximation_benchmark.py
python examples/stabilizer_clifford_entropy.py

# 量子动力学
python examples/hamiltonian_time_evolution.py
python examples/variational_quantum_dynamics_tfim.py
python examples/measurement_induced_phase_transition.py
python examples/quantum_chaos_analysis.py

# 编译器优化
python examples/compiler_lightcone_optimization.py

# 量子-经典混合
python examples/hybrid_quantum_classical_training.py
python examples/quantum_natural_gradient_optimization.py
python examples/noisy_quantum_machine_learning.py

# 噪声模拟
python examples/noisy_circuit_demo.py
python examples/noise_t1_t2_calibration.py
python examples/noisy_sampling_comparison.py

# 性能优化
python examples/memory_optimization_checkpointing.py
python examples/readout_mitigation_scalability.py
```

---

### 性能建议

1. **小规模电路（<10 qubits）**: 使用NumPy后端
   ```python
   tq.set_backend("numpy")
   ```

2. **中等规模（10-15 qubits）**: 
   - 低纠缠 → MPS模拟器
   - 高纠缠 → PyTorch GPU加速
   ```python
   tq.set_backend("pytorch")
   c.device(provider="simulator", device="matrix_product_state", max_bond=32)
   ```

3. **Clifford电路**: 使用Stabilizer模拟器
   - 内存: O(n²) vs O(2^n)
   - 速度: 可模拟100+量子比特

4. **训练和优化**: 
   - 使用PyTorch/JAX后端
   - 开启GPU加速
   - 使用JIT编译

---

## 贡献

欢迎贡献新的示例！请确保：

1. 代码清晰，有充分注释
2. 包含docstring说明用途
3. 输出有意义的结果
4. 运行时间<1分钟（benchmark除外）

---

## 文档

更多信息请参考：
- [TyxonQ文档](../docs/)
- [API参考](../docs/api/)
- [教程](../docs/tutorials/)

---

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](../LICENSE)文件。
