# TyxonQ 闭壳层 LUCJ + SQD 使用说明

TyxonQ 提供从 restricted CCSD 振幅初始化 matrix LUCJ 线路、生成量子采样，
再用 sampling-based quantum diagonalization（SQD）求解 selected-CI 子空间的完整流程。

本文以当前源码为准，只说明闭壳层 LUCJ + SQD。本文不包含 EWF、开壳层
LUCJ/UCCSD、Ext-SQD 或真实量子硬件提交。

---

## 功能概览

完整流程分为三部分：

1. 用 restricted CCSD 的 `t1/t2` 振幅初始化 matrix LUCJ 参数。
2. 构建并执行闭壳层 LUCJ 线路，得到完整 bitstring samples 或 counts。
3. 将采样映射到 alpha/beta determinant pools，在 selected-CI 子空间中迭代求解能量。

```text
分子积分 + restricted CCSD t1/t2
                 |
                 v
      compressed double factorization
                 |
                 v
      matrix LUCJ 参数 U_mu / J_mu
                 |
                 v
RHF reference -> U† -> J -> U -> optional final rotation
                 |
                 v
 TyxonQ raw counts [alpha0.. | beta0..]
                 |
       reverse_bitstring_halves()
                 |
                 v
    SQD/PySCF counts + 粒子数筛选或 recovery
                 |
                 v
       determinant pools -> selected CI
                 |
                 v
       energy、RDM、occupancies、SCI state
```

---

## 主要源码

### LUCJ

| 文件 | 作用 |
|---|---|
| `lucj/initialization.py` | 从 restricted CCSD `t1/t2` 生成 LUCJ 参数 |
| `lucj/double_factorization.py` | 对 `t2` 做显式或压缩 double factorization |
| `lucj/conversion.py` | 把 factorization 结果转换成 builder 参数字典 |
| `lucj/parameters.py` | 参数 shape、独立参数量和输入校验 |
| `lucj/topology.py` | topology、轨道 pair 和 qubit 编号 |
| `lucj/linalg.py` | unitary 参数化和相邻 Givens rotation 分解 |
| `lucj/circuit_builder.py` | 把 matrix LUCJ 参数翻译成 TyxonQ `Circuit` |

### SQD

| 文件 | 作用 |
|---|---|
| `sqd/samples.py` | counts、bitstring、概率和 determinant 整数转换 |
| `sqd/recovery.py` | 根据轨道平均占据数修复错误的 alpha/beta 电子数 |
| `sqd/subsampling.py` | 粒子数 postselection 和概率抽样 |
| `sqd/fermion.py` | SQD 主循环、PySCF selected-CI 求解和结果对象 |

完整 H2O 示例位于 `examples/h2o_sqd.py`。

---

## LUCJ

### 1. 输入和闭壳层限制

`LUCJ` 的主要输入为：

| 参数 | 含义 |
|---|---|
| `n_orbitals` | 空间轨道数 `N` |
| `n_electrons` | 总电子数，必须是正偶数 |
| `layers` | `U† -> J -> U` 重复层数 |
| `topology` | `square`、`hex` 或 `linear`，默认 `square` |

线路使用 `2N` 个 qubit，顺序固定为：

```text
[alpha0, alpha1, ..., alphaN-1 | beta0, beta1, ..., betaN-1]
```

RHF reference 会占据 alpha 和 beta 两条链上相同的最低空间轨道。

### 2. 参数初始化

推荐使用 `initialize_lucj_parameters_from_ccsd()`：

```python
from tyxonq.applications.chem.algorithms.lucj import (
    initialize_lucj_parameters_from_ccsd,
)

params = initialize_lucj_parameters_from_ccsd(
    t2_amplitudes,
    t1=t1_amplitudes,
    n_spatial_orbitals=norb,
    n_layers=1,
    topology="square",
    optimize=False,
)
```

restricted `t2` 必须具有 `(nocc, nocc, nvir, nvir)` shape，并满足
`nocc + nvir == n_spatial_orbitals`。

- `optimize=False` 使用显式 double factorization。
- `optimize=True` 使用 PyTorch 优化压缩后的 factors，默认最多迭代 `100` 次。
- 提供 `t1` 时，会生成并在线路末尾执行 `final_orbital_rotation`。
- factor 数少于 `n_layers` 时，剩余层用恒等 rotation 和零 Coulomb matrix 补齐。

参数字典包含：

| 键 | shape | 含义 |
|---|---|---|
| `orbital_rotations` | `(L, N, N)` | 每层共享给 alpha/beta 的 orbital rotation |
| `diag_coulomb_mats` | `(L, 2, N, N)` | same-spin 和 opposite-spin 两个 Coulomb channel |
| `final_orbital_rotation` | `(N, N)` 或 `None` | 从 `t1` 得到的最终 rotation |

### 3. Topology

same-spin channel 始终连接相邻空间轨道 `(p, p+1)`。topology 只改变
opposite-spin 的同轨道连接：

| Topology | 保留的 opposite-spin pair |
|---|---|
| `square` | 全部 `(p, p)` |
| `hex` | 偶数编号 `(0,0), (2,2), ...` |
| `linear` | 只保留 `(0,0)` |

### 4. 构建线路

```python
from tyxonq.applications.chem.algorithms.lucj import LUCJ

lucj = LUCJ(
    n_orbitals=norb,
    n_electrons=sum(nelec),
    layers=1,
    topology="square",
)
circuit = lucj.get_circuit(params)
```

每层的逻辑顺序固定为：

```text
RHF reference -> U† -> diagonal Coulomb J -> U
```

orbital rotation 被分解为相邻双轨道 unitary 和单 qubit phase；diagonal
Coulomb 演化被分解为 TyxonQ 支持的 `rz/rzz`。线路的化学层级记录保存在
`circuit.metadata["lucj"]["logical_ops"]`，可用于检查 reference、rotation、
Coulomb channel 和对应的底层操作编号。

---

## SQD

### 1. Bitstring 顺序

这是 LUCJ 和 SQD 之间最重要的数据契约。

TyxonQ/LUCJ raw order：

```text
[alpha0..alphaN-1 | beta0..betaN-1]
```

SQD/PySCF order：

```text
[alphaN-1..alpha0 | betaN-1..beta0]
```

因此，TyxonQ 线路产生的 raw counts 必须先转换：

```python
from tyxonq.applications.chem.algorithms.sqd import reverse_bitstring_halves

sqd_counts = {
    reverse_bitstring_halves(bitstring): count
    for bitstring, count in raw_counts.items()
}
```

随后使用默认 `sample_order="alpha_beta"` 调用 `run_sqd_fermion()`。
`sample_order` 只交换 alpha/beta 两个半区，使 SQD 内部统一为
`[beta | alpha]`；它不会反转半区内部，因此不能代替
`reverse_bitstring_halves()`。

例如，CAS(4e,4o) 的 RHF raw bitstring 为：

```text
1100|1100 -> 0011|0011
```

转换后 alpha 和 beta determinant 整数均为 `3`。

### 2. SQD 主入口

```python
from tyxonq.applications.chem.algorithms.sqd import run_sqd_fermion

result = run_sqd_fermion(
    one_body_tensor,
    two_body_tensor,
    sqd_counts,
    samples_per_batch=8,
    norb=norb,
    nelec=nelec,
    nuclear_repulsion_energy=core_energy,
    num_batches=4,
    max_iterations=5,
    symmetrize_spin=True,
    max_dim=4,
    initial_occupancies=initial_occupancies,
    seed=7,
)
```

主要控制参数：

| 参数 | 含义 |
|---|---|
| `samples_per_batch` | 每个 selected-CI batch 使用的完整 bitstring 数 |
| `num_batches` | 每轮独立抽取并求解的 batch 数 |
| `max_iterations` | occupancy-recovery 与 selected-CI 的最大迭代数 |
| `max_dim` | 每个 alpha/beta determinant pool 的容量上限 |
| `initial_occupancies` | 首轮 configuration recovery 的 alpha/beta 平均占据数 |
| `symmetrize_spin` | 合并 alpha/beta candidate pools，仅允许 `n_alpha == n_beta` |
| `include_configurations` | 强制加入 determinant pool 的 CI strings |
| `carryover_threshold` | 下一轮保留重要 CI strings 的振幅阈值 |
| `energy_tol` | 能量收敛阈值 |
| `occupancies_tol` | 轨道平均占据数收敛阈值 |

### 3. 迭代流程

没有 `initial_occupancies` 时，SQD 只保留 alpha/beta 电子数正确的完整
bitstrings。提供 `initial_occupancies` 时，SQD 会先修复粒子数错误的 samples。

每轮执行：

1. 对 samples 做粒子数 postselection 或 configuration recovery。
2. 按概率抽取若干完整 bitstring batches。
3. 分别生成 alpha 和 beta determinant pools。
4. 合入指定 configurations 和上一轮的重要 CI strings。
5. 用 PySCF `selected_ci.kernel_fixed_space()` 对角化每个子空间。
6. 选择当前能量最低的 batch，并用其 occupancies 驱动下一轮 recovery。
7. 能量和 occupancies 同时收敛时停止。

严格 HF occupancies 可能让所有翻转权重变为零。当前 recovery 会在确实需要
修复电子数时退化为均匀选择，因此不会因为零权重跳过修复或产生 `NaN`。

### 4. 输出

`run_sqd_fermion()` 返回 `SCIResult`：

| 属性 | 含义 |
|---|---|
| `energy` | selected-CI 电子能 |
| `total_energy` | 电子能加 `nuclear_repulsion_energy` |
| `orbital_occupancies` | alpha/beta 轨道平均占据数 |
| `rdm1`、`rdm2` | spin-summed 1-RDM 和 2-RDM |
| `sci_state` | CI amplitudes 与 alpha/beta CI strings |

`SCIState` 还支持 `save()`、`load()`、`rdm()`、`spin_square()` 和
`orbital_occupancies()`。

---

## H2O 完整示例

直接运行：

```bash
python examples/h2o_sqd.py --no-optimize
```

默认示例使用：

| 设置 | 值 |
|---|---|
| 分子 | H2O，`6-31g(d,p)` |
| active space | CAS(4e,4o) |
| LUCJ layers | `1` |
| topology | `square` |
| shots | `4096` |
| probability-level noise | `0.05` |
| SQD batches | `4` |
| samples per batch | `8` |
| SQD max iterations | `5` |
| max dimension | 每个 spin pool 为 `4` |

该示例先用 `UHF/UMP2` 生成自旋求和的 natural orbitals，再构造 RHF-like
reference 并运行 restricted CCSD。后续 LUCJ 仍是闭壳层、spin-balanced 线路，
SQD 使用 `nelec=(2,2)`；这不表示当前 LUCJ 已支持开壳层输入。

`lucj_noisy_energy` 和 noisy counts 使用 probability-level depolarizing mixture，
用于演示采样扰动，不等同于真实硬件噪声或逐门噪声模型。

---

## 安装与运行环境

在源码目录安装：

```bash
python -m pip install -e .
```

LUCJ + SQD 主要依赖 NumPy、SciPy、PySCF 和 OpenFermion。压缩 double
factorization 的 `optimize=True` 路径还需要 PyTorch；缺少 PyTorch 时可使用
`optimize=False` 或示例的 `--no-optimize`。

---

## 当前验证结果

以下结果来自 H2O CAS(4e,4o)、`optimize=True`、`maxiter=100`、`shots=4096`、
固定随机种子 `7` 的默认配置验证：

```text
HF energy: -76.022598397860 Ha
FCI energy: -76.068480264166 Ha
Initialized LUCJ noiseless energy: -76.048175065188 Ha
Initialized LUCJ noisy mixed energy: -75.879490138282 Ha
SQD energy: -76.062229784231 Ha
```

验证命令：

```bash
python examples/h2o_sqd.py
python -m pytest tests_applications_chem -q
python -m ruff check \
  src/tyxonq/applications/chem/algorithms/lucj \
  src/tyxonq/applications/chem/algorithms/sqd \
  examples/h2o_sqd.py \
  tests_applications_chem/test_lucj_sqd_closed_shell.py
```

当前化学测试集结果为 `10 passed`。

---

## 能力边界

- LUCJ builder 当前只接受闭壳层正偶数总电子数。
- 当前初始化入口只接受 restricted CCSD `t1/t2`。
- 本文不声明 EWF 到 LUCJ/SQD 的自动集成。
- 本文不声明开壳层 LUCJ/UCCSD 或完整开壳层工作流。
- 当前 H2O 示例运行本地 statevector 和概率层噪声，不提交真实量子硬件。
- 外部 counts 是否需要 `reverse_bitstring_halves()` 取决于其真实数据 schema；
  只有 TyxonQ/LUCJ raw order 应执行这里描述的半区反转。

## 参考来源

- LUCJ 实现参考：[qiskit-community/ffsim](https://github.com/qiskit-community/ffsim)。
- SQD 实现参考：[Qiskit/qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd)。
