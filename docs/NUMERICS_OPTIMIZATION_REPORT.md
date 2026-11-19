# TyxonQ 全框架数值加速优化报告

日期：2026-11-19  
对应审计：`docs/NUMERICS_ACCELERATION_AUDIT.md`  
统一基准：`examples/numerics_acceleration_benchmark.py`

## 1. 结论摘要

本轮采用“真实负载 profile → Amdahl 收益上限 → 独立修改 → 数值/接口回归 → 两个独立进程复测”的流程，不把孤立算子提速当作应用收益。

采用三项修改：

1. **MPS 后端实例贯穿与原生 SVD**：删除宽泛异常触发的隐式 NumPy SVD，避免设备、dtype 和全局后端串扰；NumPy 实际电路负载在中大规模双轮达到整体改善门槛。
2. **EWF DMET 分片贡献复用**：同一 kernel 不再为分片明细和总能量重复计算 cumulant、projector 与四阶 ERI 收缩；中大规模整体收益稳定超过 5%。
3. **LUCJ 初值 objective 复用**：SciPy 首次调用复用为 loss history 已计算的相同 `x0` loss/gradient；restricted 与 αβ 路径均减少一次完整 PyTorch 前向和反向，优化轨迹逐值不变。

未采用后端迁移：`quantum_info`、EWF 常驻 GPU 张量、LUCJ GPU 重构、SKQD、readout、全局 crosstalk/动力学和 MLX SQD RDM。原因分别是热点占比不足、现有收缩已优化、本机设备不支持所需 FP64/complex128，或缺少 RDM 主导整体耗时的证据。

## 2. 测量协议与环境

- 机器：macOS 26.4.0，arm64 / Apple Silicon；仅本机 CPU 实测用于本轮采用决策。
- Python 3.11.8，NumPy 2.4.6，SciPy 1.15.3，PyTorch 2.8.0，PySCF 2.14.0，MLX 0.32.4。
- 解释器：`/opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python`。
- PyTorch CPU threads：5；基准不隐式修改 BLAS/OpenMP 线程环境。
- 每个负载记录一次冷启动、至少 5 次稳态原始样本、中位数和范围。
- Python 峰值内存在独立 `tracemalloc` 运行中测量，不计入性能样本；它不完整覆盖原生 BLAS、PySCF 和 PyTorch 分配，因此同时保留分析张量字节数，不把该指标解释为完整进程显存/内存。
- 采用门槛：代表性整体负载中位耗时改善至少 5%，并在独立进程复现。小规模边界结果不单独作为采用依据。
- 基准串行运行，不在单元测试中写性能阈值。

## 3. 负载矩阵

| 套件 | small | medium | large | 结果/阶段记录 |
| --- | --- | --- | --- | --- |
| MPS | 6 qubits、depth 2、χ=4 | 10、depth 4、χ=16 | 14、depth 6、χ=32 | 完整 MPS 演化、张量 checksum、bond shape |
| quantum_info | 4 qubits，dense 16 | 6，dense 64 | 8，dense 256 | pure/mixed RDM、entropy、fidelity、negativity、trace distance |
| EWF | H2/STO-3G，FCI | LiH/STO-3G，CCSD | H2/cc-pVDZ，CCSD | SCF setup、fragment build/solve、DMET energy、调用次数 |
| LUCJ | 3 orbitals、1 term | 4 orbitals、1 term | 5 orbitals、2 terms | restricted + αβ、SciPy `nfev`、实际 objective、loss history |
| SKQD | 4 orbitals | 6 orbitals | 8 orbitals | 合法八重对称 ERI 的五操作数旋转 |
| readout | 16 维校准矩阵 | 64 维 | 256 维 | 100000 shots、SciPy FP64 `pinv`、counts 往返 |

EWF 的 SCF 在计时前完成，其 setup 单独记录。LUCJ 的 SciPy 参数和梯度仍为 NumPy CPU 边界；没有宣称整个优化器位于 GPU。

## 4. 采用项整体性能

表中时间均为稳态中位数，单位 ms。改善为相对同一基线的 `(baseline - optimized) / baseline`。`R1/R2` 是两个独立进程；`Final` 是完整回归后的六套件合并运行。

### 4.1 MPS

| 规模 | Baseline | R1 | R2 | R1/R2 改善 | Final |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 0.314 | 0.298 | 0.299 | 5.07% / 4.92% | 0.301 |
| medium | 1.070 | 0.982 | 0.968 | 8.20% / 9.55% | 1.013 |
| large | 5.074 | 4.412 | 4.215 | 13.04% / 16.93% | 4.369 |

大规模在两轮均稳定超过门槛，最终合并运行改善 13.89%。小规模处于约 5% 边界，不单独作为证据。三次运行各规模的张量 checksum 与基线完全一致。

主要收益来自高频小数组操作不再经过全局 `NumericBackend` 代理，以及删除每个双比特门的宽泛异常保护；同时修复引擎后端未传入 kernel 的语义问题。基准使用 NumPy complex128，因此这些数据**不是 GPU 加速数据**。

### 4.2 EWF DMET

| 规模 | Baseline | R1 | R2 | R1/R2 改善 | Final |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 8.197 | 7.941 | 7.718 | 3.13% / 5.84% | 8.687 |
| medium | 57.802 | 54.841 | 53.003 | 5.12% / 8.30% | 53.387 |
| large | 66.902 | 57.999 | 58.150 | 13.31% / 13.08% | 59.507 |

medium/large 在双轮均通过门槛，最终合并运行分别改善 7.64%/11.05%。small 收益不稳定，未用于采用论证。每个真实 kernel 的 `fragment_dmet_energy` 调用由 4 次降为 2 次；所有能量与基线偏差均为 0。

保留 `dmet_total_energy` 的全局密度和有效势修正，只通过兼容的可选 `fragment_energies` 复用同一调用内的分片结果。`compute_dmet_energy(..., return_fragment_energies=True)` 同样消除了重复计算。

### 4.3 LUCJ compressed DF

| 规模 | Baseline | R1 | R2 | R1/R2 改善 | Final |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 14.779 | 11.749 | 11.857 | 20.50% / 19.77% | 11.729 |
| medium | 13.051 | 10.903 | 11.190 | 16.46% / 14.26% | 11.446 |
| large | 18.576 | 16.553 | 16.429 | 10.89% / 11.56% | 17.050 |

所有规模双轮超过门槛。实际 objective 调用由 `SciPy nfev + 1` 降为 `nfev`：

- small/medium：restricted `6→5`，αβ `4→3`；
- large：restricted `10→9`，αβ `4→3`。

restricted/αβ 的初值和每个接受点的 loss history 与基线逐值完全一致。restricted target 还提升到多 stage 循环外，避免重复 host 转换；每次 SciPy objective 的 NumPy 参数、CPU gradient 返回边界保持真实存在。

## 5. 原始稳态样本

以下保留采用验收的 baseline/R1/R2 原始样本，单位 ms，顺序为实际执行顺序。

### MPS

| 规模 | Baseline | R1 | R2 |
| --- | --- | --- | --- |
| small | 0.362292, 0.313959, 0.305416, 0.314625, 0.304041 | 0.340333, 0.299375, 0.298042, 0.292000, 0.291250 | 0.347083, 0.313333, 0.298500, 0.290583, 0.290625 |
| medium | 1.164084, 1.090834, 1.069792, 1.049750, 1.045792 | 1.013084, 0.987250, 0.978500, 0.982041, 0.980708 | 0.987084, 0.967666, 0.975917, 0.963333, 0.956084 |
| large | 5.093666, 5.073750, 5.482709, 4.580958, 4.507417 | 4.331833, 4.412250, 4.449042, 4.259875, 4.486125 | 4.369083, 4.384834, 4.214750, 4.202209, 4.104584 |

### EWF

| 规模 | Baseline | R1 | R2 |
| --- | --- | --- | --- |
| small | 9.546416, 9.379000, 8.196834, 7.943542, 7.219458 | 9.420125, 7.955458, 7.875666, 7.940583, 7.280208 | 9.315500, 8.782958, 7.718084, 7.102209, 6.889208 |
| medium | 57.952833, 57.802167, 57.538875, 57.950833, 56.451833 | 70.665167, 62.029084, 54.841458, 54.574250, 52.748625 | 53.003375, 52.477250, 53.110125, 52.783125, 53.113750 |
| large | 68.026750, 66.034542, 66.520917, 66.901708, 67.797291 | 57.396458, 58.093541, 58.582167, 57.999125, 57.813875 | 59.143209, 58.524250, 58.022542, 57.201833, 58.149875 |

EWF medium R1 存在前两次暖机波动，中位数和独立 R2 均仍通过门槛；报告不删除这些样本。

### LUCJ

| 规模 | Baseline | R1 | R2 |
| --- | --- | --- | --- |
| small | 15.114083, 15.438792, 14.598709, 14.778667, 13.711875 | 12.900334, 12.394458, 11.748750, 11.508959, 11.269875 | 12.438042, 12.868625, 11.622125, 11.579292, 11.856750 |
| medium | 13.647000, 13.822250, 13.001334, 12.880333, 13.051292 | 11.436209, 11.183167, 10.320417, 10.903167, 10.634458 | 10.872125, 11.288583, 10.733541, 11.190375, 11.199583 |
| large | 19.275250, 18.985875, 18.252917, 18.167875, 18.576000 | 17.456792, 16.553417, 16.124042, 16.888166, 15.959500 | 16.458375, 16.368834, 16.429416, 16.757250, 16.122125 |

### 暂缓项 Final 原始样本

| 套件 | small | medium | large |
| --- | --- | --- | --- |
| quantum_info | 0.129833, 0.121625, 0.121250, 0.113083, 0.139292 | 1.845000, 1.721667, 1.601166, 1.810583, 1.633417 | 62.740875, 60.132750, 60.363042, 61.261708, 60.191250 |
| SKQD | 0.123000, 0.105667, 0.101583, 0.100375, 0.099666 | 0.146792, 0.117542, 0.113792, 0.112333, 0.109000 | 0.144333, 0.140958, 0.137084, 0.140417, 0.136500 |
| readout | 0.085458, 0.070917, 0.065375, 0.062458, 0.061291 | 0.324208, 0.305292, 0.296042, 0.294500, 0.293084 | 5.113875, 5.079750, 5.666708, 5.393458, 5.432834 |

## 6. 内存、精度与接口验收

### 6.1 独立 Python 峰值内存

单位 MB；仅比较同量级异常，不作为完整原生内存证明。

| 套件 | 规模 | Baseline | Final |
| --- | --- | ---: | ---: |
| MPS | small / medium / large | 0.009 / 0.043 / 0.509 | 0.009 / 0.043 / 0.509 |
| EWF | small / medium / large | 0.187 / 0.290 / 0.768 | 0.188 / 0.284 / 0.745 |
| LUCJ | small / medium / large | 0.388 / 0.322 / 0.501 | 0.423 / 0.309 / 0.433 |

LUCJ small 增加约 0.034 MB，medium/large 下降；未观察到随规模失控增长。EWF large 的独立 Python 峰值下降约 0.023 MB，但不能据此推断全部 PySCF native 峰值。

### 6.2 数值和梯度

- MPS：NumPy/PyTorch CPU 同输入态矢最大误差 `1.97e-17`；新增后端隔离和 SVD 错误传播测试。
- EWF：Vayesta 能量 golden、WF/DMET、开壳层、密度梯度、fixed-space gradient 和 scanner 全部通过；采用项能量偏差为 0。
- LUCJ：实际梯度仍由原 PyTorch autograd 计算；缓存只消费相同 `x0` 的首次结果，后续点不复用。restricted/αβ loss history、最终 loss、callback 和线路构建测试通过。
- 公共 API：新增参数均为带兼容默认值的关键字参数；没有新增应用层硬件分发体系。

## 7. 未采用项与 Amdahl 分析

### 7.1 quantum_info：保留 NumPy complex128

最终 large 总中位耗时 60.363 ms。阶段为：

| 阶段 | 时间 ms | 整体占比约 |
| --- | ---: | ---: |
| fidelity（两次 Hermitian `eigh`） | 36.478 | 60.4% |
| trace distance | 8.449 | 14.0% |
| negativity | 7.773 | 12.9% |
| entropy | 7.143 | 11.8% |
| pure + mixed RDM | 0.330 | 0.55% |

即使完全消除 RDM 时间，Amdahl 整体上限也仅约 0.55%，低于 5% 门槛，因此未改 `reduced_density_matrix`。谱分解虽占主导，但 MLX Metal 不支持 complex128，PyTorch MPS 也不提供该路径所需的 complex128 Hermitian 实现；不能用 FP32 结果冒充同精度加速。

稠密矩阵维数为 \(d=2^n\)，单个 complex128 矩阵内存为 \(16d^2\) bytes，谱分解复杂度约为 \(O(d^3)\)。large 的 \(d=256\) 单矩阵为 1.049 MB，独立 Python 峰值为 6.304 MB；论文必须同时呈现该指数增长，不能只报告时间。

### 7.2 EWF/LUCJ GPU 常驻：不进入实现

EWF 的主要四阶收缩已经使用 `optimize=True`，本轮实测收益来自删除重复工作，而非替换 BLAS。LUCJ 使用 FP64/complex128 autograd；本机 PyTorch MPS/MLX 不满足同精度要求。CUDA/CuPy 只能在 NVIDIA 环境另做真实验收。

### 7.3 P2：SKQD 与 readout 保留

| 工作流 | small | medium | large | 决策 |
| --- | ---: | ---: | ---: | --- |
| SKQD 积分旋转 | 0.104 ms | 0.111 ms | 0.141 ms | 现有 optimized contraction 已很快；无整体迁移证据 |
| readout inverse | 0.085 ms | 0.319 ms | 5.119 ms | FP64 `pinv` 和 host counts/serialization 边界保留 |

readout large 的校准矩阵为 0.524 MB，独立 Python 峰值 3.699 MB。即使未来使用 CUDA，也必须单独验收病态矩阵和单纯形约束，不能直接替换为不受支持精度的 Metal 路径。

全局 ZZ crosstalk、动力学、QNG 等 P2 同样涉及 complex128/FP64 或 SciPy host 边界；本机没有满足精度且已证明整体收益的目标设备，继续保留。

### 7.4 SQD RDM：保留 CPU

既有 H2O CAS(4e,18o) 真实整体运行中，Metal/CPU solver 两轮为 73.051/74.262 ms 与 101.266/105.867 ms，而整体约 500 ms。setup、CPU RDM、validation 和子进程路径共同稀释 solver 收益，当前数据没有证明 RDM 单独占主导。因此不实现 MLX RDM，也不改变已完成的 CUDA → Metal → PyTorch CPU SQD 优先级。详细边界见 `docs/SBD_METAL_OPTIMIZATION.md`。

## 8. 回归与静态复核

最终命令覆盖公共 numerics、模拟器分发/噪声、EWF、LUCJ、LUCJ-SQD 和 SKQD：

```text
217 passed, 3 skipped, 26 warnings in 6.24 s
```

跳过项：CuPy 未安装 1 项；ffsim 未安装 2 项。26 个 warning 为既有 PySCF OpenMP 环境提示，与 Metal 是否执行无因果关系。

LSP 语义复核：

- `dmet_total_energy` 的 driver、公开导出、按需 `get_dmet_energy` 和 `compute_dmet_energy` 调用方一致；
- MPS `apply_2q_nn` 的邻近门与 SWAP 路由均传播同一 backend；
- LUCJ 初值包装器只被 restricted 和 αβ 两条压缩路径使用；
- `git diff --check` 通过。

本机没有 CUDA/CuPy，未生成推测性能数字。CuPy 的既有契约测试跳过，未来 NVIDIA 验收必须重新执行同输入、同精度和整体工作流。

## 9. 代码变更

- `src/tyxonq/libs/quantum_library/kernels/matrix_product_state.py`
- `src/tyxonq/devices/simulators/matrix_product_state/engine.py`
- `src/tyxonq/applications/chem/algorithms/ewf/energy.py`
- `src/tyxonq/applications/chem/algorithms/ewf/driver.py`
- `src/tyxonq/applications/chem/algorithms/lucj/double_factorization.py`
- `tests_core_module/test_simulators_e2e.py`
- `tests_applications_chem/test_ewf_vayesta_migration.py`
- `tests_applications_chem/test_lucj_initialization.py`
- `examples/numerics_acceleration_benchmark.py`

未修改无关 compiler 平台方言、独立 SQD CPU golden 或公共 `ArrayBackend` 能力集合。

## 10. 复现命令

以下命令均在仓库根目录串行执行：

```bash
export PYTHONPATH=src
QC=/opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python

# 最终合并基准；输出包含冷启动、原始 samples、阶段、内存和结果摘要。
$QC examples/numerics_acceleration_benchmark.py \
  --suite all --scale all --repeats 5 --output numerics-final.json

# 独立进程复测示例。
$QC examples/numerics_acceleration_benchmark.py \
  --suite ewf --scale all --repeats 5 --output ewf-round2.json

# 完整受影响回归。
$QC -m pytest \
  tests_core_module/test_numerics.py \
  tests_core_module/test_numerics_gpu_backends.py \
  tests_core_module/test_numerics_metal_arrays.py \
  tests_core_module/test_simulators_e2e.py \
  tests_core_module/test_engine_dispatch_completeness.py \
  tests_core_module/test_noise_integration.py \
  tests_applications_chem/test_ewf_vayesta_migration.py \
  tests_applications_chem/test_ewf_density_gradient.py \
  tests_applications_chem/test_ewf_fixed_space_gradient.py \
  tests_applications_chem/test_ewf_open_shell.py \
  tests_applications_chem/test_ewf_scanner.py \
  tests_applications_chem/test_lucj_initialization.py \
  tests_applications_chem/test_lucj_circuit_builder.py \
  tests_applications_chem/test_lucj_sqd_closed_shell.py \
  tests_applications_chem/test_skqd.py -q
```

## 11. 论文/白皮书可用边界

可用于论文的结论：

- 在明确列出的 NumPy CPU 电路上，MPS 后端传播整改伴随中大规模整体中位耗时改善；
- 在 H2/LiH 的指定 EWF DMET 工作流中，删除重复分片能量使 medium/large 整体耗时稳定下降；
- 在指定 LUCJ compressed DF 优化中，初值缓存减少一次完整 autograd，所有规模整体改善且优化轨迹不变；
- profile 驱动的否决机制避免了热点占比不足或精度不兼容的 GPU 迁移。

不可外推：

- 不把上述结果称为 CUDA、Metal 或通用 GPU 加速；
- 不把中小测试规模外推到任意 qubit、bond dimension、分子或优化迭代；
- 不把单次固定 CAS SQD 结果外推为完整迭代 SQD；
- 不声称 CUDA 性能、全设备 FP64/complex128 支持或完整 native 峰值内存已经验收；
- 不用 cold time、单一最好样本或 isolated operator 数据替代上述整体中位数与双轮证据。
