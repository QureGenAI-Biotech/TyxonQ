# Metal SBD 优化记录与交接

状态日期：2026-11-19。本文是 Metal SBD 的研发主记录，用于后续性能工作、人员交接和论文写作。复现命令及固定 CUDA 版本见 `docs/SBD_REPRODUCIBILITY.md`。

## 1. 当前结论

1. Metal SBD 的 H@C、S² 和 Davidson 已真实提交到 Apple GPU；证据来自实际 `mx.gpu` stream、MSL 原生调用、完成后的 `mx.eval` 以及独立 CPU FP64 golden，而不是静态后端名称。
2. 最终三组统一基准中，36–2025 determinants 的 H@C 和固定工作量 Davidson 均达到 **Metal/MPS ≤ 1.10** 的性能相当目标；实际四组 solver 比值为 0.38/0.38/0.37/0.41。H@C 仍慢于 PyTorch CPU，不能外推为快于 CPU 或 CUDA。
3. PySCF 的 `OpenMP is not available` warning 来自 CPU 线程设置，与 MLX/Metal 是否使用 GPU 无因果关系。CPU golden、连接表构建和 RDM 都可能触发它。
4. 正确性、真实执行和性能是三个独立命题。`native_metal_calls > 0` 或 GPU stream 不能代替性能数据；结构更“融合”也不保证更快。
5. 后续优化必须维持 H@C、S²、能量和残差 golden。不得通过放宽容差、减少同步、减少相同工作量或隐藏 CPU 时间来制造加速。

## 2. 架构与迁移边界

```text
tyxonq.numerics.sbd                 公共算法出口与兼容 API
├── cuda/                           CUDA SBD、构建、固定源码、补丁和 CUDA/MPI 资源
└── metal/                          单 Apple GPU MLX/MSL SBD
    ├── solver.py                   连接表、H@C、S²、Davidson 与执行证据
    └── kernels.metal               原生 MSL 内核
```

通用数组后端与 SBD 算法分离。SQD 自动路径严格按 CUDA SBD → Metal SBD → PyTorch CPU 的硬件顺序选择；原生依赖或数值错误不触发降级。

Metal 的范围是单设备、实数 restricted selected-CI。连接表构建和小矩阵 Hermitian FP64 `eigh` 保留主机端；长向量、H@C、S² 和 Davidson 主迭代使用 MLX/Metal。当前 RDM 是 CPU PySCF 验证路径，报告为 `rdm_backend="cpu"`。不承诺 MPI/NCCL、多节点、多 GPU 或端到端 FP64 等价迁移。

## 3. 关键时间线

### 3.1 从空壳到真实求解

- 删除未接入求解链的 `MetalSolverBackend`、Objective-C++/pybind11 和独立 metallib 路线。
- 使用 MLX 数组与 `mx.fast.metal_kernel` 实现三类 H@C、S² 和 Davidson。
- H@C 覆盖 alpha same-spin、beta same-spin 和 mixed-spin，而非只演示 beta 相位。
- CPU FP64 golden 验证 H@C、S²、能量和最终独立残差。

### 3.2 证明 GPU 执行而非依赖默认设备

- 所有原生算子及端到端入口使用局部 `mx.stream(mx.gpu)`，不永久修改调用者全局设备。
- `native_metal_execution` 在真实 H@C/S² 算子中记录 device、stream 和 `completed_evaluations`。
- 测试先把全局默认设备设为 `mx.cpu`，再执行 Metal H@C、S² 和端到端求解；实际算子仍记录 GPU stream，恢复调用方设备后结束。
- `mx.eval` 完成后才读取结果和停止计时，避免把惰性任务排队当成完成。

### 3.3 建立可信性能协议

旧 9 determinant 用例的总时间约 0.4 秒，其中独立 CPU validation 占绝大部分，不能评估 GPU 内核。新增 `examples/sqd_metal_scaling.py`：

- 使用确定性因子分解实数积分，`nelec=(2,2)`；`norb=4/6/8/10` 对应 36/225/784/2025 determinants。
- JIT、连接表 setup 和 CPU golden 不进入稳态 H@C 或 solver-only 计时。
- H@C 对比 Metal FP32 与 PyTorch CPU FP32；CPU FP64 独立负责正确性。
- solver 报告迭代次数、H@C 次数、收敛和独立残差，避免比较不同工作量。
- 记录中位数、最小/最大值和原始 samples，而非只保留单次冷启动。

### 3.4 第一轮优化：保留

减少每次 matvec 的 gather/scatter、临时数组和索引运算，同时保留 FP32 内层累加顺序。详细布局见第 4 节。

### 3.5 第二轮优化：CSR 融合散射，回退

为每个输出 determinant 预计算稳定 CSR output→link 映射，在单个 MSL 线程内累加该输出的 links。数值 golden 全部通过，但多规模实测比第一轮慢，因此完整回退。该实验说明：间接寻址和每个输出线程的循环成本可能高于 MLX 批量散射收益，不能仅凭 kernel 数量判断性能。

## 4. H@C 数据布局与内核关系

### 4.1 三类贡献

- `taskType=2`：alpha same-spin 单/双激发，beta 不变。
- `taskType=1`：beta same-spin 单/双激发，alpha 不变。
- `taskType=0`：alpha 单激发与 beta 单激发组成 mixed-spin 双激发。

CPU/PySCF 在 setup 阶段生成连接表。Metal 缓存 pair、outer、address、sign、pair 矩阵和工作向量。

### 4.2 原始计算链

same-spin 每个 beta chunk 依次执行：

```text
按 address/sign gather
  → scatter 到 (pair, outer, beta_chunk) 临时布局
  → pair 矩阵收缩
  → 按 pair/outer 重新 gather 并乘 sign
  → 按输出 address scatter-add
```

mixed-spin 还需要展开全部 alpha 列，原始最终贡献是 `(beta_link, alpha)` 标量矩阵；`outer` 是该 link 固定的 beta 列，不是额外线程输出轴。

### 4.3 保留的 slot 映射

`gather_slots` 不再先 gather 到紧凑 link 数组再 scatter。setup 预计算完整：

```text
slot → source determinant address
slot → fermionic sign
```

未被连接表覆盖的 pair/outer 空洞设置 `sign=0`，因此直接 gather 即得到完整、带零空洞的临时矩阵。映射常驻 GPU，不在 matvec 中重建 NumPy 元数据。

### 4.4 保留的融合收缩

- same-spin：`pair_contract_links` 直接生成带符号的 link contributions，保留 outer 列偏移和完整临时行宽。
- mixed-spin：`pair_contract_mixed` 直接生成 `(link, alpha)` 标量贡献，避免二次 gather 和逐元素乘法。
- Davidson：通用 `pair_contract` 保留确定性 FP32 内层累加，不能误用 H@C 的 link 专用布局。
- S²：`gather_spin` 生成跃迁项，再按目标地址累加。

## 5. Golden 捕获的三个关键布局错误

这些错误都曾导致端到端 RDM 能量不一致，不能靠最终能量偶然接近来放行。

1. **临时数组空洞**：连接表 slots 不一定覆盖完整 pair/outer 空间。直接写未初始化输出会留下随机值。修复为完整 slot 映射，空洞 sign 为零。
2. **完整临时行宽**：临时布局每行包含全部 `outer × chunk`，不是只有 chunk。融合内核必须同时传入 outer 索引和 `temporary_width`。
3. **mixed-spin 轴语义**：原始贡献是 `(beta_link, alpha)`。先按 same-spin 解释会漏掉 alpha 展开；把 outer 当输出轴也会产生错误形状。最终按原始 `flat_output` 的顺序直接生成 `(link, alpha)`。

方法论：每个融合步骤先逐阶段对照临时矩阵和贡献，再跑完整 H@C，最后跑端到端 RDM/能量/残差。不要只在端到端失败后盲调内核。

## 6. 性能实验记录

### 6.1 协议差异

未优化基线使用 3 repeats；第一轮和 CSR 实验使用 5 repeats。下表是相同输入下的探索性中位数对比，不包装为严格统计显著性。最终结论必须用统一 repeats 重新测量。

### 6.2 H@C 中位耗时（毫秒）

| norb | determinants | 未优化，3 repeats | slot/link 融合，5 repeats | CSR 散射，5 repeats | PyTorch CPU FP32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 36 | 0.822 | 0.330 | 0.639 | 约 0.045–0.050 |
| 6 | 225 | 0.460 | 0.335 | 0.710 | 约 0.053–0.055 |
| 8 | 784 | 0.461 | 0.378 | 0.519 | 约 0.108–0.120 |
| 10 | 2025 | 0.976 | 0.692 | 1.083 | 约 0.207–0.243 |

结论：第一轮优化降低了 Metal 自身时间，尤其在 2025 determinants 从约 0.976 ms 降至 0.692 ms；但仍慢于 CPU FP32。CSR 方案正确却更慢，已回退。当前没有有效的 4356/8281 determinant 结果，不得填入推测数字。

### 6.3 Solver-only 探索数据

| determinants | Metal 未优化（秒） | 第一轮融合（秒） | CPU FP64（秒） | 迭代 | 收敛 | Metal 独立残差 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 36 | 0.0467 | 0.0492 | 约 0.003 | 35 | 是 | 约 1.2e-5 |
| 225 | 0.0635 | 0.0710 | 约 0.004 | 40 | 否 | 约 2.52e-1 |
| 784 | 0.0840 | 0.0958 | 约 0.007 | 40 | 否 | 约 1.32e-1 |
| 2025 | 0.1572 | 0.1316 | 约 0.018 | 40 | 否 | 约 2.81e-3 |

未收敛条目只能用于固定 40-cycle 工作量 profile，不能用于能量正确性、收敛能力或加速宣传。小 case 的 solver 时间还会受到短向量、标量同步和 host `eigh` 的显著影响。

### 6.4 MLX 内置 GEMM 实验

以 `mx.matmul` 替换确定性 H@C pair 收缩只带来有限性能变化，但默认配置下 H@C 相对误差约 `7e-4`，绝对误差约 `2e-3–3e-3`，不满足 `2e-5` golden，因此未采用。Davidson 默认路径也曾使 2025 determinants 独立残差由约 `2.8e-3` 恶化到 `0.33`。

2026-11-19 外部源码复核找到 MLX 0.32.4 官方 `Numerical Precision` 文档及 `metal/matmul.cpp`：带专用矩阵单元的硬件默认可能让 FP32 输入进入 NAX 降精度路径；进程启动时设置 `MLX_ENABLE_TF32=0` 后，FP32 改用普通 STEEL FP32 或 split-K。独立进程内仅将 `_metal_matmul` 替换为 `mx.matmul` 后，全部 12 项 Metal golden 已通过。这修正了“MLX GEMM 不可用于 SBD”的笼统表述：默认降精度路径不可用，完整 FP32 路径已通过正确性门槛。性能、扩展规模和独立 solver 残差仍须单独验收，当前生产默认不变。

### 6.5 三组 Metal/CPU/MPS 基准（2026-11-19）

新增 PyTorch MPS FP32 作为同精度 GPU 参照；MPS 只用于 benchmark，不改变 SQD 必须使用原生 Metal SBD 的生产分发。MPS 每轮显式 `torch.mps.synchronize()`，`PYTORCH_ENABLE_MPS_FALLBACK=0`。MPS 不支持 `torch.linalg.eigh`，因此与 MLX 一样，仅将 `max_space × max_space` 小矩阵显式放到 CPU FP64 `eigh`；长向量、H@C 和 Davidson 主迭代仍在 MPS。

主 solver 对照使用相同 40 cycles、初猜、max_space 和极小固定 tolerance，Metal/MPS 都执行 40 次 H@C。正常 tolerance 的收敛测试另行运行；未收敛的固定循环能量不作为基态结果。每组统一 5 repeats，JIT/setup/golden 外置。

### 6.6 三组最终耗时

H@C 单位为毫秒；solver 单位为秒。比值小于 1 表示原生 Metal 更快。

| norb | determinants | Metal H@C | CPU FP32 H@C | MPS H@C | Metal solver | MPS solver | CPU FP64 solver | Metal/MPS H@C | Metal/MPS solver |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 36 | 0.339 | 0.046 | 1.007 | 0.04357 | 0.11328 | 0.00299 | 0.336 | 0.385 |
| 6 | 225 | 0.369 | 0.053 | 0.995 | 0.04403 | 0.11741 | 0.00418 | 0.370 | 0.375 |
| 8 | 784 | 0.377 | 0.104 | 1.383 | 0.04506 | 0.11712 | 0.00652 | 0.272 | 0.385 |
| 10 | 2025 | 0.446 | 0.267 | 1.314 | 0.04909 | 0.14207 | 0.01742 | 0.339 | 0.346 |

四个规模的 H@C 与 solver 全部达到预先定义的 `Metal/MPS ≤ 1.10` 目标。正常收敛测试中 36 determinants 的 Metal/MPS 都是 35 iterations，最终复测独立残差约 `1.18e-5/1.17e-5`，通过 `1e-4` 外部残差门槛。其余 40-cycle 正常测试未收敛，仍只能作为固定工作量 profile。

### 6.7 达到 MPS 相当之前的最后瓶颈

加入 MPS 后的基线中，H@C 已快于 MPS，但 2025 determinants solver 的 Metal/MPS 比值为 `1.128`，略超目标。由于同规模 H@C 比 MPS 快约三倍，瓶颈定位到 Davidson 的投影与正交化，而不是 H@C。

原 `K×N @ N×K` 和 `K×N @ N×1` 长内积每个输出只用一个线程，分别只有 `K²` 或 `K` 个线程。最终采用每输出 256-lane 的确定性 tree reduction：lane 内固定 stride 累加，threadgroup 中按 128→1 树形归约。该修改通过全部 12 项 golden，并将 2025 determinants solver 从约 `0.1425s` 降至 `0.05194s`，最终 Metal/MPS 为 `0.406`。

此前将残差范数和修正向量范数合并为一次 `mx.eval`，在相同迭代和残差下获得约 5%–14% solver 收益，也予以保留。

### 6.8 新增负结果

- **四输出共享 matrix 读取**：每个 MSL 线程计算四个相邻输出，数值 golden 通过，但四个规模均变慢；分支、寄存器和临时访问成本抵消复用收益，已回退。
- **Davidson 直接使用 MLX GEMM**：2025 determinants solver 虽从约 `0.14s` 降至 `0.058s`，但独立残差从约 `2.8e-3` 恶化到 `0.33`，不能采用。性能提升不能抵消求解错误。

这些负结果进一步说明必须分别检查 H@C golden 和 solver 独立残差；H@C 正确并不能保证使用低精度长向量 GEMM 后 Davidson 仍正确。

### 6.9 大规模 CPU FP32 break-even（2026-11-19）

为排除 FP32 GPU 与 FP64 CPU 的精度混淆，新增 CPU FP32 固定 40-cycle Davidson，与 Metal/MPS 使用相同初猜、`max_space=12` 和 40 次 H@C；CPU FP64 仅保留为求解参考。下表均为当前自定义 Metal 内核、5 repeats 的中位数，H@C 单位为毫秒，solver 单位为毫秒。`H ratio` 和 `solver ratio` 均为 Metal/CPU FP32，小于 1 才表示 Metal 更快。

| determinants | Metal H@C | CPU32 H@C | H ratio | Metal solver | CPU32 solver | solver ratio | 轮次 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4356 | 1.531 | 0.646 | 2.372 | 81.773 | 29.896 | 2.735 | 首轮 |
| 8281 | 3.257 | 1.463 | 2.226 | 150.367 | 65.005 | 2.313 | 首轮 |
| 11025 | 3.293 | 2.114 | 1.558 | 161.529 | 101.629 | 1.589 | 首轮 |
| 11025 | 3.796 | 2.421 | 1.568 | 93.841 | 103.247 | 0.909 | 独立复测 |
| 14400 | 3.734 | 3.273 | 1.141 | 141.829 | 150.682 | 0.941 | 首轮 |
| 14400 | 4.467 | 3.210 | 1.392 | 140.523 | 151.588 | 0.927 | 独立复测 |
| 18496 | 5.136 | 5.615 | 0.915 | 206.774 | 234.903 | 0.880 | 首轮 |
| 18496 | 4.452 | 5.646 | 0.789 | 206.272 | 245.956 | 0.839 | 独立复测 |
| 23409 | 6.823 | 7.874 | 0.867 | 305.520 | 326.128 | 0.937 | 首轮 |
| 23409 | 6.890 | 8.131 | 0.847 | 306.913 | 327.687 | 0.937 | 独立复测 |

结论：11025 determinants 的 solver 比值两轮跨过 1，属于不稳定交叉点；14400 determinants 两轮均达到 CPU FP32 solver break-even，因此最早稳定点为 14400。H@C 在 18496 和 23409 determinants 达到并经复测，实测交叉区间为 14400–18496。新增规模 H@C 相对 FP64 golden 误差约 `1.68e-7–2.44e-7`。这些 40-cycle 大规格求解均未收敛，独立残差仍较大，只能用于性能 profile，不能作为基态能量结果。

大规格也改变了 MPS 对照：23409 determinants 的 Metal/MPS solver 比值两轮为 `1.188/1.463`，即 MPS solver 更快；早期 36–2025 determinants 的 Metal 快于 MPS 结论不得外推。14400 H@C 和 11025 solver 的两轮波动也要求同时保留原始 samples，不能只选择较快一轮。

### 6.10 完整 FP32 MLX GEMM A/B 与采用决定

`MLX_ENABLE_TF32=0` 的 `mx.matmul` 已通过 36–23409 determinants 的全部实测 H@C golden，误差与自定义路径同量级；36 determinants 正常收敛独立残差为 `1.16e-5`，固定 solver 残差也没有精度恶化。但性能收益不具备跨进程一致性。下表为 solver 中位耗时（毫秒），比值小于 1 表示 MLX 更快。

| determinants | 自定义 Metal | 完整 FP32 MLX | MLX/custom | 轮次 |
| ---: | ---: | ---: | ---: | --- |
| 36 | 43.671 | 41.043 | 0.940 | 首轮 |
| 225 | 43.043 | 44.560 | 1.035 | 首轮 |
| 784 | 45.222 | 47.767 | 1.056 | 首轮 |
| 2025 | 50.084 | 60.856 | 1.215 | 首轮 |
| 4356 | 81.773 | 90.507 | 1.107 | 首轮 |
| 8281 | 150.367 | 96.810 | 0.644 | 首轮 |
| 8281 | 150.450 | 166.557 | 1.107 | 独立复测 |
| 11025 | 161.529 | 131.471 | 0.814 | 首轮 |
| 11025 | 93.841 | 96.023 | 1.023 | 独立复测 |
| 14400 | 141.829 | 146.433 | 1.032 | 首轮 |
| 18496 | 206.774 | 215.185 | 1.041 | 首轮 |
| 23409 | 305.520 | 316.221 | 1.035 | 首轮 |

8281/11025 的首轮收益在独立复测中消失，其余多数规模较慢；H@C 也没有稳定、可归因的一致收益，break-even 区间未提前。因此生产继续保留自定义确定性归约，完整 FP32 MLX 仅保留为 `examples/sqd_metal_scaling.py --metal-matmul mlx-full-fp32` 的研究路径。该候选依赖进程级 `MLX_ENABLE_TF32=0`，不得在库导入时静默改变用户其他 MLX 运算。

后续源码阅读优先级为 MLX 官方 STEEL `gemm.h/loader.h/mma.h/params.h`，其次是 MIT 许可的 `tsotchke/tensorcore` FP32 SIMD-group 实现。`LaurentMazare/gemm-metal` 和 `lenguyen1807/metal-lab` 用于研究分块与 profiling，但未见明确许可证且后者 `1e-3` 验证容差不满足 SBD，不能直接搬代码或精度结论。DeepSeek DeepGEMM 面向 NVIDIA SM90/SM100、CUDA 和 FP8/FP4/BF16；GLM-5 官方仓库是模型/部署入口，均未发现可直接用于 Apple FP32 GEMM 的实现。

### 6.11 真实分子 CAS 的单次 SQD 整体加速（2026-11-19）

仅比较热点矩阵不足以说明用户收益，因此 `examples/sqd_gpu_benchmark.py` 新增 `--active-space NELEC NORB`，从真实分子的 canonical HF 轨道构造 CAS 有效积分和完整组态空间。整体墙钟使用公共 `make_gpu_sci_solver(backend="gpu"/"cpu")`，包含链接表/setup、正常收敛 Davidson、CPU RDM 和 validation；冷启动排除，稳态 5 repeats。Metal 为 FP32 长向量混合精度，CPU 为 PyTorch FP64，两者采用相同 `2e-5` 求解容差、`1e-4` 独立残差和 RDM 能量门槛。小体系 H2O CAS(2e,4o) 与 CAS(4e,4o) 另以 `full_cpu` 独立 operator/RDM validation 通过。

H2O（3 atoms、24 AO、6-31g(d,p)）的整体中位墙钟如下；比值是 CPU/Metal，大于 1 表示 Metal 加速。

| CAS | determinants | Metal wall (ms) | CPU wall (ms) | CPU/Metal | 轮次 |
| --- | ---: | ---: | ---: | ---: | --- |
| (4e,4o) | 36 | 400.342 | 399.081 | 0.997 | 首轮 |
| (4e,8o) | 784 | 407.628 | 401.139 | 0.984 | 首轮 |
| (4e,12o) | 4356 | 424.951 | 407.686 | 0.959 | 首轮 |
| (4e,16o) | 14400 | 466.265 | 469.065 | 1.006 | 首轮 |
| (4e,16o) | 14400 | 454.003 | 462.736 | 1.019 | 独立复测 |
| (4e,18o) | 23409 | 505.262 | 541.463 | 1.071 | 首轮 |
| (4e,18o) | 23409 | 504.416 | 564.630 | 1.119 | 独立复测 |

所有上述运行均正常收敛并通过残差/RDM 验收，迭代次数一致。H2O CAS(4e,18o) 的 solver 阶段两轮为 Metal `73.051/74.262 ms`、CPU `101.266/105.867 ms`，但 CPU RDM、setup、validation 和约 390–415 ms 的当前子进程路径稀释了整体收益。因此论文应同时展示整体和分阶段数据，不能把 solver 加速比直接当作整体加速比。

相同 CAS(4e,16o)、14400 determinants 下的分子系列：

| 分子 | Atoms | AO | Metal wall (ms) | CPU wall (ms) | CPU/Metal |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2O | 3 | 24 | 454.003 | 462.736 | 1.019 |
| NH3 | 4 | 29 | 454.455 | 464.268 | 1.022 |
| CH4 | 5 | 34 | 445.351 | 465.436 | 1.045 |

这些结果说明在真实分子单次固定 CAS 的 SQD 求解中，14400 determinants 附近已观察到整体 break-even，23409 determinants 的整体墙钟两轮降低约 `6.7%/10.6%`。分子系列的积分和迭代数并不完全相同，不能把差异 solely 归因于原子数。当前结果不覆盖完整迭代 SQD 采样/恢复、PCM 或 CUDA；CUDA 对应整体数据仍需在 NVIDIA 环境补测。

## 7. 测试与证据清单

`tests_applications_chem/test_sqd_metal_backend.py` 当前覆盖：

- 完整/部分 determinant 空间和多个 `(alpha,beta)` 电子数；
- alpha same-spin、beta same-spin、mixed-spin；
- 不同 beta chunk；
- H@C 与 PySCF FP64 对照；
- S² 与 PySCF 对照；
- 端到端能量、独立最终残差和 RDM 能量；
- 自旋惩罚与空自旋 sector；
- 强制全局 `mx.cpu` 后的真实 GPU stream；
- 原生 H@C gather/pair-contract 和 S² gather 调用次数；
- eval 完成计数和调用方设备恢复。

注意：这些测试是正确性和真实执行门槛，不是性能测试。最终相关测试（含 MPS、单变量分发、CPU 回退、PySCF/scanner、公共 numerics 和 SBD 适配）为 `111 passed, 29 skipped`；跳过项均因本机无 CUDA 或未安装 CuPy，不是 Metal/MPS 失败。

## 8. 后续优化路线

每项优化单独建立假设、测量、golden 和回退点，不将多个改动混成无法归因的提交。

### 已完成：Davidson 长内积并行归约

小输出长内积已采用固定 256-lane、确定性 tree reduction，解决 `K²/K` 线程不足，并完成四规模 MPS 相当验收。四输出 tile 方案为负结果，不应直接恢复。后续若继续优化 pair 收缩，应重新基于 final profile 选择 same-spin/mixed-spin 各自的数据复用方式，并维持逐元素 golden 和 solver 独立残差。

### P0：减少小内核与 host 往返

- 重新 profile gather、pair、MLX `.at.add` 和 Davidson 各阶段。
- 批量物化可以独立计算的能量、范数和残差；不得为了省同步而使用未归一化或过期系数。
- 评估小 scalar 是否真正需要当轮读取，避免无意义 host 往返。

验收：工作量、迭代轨迹和最终残差不变；任何收敛轨迹变化都必须先排除精度问题。

### 已完成：扩大规模与寻找真实交叉点

- 已覆盖到 23409 determinants，并为所有正式性能点执行独立 FP64 H@C golden。
- 自定义 Metal 的固定 solver 从 14400 determinants 起稳定达到 CPU FP32 break-even；H@C 从 18496 determinants 起经双轮确认超过 CPU FP32。
- 若继续扩大规模，必须保持相同协议并记录内存、原始 samples 和收敛状态；不得把当前交叉点外推到未测规模。

### P1：RDM 原生化的前置条件

当前 RDM 在 CPU。只有 profiling 证明 RDM 是实际应用瓶颈时才实现 MLX RDM；否则先优化 H@C/Davidson。原生 RDM 必须另行验证 PySCF 指标约定、粒子数、能量和 S²，不得沿用 H@C 正确就推断 RDM 正确。

## 9. 实验记录模板

每次实验至少记录：

1. 日期、代码/内核哈希、MLX/PyTorch/PySCF 版本和硬件。
2. 假设与只改变的变量。
3. determinant 形状、chunk、dtype、repeats、warmup 和同步方法。
4. 原始 samples、中位数、范围、setup 和 solver 工作量。
5. H@C/S²/能量/残差 golden 结果。
6. 保留或回退决定及原因。

## 10. 论文写作边界

可以说：

- 提出了单 Apple GPU 的 MLX/MSL SBD 核心实现。
- 给出了真实 GPU stream、原生调用和独立 CPU FP64 golden。
- 量化了 slot/link 融合对 Metal 自身耗时的影响，并记录了 CSR 和四输出方案的负结果。
- 在 36–2025 determinants 的统一 FP32 固定工作量基准中，原生 Metal 的 H@C 与 Davidson 均达到预设的 MPS 性能相当目标；该结论不外推到更大规模或未收敛能量。
- 自定义 Metal 的固定 40-cycle solver 在 14400 determinants 经双轮确认快于 CPU FP32，H@C 在 18496/23409 determinants 经双轮确认快于 CPU FP32；这些大规格能量均未收敛。
- 真实分子正常收敛的单次固定 CAS SQD 求解中，H2O CAS(4e,16o) 附近观察到整体 break-even；CAS(4e,18o) 的整体墙钟两轮降低 `6.7%/10.6%`。H2O/NH3/CH4 的分子系列和分阶段耗时见 §6.11。
- 完整 FP32 MLX GEMM 可通过全部实测 golden，但性能收益不稳定，当前没有采用。

当前不能说：

- Metal 在所有规模都已快于 PyTorch CPU，或已经验证快于 CUDA。
- Metal 在更大规模仍快于 MPS；23409 determinants 的 MPS solver 实测更快。
- 使用 GPU 就等价于获得加速。
- 未收敛的 40-cycle 能量是正确基态结果。
- 当前 CPU RDM 是全链路原生 Metal RDM。
- 单设备结果覆盖 CUDA MPI/NCCL、多节点、多 GPU 或 FP64 全功能。
- 单次固定 CAS 的整体求解收益等价于完整迭代 SQD 采样/恢复收益；CUDA 整体对照尚未完成。

后续论文图表必须来自统一 repeats 的最终数据和原始 JSON，并同时展示工作量、精度和残差，不只展示 wall time。

## 11. 交接复现入口

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python -m pytest \
  tests_applications_chem/test_sqd_metal_backend.py -q

/opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python \
  examples/sqd_metal_scaling.py \
  --norb 4 6 8 10 12 14 15 16 17 18 --repeats 5 --output metal_scaling.json

MLX_ENABLE_TF32=0 /opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python \
  examples/sqd_metal_scaling.py --metal-matmul mlx-full-fp32 \
  --norb 4 6 8 10 12 14 15 16 17 18 --repeats 5 --output metal_mlx_full_fp32.json

/opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python \
  examples/sqd_gpu_benchmark.py --case water_pol --active-space 4 18 \
  --backend gpu cpu --repeats 5 --reference none --validation gpu \
  --tolerance 2e-5 --operator-residual-tolerance 1e-4 \
  --rdm-energy-tolerance 1e-4 --max-cycles 100 --max-space 20 \
  --output water_cas4e18_overall.json

/opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python \
  examples/sqd_gpu_benchmark.py --case water_pol ammonia_pol methane_pol \
  --active-space 4 16 --backend gpu cpu --repeats 5 --reference none \
  --validation gpu --tolerance 2e-5 --operator-residual-tolerance 1e-4 \
  --rdm-energy-tolerance 1e-4 --max-cycles 100 --max-space 20 \
  --output molecule_cas4e16_overall.json
```

继续工作前先阅读当前 `metal/solver.py` 和 `metal/kernels.metal`，不要从已回退的 CSR 实验重新起步。固定 CUDA 构建和论文来源见 `docs/SBD_REPRODUCIBILITY.md`。
