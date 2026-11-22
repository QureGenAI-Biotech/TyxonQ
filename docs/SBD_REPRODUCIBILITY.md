# SBD 来源、构建与数值复现

Metal 优化时间线、数据布局、保留/回退实验和后续路线见 `docs/SBD_METAL_OPTIMIZATION.md`。

## 原始算法与固定版本

SBD（Selected Basis Diagonalization）核心上游为 RIKEN `r-ccs-cms/sbd`。GPU 算法参考论文为 **arXiv:2601.16637**；实现调研覆盖 TPB H@c 三类任务、连接表、Davidson 和 correlation/RDM 内核。

固定资源由 `src/tyxonq/numerics/sbd/cuda/SOURCES.json` 唯一记录：

- 核心：`https://github.com/r-ccs-cms/sbd.git`
- 核心提交：`93ebabec6c841d0ade706018b053795d8c1db9bd`
- Python 包装：`https://github.com/Qiskit/sbd-eigensolver-python`
- 包装提交：`b49c22b54813be1e0344ba52f8ffd03e3825433e`
- 许可证：Apache-2.0
- 网络受限镜像：`https://gitee.com/asskick/sbd` 与 `https://gitee.com/asskick/sbd-eigensolver-python`

`SOURCES.json` 保存所有使用文件的 SHA256。`nvcc.patch` 包含两遍修正 Gram–Schmidt、近线性相关方向处理和最终独立残差等本地稳定性修正；其哈希进入构建缓存目录和 provenance。

## CUDA 构建

依赖：NVIDIA CUDA/nvcc、OpenMPI、pybind11、mpi4py、OpenBLAS、Python 开发头文件和 git。

```bash
python -m pip install 'tyxonq[cuda]'
python -m tyxonq.numerics.sbd.cuda.build --output-dir /absolute/path/to/sbd-build
export TYXONQ_SBD_EXTENSION_DIR=/absolute/path/to/sbd-build
```

`prepare_source()` 只接受哈希完全匹配的固定源码，在独立缓存中复制并应用补丁，不修改用户传入的源码目录。构建结束写出 `sbd_build.json`，记录完整命令、nvcc/MPI 版本、绑定和内核哈希、补丁后源码哈希以及扩展哈希。

本机 Apple Silicon 验收未执行 CUDA 编译或数值测试；这些必须由 NVIDIA 环境单独完成，不得把跳过标记解释为通过。

## Metal 构建与依赖

Metal 不需要独立 C++ 扩展或 metallib 构建：

```bash
python -m pip install 'tyxonq[metal]'
```

运行时由 `mlx.core.fast.metal_kernel` JIT 编译随包提供的 `numerics/sbd/metal/kernels.metal`。直接调用原生 Metal 入口时，MLX/Metal 不可用会显式报错；通过公共 `backend="gpu"/"metal"` 分发时，环境不可用会警告并记录 CPU 回退。JIT 或原生数值内核失败始终报错，不触发数值回退。

## 固定 golden 协议

使用同一组 `h1e`、化学排序 `eri`、`ci_strings` 和电子数，先归一化 FP64 系数，再分别比较：

1. **H@c**：PySCF `direct_spin1.absorb_h1e(..., fac=0.5)` 与 `selected_ci.contract_2e` 作为独立 FP64 golden；覆盖完整和部分 determinant 空间、alpha/beta/mixed-spin 以及不同 beta 分块。
2. **S²**：PySCF `selected_ci.contract_ss`；检查跃迁、费米子符号和空占据边界。
3. **能量与残差**：收敛后重新归一化，独立计算一次最终 H@c；检查 Rayleigh 能量、最终残差、CPU H@c 逐元素误差。
4. **RDM 与自旋**：PySCF RDM 检查粒子数、能量和 S²；Metal S² 期望值必须与 CPU RDM 一致。
5. **真实执行**：断言 `native_metal_calls` 中 H@c gather/pair-contract 和 S² gather 均大于零，避免只注册接口的假通过。

Metal FP32 golden 使用分级容差；测试默认 H@c/S² 为 `2e-5/2e-6`，端到端最终残差、算符和 RDM 能量为 `1e-4`，S² 与 RDM 一致性为 `2e-5`。不得通过放宽容差掩盖显著算法错误。

本机复现命令：

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python -m pytest \
  tests_core_module/test_numerics_metal_arrays.py \
  tests_applications_chem/test_sqd_metal_backend.py -q
```

端到端复现仍可使用 `examples/sqd_gpu_benchmark.py`，分别记录 setup、求解、RDM 和 validation 时间。性能优化对照使用 `examples/sqd_metal_scaling.py`，不要以独立 CPU validation 主导的端到端墙钟评价 H@C 内核。

## 三组性能与 CPU break-even 复现（2026-11-19）

环境：Apple Silicon、MLX 0.32.4、PyTorch MPS FP32、PyTorch CPU FP32/FP64。MPS 仅作性能参照，不是 SQD 自动后端。Metal、CPU FP32 和 MPS 固定 solver 使用相同 40 cycles、初猜、`max_space=12` 和 H@C 次数；CPU FP64 仅作高精度参考，正常收敛测试另行执行。

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python \
  examples/sqd_metal_scaling.py \
  --norb 4 6 8 10 --repeats 5 --output metal_three_way.json

/opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python \
  examples/sqd_metal_scaling.py \
  --norb 12 14 15 16 17 18 --repeats 5 --output metal_break_even.json
```

每轮 Metal 显式 `mx.eval`，MPS 显式 `torch.mps.synchronize()`，`PYTORCH_ENABLE_MPS_FALLBACK=0`。MPS 不支持 `eigh`，只允许将 Davidson 小投影矩阵显式放到 CPU FP64，长向量和 H@C 不得回退 CPU。输出保存全部 samples、范围、中位数、精度、工作量、独立残差以及 `metal_to_cpu32_*_ratio` 和 break-even 布尔值。

36–2025 determinants 已达到预设 Metal/MPS `≤1.10` 目标。扩展规模中，固定 solver 在 14400 determinants 经双轮确认达到 CPU FP32 break-even；H@C 在 18496/23409 determinants 经双轮确认达到。11025 solver 两轮跨过 1，不属于稳定交叉点。23409 的 MPS solver 快于 Metal，因此不得外推小规模 MPS 结论。所有大规格固定循环均未收敛，不能作为基态能量结果。

完整 FP32 MLX GEMM 候选必须在进程启动前关闭默认降精度路径：

```bash
MLX_ENABLE_TF32=0 /opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python \
  examples/sqd_metal_scaling.py --metal-matmul mlx-full-fp32 \
  --norb 4 6 8 10 12 14 15 16 17 18 --repeats 5 \
  --output metal_mlx_full_fp32.json
```

该路径通过全部实测 golden，但性能收益未通过独立 A/B 复测，生产仍使用自定义确定性归约。详细双轮数据、原始样本波动和负结果见 `docs/SBD_METAL_OPTIMIZATION.md`。

## 真实分子 CAS 整体复现

以下基准是单次固定 CAS 的 SQD selected-CI 求解，包含 setup、正常收敛 solver、CPU RDM 和 validation，不是完整迭代 SQD 采样恢复。`gpu` 在本机自动解析 Metal，`cpu` 解析 PyTorch CPU；输入指纹、分子原子数、AO 数、活性空间和分阶段耗时均写入 JSON。

```bash
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

# 多分子平衡点：六个规模、两种后端、两个独立轮次，串行执行。
QC=/opt/homebrew/Caskroom/miniconda/base/envs/qc/bin/python
for round in 1 2; do
  for norb in 12 14 15 16 17 18; do
    PYTHONPATH=src "$QC" examples/sqd_gpu_benchmark.py \
      --case water_pol ammonia_pol methane_pol --active-space 4 "$norb" \
      --backend gpu cpu --repeats 5 --reference none --validation gpu \
      --tolerance 2e-5 --operator-residual-tolerance 1e-4 \
      --rdm-energy-tolerance 1e-4 --max-cycles 100 --max-space 20 \
      --output "molecule_n${norb}_round${round}.json" || exit 1
  done
done
```

历史单点中 H2O CAS(4e,16o) 附近观察到整体 break-even。新的六点双轮扫描进一步区分体系：NH3/CH4 在 18496 determinants 两轮均越过 CPU，H2O 在该规模仍反转，三种分子在 23409 determinants 全部稳定加速，整体墙钟降低约 `5.6%–9.4%`。所有 36 组配对负载正常收敛并通过残差/RDM 验收；完整比例和墙钟见 `docs/SBD_METAL_OPTIMIZATION.md` §6.12。原始归档为 `data/sbd_benchmarks/multi_molecule_break_even_2026-11-19.json`（SHA256 `ba09e50becc00a5dbbfe327fc5385e2408cb1d15496cb2b58d227e5fca54b58c`）。

小体系正确性轨将 `--active-space 4 4 --repeats 1 --reference none --validation full_cpu`，把独立 CPU operator/RDM 验收与性能轨分开。CUDA 整体对照仍需 NVIDIA 环境。
