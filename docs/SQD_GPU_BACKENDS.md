# SQD GPU 后端架构

## 两个独立场景

通用数组由 `tyxonq.numerics.get_backend()` 提供 NumPy、CuPy、PyTorch 和 MLX。SQD selected-CI 是专用 SBD 求解流程，不把 SBD 当作通用数组后端。

两个场景的自动硬件优先级一致：

1. CUDA 可用：通用数组使用 CuPy；SQD 使用原生 CUDA SBD。
2. CUDA 不可用、Metal 可用：通用数组使用 MLX；SQD 使用原生 MLX/Metal SBD。
3. CUDA 和 Metal 均不可用：通用数组与 SQD 使用 PyTorch CPU。

`get_backend(None)` 仍遵循全局配置和 NumPy 默认行为；只有显式 `get_backend("auto")` 使用上述自动选择。通用 PyTorch 数组仍可显式请求 CUDA、MPS 或 CPU，但 SQD 不自动选择 PyTorch CUDA/MPS。

## 单变量选择与环境不可用回退

SQD 用户只需设置同一个 `backend=` 变量：`"gpu"` 或 `"auto"` 自动按 CUDA SBD → MLX/Metal SBD → PyTorch CPU 探测，`"cpu"` 固定 PyTorch CPU；也可显式指定 `"cuda"`/`"sbd"` 或 `"metal"`。自动探测同时检查 CUDA 原生扩展和 MLX 运行时，不只看芯片名称。

手动指定的加速器、设备或必需运行库不符合当前环境时，会发出 `RuntimeWarning` 并回退 PyTorch CPU，报告以 `used_cpu_fallback=True` 和 `selection_reason` 明确记录。**这只覆盖分发前的环境不可用**；原生内核异常、不收敛或残差/RDM 超标仍直接报错，不触发其他后端。

CPU FP64 golden 是交付验收门槛，不是运行时选择条件。SQD 已删除 PyTorch MPS + Metal 热点混合路径和静态 `metal_hot_kernels` 假报告。

PyTorch MPS 只出现在 `examples/sqd_metal_scaling.py` 的三组性能研究中，用于同精度 GPU 对照，不进入 SQD 自动求解。最终 36–2025 determinants 的 H@C 与固定工作量 Davidson 均达到 Metal/MPS 中位耗时比 `≤1.10`；数据与限制见 `docs/SBD_METAL_OPTIMIZATION.md`。

## 原生 SBD 实现边界

### CUDA

`numerics/sbd/cuda/solver.py` 保留 RIKEN SBD 的 CUDA FP64 路径。固定源码、补丁和哈希见 `cuda/SOURCES.json`；构建链为 nvcc、OpenMPI、pybind11、mpi4py 和 OpenBLAS。`numerics/sbd/__init__.py` 是公共算法入口，CUDA 与 Metal 实现位于平级子包。

### 单设备 Metal

`numerics/sbd/metal/solver.py` 与 `numerics/sbd/metal/kernels.metal` 实现：

- alpha same-spin、beta same-spin 和 mixed-spin 三类 H@c；
- MLX 常驻连接表、系数、算符像和 Davidson 长向量；
- 两遍 Gram–Schmidt、厚重启和独立最终 H@c 残差；
- 原生 S² 跃迁、费米子符号与散射；
- setup 后缓存元数据，算子内部记录真实原生调用次数。

PySCF 连接表构建和 Davidson 小矩阵 Hermitian FP64 `eigh` 保留在 CPU，这与原算法的主机阶段一致。长向量矩阵收缩使用确定性 MSL FP32 累加，不依赖进程全局 MLX GEMM 精度设置。研究完整 FP32 `mx.matmul` 时必须在启动进程前设置 `MLX_ENABLE_TF32=0`；默认 NAX 降精度路径不满足 SBD golden，完整 FP32 路径虽正确但无一致性能收益。

Metal RDM 默认使用 CPU，报告明确为 `rdm_backend="cpu"`。可选 `rdm_backend="metal"` 只加速异自旋 2-RDM，1-RDM 和同自旋 2-RDM 继续使用 CPU FP64；Q60 补偿、归一化、规模上限和独立 golden 失败均显式报错，不能称为全链路原生 RDM。H@c、S²、能量和残差均通过 CPU FP64 golden。

不承诺 MPI/NCCL、多节点、多 GPU determinant slide、分布式 Allreduce 或端到端 FP64 的 CUDA 全功能等价迁移。

## target-spin 与最低根行为

`solve_sci`/`solve_sci_gpu` 的五位置参数、返回结构和后端优先级不变。Metal 的 `spin_tolerance` 是 keyword-only，默认 `1e-4`。设置 `spin_sq` 后，求解器按实际 S² 过滤目标子空间中的收敛根并选择目标根；空间不存在合法根时显式失败。无自旋约束的最低根若在 variational-bound 检查中失败，会按内部策略重试，而不是返回已知错误根。

报告区分最终所选根与全部求解尝试的 H@C、迭代和耗时；`gpu_history` 记录物理能量，不是把软自旋惩罚当作物理能量的本征值。选根与重试属于行为修正，不承诺任意初猜下都与旧实现返回同一个根。

## RDM 与持久 CPU validator

生产 RDM 通过 `spin_rdms(..., algorithm="auto")` 自动选择 factorized 路径，阈值外保留 PySCF 回退。`full_cpu` 独立验收强制 `algorithm="pyscf"`，直接调用原始 PySCF `make_rdm1s/make_rdm2s`，并在报告中记录实际参考算法；生产优化实现不能同时充当自己的 golden。

POSIX 上，独立 CPU 验收复用一个隔离 PySCF 解释器，但每个请求重新发送当前系数、积分、标签、线程数和参考模式，不复用上次数值。线程变化、异常或 worker 死亡会重建；fork 子进程只关闭继承副本，多线程调用串行提交，进程退出由父管道守护回收。其他平台保留一次性 CLI。返回的 batch solver 调用 `close()` 时同时释放设备进程池和当前进程 validator；数值错误继续向上传播。

## 报告与兼容入口

`BackendResolution` 和报告包含 `requested_backend`、`resolved_backend`、`device`、`selection_reason`、`precision_mode`、`rdm_backend`、`used_cpu_fallback`。旧 `backend` 键继续保留。Metal 还报告 `native_metal_calls` 以及 setup、solver、RDM、validation 分阶段耗时。

统一入口：

```python
from tyxonq.applications.chem.algorithms.sqd.gpu import solve_sci_gpu

result, report = solve_sci_gpu(
    ci_strings, h1e, eri, norb, nelec, backend="gpu"  # 或 "cpu"
)
```

`solve_sci_cuda`、`make_cuda_sci_solver`、`make_sbd_sci_solver`、`solve_sci_sbd` 等旧名称作为兼容别名保留。`as_pyscf_solver` 和 `SQDFCISolver` 的 `backend=` 继续透传到统一出口。

## 已移除设施

- 未接入真实求解链的 `MetalSolverBackend`、Objective-C++/pybind11 和独立 metallib 构建链；
- cuPyNumeric（上游已 EOL）及 vectorization 遗留 API；
- SQD PyTorch CUDA/MPS 自动求解和 Metal 数值异常回退；环境不可用回退不属于此移除项。

CUDA-Q 不是 NumPy 式通用数组后端，不纳入该架构。
