# Numerics 全框架加速审计

审计范围：`devices`、`libs`、`compiler`、`applications/chem`、`postprocessing`。初始证据来自逐文件静态扫描；2026-11-19 已补做真实负载 profile、双轮整体复测和独立回归。实测数据、原始 samples 与采用边界见 `docs/NUMERICS_OPTIMIZATION_REPORT.md`；静态优先级不再等同于实测加速比。

## 后端边界

- 通用自动数组：CuPy CUDA → MLX Metal → PyTorch CPU；显式 NumPy/PyTorch 行为保持兼容。
- SQD：CUDA SBD → Metal SBD → PyTorch CPU，不复用 CuPy 作为 SCI 求解器。
- MLX Metal GPU 的 FP64/complex128、`expm` 和数据依赖 `nonzero` 不可静默主机回退。
- CuPy `expm` 当前仍经 NumPy/SciPy host，不能视作全设备实现。
- CPU 独立 golden 的正确性价值高于加速价值，不迁移为共享设备内核。

## devices

| 证据 | 算子与现状 | 建议与收益 | 风险 | 优先级 |
| --- | --- | --- | --- | --- |
| `devices/simulators/statevector/engine.py:42-44,68-76`；`libs/quantum_library/kernels/statevector.py:28-59,71-129` | 1q/2q/k-qubit 门已用 `get_backend`、`einsum/matmul` | 保持 ArrayBackend；主热点已覆盖 | Metal complex128 限制 | P3 已覆盖 |
| `devices/simulators/density_matrix/engine.py:40-64,88-96`；`libs/quantum_library/kernels/density_matrix.py:20-63,117-164` | 密度矩阵门/Kraus 已用后端 `einsum` | 保持；稠密 `UρU†` 已覆盖 | 低 | P3 已覆盖 |
| `devices/simulators/matrix_product_state/engine.py`；`libs/quantum_library/kernels/matrix_product_state.py` | **已采用**：引擎 backend 贯穿初始化、门、SVD、截断和重构；SVD 错误不再隐式回退 NumPy | NumPy 整体中位耗时 large 双轮改善 13.04%/16.93%；medium 为 8.20%/9.55%；small 仅约 5% 边界 | 截断只做标量 host 秩决策；不支持精度/设备 loudly raise；不是 GPU 性能结论 | **P1 已完成** |
| `devices/simulators/statevector/engine.py:675-723,741-752,772-787` | 全局 ZZ 串扰构造 `2^n×2^n` NumPy 矩阵、`kron`、`scipy.linalg.expm` | PyTorch `matrix_exp` 或 Hermitian 谱路径；中等收益 | 稠密指数内存、复数精度、host 往返；MLX 无 `expm` | **P2** |
| `statevector/engine.py:458-462`；`density_matrix/engine.py:169-180`；`matrix_product_state/engine.py:220-233` | `expval` 转 NumPy/OpenFermion，density 还 `.toarray()` | 默认保留 CPU 稀疏；大型重复问题再评估 CuPy sparse | 稀疏格式、比特顺序、稠密化成本 | P3 |
| `statevector/engine.py:349-380,444-445`；`density_matrix/engine.py:114-145`；`matrix_product_state/engine.py:153-190` | 概率、readout、采样和 counts 序列化跨 host | 保持 host 边界，可局部复用后端 matmul | RNG 与 counts 顺序 | P3 |

## libs

| 证据 | 算子与现状 | 建议与收益 | 风险 | 优先级 |
| --- | --- | --- | --- | --- |
| `libs/quantum_library/kernels/statevector.py:28-59`；`density_matrix.py:20-63`；`matrix_product_state.py:58-70,128-137`；`gates.py:222-281` | 通用门与主要收缩已统一 | 不新增平行 NumPy/CuPy 栈 | MPS SVD 例外见上 | P3 已覆盖 |
| `libs/quantum_library/kernels/quantum_info.py` | **已 profile，保留 NumPy**：large dense=256 整体 60.363 ms；pure+mixed RDM 仅 0.330 ms（0.55%），fidelity 占约 60.4% | RDM 即使完全消除也达不到 5% 整体门槛；谱分解留待可用的同精度 CUDA 实测 | complex128 Hermitian、指数内存；MLX/MPS 不降精度冒充加速 | **暂缓** |
| `libs/hamiltonian_encoding/pauli_io.py:137-145` | `rdm_mo2ao` 连续四次 `np.tensordot` | CuPy/PyTorch 多轴收缩；高带宽收益 | 轴序和转置逐元素回归 | **P1** |
| `libs/quantum_library/dynamics.py:100-126` | 主运算已后端化，但每步 `to_numpy` 读取归一化标量 | 减少逐步同步；步数大时中等收益 | 分支和 autograd 语义 | **P2** |
| `libs/quantum_library/pulse_simulation.py:466-516,582-594,634-698`；`three_level_system.py:235-272,339-346` | SciPy `solve_ivp` 与 PyTorch 可微 Euler 双栈，矩阵仅 2×2/3×3 | 保留；GPU 不划算 | 积分精度和梯度语义不同 | P3 |
| `libs/circuits_library/variational.py:78-94,107-134` | QNG/时间变分的 `eigh`、Jacobian、`expm` | PyTorch CPU/CUDA；重复大问题时中等收益 | SciPy 优化器、复数梯度、正则化 | **P2** |

## compiler

主 gate/pulse 编译、分解、布局、调度和 lowering 没有显著数组热点。`compiler/gate_compiler/stages/decompose/matchgate.py:46-62,84-112` 只有 4×4/2×2 小矩阵；LQCloud/Guodun 的 NumPy 主要是波形序列化和 SDK 边界，均不迁移。

| 证据 | 算子与现状 | 建议与收益 | 风险 | 优先级 |
| --- | --- | --- | --- | --- |
| `compiler/gate_compiler/interop/mpo_converters.py:40-78` | 可选 MPO 外部格式转换使用 `np.einsum` 并物化 `2^n×2^n` | 仅大型 MPO 评估 CuPy/PyTorch | 外部 duck typing、指数输出和 NumPy 契约 | P3 |
| `compiler/platforms/lqcloud/lqwave_adapter.py:151-152,230-265`；`compiler/platforms/guodun/pulse.py:135-137,159-160,209-211` | 波形转换和有限性检查 | 保持平台方言/SDK 边界 | 低 | P3 |

## applications/chem：SQD 本轮结果

| 证据 | 状态 | 后续建议 |
| --- | --- | --- |
| `applications/chem/algorithms/sqd/gpu/api.py`；`numerics/accelerator.py` | **P0 已完成**：三级硬件分发，依赖和数值错误不降级 | 不新增 CuPy SCI 平行栈 |
| `numerics/sbd/metal/solver.py`；`numerics/sbd/metal/kernels.metal` | **P0 已完成**：三类 H@c、Davidson、S²；36–2025 det 达到 Metal/MPS ≤1.10；固定 solver 在 14400 det、H@C 在 18496 det 经复测达到 CPU FP32 break-even；H2O CAS(4e,18o) 正常收敛整体墙钟双轮降低 6.7%/10.6% | 大规格固定循环未收敛；23409 det 的 MPS solver 更快；分子结果仅为单次固定 CAS，不外推完整 SQD/CUDA；完整 FP32 MLX GEMM 正确但无一致收益，暂不采用 |
| `applications/chem/algorithms/sqd/gpu/solver.py` | **P0 已完成**：PyTorch CPU FP64 可真实运行，无 A100/显存等待限制 | 无 CUDA 时只考虑减少 CPU RDM 重复工作 |
| `applications/chem/algorithms/sqd/gpu/sbd_solver.py` | **本轮保留**：Metal RDM 仍为 CPU；既有真实 CAS 中 solver 仅约 73–106 ms，整体约 500 ms，尚未证明 RDM 单独主导 | 只有新增分阶段 profile 证明 RDM 占主导才实现 MLX RDM；须另行验证 FP32 误差和 PySCF 约定 |
| `applications/chem/algorithms/sqd/gpu/cpu_validate_sci.py` | 独立 PySCF H@c/RDM golden | **明确保留**，不作为性能优化对象 |
| `applications/chem/algorithms/sqd/fermion.py:51-67,666-678`；`gpu/validation.py:173-201` | RDM 能量和不变量直接 NumPy `einsum` | 大轨道且 profiler 证明显著时再评估，P1 |
| `applications/chem/algorithms/sqd/pcm.py:179-189,208-241` | AO/MO/PCM 势 NumPy 收缩，含平台 BLAS 规避 | 谨慎 P1，不可破坏既有 LoongArch64 数值规避 |
| `applications/chem/algorithms/sqd/determinants.py:158-175` | SciPy sparse `eigsh` | 默认保留；大型稀疏用例再评估，P2 |

## applications/chem：其余模块实测状态与 backlog

| 证据 | 算子与现状 | 建议与收益 | 风险 | 优先级 |
| --- | --- | --- | --- | --- |
| `algorithms/ewf/driver.py`；`energy.py`；`density.py`；`bath.py` | **已采用算法级复用**：DMET kernel 复用分片贡献，不再重复 cumulant/ERI 收缩；现有大型 `einsum` 多数已 `optimize=True` | medium 整体双轮改善 5.12%/8.30%，large 为 13.31%/13.08%；small 不稳定 | 全局 DMET 修正仍重算；能量偏差 0；未做无依据 GPU 常驻 | **P1 已完成** |
| `algorithms/ewf/linalg.py`；`bath.py` | 自然轨道与子空间谱分解已随完整 EWF 回归验证 | 本机 FP64 目标下未证明额外迁移收益，保留 NumPy | 本征向量符号、截断和正交性 | **暂缓** |
| `algorithms/lucj/double_factorization.py` | **已采用**：复用 `x0` 的 loss/gradient 供 SciPy 首次调用；restricted target 跨 stage 常驻 | small/medium/large 双轮整体改善 10.89%–20.50%；实际 objective 从 `nfev+1` 降为 `nfev` | loss history 逐值一致；SciPy NumPy 参数/梯度边界保留；不做 CuPy autograd 替代 | **P1 已完成** |
| `algorithms/lucj/linalg.py:59-75`；`double_factorization.py:978-995,1203-1228` | SciPy `logm/expm/eigh/svd` | PyTorch 谱/指数；`logm` 暂保留 SciPy | 分支、截断和旋转相位 | **P2** |
| `algorithms/skqd/basis.py`；`circuit_builder.py` | **已 profile，保留**：合法对称 ERI 旋转在 4/6/8 轨道仅 0.104/0.111/0.141 ms | 现有五操作数 `optimize=True` 已足够快；更大规模须先证明整体收益 | 对称性、实数正交性；不得用非法随机 ERI 做基准 | **P2 暂缓** |
| `dynamics/exact.py:41-50,60-62,75-87,98-103`；`dynamics/observables.py:56-69` | Pauli `kron`、谱投影、多时间点 `einsum` | CuPy/PyTorch；维度 `2^n` 时高收益 | 指数内存、复数精度 | **P2** |
| `dynamics/sqw.py:69-99,124-132,280-282`；`dynamics/spectroscopy.py:46-50,136-139` | 相位矩阵、显式 DFT GEMM、时空关联 `einsum` | CuPy/PyTorch GEMM；均匀网格另评估 FFT | 频率网格、归一化、窗和符号约定 | **P2** |
| `dynamics/ehrenfest.py:49-84`；`dynamics/models/pes.py:67-68`；`dynamics/surface_hopping.py:79-111` | 单轨迹小矩阵逐步分解 | 保持 NumPy；批量轨迹再评估 | launch/同步成本超过收益 | P3 |
| `algorithms/vqe/wavefunction/statevector_ops.py:21,70-83,217-224`；`ucc_base.py:53,1086-1088`；`civector_ops.py:182-195,280-281,318-323` | NumPy/NumericBackend 双栈，PySCF CI 点积仍在 host | statevector 复用统一后端；CI host 边界保留 | autograd 和 PySCF host 契约 | **P2** |
| `classical_chem_cloud/server/gpu_chem.py:18-24,48-56,143-159` | gpu4pyscf 已承担 HF/MP2/DFT | 保持外部加速边界 | 外部对象生命周期 | P3 已覆盖 |

## postprocessing

| 证据 | 算子与现状 | 建议与收益 | 风险 | 优先级 |
| --- | --- | --- | --- | --- |
| `postprocessing/readout.py:49-60` | 校准矩阵 `kron` 已使用统一后端 | 保持 | 模块加载时固定 backend 的语义 | P3 已覆盖 |
| `postprocessing/readout.py` | **已 profile，保留**：16/64/256 维 inverse 中位耗时 0.085/0.319/5.119 ms，SciPy FP64 `pinv` 与 counts host 边界明确 | 仅未来 CUDA 同精度整体实测有收益时再迁移；SLSQP 保留 CPU | 病态矩阵、指数内存、单纯形约束和 list/counts 返回；Metal 不支持所需 FP64 | **P2 暂缓** |
| `postprocessing/io.py:125-143,163-190` | shots 位展开、加权求和、`np.unique` | 极大 shots 才考虑 CuPy/PyTorch | 排序、整数位宽、序列化抵消收益 | P3 |
| `postprocessing/counts_expval.py:55-84` | Pauli term 逐位遍历 `2^n` 概率 | 先做 NumPy 向量化，超大数组再评估后端 | 位序、空 term、归一化 | P3 |
| `postprocessing/error_mitigation.py:22-23,47-76`；`metrics.py:49`；`classical_shadows.py:41-45,87-103` | 标量统计和 Python shots 循环 | 不做 GPU 迁移 | 低 | P3 |

## 后续实施顺序

1. **已完成**：MPS 后端/SVD 语义、EWF DMET 分片复用、LUCJ 初值 objective 复用；数据和回归见最终报告。
2. **条件性 CUDA 任务**：quantum_info 大型 Hermitian 谱分解、EWF/LUCJ 同精度常驻和 readout 大型校准，必须在 NVIDIA 环境做整体双轮验收。
3. **保持现状**：SKQD 小中型旋转、host/序列化/小矩阵路径，以及所有未证明整体占比的 P2/P3。
4. **SQD 后续**：仅在新的分阶段 profile 证明 RDM 占主导时实现 MLX RDM；继续保留独立 CPU golden 和既有硬件优先级。
