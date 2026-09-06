# SQD 接入 MD 生态：已知问题备忘（已解决）

> 状态：**已解决**（2026-09，分支 `mm-intergate`）。维护者在 main（提交
> `0f5dc00` "correct closed-shell LUCJ-SQD workflow"，属 PR #16）用
> `reverse_bitstring_halves()` 等价方案修复；本分支已手动同步该提交，并把反转
> 接入本分支独有的 `lucj_sampler`。下方根因分析保留作历史记录，实际修复见文末
> 「解决记录」。

## 问题：采样路径的位串位序与 PySCF selected-CI 串约定相反

### 现象（端到端实测，H2O / STO-3G / CAS(4,4)）

把**纯 HF** 的 counts 喂进 `run_sqd_fermion`，得到的能量比全 CAS 参考
**高 3.02 Hartree**——物理上不可能（HF 串只应比全 CAS 高约 7.4e-3 Ha）：

```python
counts = {"11001100": 4096}   # [alpha|beta]，轨道 0,1 占据 = HF
r = run_sqd_fermion(h1, h2, counts, samples_per_batch=8, norb=4,
                    nelec=(2, 2), num_batches=1, max_iterations=1,
                    max_dim=4, seed=7)
# 实测：r.energy - E_fullCAS = +3.0237 Ha   （应为 +7.4e-3）
# 实际冻结串：ci_strs_a = [12]，ci_strs_b = [12]   （应为 [3], [3]）
```

连带症状：`qc_scanner(method="sqd", sampler=lucj_sampler(...))` 首帧采样
只冻结 2×2 个串，能量比全空间参考高 1.44 Ha；`refresh` 模式的「力噪声
对比演示」也因此测不出 ~1e-3 Ha 的散布（子空间塌缩，几乎无采样自由度）。

### 根因

- `src/tyxonq/applications/chem/algorithms/sqd/samples.py`
  的 `bitstring_matrix_to_integers()` 用 **MSB 优先**把位串列转整数：
  `result += matrix[:, i] * (1 << (n_bits - 1 - i))`（第 i 列得最高位）。
- PySCF selected-CI 的串约定是 **LSB 优先**：bit p = 轨道 p。
- 于是 `'1100'`（轨道 0,1 占据，正确整数 `0b0011 = 3`）被转成
  `0b1100 = 12`（轨道 2,3 占据），轨道序整体反转。

交叉验证（同一积分下 `solve_sci` 精确对角化）：

| 串 (alpha, beta) | 能量 | 与全 CAS 差 |
|---|---|---|
| (3, 3)  正确 HF 串 | -74.96302 | +7.4e-03 ✓ |
| (12, 12)  误读串 | -71.94677 | +3.02 ✗（= 实测值）|
| (12, 6)/(12, 6)  E1 曾冻结的串 | -73.52568 | +1.44 ✗ |

### 归属与影响范围

- **既有代码**：提交 `f875fcd`（"Add LUCJ SQD workflow"），不是本次
  MD 集成分支引入的改动。
- **受影响**：一切经 `run_sqd_fermion` counts → 串 的采样路径
  （含 `examples/h2o_sqd.py` 链路、`lucj_sampler` 首帧采样、
  `MD_INTEGRATION_RESEARCH.md` 中依赖采样确定子空间的结论需复核）。
- **不受影响**：显式 `ci_strs` 直传 `solve_sci` 的冻结路径
  （`sqd/pyscf_solver.py` + `qc_scanner` 的显式串用法），
  `tests_applications_chem/test_sqd_pyscf_solver.py` 用例 1-4 仍全绿。

### 恢复时的修复建议

1. 把 `bitstring_matrix_to_integers` 改为 LSB 优先
   （`result += matrix[:, i] * (1 << i)`），并全仓 grep 该函数的
   使用点（`fermion.py` L285/L288）确认无其它依赖旧约定的代码。
2. 加一条守卫测试：`run_sqd_fermion` 只喂单个 HF 串，断言能量等于
   `solve_sci` 同串结果（本备忘中的最小复现可直接改写成测试）。
3. 复核 `h2o_sqd.py` 的历史输出数字是否受此影响。
4. 修复后再恢复：`examples/qmmm/` 下的 SQD 教程（冻结子空间、
   力噪声对比、冻结 SQD 驱动的 AIMD）。

### 复现脚本要点

```bash
conda run -n qc env PYTHONPATH=src python -c "..."   # 见「现象」一节代码
```

其它已确认无问题的基础设施（修复后可直接使用）：
`qc_scanner` 门面、`as_pyscf_scanner()` 接入 `pyscf.md.NVE`、
MM 静电嵌入（`add_mm_charges` + 显式 `unit=`）、随机子空间守卫。

---

## 解决记录（2026-09）

### 实际采用的修复（与上方「修复建议」第 1 条不同）

维护者**未**改 `bitstring_matrix_to_integers` 的位序（仍 MSB 优先），而是新增
`reverse_bitstring_halves(bitstring)`：把每个自旋半区各自反转，在 **调用方**
把 TyxonQ/LUCJ raw order `[α0..αN-1 | β0..βN-1]` 转成 SQD/PySCF order
`[αN-1..α0 | βN-1..β0]`，再交给 `run_sqd_fermion`。

- `'11001100'` → `'00110011'` → MSB 优先读 → 整数 `3` ✓（HF 串正确冻结为 `[3],[3]`）
- **反转是调用方责任，`run_sqd_fermion` 内部不做**（与 `solve_sci` 显式串路径解耦）

两种方案数学等价（半区反转 ≡ 对该半区改用 LSB 读法），但维护者方案不动既有
MSB 约定、回归面更小。

### 同步范围（本分支 mm-intergate，手动逐文件应用 `0f5dc00`，只含 SQD、不含 RIVERONE）

- `sqd/samples.py`：新增 `reverse_bitstring_halves`
- `sqd/__init__.py`：导出该函数
- `sqd/recovery.py`：`_correct_partition` 全零权重均匀回退修复
- `lucj/circuit_builder.py`：`parameter_shapes` 传 `with_final_orbital_rotation`
- `examples/h2o_sqd.py`：采样后反转半区 + `REPO_ROOT` 层级修正
- 新增 `tests_applications_chem/test_lucj_sqd_closed_shell.py`、`LUCJ_SQD.md`（字节精确）

### 本分支额外接入（main 上不存在的集成路径）

`sqd/pyscf_solver.py` 的 `lucj_sampler` 是本分支 MD 集成新增、main 上没有，
`0f5dc00` 未覆盖。已在其返回处对每个位串调用 `reverse_bitstring_halves`，并加守卫
测试 `test_sqd_pyscf_solver.py::test_lucj_sampler_bitstring_order_freezes_correct_hf_determinants`。

### 验证

- 探针：`lucj_sampler` 首帧返回 `"00110011"`，冻结串 `[3],[3]`，能量 − 全 CAS =
  `+7.44e-3` Ha（匹配本备忘期望值）
- `examples/h2o_sqd.py --no-optimize`：SQD −76.0645 ≈ FCI −76.0685（差 0.004 Ha，非 +3 Ha）
- `tests_applications_chem` 全量 **53 passed**

---

## 后续交付（2026-09-04）：暂缓的 SQD QM/MM 教程已落地（E11）

「修复建议」第 4 条（恢复 `examples/qmmm/` 下的 SQD 教程：冻结子空间、力噪声
对比、冻结 SQD 驱动的 AIMD）在位序修复后全部交付：

- **功能**：`lucj_sampler` 新增 `runtime`/`provider`/`device` 三参数——SQD 的「上设备」
  与 UCCSD/HEA 架构不同（SQD 量子部分是**采样 LUCJ 电路得 counts**、经典 selected-CI
  才对角化），故设备选项挂在**采样器**上、而非 `solver_kwargs`；`runtime="device"`
  把补满 `measure_z` 的 LUCJ 电路经 `devices.base.run` 提交（与真机同一入口，切
  `provider`/`device` 即上真机），计数同样经 `reverse_bitstring_halves` 转 SQD order。
- **教程**：`examples/qmmm/md_qmmm_sqd_device.py`（E11，纯 pyscf，~1 s）——采样冻结
  子空间能量/力 + FD 自检、frozen vs refresh 机制对比、冻结 SQD 驱动 `pyscf.md.NVE`
  AIMD、静电嵌入（`mm_charges`/`mm_gradient`）、device 采样档穿透 + 真机模板。
- **测试**：`tests_applications_chem/test_qc_scanner_sqd_device.py`（4 例，采样路径在
  qc_scanner 门面里的能量/梯度/嵌入/device 穿透）；全量回归 **57 passed**。

### 关于第 26-27 行「refresh 力噪声演示测不出散布」的复核结论

位序修复后复核发现：本体系（H₂O / CAS(4,4) / LUCJ 1 层）的采样分布高度集中于 HF
行列式，且 `run_sqd_fermion` 的能量后选会**滤掉**随机激发（高能串被丢弃），故
**同一几何**重复求值的能量散布被压到 ~1e-9 Ha，无法直接量出 §4.4 D 的 ~4.3e-3
Ha/Bohr 力噪声——这不是位序缺陷残留，而是小活性空间 + 后选去噪的固有性质。
因此 E11 第 2 节改为**确定性地演示力噪声的成因（机制）**而非散布量级：统计
采样器（= `run_sqd_fermion`）触发次数——`frozen` 全程仅首帧 1 次、`refresh` 每帧
1 次，正是「随机子空间随几何跳变 → E(R) 非光滑 → 解析力无定义」的直接证据；
换更大活性空间 / 更强关联体系即显现 §4.4 D 量级的力噪声。
