# E8-B 嵌入层验证记录（`pyscf.qmmm.pbc` Ewald 嵌入，RB1–RB10）

本文件是 `MD_INTEGRATION_PLAN.md` §6.1 **阶段 B 验证门**的取证归档，与生产
教程（本目录 `README.md` + `run.py`）放在一起。阶段 A（区域划分 + 孤立点电荷
簇嵌入，见 `../md_lammps_qmmm_embedded/`）已交付；阶段 B 把嵌入层换成
**Ewald 周期求和**，使 QM 区感受到 MM 点电荷的全部周期镜像，适用于周期性
固体材料表面的 QM/MM。

当前状态：**验证门已执行完毕（2026-08-31），RB5 降级已于 2026-09-01 复核后
收回，实现已落地（`QCScanner` 的 `mm_lattice` 分支，用例 12），三进程教程已
交付**（本目录，实跑精度 1.04e-5 Ha）。

可复现取证脚本：`verify_gates.py`（几分钟跑完）::

    conda run --no-capture-output -n qc python verify_gates.py

---

## 1. 嵌入接口与能量分工

```
pyscf.qmmm.pbc.itrf.add_mm_charges(mf, mm_coords, a, charges,
                                   rcut_ewald=..., rcut_hcore=..., unit=...)
```

- 只能装饰 **SCF 对象**（`scf.RHF` 等）；对 post-HF 对象直接
  `NotImplementedError`（`pbc/itrf.py` L100-101）。正确姿势：
  `RHF → add_mm_charges → kernel → mcscf.CASCI(mf) → 挂 TyxonQ 求解器`。
- 上游返回的总能量 = QM 能量 + QM 核–MM 静电 + 电子密度–MM 静电
  + **QM 与 MM 周期镜像的 Ewald 修正**；不含 MM–MM 静电/MM 能/vdW
  （这些继续由 MM 引擎负责，与阶段 A 相同）。
- 实空间只生成最近邻 27 胞镜像（`get_lattice_Ls`），故
  `rcut_ewald` 必须小于盒边（上游 `assert`）。

## 2. 风险清单与验证结果（首跑：2026-08-31）

体系：H₂O（STO-3G，CAS(4,4)）+ 两个 ±0.5 e 点电荷，20 Å 立方盒，
`rcut_ewald = 12 Å`，`rcut_hcore = 9 Å`。完整证据链见计划文档
§6.1「验证门执行结论」表，此处是速查版：

| 门 | 内容 | 结果 |
|---|---|---|
| RB1 | SCF 层装饰后架 CASCI / UCCSD；大盒极限收敛到簇嵌入 | ✅ L=40/80 Å 时 dE ≤ 1.3e-10 Ha；scanner 梯度差 ≤ 8e-8 Ha/Bohr；UCCSD 与 FCI 参考差 3.1e-7 Ha |
| RB2 | `rcut_ewald < min(盒边)` 上游强断言 | ✅ 确认存在，实现时加带源码位置的守卫 |
| RB3 | `rcut_hcore` 过小截断 QM 区 / 过大撞 QM 镜像 | ✅ 实测两条断言都会触发（见 §3） |
| RB4 | 缺省 Ewald 参数只是启发式，须显式传参并查收敛 | ✅ ±20% 能量变化 ≤ 1.4e-11 Ha |
| RB5 | MM 反作用力有限差分验证 | ⚠️ 缺口确认但**降级已收回**：CASCI 差 4.3e-5、HF 差 9.3e-10 → 仍交付 `mm_gradient`，文档化该近似（归因见 §4） |
| RB6 | 双计数：LAMMPS 侧 QM 原子电荷置 0 | ✅ 沿用阶段 A 策略 |
| RB7 | 重入更新需重置上游 5 个缓存 | ✅ 重入 + **重跑 `mf.kernel()`** 后与新鲜构建差 3.2e-9 Ha |
| RB8 | 周期应力（stress）不在范围 | ✅ 维持不声明 |
| RB9（新） | `QMMMSCF.as_scanner = NotImplemented` 弄断 scanner 链 | ⚠️ 缓解：用基类 `scf.hf.SCF.as_scanner` 替换（实测能量/梯度正确） |
| RB10（新） | 盒子必须明显大于 2×QM 区尺寸 | ⚠️ `rcut_hcore` 需同时 > QM 区半径、< 质心到最近 QM 镜像距离（≈半盒边） |

## 3. `get_hcore` 的两条几何断言（RB3/RB10 实测）

`pbc/itrf.py` `get_hcore`（L176、L183）：

1. `rcut_hcore² < min|QM质心 − QM镜像|²` —— 即 **rcut_hcore 必须小于
   半盒边**（半对角线不够！实测 20 Å 盒传 rcut_hcore=20 Å 直接炸）；
2. `rcut_hcore² > max|QM原子 − QM质心|²` —— QM 区必须整体落在
   rcut_hcore 内。

实现阶段将这两条翻译成带源码位置的显式校验，替代让上游 `AssertionError`
裸炸。

## 4. RB5 缺口的完整归因（本文件最重要的结论）

**现象**：CASCI 下「解析 MM 力」与「总能量对 MM 坐标的中心差分」差
4.3e-5 Ha/Bohr，1e-2 ~ 3e-4 Bohr 步长下恒定 → 不是差分截断误差。

**归因链**（每步都是数值实验，见 `verify_gates.py`）：

1. HF 层同口径验证：9.3e-10（机器精度）→ 上游解析式对 HF 自洽；
2. 分项：Ewald+核分项与解析一致到 9e-9 → 残差全在电子项；
3. 冻结 CASCI 1-RDM 时，电子能量泛函对 MM 位移的梯度 == 解析式
   （Hellmann-Feynman）→ 解析式没有算错；
4. 残差 = 完整重优化路径 FD − 冻结泛函 FD ≈ 4.1e-5 → 缺失项是
   **SCF 轨道对 MM 位移的响应**：CASCI 对 HF 轨道非变分，嵌入势随
   MM 移动改变轨道、进而改变 CASCI 能量，上游解析式（上游示例全是
   HF/DFT）没有该耦合项，等效缺一个 post-HF 版 CPHF。

**复核结论（2026-09-01，降级收回）**：该缺口与 CI 求解器质量无关——
把 CASCI 换成 FCI 同样存在（根源是 post-HF 架在 HF 轨道上、对轨道非变分）。
量级判断：

- 绝对量 4.3e-5 Ha/Bohr ≈ **2.2 meV/Å**，≪ kT(300K) ≈ 25 meV；
- 相对本基准 MM 净嵌入力（~1.4e-3 Ha/Bohr）约 **3%**；
- 数值实验证实是几何的**光滑函数**（不随差分步长变化）→ Verlet 积分可用；
- 低于 UCCSD 数值梯度自身抖动（~1e-4 Ha/Bohr）与真硬件 shot noise 一到两个量级。

代价如实标注：力与所报能量差一个小的非保守分量，**严格 NVE 能量守恒诊断失效**
（恒温/恒压 MD 不受影响）；误差随嵌入强度增长（MM 电荷靠近 QM 区会变大），
没有普适上界。**决策：pbc 模式仍交付 `mm_gradient()`，按测试规范（全程 UCCSD、
5% 相对 + 绝对容差覆盖偏置）验收。**若未来需要严格能量-力一致，升级路径是：
FD 反作用力（每步 6N 次重算，贵）或实现轨道响应项（post-HF 版 CPHF/z-vector）
或改用 `mcscf.CASSCF`（轨道共优化，上游有完整解析梯度）。

## 5. 单位约定（两个真实踩过的坑）

1. **Bohr/Å 混用会把距离搞错 1.8897 倍**。`add_mm_charges` 缺省按
   `mol.unit` 解读 MM 坐标：阶段 A 曾因此把 3 Å 的点电荷读成 3 Bohr
   （1.59 Å），能量差 0.05 Ha。规则：**调用 `add_mm_charges` 永远显式
   传 `unit=`**；scanner 门面中 MM 坐标单位跟随 `QCScanner.unit`。
2. **pyscf scanner 收裸坐标数组时按 `mol.unit` 解读**。`mol` 以 Å 建、
   却传 `atom_coords()`（Bohr）会把几何放大 1.89 倍（RB1 干跑实测，
   能量从 −74.97 变回裸水的 −70.46）。规则：坐标数组单位必须与
   `mol.unit` 一致；i-PI/ASE 链路内部统一在边界处一次性换算。
3. 有限差分导数若要 Ha/Bohr，Å 位移记得除以 1.8897261339。

## 6. 实现与交付清单（已全部落地）

1. **实现**（`src/tyxonq/applications/chem/interfaces/scanner.py`）：
   - `QCScanner` 增加 `mm_lattice` / `rcut_ewald` / `rcut_hcore` 参数，
     内部走 `pbc.itrf.add_mm_charges`，含 RB2/RB3/RB10 显式守卫（带源码位置）；
   - `set_mm_charges` 的 pbc 分支走重入路径（RB7 协议：重入重置 5 个缓存 +
     重跑 `mf.run()`）；
   - `as_scanner` 缺口按 RB9 缓解落地；
   - `mm_gradient()` 在 pbc 模式下含 Ewald 项，文档标注轨道响应缺口与量级。
2. **测试**：用例 12（`tests_applications_chem/test_qmmm_pbc_embedding.py`，
   簇/周期一致性、重入一致性、守卫报错，全程 `method="uccsd"`，
   容差沿用测试规范）。
3. **教程**（本目录）：`in/ + input.xml + run.py` 三进程编排
   （LAMMPS `fix ipi` ↔ i-PI ↔ TyxonQ driver），LAMMPS 侧用 `kspace_style pppm`
   承担 MM–MM 周期静电，与 PySCF Ewald（QM–MM 周期）互补不重叠；
   run.py 用解析 Ewald 和（双 κ 互检）独立复核，实跑 1.04e-5 Ha；
   QM 原子电荷置 0 防双计数（RB6）。
4. 结论持续回写 `MD_INTEGRATION_PLAN.md` §6.1。
