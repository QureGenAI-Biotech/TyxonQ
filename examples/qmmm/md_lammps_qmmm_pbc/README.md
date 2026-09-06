# E8-B：周期性固体 QM/MM 三进程教程（pbc Ewald 嵌入 + LAMMPS pppm）

TyxonQ 经 i-PI 桥接接入 LAMMPS 的**真周期**静电嵌入 QM/MM：QM 区感受 MM
点电荷的**全部周期镜像**（`pyscf.qmmm.pbc` Ewald 求和），MM–MM 周期静电
由 LAMMPS `kspace_style pppm` 负责。与阶段 A（`../md_lammps_qmmm_embedded/`，
孤立点电荷簇嵌入）的差异只在嵌入层的周期性，拓扑与区域划分约定完全相同。

嵌入层的验证记录（风险清单、RB5 轨道响应缺口归因、单位约定）见同目录
`VALIDATION.md`，可复现取证脚本为同目录 `verify_gates.py`；
本教程是它的生产级落地示例。

## 体系与能量分工

6 原子双水，20 Å 立方周期盒（几何整体 +10 平移进盒）：

- 原子 0-2 = QM 水：CAS(4,4) **UCCSD**（TyxonQ 目标算法），pbc Ewald 嵌入；
- 原子 3-5 = MM 水：TIP3P（O −0.834 / H +0.417；LJ 仅 O：ε=0.1521 kcal/mol，σ=3.1507 Å）。

```
E = E_QM(QM 水；pbc Ewald 嵌入于 MM 点电荷及其周期镜像)   ← TyxonQ driver
  + E_MM(MM-MM 周期静电(pppm) + QM-MM vdW)                ← LAMMPS fix ipi
```

**防双计数（周期版）四要点**：

| 项 | 处理 | 依据 |
|---|---|---|
| QM-MM 库仑 | LAMMPS 侧 QM 原子电荷置 0；只由嵌入算一份（含周期镜像） | RB6 |
| QM-MM vdW | `pair_coeff * * 0.0` 后只打开 O-O 通道（关 QM-QM LJ） | 同阶段 A |
| MM-MM 静电 | 全归 LAMMPS pppm；pbc 嵌入不含 MM-MM（上游 docstring；RB1 大盒极限与簇嵌入一致也印证） | RB1/RB6 |
| 两个周期求和 | 互补不重叠：pppm 只对 MM 电荷（QM 电荷为 0），PySCF Ewald 只算 QM-MM 交叉项 | 本教程核心设计 |

## 嵌入参数与守卫（driver `-o` 串）

```
mm_lattice=20 0 0 0 20 0 0 0 20,rcut_ewald=8.0,rcut_hcore=9.0
```

- `mm_lattice`：行主序 9 个数（`-o` 串逗号是分隔符，只能空格），3×3 对角
  （上游只支持正交盒，`pbc/mm_mole.py` L54）；
- `rcut_ewald < 最小盒边`（上游 assert，mm_mole.py L63）——实空间只有最近邻
  27 胞镜像；
- `QM 区半径 < rcut_hcore < 半盒边`（`get_hcore` L176/L183：QM 原子要罩住、
  QM 镜像在盒边处不能撞）——本教程 1.06 < 9.0 < 10 ✓。
- 违反任一条，`QCScanner` 报带源码位置的 `ValueError`（不会让上游裸炸）。

**已知近似（RB5，必读）**：`mm_gradient` 的 pbc 分支缺 post-HF 轨道响应项，
基准偏置 ~4.3e-5 Ha/Bohr（相对净嵌入力 ~3%，随几何光滑）。恒温/恒压 MD 可用；
**严格 NVE 能量守恒诊断不成立**。详见同目录 `VALIDATION.md` §4。

## 运行

```bash
conda run --no-capture-output -n qc env PYTHONPATH=src \
    python examples/qmmm/md_lammps_qmmm_pbc/run.py
```

`run.py` 一键编排：算参考值 → 替换端口占位符 → 拉起三进程（i-PI server →
TyxonQ driver → LAMMPS）→ 回读 `qmmm.md` 校验 → 收尾（数据写齐即验收，
30 s 宽限后主动结束三进程，同阶段 A）。

**验收口径**：step-0 总势能 vs 独立参考值，容差 2e-4 Ha。参考值三件套：

1. `E_QM`：`qc_scanner(..., mm_charges=..., mm_lattice=..., rcut_*=...)` 直接算；
2. `E_LJ`：O_qm-O_mm 一对 LJ 解析式；
3. `E_ewald`：MM-MM 周期静电的**精确 Ewald 和**（实空间 erfc + 倒空间 +
   自能，tinfoil 边界）——与 LAMMPS pppm 完全独立；脚本内部用两个不同 κ
   互检（精确总和与 κ 无关，差 <1e-9，是 Ewald 实现的正确性自检）。

实跑结果（2026-09-01）：**|pot₀ − E_ref| = 1.04e-5 Ha**，其中包含
pppm（精度 1e-6）对精确 Ewald 的偏差、i-PI 单位常数回转（~1e-7 相对）
与 UCCSD 数值抖动（~1e-8 Ha）。

## 手动三窗口启动（调试用）

```bash
# 窗口 1：i-PI server（先把 in/input.xml 的占位符换成实际端口/路径）
i-pi input.xml
# 窗口 2：TyxonQ driver
i-pi-driver-py -a 127.0.0.1 -p <PORT_TQ> -m custom \
    -P src/tyxonq/applications/chem/interfaces/ipi_driver.py \
    -o "water_dimer.xyz,basis=sto-3g,active_space=4 4,method=uccsd,\
qm_indices=0 1 2,atom_charges=0.0 0.0 0.0 -0.834 0.417 0.417,\
mm_lattice=20 0 0 0 20 0 0 0 20,rcut_ewald=8.0,rcut_hcore=9.0"
# 窗口 3：LAMMPS（in.qmmm 的端口占位符同换）
lmp -in in.qmmm
```

LAMMPS 结束时若报 `Got EXIT message from i-PI` 属正常退出路径。

## 迁移到你的体系

1. 换几何/电荷：同步改 `water_dimer.xyz`、`water_dimer.lmp`、driver 的
   `qm_indices`/`atom_charges`，以及 `run.py` 的参考值段；
2. 换盒子：`mm_lattice`、i-PI `<cell>`、LAMMPS 盒边界**三处同改**；重选
   `rcut_ewald`（< 最小盒边）与 `rcut_hcore`（QM 区半径 < rcut < 半盒边）；
3. pppm 精度（`kspace_style pppm 1e-6`）与 PySCF 侧 `rcut_ewald` 是两个
   **独立的**周期求和（分别管 MM-MM 与 QM-MM），无需参数一致，但各自的
   收敛性要分别检查（改参数看总能量变化，参考 §6.1 RB4）；
4. 若 MM 区有多个分子/净电荷非零：pppm 要求参考胞净电荷为 0（本教程
   TIP3P 水恰好中性）；带电体系需另加均匀背景电荷处理，超出本教程范围。
