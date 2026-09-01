# E8-A — LAMMPS + i-PI + TyxonQ：区域划分 + 静电嵌入的 QM/MM

生产级固体物理 QM/MM 的**阶段 A**（簇模型）教程：在既有三进程拓扑
（见 `../md_lammps_fix_ipi/`，E7）之上，加上两件事：

1. **QM/MM 区域划分**——i-PI 把全体系坐标广播给两个 client；TyxonQ
   driver 按 `qm_indices` 切出 QM 子集做量子计算，返回**全原子**力
   （QM 梯度 ⊕ MM 反作用力）；LAMMPS 负责 MM 部分。
2. **静电嵌入**——MM 点电荷经 `pyscf.qmmm.add_mm_charges` 进入 QM
   哈密顿量（QM 电子密度被 MM 电荷极化），这是标准的静电嵌入，不是
   机械嵌入。

体系：水二聚体。原子 0-2 = QM 水（CAS(4,4) UCCSD），原子 3-5 = MM 水
（TIP3P 电荷 + O 原子 LJ）。

## 能量分工（严格无重叠）

```
E = E_QM(QM 水；哈密顿量嵌入 MM 点电荷)     ← TyxonQ 返回
  + E_MM(MM-MM 全部：库仑 + vdW；QM-MM vdW)  ← LAMMPS 返回
```

防双计数三策略（缺任何一条都会重复计数）：

| 相互作用 | 归谁 | 实现 |
|---|---|---|
| QM-QM 全部 | TyxonQ | 量子计算本身 |
| QM-MM 静电 | TyxonQ 嵌入 | `water_dimer.lmp` 中 QM 原子电荷置 0 |
| QM-MM vdW | LAMMPS | `pair_coeff * * 0.0` 后只打开 O-O 通道 |
| MM-MM 全部 | LAMMPS | 照常计算（含单个 MM 水分子内的三对库仑） |

注意上游语义：`pyscf.qmmm.add_mm_charges` 只给 QM 哈密顿量加 MM 点电荷
势，**不含** MM-MM 静电、MM 能与 vdW（上游 docstring），所以这些必须
由 LAMMPS 补上。TyxonQ 额外返回的 MM 反作用力（`QCScanner.mm_gradient`，
复用上游 `QMMMGrad` 的 `grad_hcore_mm + grad_nuc_mm`）保证 MM 原子也受
到嵌入势的力，能量-力一致。

## 文件

```
in/water_dimer.xyz   全体系初始结构（6 原子；原子序即区域划分的依据）
in/water_dimer.lmp   LAMMPS data 文件（4 种原子类型；盒子必须包住所有原子，
                     否则 read_data 会丢原子并报 "Did not assign all atoms"）
in/in.qmmm           LAMMPS 输入（atom_style charge；防双计数的关键配置）
in/input.xml         i-PI server 输入（两个 ffsocket 力分量相加，端口占位符）
run.py               一键编排：参考值 → 三进程 → step-0 能量验收
```

## 运行

```bash
conda run -n qc env PYTHONPATH=src python examples/qmmm/md_lammps_qmmm_embedded/run.py
```

（提示：`conda run` 默认缓冲子进程输出，要实时看进度请加
`--no-capture-output`。）

验收口径：第 0 步总势能 ≈ E_QM(嵌入) + E_LJ(QM-MM vdW) + E_coul(MM-MM 库仑)，
容差 2e-4 Ha（覆盖 i-PI 单位回转误差与 UCCSD 数值抖动）；实测一致到 ~1e-5 Ha。

手动三窗口启动（调试用）：

```bash
# 1) server
i-pi input.xml        # 先把 __PORT_TQ__/__PORT_LMP__/__XYZ__ 占位符填好
# 2) TyxonQ driver（QM 力 + 区域划分）
i-pi-driver-py -a 127.0.0.1 -p <PORT_TQ> -m custom \
  -P src/tyxonq/applications/chem/interfaces/ipi_driver.py \
  -o "water_dimer.xyz,basis=sto-3g,active_space=4 4,method=uccsd,qm_indices=0 1 2,atom_charges=0.0 0.0 0.0 -0.834 0.417 0.417"
# 3) LAMMPS（MM 力）
lmp -in in.qmmm       # 把 __PORT_LMP__ 填好
```

`-o` 参数串里空格分隔的列表**不能用逗号**（逗号是上游参数分隔符）；
`qm_indices`/`atom_charges` 由 `ipi_driver._normalize` 解析。

## 换真实体系时改哪里

- **区域划分**：`qm_indices`（QM 原子序号，与模板原子序一致）与
  `atom_charges`（全原子电荷；QM 原子的值只是占位）。
- **活性空间/方法**：`active_space`、`method`（`uccsd`/`rouccsd`/`hea`/`sqd`）。
- **MM 力场**：`water_dimer.lmp` 的电荷与 `in.qmmm` 的 pair 参数；QM 原子
  电荷保持 0、QM-QM pair 保持 0 这两条防双计数规则不能破。
- **多分子/周期体系**：真周期（Ewald 嵌入）是阶段 B
  （`pyscf.qmmm.pbc`），风险与验证门见 `MD_INTEGRATION_PLAN.md` §6.1。

## 本期不做（见计划文档）

- 周期边界与应力（R3/RB8）；
- QM 区电荷响应（嵌入只用固定点电荷，无极化力场）；
- 自适应区域（QM 区固定）。
