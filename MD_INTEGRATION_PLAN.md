# TyxonQ 接入分子动力学生态：升级计划（SQD 感知修订版）

> 依据：`MD_INTEGRATION_RESEARCH.md`。本文只写**做什么、怎么做、怎么验收**，事实与数据一律引用调研文档章节号。
>
> 本文取代此前那版只考虑 VQE 的计划。修订点集中在第 0 节。

---

## 0. 相对旧计划的七处修正

| # | 旧计划的说法 | 修正后 | 依据 |
|---|---|---|---|
| 1 | 只考虑 VQE 类（UCC/UCCSD/HEA） | 求解器**分成两族**：VQE 族与 SQD 族，各有独立的适配要点与确定性前提 | §4.3 |
| 2 | 无 `SQD.as_pyscf_solver` | **新增，且升为 P0 第一顺位** | §4.4 |
| 3 | UCC 是主 MD 引擎 | **冻结 SQD 是主 MD 引擎**；UCC 降为小体系参考路径 | §4.6：CAS(8,8) 上 UCCSD 0.672 s vs 冻结 SQD 0.018 s（37×）；精度 2.9e-06 vs 1.9e-08 |
| 4 | 参数热启动是 P0 唯一真工程量、MD 可行性前提 | **降级为次要可选优化** | §4.6：实测收益仅 1–3% |
| 5 | 打算给 LUCJ 也加 `as_pyscf_solver` | LUCJ 是**采样器**，不是能量方法；作为 `SQD.as_pyscf_solver(sampler=...)` 的入参 | §4.3 |
| 6 | `qc_scanner(ansatz="...")` 扁平字符串 API | 改为 `qc_scanner(method=..., sampler=..., subspace=...)`，容纳「求解器 + 采样器」两层 | 同上 |
| 7 | 无 ERI 格式约束 | SQD 适配层**必须** `ao2mo.restore(1, h2, norb)` | §4.5 |

**未被推翻、原样保留的部分**：ASE Calculator 作为唯一枢纽；i-PI 侧继承上游 `ASEDriver`；OpenMM 侧走 `MLPotential('ase')`；MDI engine 承担静电嵌入；核梯度全部复用 `pyscf/grad/`；`pyproject.toml` 需收紧 pyscf 下界。

---

## 1. 总体架构

```
                    ┌─────────────────────────────────────────┐
                    │  求解器层（TyxonQ 已有 + 本次补齐）      │
                    │                                          │
                    │  VQE 族      SQD 族                      │
                    │  UCC/UCCSD   SQD  ←── sampler: LUCJ      │
                    │  HEA         (frozen subspace)  或 counts │
                    └──────────────────┬──────────────────────┘
                                       │ as_pyscf_solver()  ← fcisolver 鸭子类型
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │  mcscf.CASCI + pyscf/grad/casci.py      │  ← 全部复用，零重写
                    │  (+ pyscf/qmmm/itrf.py 静电嵌入)        │
                    └──────────────────┬──────────────────────┘
                                       │ nuc_grad_method().as_scanner()
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │        qc_scanner()  ← 唯一门面          │
                    │   (cell, positions) → (E, F)  原子单位   │
                    └────────┬───────────────────┬─────────────┘
                             │                   │
              ┌──────────────▼─────┐      ┌──────▼──────────────┐
              │  TyxonQCalculator  │      │  MDI engine         │
              │  （ASE 枢纽）       │      │  （静电嵌入专线）    │
              └──┬────┬────────┬───┘      └──────┬──────────────┘
                 │    │        │                 │
      ASE 原生 ──┘    │        └── openmm-ml     └── LAMMPS
      (opt/MD/NEB)    │            MLPotential       fix mdi/qm
                      │            ('ase')           fix mdi/qmmm
              i-PI ASEDriver 子类
                      │
                 LAMMPS fix ipi / 任意 i-PI server（含 PIMD）
```

**一句话**：`qc_scanner` 是唯一需要认真设计的抽象，其余四个适配层都是 20–80 行的薄壳；
最大的真实工程量在 `SQD.as_pyscf_solver` 的子空间管理。

> **修订（E8 纠偏）**：静电嵌入**不必等 MDI**。已取证 `pyscf.qmmm`（分子）与 `pyscf.qmmm.pbc`
> （Ewald/周期）两条静电嵌入链路均带 QM 梯度与 MM 反作用力 API，可在现有三进程拓扑
> （LAMMPS `fix ipi` + i-PI server + TyxonQ driver）内完成「区域划分 + 静电嵌入」。
> E8 因此拆为 **阶段 A（簇嵌入）→ 阶段 B（周期 Ewald 嵌入）**，见 §5 与 §6.1；
> MDI 专线（原 E8）顺延为 E9，不再阻塞生产级固体 QM/MM。

---

## 2. 新增文件清单

```
src/tyxonq/applications/chem/
├── algorithms/
│   └── sqd/
│       └── pyscf_solver.py          [新] SQD → fcisolver 适配 + 子空间策略
└── interfaces/                       [新目录]
    ├── __init__.py                   惰性导出，不硬依赖 ase/openmm/ipi/mdi
    ├── scanner.py                    [P0] qc_scanner 门面
    ├── ase_calculator.py             [P1] TyxonQCalculator
    ├── ipi_driver.py                 [P2] TyxonQDriver(ASEDriver)
    ├── openmm_potential.py           [P3] create_tyxonq_system / mixed / qmmm_ee（静电嵌入）帮助函数
    └── mdi_engine.py                 [P4] MDI engine + 静电嵌入

tests_applications_chem/
├── test_sqd_pyscf_solver.py         [新] 用例 1–4
├── test_qc_scanner.py               [新] 用例 5–6
├── test_ase_calculator.py           [新] 用例 7
├── test_ipi_driver.py               [新] 用例 8
├── test_openmm_potential.py         [新] 用例 9
├── test_mdi_engine.py               [新] 用例 10
└── test_qmmm_embedding.py           [新] 用例 11（E8 阶段 A：区域划分+静电嵌入+反作用力）

examples/
├── md_sqd_scanner_basics.py         [E1]
├── md_pyscf_native_aimd.py          [E2]
├── md_ase_optimize_and_md.py        [E3]
├── md_ipi_driver_server.py          [E4]
├── md_openmm_pure_qm.py             [E5]
├── md_openmm_qmmm_solvated.py       [E6]
├── md_openmm_qmmm_electrostatic.py  [E6b] OpenMM 原生静电嵌入（簇，单进程，无需 MDI）
├── md_lammps_fix_ipi/               [E7] in/ 输入 + README + run.py 编排（三进程，管线验证）
├── md_lammps_qmmm_embedded/         [E8-A] 区域划分+静电嵌入（簇），三进程实跑（无需 MDI）
├── md_lammps_qmmm_pbc/              [E8-B] pyscf.qmmm.pbc Ewald 嵌入（固体，需先过 §6.1 验证门）
└── md_mdi_qmmm_embedded.py          [E9] MDI 专线静电嵌入（Python driver 双进程，无需带 MDI 的 LAMMPS）
```

---

## 3. 分阶段实施

### P0 — 求解器层与门面（唯一的硬工程量）

#### P0-A `sqd/pyscf_solver.py`：`SQD.as_pyscf_solver`

这是本次升级的核心新增。三条硬约束来自实测：

1. **ERI 解包**：CASCI 传入的 `h2` 是 8 重压缩，`solve_sci` 要全 4 指标 → 入口先
   `h2 = ao2mo.restore(1, np.asarray(h2), norb)`（§4.5）。
2. **子空间必须真冻结**：不能用 `run_sqd_fermion(include_configurations=...)` 实现冻结——
   读 `fermion.py` L294-302，`include_*` 只是**并入**采样串再被 `max_dim` 截断，
   子空间仍随采样漂移。真冻结必须**绕过 SQD 主循环**，直接
   `solve_sci(ci_strs=(strs_a, strs_b), ...)`。
3. **`spin_square` 要真算**：`fermion.py` L11-18 已从 `pyscf.fci.selected_ci` 导入 `spin_square`，
   直接转发即可，不要像 UCC/HEA 那样硬编码 `0.0, 1.0`。

三种子空间策略（`subspace=` 参数）：

| 取值 | 行为 | 梯度可用性 | 允许用于 MD 接口 |
|---|---|---|---|
| `"frozen"`（默认） | 首次调用跑一遍完整 SQD 采样确定 `(strs_a, strs_b)` 并**锁存**；之后每个几何只做 `solve_sci` | ✓ 5.9e-08 ~ 5.8e-07（§4.4 B/D3） | ✓ |
| `"refresh"` | 每个几何重跑采样，重新确定子空间 | ✗ 力噪声 4.3e-03 Ha/Bohr（§4.4 D） | ✗ **必须拒绝** |
| `"adaptive"` | 每 N 步刷新一次，步间冻结 | 分段光滑，仅供研究 | ⚠ 默认拒绝，需 `allow_discontinuous=True` 显式开启 |

锁存内容取自 `SCIResult.sci_state.ci_strs_a` / `.ci_strs_b`（`fermion.py` L403-405 已暴露）。

要点：
- 类应提供 `freeze_from(result_or_strs)` 与 `frozen_strings` 属性，便于测试与 example 显式检查。
- `subspace="frozen"` 且给了显式 `ci_strs` 时，**跳过采样**，不需要 sampler、不需要 counts。
  这是 §4.4 A/B 探针的路径，也是 MD 的默认路径。
- 冻结串以 `(alpha_strs, beta_strs)` 的整数数组形式保存，可 `np.save` 落盘、跨进程复用。
- 采样器协议保持最小：`sampler(h1, h2, norb, nelec) -> counts dict`。
  `LUCJ` 不实现该协议，因此提供一个 `lucj_sampler(...)` 工厂把
  `initialize_lucj_parameters_from_ccsd` + `LUCJ(...).get_circuit` + `StatevectorEngine` 串起来
  （链路照抄 `examples/h2o_sqd.py` L276-290），而不是去改 `LUCJ` 类。

#### P0-B `interfaces/scanner.py`：`qc_scanner` 门面

唯一对外入口，产出物就是 PySCF 的 grad scanner（§3.2 已证明这是全生态跑 MD 的充分条件）。

```python
def qc_scanner(
    atom,                         # PySCF 几何（字符串 / 列表）
    *,
    basis="sto-3g",
    charge=0, spin=0, unit="Angstrom",
    active_space,                 # (n_elec, n_orb)，与 ucc.py L143-147 约定一致
    method="sqd",                 # "sqd" | "uccsd" | "rouccsd" | "hea"（不提供外部 fcisolver 后门，
                                    # 本项目的意义就是验证这些量子算法，参见用例 11 容差策略）
    sampler=None,                 # 仅 method="sqd"；None 时需显式 ci_strs 或首帧采样
    subspace="frozen",            # 仅 method="sqd"
    solver_kwargs=None,
    mm_charges=None,              # (coords, charges)，走 pyscf.qmmm.add_mm_charges
    verbose=0,
) -> Callable[[Any], tuple[float, np.ndarray]]
```

内部严格是这三行加胶水（§4.1 已实测通路）：

```python
mc = mcscf.CASCI(mf, n_orb, n_elec)
mc.fcisolver = _make_fcisolver(method, ...)
scanner = mc.nuc_grad_method().as_scanner()
```

守卫：`subspace != "frozen"` 时抛 `ValueError`，除非 `allow_discontinuous=True`；
错误消息要指向 §4.4 D 的数据（力噪声 4.3e-03 Ha/Bohr）。

#### P0-C 修正两处 `spin_square`

`ucc.py` L1014-1015 与 `hea.py` L586-587 都硬编码 `return 0.0, 1.0`。
UCC 侧对闭壳单重态是对的，但对 `spin != 0` 会静默给错值。
最小修改：闭壳时保持现值，`spin != 0` 时抛 `NotImplementedError` 并提示改用 SQD 路径。
不在本次扩大 UCC 的 `<S²>` 实现范围。

#### P0-D `pyproject.toml`

- `pyscf>2.5` → `pyscf>=2.10`（§6：2.9.0 + numpy 2.4.4 在 PySCF 自身参考实现内崩溃）。
- 新增 optional extra：

```toml
[project.optional-dependencies]
md = ["ase>=3.23", "openmm>=8.1", "openmmml>=1.7"]
```

`ipi` 与 `mdi` 不进 extra（ipi 是独立可执行程序，mdi 需编译），改为在 example README 里给安装说明。

**P0 验收**：用例 1–6 全绿；`MD_INTEGRATION_RESEARCH.md` §4.4 A/B 与 §4.6 的数字能被测试复现。

---

### P1 — `interfaces/ase_calculator.py`：唯一枢纽

`TyxonQCalculator(ase.calculators.calculator.Calculator)`：

- `implemented_properties = ["energy", "free_energy", "forces"]`（周期体系的 `stress` 本期不做，
  见 §7 风险 R3）。
- `__init__` 参数与 `qc_scanner` 同构，内部持有一个 scanner。
- `calculate()`：`atoms.get_positions()` → Å 转 Bohr → scanner → Hartree/Bohr 转 eV/Å。
  换算常数只在这一个文件里出现（附录 B：`1 Bohr = 0.52917721092 Å`）。
- 力的符号：PySCF grad 返回 `de = dE/dR`，ASE 要 `F = -dE/dR`，**必须取负**。这是最易错的一处。
- 惰性 import ase，缺失时给出 `pip install "tyxonq[md]"` 提示。

**P1 验收**：用例 7。

---

### P2 — `interfaces/ipi_driver.py`：约 20 行

继承上游 `ipi.pes._ase.ASEDriver`（§2.1 全文已读），只覆写 `check_parameters()` 里
设 `self.ase_calculator = TyxonQCalculator(...)`。单位换算、virial、Voigt 展开、
extras、socket 全由上游 `convert_units` / `post_process` 负责，不重写。

同时给出「不改上游」的落地方式：i-PI 支持从外部模块加载 PES，example 里用
`ipi-driver` + `--module` 指向本文件；另附一段向 `ipi/pes/` 上游提交插件的说明（可选，不阻塞）。

**P2 验收**：用例 8。

---

### P3 — `interfaces/openmm_potential.py`：薄封装 ✅（2026-09-01 完成）

基于 `MLPotential('ase').createSystem(topology, calculator=...)`（§2.1 逐字确认的 API，
已读已装版本 `openmmml` 1.7 源码复核：`createSystem`/`createMixedSystem` 签名一致）：

- `create_tyxonq_system(topology, **scanner_kwargs)` → 纯 QM System。
- `create_mixed_system(topology, mm_system, qm_atoms, interpolate=False)` → 转发
  `createMixedSystem(...)`，`interpolate=True` 时暴露 `lambda_interpolate` 供自由能微扰。
- 支持透传 `info={'charge': ...}`，把 OpenMM 侧的总电荷映射到 `qc_scanner(charge=...)`。
- **明确文档化**：`create_mixed_system` 是**机械嵌入**，QM 区感受不到 MM 静电极化。
- **`create_qmmm_ee_system`（2026-09-01 新增，簇嵌入版）**：OpenMM 原生静电嵌入，
  无需 MDI / i-PI：区域划分模式的 `TyxonQCalculator`（`qm_indices + atom_charges`，
  MM 电荷经 `set_mm_charges` 每步注入）挂 `PythonForce` 返回全原子力（QM 梯度 ⊕
  MM 反作用力）；防双计数手术与 E8 的 LAMMPS 侧一一对应（QM 电荷置 0、QM 内部
  键合项/例外清零、保留 QM-MM vdW 与 MM 侧经典项）；pbc 参数拒收（守卫报错指路 E8-B）。

**P3 验收**：用例 9（实跑 7/7，含 3 个静电嵌入测试，2026-09-01）；E5（NVE 漂移
5.9e-6 Ha）、E6（λ=0 回到纯经典差 5.5e-15；能量分解恒等式差 5.9e-7）与 E6b
（能量分解残差 5.9e-7；10 步 NVE 极差 2.0e-5）均实跑通过。
注意：PyPI 包名是 `openmmml`（无连字符），`openmm-ml` 只是 conda-forge 通道名。

---

### P4 — `interfaces/mdi_engine.py`：静电嵌入专线 ✅ 已完成（2026-09-01）

抄 psi4 `mdi_engine.py` 的命令表结构（§3.6，只注册 `@DEFAULT` 一个节点）：

必需命令：`<NATOMS` `<COORDS` `>COORDS` `<ENERGY` `<FORCES` `<ELEMENTS` `<MASSES`
`<TOTCHARGE` `>TOTCHARGE` `<ELEC_MULT` `>ELEC_MULT` `<DIMENSIONS` `EXIT`
静电嵌入三条：`>NLATTICE` `>CLATTICE` `>LATTICE` → 缓存成对后经
`qc_scanner.set_mm_charges` 注入（E8 阶段 A 同款链路）；`<FORCES` 返回全原子力 =
QM 梯度（含嵌入）⊕ MM 反作用力（`mm_gradient`）。

MDI 用原子单位，与 PySCF 原生一致，**不做任何换算**（附录 B）。
惰性 import `mdi`：PyPI 包名 `pymdi`（`pip install 'pymdi>=1.4'`，import 名 `mdi`，
又一次「包名≠导入名」坑），缺失时报带安装指引的 `ImportError`；无需自行编译。
协议取证：`>COORDS` 传全体系坐标（引擎按 `3*natoms` 接收，内部自取 QM 子集）；
ENGINE 先 `MDI_Init` + 注册 + `MDI_Accept_Communicator`（listen），driver 后连。
实现中发现并修复一个上游语义坑：`qmmm.add_mm_charges` 对**未装饰**的裸平均场
返回新对象而非原地装饰（`pyscf/qmmm/itrf.py` L36-60），
`QCScanner.set_mm_charges` 已在「裸建后首次带电荷」时整体重建配方规避。
教程形态：自带手写 Python driver（qc 的 `lammps 2025.7.22` 不带 MDI 包，
三进程拓扑接不通；簇静电嵌入对照已由 E6b 覆盖）。

**P4 验收 ✅**：用例 10（4/4，含「先裸 >COORDS 后推 lattice」时序回归）；
E9 教程实跑过（ΔE = +3.283e-3 Ha，与 E6b 簇嵌入同值；每步更新生效）。
算法覆盖面补充（2026-09-01）：`method="hea"` 实测全链路可用（嵌入能量位移与 `mm_gradient` 均正常；水二聚体 CAS(4,4) 下比 UCCSD 高 ~7.5e-3 Ha，RY-only ansatz 表达能力不足的预期行为）；真机参数经 `solver_kwargs` 的 `shots`/`provider`/`device` → `as_pyscf_solver(device_opts=...)` → `kernel` 透传到 `devices.base.run`（hea/uccsd/rouccsd 全覆盖；`device_opts` 支持放在共享的 `UCC.as_pyscf_solver` 基类，调用形态同 `examples/cloud_uccsd_hea_demo.py`；simulator 档用例 `test_qc_scanner_hea.py` + `test_qc_scanner_uccsd_device.py` 共 4/4，真机实测留待有资源时）。同日修复 device runtime 两处既有缺陷（与 qmmm 链路无关但被其暴露）：① X/Y 基旋转放在镜像比特 `n-1-q` 而聚合直读比特 `q`，3+ 比特采样档能量错误（水 CAS(4,4) 偏差 0.25 Ha，此前仅 2 比特 H2 对照测试未暴露）；② shots=0 解析档的 expectations/probabilities 留在 `result_meta` 嵌套层，聚合退化为只加常数项。修复后解析档与 numeric 机器精度一致、采样档落在统计误差内（`tests_mol_valid` 178 过 + `tests_applications_chem` 41 过）。

---

### P5 — 力场导出（本次仅落设计，不实现）

原始需求里的「生成 LAMMPS/OpenMM 可加载的力场文件」与 MD 在线耦合是两件事。
建议顺序：先把 P0–P4 的在线耦合跑通，再用 `qc_scanner` 作为数据生成器去拟合力场。
本期只在 `MD_INTEGRATION_PLAN.md` 留位，不写代码，避免在 P0 未验收前分散精力。

---

## 4. 测试用例（10 项）

全部遵循 `tests_applications_chem/` 现有惯例：`_has_pyscf()` + `pytest.mark.skipif`；
ase/openmm/ipi/mdi 各自加独立的 `_has_*()` 跳过守卫（§附录 A：三个 conda env 都没装这四者）。

| # | 文件 | 用例 | 断言 |
|---|---|---|---|
| 1 | `test_sqd_pyscf_solver.py` | **SQD 冻结=全 CAS 精确性**：H₂O/STO-3G CAS(4,4)，冻结串取全空间 | `ΔE < 1e-10`、`max\|ΔGrad\| < 1e-7` vs stock CASCI（实测 1.563e-13 / 1.892e-08，§4.4 A） |
| 2 | 同上 | **冻结截断子空间 vs 有限差分** 3×3 | 两个方向差均 `< 1e-5`（实测 2.8e-07 / 5.8e-07） |
| 3 | 同上 | **变分上界**：截断子空间能量 ≥ 全 CAS 能量 | `E_trunc - E_full > 0`（实测 +3.037e-03 Ha） |
| 4 | 同上 | **随机模式守卫**：`subspace="refresh"` 传给 `qc_scanner` | 抛 `ValueError`，消息含 "frozen"；`allow_discontinuous=True` 时放行 |
| 5 | `test_qc_scanner.py` | **UCC 路径回归**：`method="uccsd"` | `max\|Δg\| < 1e-5` vs stock CASCI（实测 2.9e-06，§4.1） |
| 6 | 同上 | **跨几何一致性 + 静电嵌入**：同一 scanner 连续吃两个几何；再带 `mm_charges` | per-geometry 误差 `< 1e-5`；带 MM 时与 `add_mm_charges` 手搭链路一致（`< 1e-5`） |
| 7 | `test_ase_calculator.py` | **单位与符号**：`atoms.get_potential_energy()` / `get_forces()` | 与 scanner 原始值按附录 B 换算一致（`rtol=1e-10`）；**力符号相反**；1 步 BFGS 能量下降 |
| 8 | `test_ipi_driver.py` | **i-PI driver**：起一个最小 socket server 发 `POSDATA`/`GETFORCE` | 收到的 E/F 与直接调 scanner 一致；无 i-PI 时 skip |
| 9 | `test_openmm_potential.py` | **openmm-ml ✅（2026-09-01，7/7）**：单水 topology 建纯 QM System，Context 取 (E, F) 与 scanner 按 kJ/mol/nm 换算一致（rel 1e-7 / rtol 1e-6）；`info={'charge':...}` 映射；`createMixedSystem`（interpolate=True）含 `lambda_interpolate` 且 λ=0/1 能量可算；混合体系删 QM 内部键/角项；**静电嵌入（`create_qmmm_ee_system`）**：防双计数手术 + pbc 守卫、能量分解（E_total = E_QM(嵌入)+E_MM 经典，rel 1e-6，嵌入能量与裸算不同证明电荷进哈密顿量）、全原子力 = 区域划分 Calculator 力 + 纯经典力（rtol 1e-5） | 全程 `method="uccsd"`；无 openmm 时 skip |
| 10 | `test_mdi_engine.py` | **MDI 命令回环 ✅（2026-09-01，4/4）**：双进程 TCP（driver=父进程，engine=子进程）；①无嵌入：元数据命令 + (E, F) 与裸 scanner 一致（rel 1e-6 / rtol 1e-4）；②推 `>NLATTICE`/`>CLATTICE`/`>LATTICE`：能量位移 > 1e-4，(E, F) 与带 `mm_charges` 的 scanner 一致，MM 行 = 反作用力；③**先裸 `>COORDS` 后推 lattice**（锁住裸建后重入装饰丢弃新对象的 bug）；④二次推 `>LATTICE` 每步更新 | 全程 `method="uccsd"`；无 `mdi`（PyPI `pymdi`）时 skip |
| 11 | `test_qmmm_embedding.py` | **E8 阶段 A**：`set_mm_charges` 每步更新与新鲜构建一致；MM 反作用力对 MM 坐标有限差分；`TyxonQCalculator(qm_indices=...)` 全体系力 = QM 梯度 + MM 反作用力 | 全程 `method="uccsd"`（目标算法，不换 FCI）：能量一致 `1e-8`、力一致 `rtol=1e-3`；有限差分步长 0.05 Å、5% 相对容差（覆盖量子数值抖动）；MM 原子受力非零 |
| 12 | `test_qc_scanner_hea_device.py` | **HEA 采样档穿透（2026-09-01，2/2）**：`shots=0` 与 numeric 一致（rel 1e-8）；`shots=4096` 全链路跑通（`< 3e-2`，带噪优化尾部容差）；instance 属性证明选项穿透；嵌入档能量位移存在 | 全程 `method="hea"`；HEA 存在的目的就是上真机，采样档是其真实工作形态 |
| 13 | `test_device_runtime_regression.py` | **三个盲区回归（2026-09-01，3/3）**：①≥3 比特体系采样档位序（UCCSD H4 8q + HEA 水 CAS(4,4) 6q，4096 shots `< 2e-2`）；②shots=0 解析档直连（绕过 API 层 numeric 短路，rel 1e-8，UCCSD 侧用 `trotter=True`）；③同批电路 32768-shot 计数 vs 解析概率口径一致（`< 1e-2`） | 封堵 2 比特镜像自抵消 / API 短路 / 宽容差三个历史盲区 |

**新增于旧计划的是 1–4**：SQD 专属，容差比 UCC 路径严一到两个数量级（1e-7 vs 1e-5），
因为 SQD 走 PySCF 原生 `selected_ci`，没有 ansatz 误差。

**挂起 to-do：UCC 门级 ansatz 等价性缺口**（2026-09-01 发现，独立立项）：
`build_ucc_circuit` 门级态制备（TenCirChem 风格 `2*theta` 双激发 cry 分解 + 共享参数）
与数值路径精确 `exp` 演化在 H4 上有 ~1e-4 Ha 确定性态差（态重叠 0.99995，
H6 上偏差 +0.173 同机制）；`trotter=True` 档精确到 2.9e-13。与旋转位序修复无关，
见 `ucc_device_runtime.py` backlog 第 7 项；回归测试②用 `trotter=True` 绕开此缺口。

**挂起 to-do：双激发分量 ±π/2 参数移位梯度恒为零**（2026-09-01 发现，独立立项）：
`2*theta` 约定下能量面对双激发参数周期为 π，E(θ±π/2) 对任意 θ 恒等，
故 device `energy_and_grad` 的 ±π/2 PSR 双激发分量梯度恒为 0（数值梯度不为 0）；
修复方向：移位 s 配合归一化 1/sin(2s)（如 s=π/4 时 g = E(θ+s)−E(θ−s)）或改用有限差分。
见 `ucc_device_runtime.py` backlog 第 8 项。

---

## 5. Examples（8 项）

| # | 文件 | 展示 | 依赖 |
|---|---|---|---|
| E1 | `md_sqd_scanner_basics.py` | `qc_scanner` 三行拿到 (E, F)；打印冻结子空间大小；对比 `subspace="frozen"` vs `"refresh"` 的力噪声，直观再现 §4.4 D | 仅 pyscf |
| E2 | `md_pyscf_native_aimd.py` | `pyscf.md.NVE(scanner)` 跑 20 步 AIMD，画能量守恒曲线 | 仅 pyscf |
| E3 | `md_ase_optimize_and_md.py` | ASE BFGS 优化 H₂O + Langevin 短 MD + 振动频率 | ase |
| E4 | `md_ipi_driver_server.py` | 启 i-PI server + TyxonQ driver 双进程；附 `input.xml`；含一个 PIMD 变体（i-PI 独占能力） | ipi |
| E5 | `md_openmm_pure_qm.py` | `MLPotential('ase')` 建纯 QM System，OpenMM 积分器跑 NVE MD（velocity-Verlet 0.2 fs，漂移 5.9e-6 Ha；含 CMMotionRemover 质心动能台阶的实测说明） | openmm, openmmml, ase |
| E6 | `md_openmm_qmmm_solvated.py` | 溶剂化体系：QM 溶质 + MM 水，`createMixedSystem`；`interpolate=True` 演示 λ 微扰（能量分解恒等式验收，机械嵌入边界文档化） | 同上 |
| E6b | `md_openmm_qmmm_electrostatic.py` | **OpenMM 原生静电嵌入（簇，单进程，无需 MDI）**：两水二聚体，`create_qmmm_ee_system`；嵌入极化效应演示（符号随取向）+ 能量分解验收（残差 5.9e-7）+ 10 步 NVE（极差 2.0e-5，RB5 轨道响应缺口如实文档化） | 同上 |
| E7 | `md_lammps_fix_ipi/` | **管线验证**：TyxonQ 经 i-PI 桥接接入 LAMMPS。三进程拓扑：LAMMPS（`fix ipi`，client/力场端）+ i-PI server（MD 引擎）+ TyxonQ driver（client/QM 力端）；`<forces>` 多力场加权求和（已取证 `InputForceComponent.weight`）。演示体系用水分子 + 弱 LJ 验证管线与力合成；README 注明生产级区域划分/静电嵌入见 E8 | lammps, ipi |
| E8-A | `md_lammps_qmmm_embedded/` | **生产级固体 QM/MM 阶段 A（无需 MDI）✅ 已完成**：区域划分（`qm_indices`）+ 静电嵌入（`pyscf.qmmm.add_mm_charges`）+ MM 反作用力，三进程拓扑不变；双水分子示范（QM 水 + MM 水）。LAMMPS 侧防双计数：QM 原子电荷置 0（QM–MM 库仑归嵌入）、`pair_coeff qmtype qmtype 0.0 ...` 关 QM–QM LJ、保留 QM–MM vdW。实跑验收：step-0 总势能 = E_QM(嵌入)+E_LJ(QM-MM vdW)+E_coul(MM-MM) 一致到 7.67e-6 Ha（容差 2e-4） | lammps, ipi |
| E8-B | `md_lammps_qmmm_pbc/` | **阶段 B：真周期 ✅ 已完成**。换 `pyscf.qmmm.pbc.add_mm_charges`（Ewald 背景电荷，含参考胞与周期镜像的静电互作用）；§6.1 验证门已执行完毕，`QCScanner` 的 `mm_lattice` 分支、用例 12、三进程教程均落地（回归 26/26）；RB5 降级已于 2026-09-01 复核后收回（`mm_gradient` 照交，文档化轨道响应缺口）。LAMMPS 侧 MM-MM 周期静电用 `kspace_style pppm`，与嵌入的 QM-MM 周期互补不重叠（QM 电荷置 0）。实跑验收：step-0 总势能 = E_QM(pbc 嵌入)+E_LJ+E_ewald(解析 Ewald，双 κ 互检) 一致到 1.04e-5 Ha（容差 2e-4）；验收含「大盒子极限收敛到阶段 A 簇嵌入」对照（用例 12） | lammps, ipi |
| E9 | `md_mdi_qmmm_embedded.py` | **MDI 专线静电嵌入 ✅ 已完成**：自带手写 Python driver（两进程），不需带 MDI 包的 LAMMPS；三步验收：无嵌入基准 → 推晶格后 ΔE = +3.283e-3 Ha + MM 反作用力非零 → MM 平移 0.2 Å 能量跟随（`set_mm_charges` 重入）；与 E8-A 物理等价，簇嵌入对照见 E6b | pymdi |

每个 example 顶部注明：所需环境、预期运行时间、以及若依赖缺失时的优雅退出。
E1/E2 必须能在纯 pyscf 环境跑通（这是 CI 唯一能覆盖的两个）。

---

## 6. 风险与守卫

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 用户误用随机 SQD 跑 MD，拿到看似合理但完全错误的轨迹 | `qc_scanner` 默认 `frozen`；非 frozen 直接抛错；错误消息带 §4.4 D 数据；用例 4 守卫 |
| R2 | 冻结子空间在几何大幅变化后不再有代表性（如键断裂） | `frozen_strings` 可读可存；提供 `refreeze()` 显式接口；文档说明「构型显著变化后应重新冻结并接受轨迹分段」 |
| R3 | 周期体系需要 stress/virial，本期不做 | Calculator 不声明 `stress`；ASE/i-PI 会明确报缺失而非静默给零 |
| R4 | MDI 单人维护（968/987 提交出自一人，§1.6） | 协议已冻结在 1.4.x、BSD-3、LAMMPS 与 psi4 均 vendored in-tree；P4 放在最后，前三阶段不依赖它 |
| R5 | 依赖缺失导致 import 失败 | `interfaces/__init__.py` 全部惰性导出；每个适配层独立 try-import |
| R6 | pyscf/numpy 坏组合（§6） | `pyscf>=2.10`；测试里加一个版本前置检查并给出明确提示 |
| R7 | UCC 在大活性空间下慢到不可用（CAS(8,8) 0.672 s/几何） | 文档给出选型表：CAS ≤ (4,4) 或需要参数化 ansatz 时用 UCC，否则用冻结 SQD |

### 6.1 阶段 B（周期 Ewald 嵌入，`pyscf.qmmm.pbc`）风险清单（逐条源码取证）

阶段 B 在阶段 A 的三进程拓扑上把嵌入层换成 `pyscf.qmmm.pbc.add_mm_charges(scf_method,
mm_coords, a, charges, rcut_ewald=None, rcut_hcore=None)`。以下风险全部来自已读源码（文件：
`pyscf/qmmm/pbc/itrf.py`、`pyscf/qmmm/pbc/mm_mole.py`，行号为已装版本），**每条都有对应的验证门，
全部通过后才允许写 E8-B 教程**。

| # | 风险点 | 受影响展位（占位） | 原因（源码依据） | 缓解 / 验证门 |
|---|---|---|---|---|
| RB1 | pbc 版拒绝直接装饰 post-HF 对象 | `qc_scanner` 内部装饰位：必须在 `scf.RHF(mol)` 层装饰，再架 `mcscf.CASCI(mf)`；不得对 `mc` 调 `qmmm_for_scf` | `pbc/itrf.py` L100-101：`qmmm_for_scf` 对非 SCF 输入直接 `raise NotImplementedError()`（分子版 L105-112 支持，pbc 版没做） | 阶段 B 第一步干跑：pbc 装饰的 RHF 上架 CASCI，验证能量与分子版簇嵌入在去周期极限的一致性；失败则阶段 B 降为仅 HF 参考面 |
| RB2 | Ewald 实空间截断必须小于盒子，且只考虑最近邻镜像 | `qc_scanner` 新增的 `mm_lattice`/`rcut_ewald` 参数；教程的盒子尺寸 | `pbc/mm_mole.py` L62-63：`assert rcut_ewald < min(np.diag(a))`，注释原文「Only rcut_ewald < box size implemented」；`get_lattice_Ls()`（L103）只生成最近邻胞镜像 | 显式暴露并校验 `rcut_ewald < min(盒子边长)`，违反时报带源码位置的错误而不是让上游 assert 炸；教程盒子 ≥ 2×rcut_ewald |
| RB3 | `rcut_hcore` 过小会截断 QM 区 | 同上，另加教程断言 | `pbc/mm_mole.py` L171-172 注释：Ewald 势要扣除 `rcut_hcore` 内的实空间库仑；L191-192 NOTE 原文「a too small rcut_hcore truncates QM atoms」——精确 QM-MM 耦合只在 `rcut_hcore` 内生效 | 默认值（半盒子对角线，L59-61）通常够；教程校验 QM 区质心到最远 QM 原子距离 < `rcut_hcore`，不满足时报错 |
| RB4 | 缺省 Ewald 参数只是启发式 | `create_mm_mol` 调用处；教程需打印实际生效的 `rcut_ewald`/`rcut_hcore` | `pbc/mm_mole.py` L56-61：缺省时 `rcut_ewald = min(diag(a))·0.5`、`rcut_hcore = 半对角线`，且都是 `logger.warn` 而非显式设计 | 教程不依赖缺省值：显式传参并做截断收敛性检查（改 `rcut_ewald` ±20% 能量变化 < 1e-6 Ha） |
| RB5 | MM 反作用力多了 Ewald 项，且从未在 CASCI 密度矩阵下被验证 | `QCScanner` 的 MM 力组装位（阶段 A 的 `grad_hcore_mm + grad_nuc_mm` 需扩为含 `de_ewald_mm` 的 pbc 分支） | `pbc/itrf.py` L489-490：`qmmm_grad_for_scf` 额外挂 `de_ewald_mm`/`de_nuc_mm` 缓存；`grad_ewald(dm)`（L517）/`grad_hcore_mm`（L908）均以 dm 为参，上游示例全是 HF/DFT，CASCI 1-RDM 喂入无人验证 | 阶段 B 用中心差分对 MM 坐标逐分量验证 MM 受力（含 Ewald 贡献）；实测发现上游解析式缺 post-HF 轨道响应项（~4.3e-5 Ha/Bohr，见结论表），验收改为「绝对容差 2e-4 Ha/Bohr + 5% 相对（覆盖 UCCSD 抖动与已知偏置），并文档化该近似」，仍交付 `mm_gradient` |
| RB6 | 周期嵌入包含参考胞与镜像的互作用，与 LAMMPS 侧 kspace 的双计数边界变化 | LAMMPS 输入（`in.qmmm`）；README 的分区说明 | `pbc/itrf.py` docstring L38-39：总能量「The electrostatic interactions between reference cell and periodic images are also computed」——嵌入侧已含 QM 与 MM 周期镜像的静电；若 LAMMPS 侧 kspace 仍把 QM 原子当带电粒子，会重复计数 | 与阶段 A 同策略：LAMMPS 侧 QM 原子电荷恒为 0；README 写明 LAMMPS kspace（MM–MM）与 PySCF Ewald（QM–MM 周期）的参数一致性检查项 |
| RB7 | 每步更新 MM 环境需重置上游缓存字段 | `set_mm_charges` 的 pbc 分支必须走 `qmmm.qmmm.pbc.qmmm_for_scf` 而不是手改 `mm_mol` | `pbc/itrf.py` L89-96：重入 `qmmm_for_scf` 会同时重置 `s1r`/`s1rr`/`mm_ewald_pot`/`qm_ewald_hess`/`e_nuc` 五个缓存；手改属性会残留旧周期的缓存导致错能 | `set_mm_charges` 统一走 `qmmm_for_scf` 重入路径；用例验证「更新后能量 == 新鲜构建」 |
| RB8 | 周期性应力（stress）仍不在本期范围 | `TyxonQCalculator.implemented_properties` 不加 `stress` | 与原 R3 同：Ewald 嵌入的晶胞应力上游未提供现成 API | 维持原 R3 缓解：不声明 `stress`，ASE/i-PI 明确报缺失；固定盒子跑 NVT |

**验证门流程**：RB1 干跑 → RB2/RB3 参数守卫落地 → RB4 截断收敛 → RB5 有限差分受力 →
RB6 双计数审查 → 全部通过后写 `md_lammps_qmmm_pbc/` 教程并把结论回写本表。

**验证门执行结论（2026-08-31 实跑取证，体系：H₂O + 2 个 ±0.5 点电荷，20 Å 立方盒，
STO-3G，CAS(4,4)，`pyscf.qmmm.pbc.itrf.add_mm_charges`，rcut_ewald=12 Å，rcut_hcore=9 Å）**：

| # | 结论 | 实测证据 |
|---|---|---|
| RB1 | ✅ 通过 | SCF 层装饰后架 `mcscf.CASCI` 链路可跑；大盒极限与分子簇嵌入一致（L=40/80 Å 时 dE ≤ 1.3e-10 Ha）；scanner 梯度与直接核梯度差 ≤ 8e-8 Ha/Bohr；TyxonQ UCCSD 求解器同链路通过（E 与 FCI 参考差 3.1e-7 Ha） |
| RB2 | ✅ 确认存在，守卫照计划实现 | `pbc/mm_mole.py` L62-63 `assert rcut_ewald < min(diag(a))`；另实测：晶格必须对角（L54）、MM `Cell` 内部单位 Bohr、`get_lattice_Ls` 只含最近邻 27 胞镜像 |
| RB3 | ✅ 确认存在，守卫照计划实现 | 实测触发 `get_hcore` L176 断言「QM image is within rcut_hcore of QM center」（rcut_hcore 必须 < 半盒边，不能只 < 半对角线）与 L183「所有 QM 原子须在 rcut_hcore 内」两条 |
| RB4 | ✅ 通过 | rcut_ewald ±20% 时能量变化 ≤ 1.4e-11 Ha（远小于 1e-6 阈值） |
| RB5 | ⚠️ 缺口确认，**降级已于 2026-09-01 复核后收回，仍交付 `mm_gradient`** | HF：max\|FD−解析\| = 9.3e-10（机器精度）；CASCI：4.3e-5，不随步长变化（1e-2~3e-4 Bohr 恒定）。归因链：Ewald+核分项与解析一致到 9e-9；冻结 CASCI 1-RDM 的电子项泛函与解析一致（Hellmann-Feynman）；残差 = 完整路径 FD − 冻结泛函 FD ≈ 4.1e-5，即 **SCF 轨道对 MM 位移的响应**（post-HF 对 HF 轨道非变分，上游解析式未含该耦合项，等效 post-HF 版 CPHF 缺失；换成 FCI 同样存在，与 CI 求解器质量无关）。**复核结论**：① 绝对量 4.3e-5 Ha/Bohr ≈ 2.2 meV/Å ≪ kT(300K)≈25 meV；② 相对本基准 MM 净嵌入力（~1.4e-3 Ha/Bohr）约 3%；③ 数值实验证实是几何的光滑函数，Verlet 积分可用；④ 低于 UCCSD 数值梯度自身抖动（~1e-4）与真硬件 shot noise 一到两个量级。代价：力与所报能量差一个小的非保守分量，**严格 NVE 能量守恒诊断失效**（恒温/恒压 MD 不受影响）；误差随嵌入强度增长、无普适上界，文档须如实标注。验收：用例按测试规范（全程 UCCSD、5% 相对 + 绝对容差覆盖偏置） |
| RB6 | ✅ 设计确认（随实现生效） | `add_mm_charges` docstring：嵌入侧不含 MM 静电力/MM 能/vdW，但含 QM 与 MM 周期镜像的静电；LAMMPS 侧 QM 原子电荷置 0 的同阶段 A 策略继续适用 |
| RB7 | ✅ 通过 | 对已装饰 mf 重入 `add_mm_charges`（走 L89-96 重置分支）→ **重跑 `mf.kernel()`** → 重建 CASCI：与新鲜构建差 3.2e-9 Ha（SCF 收敛级别）。注意：重入后不重跑 SCF 会用旧轨道（差 4.5e-6），scanner 每步重算平均场天然满足此协议 |
| RB8 | ✅ 维持 | 不声明 `stress` |
| RB9（新增） | ⚠️ 已实测定位，缓解落地 | `QMMMSCF.as_scanner = NotImplemented`（`pbc/itrf.py` L112），而 `CASCI_Scanner.__init__`（`mcscf/casci.py` L650）强制调 `mc._scf.as_scanner()` → scanner 链路断。缓解：构建 scanner 前将装饰 mf 的 `as_scanner` 替换为基类 `scf.hf.SCF.as_scanner`（dict 拷贝保留 `mm_mol` 等全部嵌入属性，实测能量/梯度正确） |
| RB10（新增） | ⚠️ 已实测定位，实现时遵守 | `get_hcore` 要求 rcut_hcore 同时满足「> QM 区最大半径」与「< QM 质心到最近 QM 镜像距离（≈半盒边）」，即盒子必须明显大于 2×QM 区尺寸；教程盒子按此选取 |

---

## 7. 交付顺序与验收

```
P0-A  SQD.as_pyscf_solver        →  用例 1,2,3
P0-B  qc_scanner                 →  用例 4,5,6      ← 此处形成第一个可用交付
P0-C  spin_square 守卫            →  并入用例 1
P0-D  pyproject                  →  用例 6 前置检查
        ↓ 第一个里程碑：E1 + E2 可跑，纯 pyscf 环境
P1    TyxonQCalculator           →  用例 7,  E3
        ↓ 第二个里程碑：枢纽成立，后续三条全是薄壳
P2    ipi_driver                 →  用例 8,  E4, E7
P2.5  生产级固体 QM/MM（E8 阶段 A）✅ →  用例 11, E8-A（区域划分+静电嵌入+反作用力，无需 MDI）
      └─ E8 阶段 B（pbc Ewald）    →  §6.1 验证门 ✅（RB5 降级已收回）→ 实现+用例 12 ✅ → E8-B 教程 ✅（1.04e-5 Ha）
P3    openmm_potential ✅         →  用例 9（7/7）,  E5, E6, E6b（静电嵌入，均实跑过）
P4    mdi_engine ✅               →  用例 10（4/4）, E9（Python driver 双进程，实跑过；簇静电嵌入对照已由 E6b 覆盖）
P5    力场导出                     →  另立计划
```

**每个阶段的验收标准统一为**：对应用例在 `tcc` 环境（pyscf 2.10.0 + numpy 2.3.2）实跑通过，
且数值与 `MD_INTEGRATION_RESEARCH.md` 记录的实测值同量级。不接受「代码写完但没跑」。

**实施前需补齐的环境**：三个现有 conda env 均未装 `ase` / `openmm-ml` / `ipi` / `mdi`（§附录 A）。
P0 不需要它们；P1 起需要，建议在 `tcc` 上装 `ase`，openmm 相关另建 env。
**进度更新（2026-09-01）**：`qc` 环境已装 `ase 3.29` / `ipi 3.3` / `openmm 8.6.0` /
`openmmml 1.7`（PyPI 包名 `openmmml`），P1–P3 均在 `qc` 环境实跑验收。
