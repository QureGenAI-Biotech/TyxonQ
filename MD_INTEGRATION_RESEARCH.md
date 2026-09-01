# TyxonQ 接入分子动力学生态：调研结论

> 本文只记录**已核实的事实与实测数据**，不含实施方案。实施方案见 `MD_INTEGRATION_PLAN.md`。
>
> 所有源码引用均为逐字核对过的上游代码；所有数值均为本机实测（环境见附录 A）。
> 凡属推测的内容都显式标注为「推测」。

---

## 0. 一句话结论

TyxonQ 接入 OpenMM / LAMMPS **不需要为每个生态各写一套适配器**。上游已存在两个官方 ASE 适配层，
使 i-PI、OpenMM、LAMMPS 三条路线全部收敛到同一个枢纽（一个 ASE Calculator）；
而这个 Calculator 所需的核梯度**今天就能跑通**，完全复用 PySCF 现成代码。

唯一无法由 ASE 枢纽覆盖的能力是**静电嵌入 QM/MM**，它需要 MDI 协议，构成第二条独立分支。

---

## 1. 协议层：i-PI 与 MDI

### 1.1 i-PI 协议

- 角色命名：i-PI 自称 server，把力提供方称为 **driver**。
- 消息集合极小，固定 5 条：`STATUS` / `INIT` / `POSDATA` / `GETFORCE` / `EXIT`。
- 线格式：12 字节定长命令头 + 裸二进制载荷。
- `POSDATA` 载荷：cell(9) + celli(9) + natoms(1) + coords(3N)。
- `GETFORCE` 返回：status(12 字节) + pot(1) + nat(1) + forces(3N) + virial(9) + nextra(1)。
- **抽象层次**：把 driver 塑造成无状态纯函数 `(cell, pos) → (E, F, virial)`。
  i-PI 永远拥有 MD 主循环，driver 不被允许持有跨步状态，也不能被查询内部信息。

### 1.2 MDI 协议（MolSSI-MDI）

- 角色命名：MDI 把力提供方称为 **engine**，把控制方称为 **driver**。
- 命令约 40 条，且带**节点系统**（`@DEFAULT` / `@INIT_MD` / `@FORCES` / `@ENDSTEP` / `@PRE-FORCES`），
  driver 可以让 engine 跑到某个节点暂停、注入数据、再继续。
- 支持 `MD` / `OPTG` 等命令让 engine 跑**自己的**动力学或优化。
- 传输层三选：TCP / MPI / **LINK（编译成 `.so` 在同进程内调用，零 IPC 开销）**。
- **抽象层次**：远程操控引擎内部状态。允许 engine 是有状态、可查询的对象。

### 1.3 术语陷阱（两边命名正好相反）

| 角色 | i-PI 叫法 | MDI 叫法 |
|---|---|---|
| 提供能量与力的一方（TyxonQ 的位置） | **driver** | **engine** |
| 拥有采样/优化主循环的一方 | server | **driver** |

阅读两边文档时极易混淆，本仓库文档统一按各自原生术语并标注归属。

### 1.4 MDI 的 `-ipi` 兼容层：它到底是什么

这一层容易被误读为「MDI 通过 i-PI 接入」，**不是**。实际机制由两半拼成：

1. `MDI_Library/mdi_global.c` L642-643，每个新 communicator 的默认值：
   ```c
   // For version 1.4.0 of the MDI Library and above, will receive this from the other code
   new_comm.name_length = 12;
   new_comm.command_length = 12;
   ```
   而 MDI 自己的原生命令长度是 `mdi_global.h` 的 `#define MDI_COMMAND_LENGTH_ 256`。
   `general_send_command` 取二者较小值：
   ```c
   int count = MDI_COMMAND_LENGTH_;              // 256
   if ( this->command_length < MDI_COMMAND_LENGTH_ ) {
     count = this->command_length;               // 无 MDI 握手时停在 12
   }
   ```

2. `-ipi` 标志（`mdi_general.c` 中 `ipi_compatibility` 仅 3 处：L197 置位、L463、L527）
   在 `general_send` / `general_recv` 里关掉 MDI 自己的 4-int 载荷头
   （error flag / header type / datatype / count）。

两半合起来，MDI 在 `-ipi` 下吐出的字节流 = 12 字节命令 + 裸载荷 = **逐字节等于 i-PI 线格式**。

**关键限制**：`-ipi` 只改传输层，**不改词汇表**。i-PI 的协议语义必须由调用方手写——
参考 `tests/MDI_Test_Codes/driver_ipicomp_py/driver_ipicomp_py.py`，它逐条手拼
`INIT` / `STATUS` / `POSDATA` / `GETFORCE` / `EXIT`。

**反方向不通**：真正的 i-PI server 向通用 MDI engine（如 LAMMPS）发 `POSDATA`，
engine 会回 "Unrecognized command"，因为它的命令表里没有这个词。

**旁证**：`tests/MDI_Test_Codes/engine_ipi_cxx/engine_ipi_cxx.cpp` 干脆不链接 MDI，
自带 `ipi_sockets.c`，主循环是死读 12 字节：
```cpp
int ipi_msglen = 12;
char* command = new char[ipi_msglen];
while( not exit_signal ) {
  readbuffer(&sockfd, command, &ipi_msglen);
  execute_command(command, sockfd, NULL);
}
```

结论：**链路里没有 i-PI 这个程序，也不存在桥**。`-ipi` 是「MDI 把自己降级成说 i-PI 的话」。

### 1.5 能力对比

| 能力 | i-PI | MDI |
|---|---|---|
| 消息/命令数量 | 5 | ~40 |
| 节点系统（暂停/注入/续跑） | ✗ | ✓ |
| engine 可持有跨步状态 | ✗（设计上禁止） | ✓ |
| 传输层 | TCP / UNIX socket | TCP / MPI / **LINK（同进程）** |
| **传 MM 点电荷进 QM 哈密顿量** | **✗（结构上无此载荷）** | **✓ `>NLATTICE`/`>CLATTICE`/`>LATTICE`** |
| 静电嵌入 QM/MM | ✗ 只能机械嵌入 | ✓ |
| PIMD / RPMD / instanton / 核量子效应 | ✓（独占） | ✗ |
| 高级恒温器 / 副本交换 | ✓（独占） | ✗ |
| 让 engine 跑自己的 MD/优化 | ✗ | ✓ `MD` / `OPTG` |
| 周期体系 virial/stress | ✓ 必需 | ✓ 可选 |

**决定性差异**：`>NLATTICE` / `>CLATTICE` / `>LATTICE` 把 MM 点电荷的数量、坐标、电荷值
灌进 QM 哈密顿量。i-PI 的 5 条消息里**没有任何等价载荷**，因此 i-PI 路线只能做
机械嵌入（多个 forcefield 的力线性叠加），**QM 区感受不到 MM 的静电极化**。

### 1.6 活跃状态（实测于 2026-08）

| 指标 | i-pi/i-pi | MolSSI-MDI/MDI_Library |
|---|---|---|
| star | 318 | 35 |
| fork | 146 | 12 |
| contributors | **52** | **4** |
| commit 集中度 | ceriottm 454 / litman90 419 / EliaStocco 331 / eszterpos 193 / venkatkapil24 190 / mahrossi 169 … | **968 of 987 出自 taylor-a-barnes** |
| 2026 年 release | 6 个（最新 v3.3.0，2026-06-30） | v1.4.40（2026-06-17） |
| 最近提交性质 | 持续功能开发 | 纯维护（`Fix atomic_unit_of_force conversion`、`Add PyPI publication workflow`、`Fix pip cmake arguments`） |
| LICENSE | 无 LICENSE 文件 | BSD-3-Clause |

**MolSSI-MDI 组织的卫星仓库大面积休眠**，且几乎全 0 star：

| 仓库 | 最后 push |
|---|---|
| `MDI_QMMM_Driver` | 2020-05-21 |
| `MDI_Metadynamics` | 2020-09-22 |
| `MDI_NEB_Driver` | 2021-05-12 |
| `MDI_GCMC` | 2022-06-01 |
| `MDI_OpenMM_Plugin` | 2023-09-13 |
| `MDI_QCEngine` | 2023-09-15 |

→ 「复用 MolSSI 现成 driver」**不是**一条可依赖的价值来源。

**MDI 的真实价值来自 in-tree 集成**：

| 位置 | 体量 | 最近改动 |
|---|---|---|
| LAMMPS `src/MDI/`（16 文件） | `mdi_engine.cpp` 60,634 B<br>`fix_mdi_qmmm.cpp` 58,884 B<br>`fix_mdi_qm.cpp` 33,326 B<br>`mdi_plugin.cpp` 5,411 B | 2026-07-28 |
| LAMMPS `src/MISC/fix_ipi.cpp` | 17,205 B | 2026-07-03 |
| psi4 主干 | `psi4/driver/mdi_engine.py`（498 行）+ `external/upstream/mdi` + `tests/pytests/test_mdi.py` | 在维护 |

LAMMPS 在 MDI 上投入的代码量约为 i-PI 的 **9 倍**。
单人维护风险的缓解因素：协议已冻结在 1.4.x、BSD-3 许可、LAMMPS 与 psi4 均已 vendored in-tree、
且 akohlmey（LAMMPS 主维护者）在 4 位 contributor 之列。

---

## 2. 生态接入点现状

### 2.1 上游已存在的两个官方 ASE 适配层（本调研最重要的发现）

**i-PI 侧：`ipi/pes/_ase.py`**

```python
class ASEDriver(Dummy_driver):
    """Base class using an arbitrary ASE calculator as i-pi driver.
    Should not be called directly as it does not set a calculator."""
    def __init__(self, template, has_energy=True, has_forces=True, has_stress=True, ...)
    def check_parameters(self):     # 子类在此设置 self.ase_calculator
    def convert_units(self, cell, pos)   # Å / eV ↔ 原子单位，已实现
    def compute_structure(self, cell, pos)
    def post_process(self, properties, structure)   # virial、Voigt 展开、extras 全已实现
```

→ 提供一个 ASE Calculator，即可通过继承 `ASEDriver` 得到完整 i-PI driver，
单位换算、virial、extras、socket 全部由上游负责。

**OpenMM 侧：`openmmml/models/asepotential.py`**

```python
class ASEPotentialImpl(MLPotentialImpl):
    """This MLPotentialImpl implements potentials using an ASE calculator.
    >>> potential = MLPotential('ase')
    >>> system = potential.createSystem(topology, calculator=calculator)
    >>> system = potential.createSystem(topology, calculator=calculator, info={'charge':2})
    """
```

同时 `openmmml/mlpotential.py` 提供
`createMixedSystem(topology, system, atoms, forceGroup, interpolate=False)`，
`interpolate=True` 时引入全局参数 `lambda_interpolate`，可做 MM↔QM 自由能微扰。

openmm-ml 状态健康：180 star、51 fork、2026-08-24 有提交、2026 年发布 1.5 / 1.6 / 1.7。

### 2.2 各生态的接入方式汇总

| 目标 | 接入方式 | TyxonQ 侧成本 |
|---|---|---|
| ASE 自身（优化/MD/频率/NEB） | 直接用 Calculator | 0（枢纽本身） |
| i-PI | 继承 `ipi/pes/_ase.py` 的 `ASEDriver` | 极小 |
| LAMMPS（力提供方） | i-PI driver + LAMMPS `fix ipi` | 0 |
| OpenMM（纯 QM / 机械嵌入 QM/MM / 自由能） | `MLPotential('ase').createSystem(...)` | 极小 |
| LAMMPS（静电嵌入 QM/MM） | MDI engine + `fix mdi/qm` / `fix mdi/qmmm` | 中（需实现 MDI engine） |

### 2.3 i-PI 的 PES 插件生态

`ipi/pes/` 共 23 个插件，ML 势主导：
`_ase, _mace, metatomic, pet, so3lr, rascal, psiflow, xtb, elphmod, bath, doublewell, dummy, ...`

说明：这些插件均**惰性 import 各自后端**，i-PI 不硬依赖任何一个。
因此向上游添加一个 `tyxonq` 插件在依赖方向上没有问题。

---

## 3. PySCF 架构调研

### 3.1 三个万能钩子

| 钩子 | 实现文件数 | 契约 |
|---|---|---|
| `method.nuc_grad_method()` | 56 | 返回梯度对象 |
| `grad_obj.as_scanner()` | 20 | `scanner(mol_or_geom) -> (e_tot, de)`，Hartree / Hartree·Bohr⁻¹ |
| fcisolver 鸭子类型 | — | mcscf 只要求 `kernel` / `make_rdm1` / `make_rdm12` / `spin_square` |

### 3.2 `as_scanner()` 是 PySCF 全生态跑 MD 的唯一充分条件

`pyscf/md/integrators.py` 的 `_Integrator.__init__`（L188-195）：

```python
def __init__(self, method, **kwargs):
    if getattr(method, 'as_scanner', None):
        self.scanner = method.as_scanner()
    elif getattr(method, 'nuc_grad_method', None):
        self.scanner = method.nuc_grad_method().as_scanner()
```

之上是 `VelocityVerlet`（导出为 `pyscf.md.NVE`）与 `NVTBerendson`。
即：任何提供 `as_scanner()` 的方法都能零协议代码跑 AIMD。

### 3.3 `pyscf/grad/casci.py`：TyxonQ 复用梯度的全部代价

353 行代码中**只有一处触碰求解器**：

```python
casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)
dm_core = numpy.dot(mo_core, mo_core.T) * 2
dm_cas  = reduce(numpy.dot, (mo_cas, casdm1, mo_cas.T))   # mo_cas = mo_coeff[:,ncore:nocc]
```

→ `casdm1` 必须是 **ncas×ncas 的 CAS-MO 基**。其余全是经典部分：
`ao2mo.kernel` 取 aapa、`get_jk`、Lagrangian `Imat`、Z-vector、
`cphf.solve` 轨道响应、`hcore_generator` / `get_ovlp` 的 AO 导数积分。

同文件 L323-333 刻意为外部模块留了口子：

```python
# Initialize hcore_deriv with the underlying SCF object because some
# extensions (e.g. x2c, QM/MM, solvent) modifies the SCF object only.
def hcore_generator(self, mol=None):
    mf_grad = self.base._scf.nuc_grad_method()
    return mf_grad.hcore_generator(mol)

# Calling the underlying SCF nuclear gradients because it may be modified
# by external modules (e.g. QM/MM, solvent)
def grad_nuc(self, mol=None, atmlst=None):
    mf_grad = self.base._scf.nuc_grad_method()
    return mf_grad.grad_nuc(mol, atmlst)
```

这正是 QM/MM 能与 CASCI 梯度自动叠加的原因（已实测，见 §4.1）。

**工业验证证据**：`pyscf/grad/` 共 31 个文件（`rhf.py` 472、`rks.py` 625、`ccsd.py` 462、
`mp2.py` 329、`casci.py` 353、`casscf.py` 233 …），多处并存 slow 参考实现与优化实现互校，
并引用原始文献（如 J. Comput. Chem., 5, 589）。

### 3.4 `pyscf/qmmm/itrf.py`：静电嵌入现成

```
 35: def add_mm_charges(scf_method, atoms_or_coords, charges, radii=None, unit=None)
 87: def qmmm_for_scf(method, mm_mol)
113: class QMMM       118: class QMMMSCF(QMMM)       202: class QMMMPostSCF(QMMM)
218: def add_mm_charges_grad(scf_grad, atoms_or_coords, charges, radii=None, unit=None)
264: def qmmm_grad_for_scf(scf_grad)      280: class QMMMGrad
```

### 3.5 PySCF 有没有 i-PI / MDI 接入？——**都没有**

严格核实过程（避免误报）：

| 检查 | 结果 |
|---|---|
| 安装目录内容级 grep `i-PI` / `MDI` / `ASE` | 仅 1 处命中，且为误报 |
| 误报 1：`pyscf/df/autoaux.py:58` | 是 `MolSSI-BSE/basis_set_exchange` 的 URL 注释，与 MDI 无关 |
| 误报 2：`pyscf/solvent/pol_embed.py` | 是变量名 `e_ipip`，与 i-PI 无关 |
| GitHub 代码搜索 `pyscf ipi` | 0 结果 |
| GitHub 代码搜索 `pyscf mdi` | 0 结果 |
| GitHub 搜索 `pyscf ase calculator` | 仅 `awvwgk/dft3c`(3★)、`kangmg/PySCF4ASE`(0★)，均非官方 |

**与 psi4 的反差**：

| | i-PI 接入 | MDI 接入 | ASE 接入 |
|---|---|---|---|
| psi4 | ✗ | ✓ in-tree（`mdi_engine.py` 498 行） | 第三方 |
| PySCF | ✗ | ✗ | 仅零星第三方 |

→ 「沿用 PySCF」在协议接入这件事上**无路可沿**；反过来说，
TyxonQ 做 MDI engine 是**领先 PySCF** 的。

### 3.6 psi4 的 MDI engine 命令表（TyxonQ 的抄写模板）

```python
"<NATOMS": send_natoms,        "<COORDS": send_coords,
"<CHARGES": send_charges,      "<ELEMENTS": send_elements,
"<MASSES": send_masses,        "<ENERGY": send_energy,
"<FORCES": send_forces,        ">COORDS": recv_coords,
">NLATTICE": recv_nlattice,    ">CLATTICE": recv_clattice,
">LATTICE": recv_lattice,      ">MASSES": recv_masses,
"<DIMENSIONS": send_dimensions,
"<TOTCHARGE": send_total_charge,  ">TOTCHARGE": recv_total_charge,
"<ELEC_MULT": send_multiplicity,  ">ELEC_MULT": recv_multiplicity,
...
MDI_Register_Node("@DEFAULT")     # 只注册一个节点
```

→ QM engine **不需要碰节点系统**，498 行足够。

---

## 4. TyxonQ 现状实测

### 4.1 核梯度：不是缺口，今天就能跑

基准体系 H₂O / STO-3G，CAS(4,4)，`ncore=3`。链路为：

```python
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.fcisolver = UCCSD.as_pyscf_solver(runtime="numeric")
scanner = mc.nuc_grad_method().as_scanner()
```

| 探针 | 结果 |
|---|---|
| CASCI 解析梯度 vs stock CASCI | `max\|Δg\| = 2.911531e-06` Hartree/Bohr；`ΔE = +2.925e-07` |
| `as_scanner()` 跨两个几何 | per-geometry `max\|Δg\|` = 2.912e-06 / 8.387e-07 |
| 有限差分交叉验证 | `d/d(H1 z)` 差 7.41e-07；`d/d(H2 y)` 差 1.07e-06 |
| 静电嵌入 QM/MM（`add_mm_charges`）vs stock | `ΔE = 2.96e-07`；`max\|ΔGrad\| = 1.091e-06` |
| MM 电荷的物理效应（证明电荷真进了哈密顿量） | 能量位移 **-0.011950 Hartree**；梯度位移 **0.003894 Hartree/Bohr** |

残余误差全部来自 UCCSD 相对 FCI 的 ansatz 误差，不是基变换错误。

### 4.2 `basis="AO"` 默认值在设计上正确

`from_integral` 构造的 `_Molecule` 满足 `int1e_nuc = h1`（已在外层 CAS-MO 基）、`ovlp = I`。
内层 RHF 对角化近对角的 h1，得 `C ≈ I`（**实测偏离恒等仅 8.29e-07**）。
UCC ansatz 建在内层 MO 基，`basis="AO"` 执行 `rdm_mo2ao(rdm, C) = C·rdm·Cᵀ`
回到内层 AO 基，而内层 AO 基 ≡ 外层 CAS-MO 基 ✓

实测：`basis='MO'` 2.985e-06 vs `basis='AO'` 3.631e-06。

结论：`"AO"` 普遍正确且更鲁棒；`"MO"` 才是内层轨道一旦发生旋转就会**静默出错**的那个。
**现有默认值不需要修改。**

### 4.3 SQD 与 LUCJ 的定位

| | LUCJ | SQD |
|---|---|---|
| 本质 | 电路构建器 `get_circuit(params) -> Circuit` | 能量方法 |
| 有能量吗 | ✗ | ✓ |
| 有 RDM 吗 | ✗ | ✓ `SCIResult.rdm1` / `.rdm2` |
| 有 `as_pyscf_solver` 吗 | ✗ | ✗（**待补**） |
| 在工作流中的角色 | **采样器** | **求解器** |

→ LUCJ 不是能量方法，**不应该有 `as_pyscf_solver`**；它是喂给 SQD 的采样器。

全仓库现状：只有 `ucc.py:961` 与 `hea.py:559` 有 `as_pyscf_solver`。

SQD 的 RDM 来自 **PySCF 自己的 `pyscf.fci.selected_ci`**，且能量就是由这两个 RDM 算出的
（`sqd/fermion.py:392-408`）：

```python
myci = fci.selected_ci.SelectedCI()
_, sci_vec = fci.selected_ci.kernel_fixed_space(myci, one_body_tensor, two_body_tensor,
                                                norb, nelec, ci_strs=ci_strings, **kwargs)
dm1 = myci.make_rdm1(sci_vec, norb, nelec)
dm2 = myci.make_rdm2(sci_vec, norb, nelec)
energy = np.einsum("pr,pr->", dm1, one_body_tensor) + 0.5 * np.einsum("prqs,prqs->", dm2, two_body_tensor)
```

注意 `ci_strs=` 是**显式入参** → 在 `solve_sci` 这一层，冻结行列式子空间天然被支持。

⚠️ 但**上层的 `run_sqd_fermion(include_configurations=...)` 不是冻结机制**。
读 `fermion.py` L294-302：

```python
strs_a = np.concatenate((include_a, carryover_strings_a, samples_a))
strs_a = _unique_with_order_preserved(strs_a)[:max_dim_a]
```

`include_*` 只是把指定串**并入**采样得到的串，随后被 `max_dim` 截断，
最终子空间仍随采样漂移。因此实现「冻结」必须**绕过 SQD 主循环**，
直接调 `solve_sci(ci_strs=(strs_a, strs_b), ...)`。

冻结所需的串可从上一次结果里取：`SCIResult.sci_state.ci_strs_a` / `.ci_strs_b`（L403-405 已暴露）。

### 4.4 SQD 能否满足 PySCF 梯度契约：实测

**A) 冻结子空间 = 全 CAS（此时 selected-CI ≡ CASCI，应精确相等）**

| 量 | 值 |
|---|---|
| `ΔE` vs stock CASCI | **1.563e-13** |
| `max\|ΔGrad\|` | **1.892e-08** |

→ SQD → CASCI 梯度链路**精确**。`selected_ci` 的 RDM 约定与 `grad/casci.py` 直接兼容，
**不需要任何基变换技巧**（比 UCC 路线更干净：UCC 的 2.9e-06 是 ansatz 误差）。

**B) 冻结但截断的子空间（3×3）vs 有限差分**

| 量 | 值 |
|---|---|
| `E` 高于全 CAS | +3.037e-03 Hartree（变分上界，符号正确 ✓） |
| `d/d(H1 z)` 解析 vs FD | 差 **2.820e-07** |
| `d/d(H2 y)` 解析 vs FD | 差 **5.767e-07** |

→ **在固定行列式子空间内 Hellmann-Feynman 成立**，`grad/casci.py` 给出的解析梯度是精确的。
物理原因：SCI 解在 span(S) 内变分最优，故 `∂c/∂R` 项消失；轨道响应由 CPHF 处理。

**C) 确定性重选子空间（负对照）——出乎预料地也通过了**

| 量 | 值 |
|---|---|
| `d/d(H1 z)` 差 | 1.815e-07 |
| `d/d(H2 y)` 差 | 4.936e-07 |

原因：位移 1e-3 Å 时按幅值排序的 top-3 行列式**没有变化**，选择在局部是常量，
因此能量局部光滑。**不连续性只在排序发生翻转的点出现。**

→ 修正认识：真正的杀手不是「子空间随几何变」，而是**随机性**。

**D) 随机子空间（真实 SQD 的采样行为）——致命**

同一几何、8 次独立采样：

| 量 | 值 |
|---|---|
| 能量散布 | **3.328e-03 Hartree**（约 2 倍化学精度 1.6 mHa） |
| 能量标准差 | 1.343e-03 Hartree |
| `dE/d(H1 z)` 散布 | 5.093e-03 |
| 力分量标准差 | **4.311e-03 Hartree/Bohr** |

有限差分（正负位移各自独立采样）：

| 方向 | 解析 | FD | 差 |
|---|---|---|---|
| `d/d(H1 z)` | -0.032433 | -0.030504 | **1.929e-03** |
| `d/d(H2 y)` | -0.024300 | -0.023592 | **7.080e-04** |

固定 seed（子空间实际被冻结）后：

| 方向 | 差 |
|---|---|
| `d/d(H1 z)` | **5.929e-08** |
| `d/d(H2 y)` | 5.326e-07 |

**结论**：随机子空间下能量**不是 R 的函数**，「梯度」在数学上没有定义；
误差比冻结模式差 **4 个数量级**。冻结子空间是 MD 的**正确性前提**，不是效率优化。

⚠️ 重要细节：本探针中「固定 seed」之所以有效，是因为随机抽取与 `h1`/`h2` 无关。
在真实 SQD 工作流里，样本来自测量一个**参数依赖几何**的电路，
因此**固定 seed 并不保证得到同一子空间**。必须显式冻结行列式列表。

### 4.5 集成障碍（已定位）

CASCI 传给 fcisolver 的 `eri_cas` 是 **8 重对称压缩格式**，而 `solve_sci` 要求全 4 指标张量：

```
ValueError: einstein sum subscripts string contains too many subscripts for operand 1
  at sqd/fermion.py:396  →  np.einsum("prqs,prqs->", dm2, two_body_tensor)
```

修复：适配层必须做 `h2_full = ao2mo.restore(1, np.asarray(h2), norb)`。
（UCC 路线不受影响，因为它走 `from_integral` 的另一条通路。）

### 4.6 每几何成本标定

6 点轨迹，取后 5 点均值（首点含 SCF 冷启动）。单位：秒/几何。

| 体系 | stock CASCI | 冻结 SQD（全 CAS） | 冻结 SQD（截断） | UCCSD 冷启动 | UCCSD 热启动 |
|---|---|---|---|---|---|
| STO-3G，CAS(4,4)，dim 36 | 0.008 | 0.009 | 0.008 | 0.019 | 0.018 |
| 6-31G，CAS(6,6)，dim 400 | 0.031 | 0.018 | 0.017 | 0.079 | 0.076 |
| 6-31G，CAS(8,8)，dim 4900 | 0.034 | 0.023 | 0.018 | **0.672** | **0.666** |

三条硬结论：

1. **冻结 SQD 的成本几乎不随活性空间增长**（0.008 → 0.017 → 0.018 s），
   在 CAS(6,6) 与 CAS(8,8) 上**比 stock CASCI 还快**（截断空间的对角化比全 CAS 便宜）。
2. **UCCSD 成本快速膨胀**：0.019 → 0.079 → 0.672 s，从 CAS(4,4) 到 CAS(8,8) 涨 **35 倍**；
   在 CAS(8,8) 上是冻结 SQD 的 **37 倍**。
3. **参数热启动的收益只有 1–3%，可忽略**。原因推测：`as_pyscf_solver` 每次调用
   `from_integral` 重建实例并重算 MP2 初值，而 MP2 初值本身已经很好，
   优化器迭代数几乎不变。

→ 修正认识：**热启动不是关键工程量**；**冻结子空间 SQD 才是最现实的 MD 引擎**。

---

## 5. 被数据否掉的方案（不要再考虑）

| 方案 | 否掉的理由 |
|---|---|
| 「MDI 通过 i-PI 接入 OpenMM」 | `-ipi` 只是把 MDI 线格式降级成 i-PI 格式，不改词汇表；链路里没有 i-PI 进程，不存在桥（§1.4） |
| 「复用 MolSSI 现成 MDI driver」 | 卫星仓库大面积休眠（2020–2022），且 `-ipi` 模式砍掉了它们所需的完整命令集（§1.6） |
| 「先实现自己的核梯度」 | 已实测可用，误差 2.9e-06；PySCF 的 31 文件梯度栈可直接复用（§4.1） |
| 「`basis="AO"` 默认值有风险，需改成 MO」 | 恰好相反，`"AO"` 才是正确且鲁棒的（§4.2） |
| 「参数热启动是 MD 可行性的前提」 | 实测收益 1–3%，可忽略（§4.6） |
| 「给 LUCJ 加 `as_pyscf_solver`」 | LUCJ 不是能量方法，是采样器（§4.3） |
| 「SQD 直接用于 MD」 | 随机子空间使 E 不成为 R 的函数，力噪声 4.3e-03 Hartree/Bohr；必须先冻结子空间（§4.4 D） |

---

## 6. 已知环境地雷

`pyscf 2.9.0 + numpy 2.4.4` 会在**PySCF 参考实现内部**崩溃：

```
File "pyscf/grad/casci.py", line 152, in grad_elec
    dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
File "pyscf/lib/numpy_helper.py", line 245, in einsum
    inds, idx_rm, einsum_str, remaining = contraction[:4]
ValueError: not enough values to unpack (expected 4, got 3)
```

原因：numpy 2.4 改变了 `einsum_path` 的返回结构，pyscf 2.9 的 `lib.einsum` 未跟上。
与 TyxonQ 无关。

实测版本矩阵：

| 环境 | pyscf | numpy | 梯度可用 |
|---|---|---|---|
| `qc` | 2.9.0 | 2.4.4 | ✗ 崩溃 |
| `tcc` | 2.10.0 | 2.3.2 | ✓ |
| `openmm` | 2.8.0 | 1.26.4 | 未测 |

⚠️ 当前 `pyproject.toml` 的 `pyscf>2.5` + `numpy>=2.0.2` **允许这个坏组合**。

---

## 附录 A：实测环境

| 环境 | 关键版本 |
|---|---|
| `tcc`（主验证环境） | pyscf 2.10.0、numpy 2.3.2、scipy 1.16.1 |
| `qc` | pyscf 2.9.0、numpy 2.4.4、torch 2.8.0、tyxonq 1.0.0 |
| `openmm` | openmm 8.2、pyscf 2.8.0、numpy 1.26.4 |

TyxonQ 以 `PYTHONPATH=src` 方式从工作区源码加载。

**三个环境均未安装 `ase`、`openmm-ml`、`ipi`、`mdi`**——这四者是实施阶段需要补齐的验证依赖。

## 附录 B：单位约定

| 量 | PySCF / MDI | ASE | i-PI 内部 | OpenMM |
|---|---|---|---|---|
| 长度 | Bohr | Å | Bohr | nm |
| 能量 | Hartree | eV | Hartree | kJ/mol |
| 力 | Hartree/Bohr | eV/Å | Hartree/Bohr | kJ/mol/nm |

`1 Bohr = 0.52917721092 Å`。MDI 用原子单位，与 PySCF 原生一致，**无需换算**。
