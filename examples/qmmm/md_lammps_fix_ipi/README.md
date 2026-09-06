# E7 — TyxonQ 经 i-PI 桥接接入 LAMMPS（面向固体物理 QM/MM）

## 设计目标

让 **TyxonQ 的量子化学势能面方便地接入 LAMMPS**，服务固体物理 QM/MM 场景。
技术路线（调研结论）：走 **i-PI 桥接**——LAMMPS 官方支持 `fix ipi`，
TyxonQ 侧只需一个基于 `ipi.pes._ase.ASEDriver` 的薄壳
（`tyxonq.applications.chem.interfaces.ipi_driver`），协议零手写。

## 拓扑（三进程）

```
LAMMPS（fix ipi，client / MM 力端）
        │   i-PI socket 协议（POS/力，原子单位）
        ▼
i-PI server（MD 引擎：系综、积分器、PIMD 都在这里，input.xml）
        ▲
        │   i-PI socket 协议
TyxonQ driver（i-pi-driver-py -m custom，client / QM 力端）
```

`input.xml` 的 `<forces>` 把两个力场**加权求和**（`<force forcefield='...' weight='...'/>`，
已对 ipi 3.3.0 `InputForceComponent.weight` 取证）。这就是 i-PI 官方文档所说的
"mixing first-principles calculations and empirical force fields"——
固体 QM/MM 中 delta 学习、嵌入修正等配方的组合基础。

## 文件

| 文件 | 角色 |
|---|---|
| `in/input.xml` | i-PI server 输入：两个 `ffsocket`（`tq` / `lammps`）+ `<forces>` 求和 |
| `in/in.qmmm` | LAMMPS 输入：`lj/cut` + `fix ipi`（client） |
| `in/water.lmp` | LAMMPS data 文件（原子顺序必须与 i-PI 侧 xyz 一致：O, H, H） |
| `run.py` | 一键编排：算参考值 → 填端口占位符 → 按序拉起三进程 → 校验 |

## 运行

一键（推荐）：

```bash
python examples/qmmm/md_lammps_fix_ipi/run.py
```

手动三终端（占位符自行替换为实际端口；`run.py` 打印的工作目录里有现成的替换后文件）：

```bash
# 1. 先起 server（必须最先，绑好两个端口）
i-pi input.xml &

# 2. 再起 TyxonQ driver（QM 力端）
i-pi-driver-py -a 127.0.0.1 -p <PORT_TQ> -m custom \
    -P <repo>/src/tyxonq/applications/chem/interfaces/ipi_driver.py \
    -o "water.xyz,basis=sto-3g,active_space=4 4,method=uccsd" &

# 3. 最后起 LAMMPS（MM 力端）
lmp -in in.qmmm
```

启动顺序不能乱：两个 client 都只会连一次，连不上即退出。

## 本示例的验证逻辑

演示体系是水分子 + 很弱的 LJ（`epsilon=0.001 kcal/mol`，故意取小，
只证管线不喧宾夺主）。`run.py` 会：

1. 在初始几何上直接调 `qc_scanner` 得 `E_QM`，按 `in.qmmm` 的 LJ 参数
   解析算 `E_MM`；
2. 跑完三进程模拟后校验：**第 0 步总势能 ≈ E_QM + E_MM**。

实测（macOS arm64，LAMMPS 22Jul2025u5 + ipi 3.3.0）：

```
E_QM = -74.97045409 Ha，E_MM = +0.00258742 Ha，E_ref = -74.96786667 Ha
step-0 potential = -74.96787436 Ha，偏差 7.7e-06 Ha
```

偏差来自 i-PI 单位常数的回转精度（`ipi.utils.units` 仅约 8 位有效数字，
详见 `tests_applications_chem/test_ipi_driver.py` 的容差注释）。
LAMMPS 日志里 `E_pair = 1.6236 kcal/mol` 与解析值 2.587e-3 Ha 完全一致。

## 已知行为与边界

- **LAMMPS 的"报错"是正常收尾**：模拟结束时 LAMMPS 日志会出现
  `ERROR ... Got EXIT message from i-PI. Now leaving!`——这是 `fix ipi`
  收到 server EXIT 后的正常退出路径（`fix_ipi.cpp`），不是故障。
- **本例是"管线验证"，不是生产级固体 QM/MM**：真实固体需要
  ① QM/MM 区域划分（LAMMPS 管全部原子，TyxonQ 只管 QM 区）；
  ② 静电嵌入（MM 点电荷进 QM 哈密顿）。区域划分 + 静电嵌入已在 **E8-A**
  （簇，`md_lammps_qmmm_embedded/`）与 **E8-B**（周期 Ewald，
  `md_lammps_qmmm_pbc/`）落地，MDI 专线对照见 **E9**（`md_mdi_qmmm_embedded.py`）。
- **TyxonQ 侧本期不支持周期应力**（计划风险表 R3）：Calculator 不声明
  `stress`，i-PI 按零 virial 处理；做 NPT/变胞模拟前请先确认这一点。
- `fix ipi` 不支持 `units lj`（约化单位）；本例用 `units real`。
- i-PI 会把 20 Å 盒子的晶胞也发给 TyxonQ driver，driver 忽略它
  （分子体系无 pbc）；固体体系换晶胞即可，协议不变。

## 环境要求

- PySCF、ASE、i-PI（`pip install -U ipi`；勿用 `pip install i-PI`，那是废弃占位包）
- LAMMPS 编译含 `fix ipi`（MISC 包；conda-forge 的 `lammps` 默认包含），
  `lmp` 在 PATH 上
- 依赖缺失时 `run.py` 直接 `exit 0`，方便 CI
