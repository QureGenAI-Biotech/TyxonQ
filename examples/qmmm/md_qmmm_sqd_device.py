"""E11：QM/MM 上的 SQD——采样冻结子空间驱动势能面、AIMD、静电嵌入与真机档。

E1/E2 展示了 ``qc_scanner`` 的 VQE 族（UCCSD/HEA）本地数值路径，E10 展示了
UCCSD/HEA 如何把哈密顿量测量提交到量子设备。本教程补上**第三条算法路径 SQD**
（sample-based quantum diagonalization）：它同样是 ``qc_scanner`` 的一等方法
（``method="sqd"``，且是默认方法），但量子/经典的分工与 VQE 族**根本不同**：

    UCCSD/HEA：量子设备**测量哈密顿量期望值** → 经典优化 ansatz 参数
    SQD      ：量子设备**采样 LUCJ 电路得到 counts** → 经典 selected-CI 对角化

这个差异决定了「上设备」的选项挂在哪里：

    UCCSD/HEA：运行选项经 ``solver_kwargs``（shots/provider/device）穿透到 kernel；
    SQD      ：运行选项挂在**采样器** ``lucj_sampler(runtime=..., provider=...,
               device=..., shots=...)`` 上——因为量子部分就是「采样 LUCJ 电路」。

SQD 的 MD 语义核心是**子空间策略**（依据 ``MD_INTEGRATION_RESEARCH.md`` §4.4）：

    subspace="frozen"  （默认）首帧采样确定行列式子空间后**锁存**，能量面是几何的
                       光滑函数 → 解析核梯度有定义 → 唯一可用于 MD 的模式；
    subspace="refresh" 每帧重新采样 → 随机子空间下能量**不是几何的函数**，力噪声
                       ~4.3e-3 Ha/Bohr（§4.4 D）→ 解析梯度在数学上无定义，
                       ``qc_scanner`` 默认拒绝，需显式 ``allow_discontinuous=True``；
    subspace="adaptive" 每 N 帧刷新，能量面分段光滑，仅供研究。

本教程分五节：① 采样冻结子空间的能量/力（含有限差分自检）；② frozen vs refresh
力噪声对比（§4.4 D）；③ 冻结 SQD 驱动 PySCF 原生 AIMD（NVE）；④ 静电嵌入
（``mm_charges``/``mm_gradient``）；⑤ device 采样档穿透 + 真机模板。

所需环境：tyxonq + pyscf>=2.10（不需要 ase / openmm / ipi / mdi）。
预期运行时间：约 1 分钟量级（H2O / STO-3G / CAS(4,4)；SQD 采样只在首帧跑一次
``run_sqd_fermion``，冻结后每帧只做小 selected-CI 对角化，故 MD 很快）。
依赖缺失时本脚本优雅退出，不报错。
"""

import io
import sys
import time

import numpy as np

# ---- 依赖守卫：缺 pyscf 时优雅退出（CI 友好） ----
try:
    from pyscf import gto, mcscf, md, scf
    from pyscf.md.distributions import MaxwellBoltzmannVelocity
except ImportError:
    print("本教程需要 pyscf>=2.10：pip install 'pyscf>=2.10'")
    sys.exit(0)

from tyxonq.applications.chem.algorithms.sqd import lucj_sampler
from tyxonq.applications.chem.interfaces.scanner import qc_scanner

# ---------------------------------------------------------------------------
# 演示体系：QM 水分子 + 一个 tip3p MM 水（静电嵌入），与测试用例 / E10 同款。
# 约定：QM 几何与 MM 位置一律用 Bohr 交给 scanner（几何构建可用 Å，见 E1）。
# ---------------------------------------------------------------------------
B2A = 0.52917721092
WATER = "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692"
H2O_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])
H2O_BOHR = H2O_ANG / B2A
MM_POS_BOHR = (H2O_ANG + np.array([2.9, 0.8, 0.3])) / B2A   # MM 水位置（Bohr）
MM_CHARGES = np.array([-0.834, 0.417, 0.417])               # tip3p 电荷
SPEC = [("O", tuple(H2O_ANG[0])), ("H", tuple(H2O_ANG[1])), ("H", tuple(H2O_ANG[2]))]
BASIS, ACTIVE_SPACE = "sto-3g", (4, 4)   # (n_elec, n_orb)

# SQD 采样参数（对齐 examples/h2o_sqd.py 与守卫测试）：
LUCJ_LAYERS = 1        # LUCJ 电路层数
SHOTS = 4096           # 采样次数
SEED = 7               # numeric 档随机种子（可复现）

_T_START = time.time()


def _log(msg):
    """带累计耗时的进度日志（对齐 E10）。"""
    print(f"[{time.time() - _T_START:6.1f}s] {msg}", flush=True)


# 初始几何与参考平均场：sampler 用 mf 的 mo_coeff 做活性空间 frozen-CCSD →
# 初始化 LUCJ 参数（只算一次，跨几何复用）；每帧实际的 h1/h2 由 scanner 内部给出。
mol0 = gto.M(atom=WATER, basis=BASIS, unit="Angstrom", verbose=0)
coords0 = mol0.atom_coords()          # Bohr 坐标：调用 scanner 一律用 Bohr
mf0 = scf.RHF(mol0).run()
n_elec, n_orb = ACTIVE_SPACE
e_full = float(mcscf.CASCI(mf0, n_orb, n_elec).kernel()[0])   # 全 CAS 参考（SQD 变分下界）


def _numeric_sampler(noise_p=0.0, seed=SEED):
    """numeric 档采样器：本地 StatevectorEngine 精确概率 + rng 抽样（seed 可复现）。"""
    return lucj_sampler(mf0, n_layers=LUCJ_LAYERS, shots=SHOTS,
                        noise_p=noise_p, seed=seed, runtime="numeric")


# ===========================================================================
# 第 1 节：采样冻结子空间——三行拿到 (能量, 力)，与全 CAS 对比 + 有限差分自检
# ===========================================================================
print("=" * 72)
print("第 1 节：SQD 采样冻结子空间基础（method='sqd', subspace='frozen'）")
print("=" * 72)

_log("构建 SQD 采样扫描器并计算首帧（首帧会跑一次 run_sqd_fermion 确定子空间）…")
scan = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE, unit="Bohr",
                  method="sqd", sampler=_numeric_sampler(), subspace="frozen")
e0, de0 = scan(coords0)
de0 = np.asarray(de0)

print(f"SQD 冻结子空间 E = {e0:.10f} Hartree")
print(f"全 CAS 参考    E = {e_full:.10f} Hartree，差 {e0 - e_full:+.3e}")
# SQD 是在采样子空间内对角化，是全 CAS 的**变分上界**（子空间越小能量越高）。
assert e0 > e_full - 1e-9, "SQD 能量不应低于全 CAS（变分上界被破坏）"
assert e0 - e_full < 0.05, f"SQD 采样偏差过大（{e0 - e_full:.3e} Ha），检查位序/子空间"
print(f"核梯度范数 |dE/dR| = {np.linalg.norm(de0):.6f} Hartree/Bohr")

# 冻结子空间能量面光滑 → 解析核梯度应与中心差分一致（subspace="frozen" 的意义）。
h = 1e-3  # Bohr
dz = np.array([[0, 0, h], [0, 0, 0], [0, 0, 0]])
e_p, _ = scan(coords0 + dz)
e_m, _ = scan(coords0 - dz)
fd_z = (e_p - e_m) / (2 * h)
print(f"O 原子 z 方向：解析梯度 {de0[0, 2]:+.6f}  vs  中心差分 {fd_z:+.6f}"
      f"   差 {abs(de0[0, 2] - fd_z):.2e}")
assert abs(de0[0, 2] - fd_z) < 1e-4, "冻结子空间解析梯度与有限差分不一致！"
print("结论：frozen 子空间冻结后，能量面光滑、解析梯度可用 → 可直接驱动 MD。")


# ===========================================================================
# 第 2 节：frozen vs refresh——随机子空间下力噪声（§4.4 D）
# ===========================================================================
# 核心物理：refresh 每帧重新采样，随机子空间下能量**不是几何的函数**——同一几何
# 多次求值会得到不同能量/力（力噪声 ~4.3e-3 Ha/Bohr）。因此解析梯度无定义，
# qc_scanner 默认拒绝 refresh/adaptive，需显式 allow_discontinuous=True 才放行。
print()
print("=" * 72)
print("第 2 节：frozen vs refresh 力噪声对比（§4.4 D）")
print("=" * 72)

# 2a：守卫——非冻结子空间默认被拒绝（能量面不光滑，解析梯度无定义）。
try:
    qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE, unit="Bohr",
               method="sqd", sampler=_numeric_sampler(), subspace="refresh")
    raise AssertionError("refresh 模式本应被 qc_scanner 拒绝！")
except ValueError as exc:
    print(f"[守卫] subspace='refresh' 被拒绝（符合预期）：\n       {str(exc)[:96]}…")

# 2b：机制演示——frozen 只采样一次即锁存，refresh 每个几何都重采样。
#     §4.4 D 的力噪声正源于此：refresh 每帧重新确定子空间，子空间随几何跳变，
#     E(R) 不再是几何的光滑函数。（本体系 LUCJ 高度集中于 HF、能量后选又滤掉
#     采样噪声，同一几何的数值散差被压得很小，但**每帧重采样**这一机制是确定的；
#     换更大活性空间 / 更强关联即显现 ~4.3e-3 Ha/Bohr 量级的力噪声。）
GEOMS = [coords0,
         coords0 + np.array([[0.0, 0.0, 0.02], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
         coords0 + np.array([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.0]])]


def _counting_sampler(base):
    """包一层计数，统计 sampler 被真实调用的次数（= run_sqd_fermion 触发次数）。"""
    calls = {"n": 0}

    def wrapped(h1, h2, norb, nelec):
        calls["n"] += 1
        return base(h1, h2, norb, nelec)

    wrapped.calls = calls
    return wrapped


_log("frozen：扫描 3 个几何，观察 sampler 触发次数…")
smp_f = _counting_sampler(_numeric_sampler())
scan_f = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE, unit="Bohr",
                    method="sqd", sampler=smp_f, subspace="frozen")
for g in GEOMS:
    scan_f(g)

_log("refresh：扫描同样 3 个几何（allow_discontinuous=True 才放行）…")
smp_r = _counting_sampler(_numeric_sampler())
scan_r = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE, unit="Bohr",
                    method="sqd", sampler=smp_r, subspace="refresh",
                    allow_discontinuous=True)
for g in GEOMS:
    scan_r(g)

print(f"扫描 {len(GEOMS)} 个几何，sampler（= run_sqd_fermion）被调用次数：")
print(f"  frozen  ：{smp_f.calls['n']} 次（首帧采样后锁存子空间，后续几何全部复用）")
print(f"  refresh ：{smp_r.calls['n']} 次（每个几何都重新采样确定子空间）")
assert smp_f.calls["n"] == 1, "frozen 子空间应只在首帧采样一次"
assert smp_r.calls["n"] == len(GEOMS), "refresh 应每个几何都重采样"
print("结论：refresh 每帧重采样子空间 → E(R) 非光滑、解析力无定义（§4.4 D）→ 只能 frozen 驱动 MD。")


# ===========================================================================
# 第 3 节：冻结 SQD 驱动 PySCF 原生 AIMD（NVE 系综）
# ===========================================================================
# qc_scanner.as_pyscf_scanner() 返回标准 pyscf.lib.GradScanner，pyscf.md.NVE 原样
# 接受（与 E2 的 UCCSD 完全相同的对接）。因为子空间冻结、能量面光滑，总能量守恒。
print()
print("=" * 72)
print("第 3 节：冻结 SQD 驱动 AIMD（NVE，与 E2 同款对接）")
print("=" * 72)

N_STEPS, DT, TEMPERATURE = 15, 20, 100.0   # 步数 / 时间步(a.u.) / 初温(K)

scan_md = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE, unit="Bohr",
                     method="sqd", sampler=_numeric_sampler(), subspace="frozen")
_log("AIMD 首帧（采样确定冻结子空间）…")
scan_md(coords0)                            # 首帧：构建 + 冻结子空间
gscanner = scan_md.as_pyscf_scanner()

nve = md.NVE(gscanner)
nve.dt = DT
nve.steps = N_STEPS
nve.verbose = 0
# pyscf.md 每帧会无条件把坐标/速度表 dump 到 integrator.stdout（不受 verbose
# 控制，见 pyscf/md/integrators.py 模块级 kernel 的 _write）；重定向到内存流即可静音，
# 能量经下面的 callback 收集后由本脚本自行汇总。
nve.stdout = io.StringIO()
veloc = MaxwellBoltzmannVelocity(gscanner.mol, T=TEMPERATURE)
history = {"epot": [], "ekin": []}
nve.callback = lambda envs: (history["epot"].append(nve.epot),
                             history["ekin"].append(nve.ekin))

_log(f"NVE 积分 {N_STEPS} 步 × {DT} a.u.（≈{DT * 0.0242:.2f} fs/步）…")
nve.kernel(veloc=veloc)

epot = np.asarray(history["epot"])
ekin = np.asarray(history["ekin"])
etot = epot + ekin
drift = float(etot.max() - etot.min())
print(f"势能范围   ：[{epot.min():.8f}, {epot.max():.8f}] Hartree")
print(f"总能量漂移 ：{drift:.3e} Hartree（max - min）")
print("冻结子空间能量面光滑 → Verlet 积分总能量漂移仅积分误差量级（对比 refresh 会无规则跳变）。")


# ===========================================================================
# 第 4 节：静电嵌入——MM 点电荷进入 SQD 哈密顿量 + MM 反作用力
# ===========================================================================
# 嵌入只需构造时多给 mm_charges=(位置, 电荷)；sampler 用同一嵌入 mf 初始化 LUCJ
# 参数，参考才与实际哈密顿量一致。mm_gradient 给出当前态对 MM 粒子的反作用力，
# 供 QM/MM MD 双向耦合（MM 粒子也受力）。全部复用 pyscf.qmmm，零手写梯度。
print()
print("=" * 72)
print("第 4 节：静电嵌入（mm_charges / mm_gradient）")
print("=" * 72)

# 嵌入体系的参考平均场（sampler 用它初始化 LUCJ 参数）。
mol_emb = gto.M(atom=WATER, basis=BASIS, unit="Angstrom", verbose=0)
from pyscf import qmmm  # noqa: E402  （局部导入，保持顶部守卫简洁）
mf_emb = qmmm.add_mm_charges(scf.RHF(mol_emb), MM_POS_BOHR, MM_CHARGES, unit="Bohr").run()

scan_emb = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE, unit="Bohr",
                      method="sqd", subspace="frozen",
                      sampler=lucj_sampler(mf_emb, n_layers=LUCJ_LAYERS, shots=SHOTS,
                                           seed=SEED, runtime="numeric"),
                      mm_charges=(MM_POS_BOHR, MM_CHARGES))
_log("嵌入 SQD 首帧 + mm_gradient…")
e_emb, de_emb = scan_emb(coords0)
g_mm = np.asarray(scan_emb.mm_gradient())

shift = float(e_emb) - float(e0)
print(f"裸体系   E = {e0:.10f} Ha")
print(f"嵌入体系 E = {e_emb:.10f} Ha，静电嵌入位移 {shift:+.6f} Ha")
print(f"QM 核梯度形状 {np.asarray(de_emb).shape}，MM 反作用力形状 {g_mm.shape}"
      f"（范数 {np.linalg.norm(g_mm):.6f} Ha/Bohr）")
assert abs(shift) > 1e-4, "MM 电荷未进入哈密顿量（嵌入位移应可测）"


# ===========================================================================
# 第 5 节：device 采样档穿透 + 真机模板
# ===========================================================================
# SQD 的「上设备」= 把补满 measure_z 的 LUCJ 电路经 devices.base.run 提交采样。
# 与真机走**同一条**提交入口，故 provider="simulator" 验证通过后，切真机只需改
# lucj_sampler 的 provider/device 两个字符串（对比 E10：UCCSD/HEA 改 solver_kwargs）。
print()
print("=" * 72)
print("第 5 节：device 采样档穿透（runtime='device'）+ 真机模板")
print("=" * 72)

_log("device 采样档（simulator::statevector，与真机同款提交链路）…")
scan_dev = qc_scanner(SPEC, basis=BASIS, active_space=ACTIVE_SPACE, unit="Bohr",
                      method="sqd", subspace="frozen",
                      sampler=lucj_sampler(mf0, n_layers=LUCJ_LAYERS, shots=SHOTS,
                                           runtime="device", provider="simulator",
                                           device="statevector"))
e_dev, de_dev = scan_dev(coords0)
print(f"numeric 档 E = {e0:.10f} Ha")
print(f"device  档 E = {e_dev:.10f} Ha，差 {float(e_dev) - float(e0):+.2e}")
# 因 LUCJ(1 层) 分布高度集中于 HF，device 与 numeric 冻到同一子空间，能量吻合；
# 容差给采样统计偏差 6e-3（足以抓住曾经的 3 Ha 量级位序回归）。
assert abs(float(e_dev) - float(e0)) < 6e-3, "device 采样档与 numeric 档不一致"
assert abs(float(e_dev) - e_full) < 0.05, "device 档能量偏离全 CAS 过多"
print(f"device 档核梯度范数 |dE/dR| = {np.linalg.norm(np.asarray(de_dev)):.6f} Ha/Bohr")

print()
print("真机模板（默认注释；资源到位后放开即用，只改 provider/device）：")
print("""
    import tyxonq as tq
    tq.set_token("YOUR_TOKEN", provider="tyxonq", device="homebrew_s2")

    scan_real = qc_scanner(
        SPEC, basis=BASIS, active_space=ACTIVE_SPACE, unit="Bohr",
        method="sqd", subspace="frozen",
        sampler=lucj_sampler(mf0, n_layers=LUCJ_LAYERS, shots=8192,
                             runtime="device",
                             provider="tyxonq",        # 或 qcos / quafu
                             device="homebrew_s2"))
    e, de = scan_real(coords0)
""")
print("提示：")
print("  - SQD 上真机的 shots 建议 ≥8192：子空间由计数确定，样本越多越稳；")
print("  - provider 取值同 tyxonq.devices.base.run：tyxonq / qcos / quafu / simulator；")
print("  - 真机采样有读出噪声时，frozen 子空间仍锁存首帧结果、能量面保持光滑，")
print("    但首帧子空间质量依赖计数保真度——建议先在 simulator 档验证再切真机；")
print("  - 带嵌入的 MD（ASE/i-PI/OpenMM/MDI）把本 scanner 换进对应适配层即可，")
print("    参见 E3（ASE）、E4（i-PI）、E5/E6（OpenMM）、E9（MDI）。")

print("\nE11 完成：SQD 已像 UCCSD/HEA 一样成为 qc_scanner 的一等 QM/MM 方法，")
print("覆盖采样冻结子空间、力噪声对照、AIMD、静电嵌入与 device/真机穿透。")
_log("总耗时见上行时间戳。")
