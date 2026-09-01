"""E2：UCCSD 扫描器直接驱动 PySCF 原生 AIMD（NVE 系综）。

本教程展示 `qc_scanner` 与 `pyscf.md` 的**零胶水**对接：
`qc_scanner` 的底层产物就是标准的 `pyscf.lib.GradScanner`，
而 `pyscf.md.NVE` 的第一个参数恰好接受任何 `lib.GradScanner`——
于是「量子化学势能面 → 玻恩-奥本海默分子动力学」只需要::

    scan(mol.atom_coords())                     # 首帧：构建平均场与梯度扫描器
    nve = pyscf.md.NVE(scan.as_pyscf_scanner()) # 交给 PySCF 积分器
    nve.kernel(veloc=veloc, steps=20)

每一步积分都会回调扫描器：新几何 → 重解平均场 → UCCSD 把 ansatz 参数
优化到该几何的能量极小 → 解析核梯度。因为每一步都是**完整的变分极小**，
能量面是几何的光滑函数，总能量守恒——这就是 VQE 族方法能直接跑 MD 的原因
（对比：任何带随机性的子空间/采样方案都会破坏这一性质）。

所需环境：tyxonq + pyscf>=2.10（不需要 ase / openmm / ipi / mdi）。
预期运行时间：约 30 秒（H2O / STO-3G / CAS(4,4) / 20 步 NVE，
UCCSD 约 0.02 s/几何）。
依赖缺失时本脚本优雅退出，不报错。
"""

import sys

import numpy as np

# ---- 依赖守卫：缺 pyscf 时优雅退出（CI 友好） ----
try:
    from pyscf import gto, md
    from pyscf.md.distributions import MaxwellBoltzmannVelocity
except ImportError:
    print("本教程需要 pyscf>=2.10：pip install 'pyscf>=2.10'")
    sys.exit(0)

from tyxonq.applications.chem.interfaces.scanner import qc_scanner

# ---------------------------------------------------------------------------
# 演示分子与模拟参数
# ---------------------------------------------------------------------------
WATER = "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692"
BASIS = "sto-3g"
ACTIVE_SPACE = (4, 4)   # CAS(4,4)：STO-3G 水分子的全价活性空间
METHOD = "uccsd"        # 也可以换 "hea"（见 E1 第 3 节，速度较慢）

N_STEPS = 20            # MD 步数
DT = 20                 # 时间步长，原子单位（1 a.u. ≈ 0.0242 fs，20 a.u. ≈ 0.48 fs）
TEMPERATURE = 100.0     # 初始速度对应温度（K）。取低温让分子停留在平衡构型附近，
                        # 小振幅振动下 ansatz 的变分极小最稳定。


# ===========================================================================
# 第 1 步：构建扫描器，并在初始几何上完成首帧计算
# ===========================================================================
print("=" * 72)
print(f"第 1 步：构建 {METHOD} 扫描器（首帧）")
print("=" * 72)

mol0 = gto.M(atom=WATER, basis=BASIS, unit="Angstrom", verbose=0)

scan = qc_scanner(WATER, basis=BASIS, active_space=ACTIVE_SPACE, method=METHOD)
e0, de0 = scan(mol0.atom_coords())  # 首帧：构建 (mol, mf, mc, scanner)
print(f"初始几何能量 E = {e0:.10f} Hartree")
print(f"初始受力范数 |dE/dR| = {np.linalg.norm(de0):.6f} Hartree/Bohr")

# ===========================================================================
# 第 2 步：取出底层 pyscf GradScanner，交给 NVE 积分器
# ===========================================================================
# as_pyscf_scanner() 返回的正是 mc.nuc_grad_method().as_scanner() 的产物，
# isinstance(., lib.GradScanner) 为真，pyscf.md 会原样接受，不做任何包装。
print()
print("=" * 72)
print("第 2 步：接入 pyscf.md.NVE")
print("=" * 72)

gscanner = scan.as_pyscf_scanner()

nve = md.NVE(gscanner)
nve.dt = DT
nve.steps = N_STEPS
nve.verbose = 3          # 让 PySCF 打印每步的 Epot/Ekin/Etot/T 表格

# 初始速度：按麦克斯韦-玻尔兹曼分布采样（原子单位）。
veloc = MaxwellBoltzmannVelocity(gscanner.mol, T=TEMPERATURE)

# 用 callback 逐步收集能量，画守恒曲线。
# 注意：Epot/Ekin 存在积分器属性上（不是回调局部变量里），
# 所以直接闭包捕获 nve 对象读取。
history = {"epot": [], "ekin": []}


def _record(envs):
    history["epot"].append(nve.epot)
    history["ekin"].append(nve.ekin)


nve.callback = _record

# ===========================================================================
# 第 3 步：跑 MD
# ===========================================================================
print()
print("=" * 72)
print(f"第 3 步：NVE 分子动力学（{N_STEPS} 步 × {DT} a.u. ≈ {DT * 0.0242:.2f} fs/步）")
print("=" * 72)

result = nve.kernel(veloc=veloc)

# ===========================================================================
# 第 4 步：检查能量守恒
# ===========================================================================
print()
print("=" * 72)
print("第 4 步：总能量守恒检查")
print("=" * 72)

epot = np.asarray(history["epot"])
ekin = np.asarray(history["ekin"])
etot = epot + ekin
drift = etot.max() - etot.min()

print(f"势能范围   ：[{epot.min():.8f}, {epot.max():.8f}] Hartree")
print(f"动能范围   ：[{ekin.min():.8f}, {ekin.max():.8f}] Hartree")
print(f"总能量漂移 ：{drift:.3e} Hartree（max - min）")
print(f"最终几何末帧能量：{result.epot:.8f} Hartree")
print()
print("势能面光滑（每步完整变分极小）时，Verlet 积分的总能量漂移")
print("应只有积分误差量级；若能量面不光滑，这里会出现无规则跳变。")

# ---------------------------------------------------------------------------
# 作图（可选）：装了 matplotlib 就画能量守恒曲线，否则跳过。
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")  # 无显示环境也能运行
    import matplotlib.pyplot as plt

    steps = np.arange(len(etot))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, epot, "o-", label="E_pot")
    ax.plot(steps, ekin, "s-", label="E_kin")
    ax.plot(steps, etot, "^-", label="E_tot")
    ax.set_xlabel("MD step")
    ax.set_ylabel("Energy / Hartree")
    ax.set_title(f"{METHOD.upper()} AIMD (NVE): energy drift = {drift:.2e} Ha")
    ax.legend()
    out = "md_energy_conservation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n能量守恒曲线已保存到 {out}")
except ImportError:
    print("\n（未安装 matplotlib，跳过作图。）")

# ===========================================================================
# 讨论：扩展到更大的体系 / 更长的轨迹
# ===========================================================================
# - 每一步都要重解平均场 + 变分优化，成本 ~ 0.02 s/几何（CAS(4,4)/STO-3G），
#   活性空间增大后按 UCCSD 标度增长；大体系请考虑更小的活性空间或更快基组。
# - 本教程只演示 NVE；pyscf.md 还有 NVT/NPT 积分器，用法完全相同
#   （把 md.NVE 换成 md.NVT 并传温度参数即可）。
# - 后续教程：E3（ASE 结构优化 + MD）、E4（i-PI driver）、E5/E6（OpenMM），
#   全部以本教程的 qc_scanner 为唯一势能面入口。
print("\nE2 完成。")
