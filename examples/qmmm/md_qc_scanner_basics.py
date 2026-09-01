"""E1：qc_scanner 基础——三行拿到 (能量, 力)，UCCSD 与 HEA 两条 VQE 路径。

本教程展示把 TyxonQ 的量子求解器接入分子动力学生态的**最小门面**
``qc_scanner``（``tyxonq.applications.chem.interfaces.scanner``）。
它是后续所有适配层（ASE Calculator、i-PI driver、OpenMM、MDI）的唯一入口，
契约与 ``pyscf.grad`` 的 scanner 完全一致::

    scanner(geometry_bohr) -> (e_tot, de)   # Hartree / Hartree·Bohr^-1

内部链路（全部复用 PySCF 现成代码，零重写）::

    mc = mcscf.CASCI(mf, n_orb, n_elec)   # 活性空间
    mc.fcisolver = UCCSD / HEA 求解器      # TyxonQ 提供（as_pyscf_solver）
    scanner = mc.nuc_grad_method().as_scanner()   # 解析核梯度

可用方法（``method=``）：``"uccsd"``（闭壳层，本教程主线）、``"rouccsd"``
（开壳层）、``"hea"``（硬件高效 ansatz）。VQE 族方法每个几何都会把 ansatz
参数优化到能量极小，因此能量面是几何的光滑函数，解析梯度有明确定义，
可以直接用于结构优化与分子动力学。

所需环境：tyxonq + pyscf>=2.10（不需要 ase / openmm / ipi / mdi）。
预期运行时间：约 30 秒（H2O / STO-3G / CAS(4,4)）。
依赖缺失时本脚本优雅退出，不报错。
"""

import sys

import numpy as np

# ---- 依赖守卫：缺 pyscf 时优雅退出（CI 友好） ----
try:
    from pyscf import gto, mcscf, scf
except ImportError:
    print("本教程需要 pyscf>=2.10：pip install 'pyscf>=2.10'")
    sys.exit(0)

from tyxonq.applications.chem.interfaces.scanner import qc_scanner

# ---------------------------------------------------------------------------
# 演示分子：水分子，STO-3G 基组，活性空间 CAS(4, 4)（4 电子 4 轨道）。
# ---------------------------------------------------------------------------
WATER = "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692"
BASIS = "sto-3g"
ACTIVE_SPACE = (4, 4)  # (n_elec, n_orb)，与 UCC/HEA 的 active_space 约定一致

mol0 = gto.M(atom=WATER, basis=BASIS, unit="Angstrom", verbose=0)
coords0 = mol0.atom_coords()  # Bohr 坐标：调用 scanner 一律用 Bohr


# ===========================================================================
# 第 1 节：最小用法——三行拿到 (能量, 力)
# ===========================================================================
print("=" * 72)
print("第 1 节：最小用法（method='uccsd'）")
print("=" * 72)

# 三行：构建 scanner -> 调用 -> 得到原子单位下的能量与核梯度。
scan = qc_scanner(WATER, basis=BASIS, active_space=ACTIVE_SPACE, method="uccsd")
e, de = scan(coords0)

print(f"E = {e:.10f} Hartree")
print("dE/dR (Hartree/Bohr):")
print(de)

# 自检 1：与 stock CASCI（同一活性空间的精确对角化）对比。
# CAS(4,4) 很小，UCCSD ansatz 几乎能覆盖全空间，两者应高度一致。
mf = scf.RHF(mol0).run()
n_elec, n_orb = ACTIVE_SPACE
mc_ref = mcscf.CASCI(mf, n_orb, n_elec)  # CASCI 的参数顺序是 (n_orb, n_elec)
e_ref = mc_ref.kernel()[0]
print(f"\nstock CASCI 参考：E = {e_ref:.10f} Hartree，差 {e - e_ref:+.3e}")

# 自检 2：用中心差分复核解析梯度。能量面光滑时两者应一致到 ~1e-5。
h = 1e-3  # Bohr
dz = np.array([[0, 0, h], [0, 0, 0], [0, 0, 0]])
e_p, _ = scan(coords0 + dz)
e_m, _ = scan(coords0 - dz)
fd_z = (e_p - e_m) / (2 * h)
print(f"O 原子 z 方向：解析梯度 {de[0, 2]:+.6f}  vs  中心差分 {fd_z:+.6f}"
      f"   差 {abs(de[0, 2] - fd_z):.2e}")
assert abs(de[0, 2] - fd_z) < 1e-4, "解析梯度与有限差分不一致！"


# ===========================================================================
# 第 2 节：同一个 scanner 跨几何复用（结构优化 / MD 的前提）
# ===========================================================================
# scanner 持有构建配方与求解器对象，每次调用在新几何上重建分子与平均场，
# 因此可以像普通函数一样沿反应坐标连续调用——这正是 ASE/i-PI/OpenMM
# 适配层对它的用法。
print()
print("=" * 72)
print("第 2 节：同一个 scanner 跨几何扫描（拉伸 O-H 键）")
print("=" * 72)

for stretch in (0.0, 0.05, 0.10, 0.15):
    # 把两个 H 沿各自 O-H 方向外推（Å），重建 PySCF 原子规格
    geom = f"O 0.0 0.0 {0.1173}; H 0.0 {0.7572 + stretch} {-0.4692 - stretch}; H 0.0 {-0.7572 - stretch} {-0.4692 - stretch}"
    mol_s = gto.M(atom=geom, basis=BASIS, unit="Angstrom", verbose=0)
    e_s, de_s = scan(mol_s.atom_coords())
    print(f"拉伸 {stretch:+.2f} Å：E = {e_s:.8f} Ha，|力| = {np.linalg.norm(de_s):.6f} Ha/Bohr")


# ===========================================================================
# 第 3 节：换方法只改一个字符串——HEA 路径
# ===========================================================================
# HEA（硬件高效 ansatz）走参数化量子电路优化。注意两点：
#   1. runtime="numeric" 让电路在 statevector 上精确求值（确定性、无采样噪声）；
#   2. ansatz 表达能力有限时能量会高于 UCCSD/CASCI，这是变分上界性质，
#      但能量面依然光滑，梯度依然可用。
print()
print("=" * 72)
print("第 3 节：method='hea'（硬件高效 ansatz）")
print("=" * 72)

scan_hea = qc_scanner(
    WATER, basis=BASIS, active_space=ACTIVE_SPACE, method="hea",
    solver_kwargs={"runtime": "numeric", "n_layers": 1},
)
e_hea, de_hea = scan_hea(coords0)
print(f"HEA   E = {e_hea:.10f} Hartree")
print(f"UCCSD E = {e:.10f} Hartree")
print(f"CASCI E = {e_ref:.10f} Hartree（变分上界：E_hea >= E_uccsd >= E_casci 应成立）")
assert e_hea >= e - 1e-8, "变分上界被破坏！"

print()
print("方法选型建议：")
print("  - 闭壳层小体系、要精度：method='uccsd'（本例与 CASCI 同量级，约 0.02 s/几何）")
print("  - 开壳层：method='rouccsd'")
print("  - 研究自定义电路 ansatz：method='hea'")

print("\nE1 完成。下一步见 md_pyscf_native_aimd.py（E2）：把 UCCSD 扫描器直接喂给 PySCF 原生 AIMD。")
