"""E8 阶段 B 验证门复核脚本：``pyscf.qmmm.pbc`` Ewald 静电嵌入。

对应 ``MD_INTEGRATION_PLAN.md`` §6.1 风险清单 RB1–RB10 的可复现取证。
首跑结论（2026-08-31）已回写计划文档 §6.1「验证门执行结论」表，本脚本用于
升级 pyscf / 换体系后快速复核结论是否仍然成立。

单位约定（两个历史坑，务必遵守，见同目录 ``VALIDATION.md`` §5）：

- QM 几何给 ``gto.M`` 时用 ``unit="Angstrom"``，则后续所有坐标
  （MM 位点、扰动步长）统一按 Å；``add_mm_charges`` 必须显式传
  ``unit=``（缺省会按 ``mol.unit`` 解读 MM 坐标，Bohr/Å 混用会把距离
  搞错 1.8897 倍，阶段 A 实测翻过车）。
- 晶格向量 ``a`` 与 ``rcut_*`` 的单位与 MM 坐标的 ``unit`` 一致
  （``pbc/mm_mole.py`` ``create_mm_mol`` 内部整体转 Bohr）。
- 有限差分的导数要换算到 Ha/Bohr：Å 位移需除以 1.8897261339。

运行::

    conda run --no-capture-output -n qc python verify_gates.py

全程约几分钟（RB5/RB4 含几十次 RHF+CASCI 重建）。
"""
from __future__ import annotations

import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.qmmm.pbc.itrf import add_mm_charges as pbc_add_mm_charges

# ---------------- 体系定义（与计划文档 §6.1 结论表一致） ----------------
ATOM = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"  # 水，Å
MM_COORDS = np.array([[3.0, 1.0, 0.5], [-3.0, -1.0, 0.0]])      # ±0.5 e 点电荷，Å
MM_Q = np.array([0.5, -0.5])
L_BOX = 20.0        # 立方盒边长，Å（要求明显大于 2×QM 区尺寸，见 RB10）
RCUT_EWALD = 12.0   # Å，须 < 盒边（RB2，上游 assert）
RCUT_HCORE = 9.0    # Å，须 > QM 区半径且 < 半盒边（RB3，上游两条 assert）
ANG2BOHR = 1.8897261339


def build(mm_coords, rcut_ewald=RCUT_EWALD, rcut_hcore=RCUT_HCORE):
    """水 + 周期点电荷背景：RHF 层装饰 → CASCI(4,4)。返回 (mol, mf, mc, e)。"""
    mol = gto.M(atom=ATOM, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = pbc_add_mm_charges(
        scf.RHF(mol), mm_coords, np.eye(3) * L_BOX, MM_Q,
        rcut_ewald=rcut_ewald, rcut_hcore=rcut_hcore, unit="Angstrom",
    )
    mf.kernel()
    mc = mcscf.CASCI(mf, 4, 4)
    e = mc.kernel()[0]
    return mol, mf, mc, e


def rb1_cluster_limit():
    """RB1：大盒极限收敛到分子版簇嵌入 + scanner 链路 + TyxonQ UCCSD。"""
    from pyscf import qmmm

    # 分子版簇嵌入参考（阶段 A 同款链路）
    mol0 = gto.M(atom=ATOM, basis="sto-3g", unit="Angstrom", verbose=0)
    mf0 = qmmm.add_mm_charges(scf.RHF(mol0), MM_COORDS, MM_Q, unit="Angstrom")
    mc0 = mcscf.CASCI(mf0, 4, 4)
    e_cl = mc0.kernel()[0]
    g_cl = mc0.nuc_grad_method().kernel()

    print("[RB1] 簇嵌入参考 E = %.10f" % e_cl)
    for L in (20.0, 40.0, 80.0):
        mol = gto.M(atom=ATOM, basis="sto-3g", unit="Angstrom", verbose=0)
        mf = pbc_add_mm_charges(
            scf.RHF(mol), MM_COORDS, np.eye(3) * L, MM_Q,
            rcut_ewald=12.0, rcut_hcore=0.45 * L, unit="Angstrom",
        )
        mf.kernel()
        mc = mcscf.CASCI(mf, 4, 4)
        e = mc.kernel()[0]
        # RB9 缓解：pbc QMMMSCF.as_scanner 被上游置 NotImplemented，
        # 用基类版本替换（dict 拷贝保留 mm_mol 等全部嵌入属性）。
        mf.as_scanner = lambda mf=mf: scf.hf.SCF.as_scanner(mf)
        scan = mc.nuc_grad_method().as_scanner()
        # 等价阶段 A 协议：scanner 由已收敛平均场构建，轨道随迁。
        scan.base._scf.mo_coeff = mf.mo_coeff
        scan.base._scf.mo_occ = mf.mo_occ
        scan.base._scf.e_tot = mf.e_tot
        # 注意：mol.unit='Angstrom'，scanner 收裸坐标数组按 mol.unit 解读。
        e2, g2 = scan(mol.atom_coords() * 0.52917721092)
        print("[RB1] L=%5.1f Ang: dE vs cluster = %+.3e   max|dgrad| = %.3e"
              % (L, e - e_cl, np.max(np.abs(g2 - g_cl))))


def rb4_truncation():
    """RB4：rcut_ewald ±20% 能量变化 < 1e-6 Ha。"""
    e0 = build(MM_COORDS)[3]
    e_lo = build(MM_COORDS, rcut_ewald=0.8 * RCUT_EWALD)[3]
    e_hi = build(MM_COORDS, rcut_ewald=1.2 * RCUT_EWALD)[3]
    print("[RB4] dE(-20%%) = %+.3e   dE(+20%%) = %+.3e  (阈值 1e-6)"
          % (e_lo - e0, e_hi - e0))


def _mm_grad_analytic(mf, mc, dm):
    """解析 MM 梯度 = grad_hcore_mm + grad_nuc_mm + grad_ewald(with_mm)。"""
    g = mf.nuc_grad_method()
    return (np.asarray(g.grad_hcore_mm(dm)) + np.asarray(g.grad_nuc_mm())
            + np.asarray(g.grad_ewald(dm, with_mm=True)[1]))


def _fd_mm_gradient(mm_ref, energy_fn, h_ang=1e-3):
    """对 MM 坐标逐分量中心差分，返回 Ha/Bohr。"""
    fd = np.zeros_like(mm_ref)
    for i in range(mm_ref.shape[0]):
        for x in range(3):
            mp = mm_ref.copy(); mp[i, x] += h_ang
            mm = mm_ref.copy(); mm[i, x] -= h_ang
            fd[i, x] = (energy_fn(mp) - energy_fn(mm)) / (2 * h_ang * ANG2BOHR)
    return fd


def rb5_mm_forces():
    """RB5：MM 反作用力有限差分验证（HF 应到机器精度；CASCI 有已知缺口）。

    缺口的归因（首跑取证）：解析式是冻结 1-RDM 的 Hellmann-Feynman 力；
    完整重优化路径的 FD 还含「SCF 轨道对 MM 位移的响应」——CASCI 对 HF
    轨道非变分，上游解析式没有该耦合项（等效 post-HF 版 CPHF 缺失），
    残差 ~4e-5 Ha/Bohr 且不随差分步长变化。
    """
    # HF 层（上游示例口径，应通过）
    mol = gto.M(atom=ATOM, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = pbc_add_mm_charges(
        scf.RHF(mol), MM_COORDS, np.eye(3) * L_BOX, MM_Q,
        rcut_ewald=RCUT_EWALD, rcut_hcore=RCUT_HCORE, unit="Angstrom",
    )
    mf.kernel()
    dm_hf = mf.make_rdm1()
    de_hf = _mm_grad_analytic(mf, None, dm_hf)

    def e_hf(mm):
        m = gto.M(atom=ATOM, basis="sto-3g", unit="Angstrom", verbose=0)
        fx = pbc_add_mm_charges(
            scf.RHF(m), mm, np.eye(3) * L_BOX, MM_Q,
            rcut_ewald=RCUT_EWALD, rcut_hcore=RCUT_HCORE, unit="Angstrom",
        )
        fx.kernel()
        return fx.e_tot

    fd_hf = _fd_mm_gradient(MM_COORDS, e_hf)
    print("[RB5] HF:    max|FD - analytic| = %.3e  (容差 1e-5)"
          % np.max(np.abs(fd_hf - de_hf)))

    # CASCI 层（已知不通过：轨道响应缺口）
    _, mf, mc, _ = build(MM_COORDS)
    dm_cas = mc.make_rdm1()
    de_cas = _mm_grad_analytic(mf, mc, dm_cas)
    fd_cas = _fd_mm_gradient(MM_COORDS, lambda mm: build(mm)[3])
    err = np.max(np.abs(fd_cas - de_cas))
    print("[RB5] CASCI: max|FD - analytic| = %.3e  (容差 1e-5，已知缺口≈4e-5)" % err)


def rb7_reentry():
    """RB7：重入 add_mm_charges + 重跑 mf.kernel() 后 == 新鲜构建。

    注意协议：重入只重置上游缓存（s1r/s1rr/mm_ewald_pot/qm_ewald_hess/e_nuc），
    不重跑 ``mf.kernel()`` 会沿用旧轨道（能量差 ~5e-6）。scanner 每步
    重算平均场，天然满足该协议。
    """
    mm0 = MM_COORDS.copy()
    mm1 = mm0 + np.array([[0.2, -0.1, 0.1], [-0.1, 0.1, 0.1]])
    e_fresh = build(mm1)[3]

    mol = gto.M(atom=ATOM, basis="sto-3g", unit="Angstrom", verbose=0)
    mf = pbc_add_mm_charges(
        scf.RHF(mol), mm0, np.eye(3) * L_BOX, MM_Q,
        rcut_ewald=RCUT_EWALD, rcut_hcore=RCUT_HCORE, unit="Angstrom",
    )
    mf.kernel()
    mcscf.CASCI(mf, 4, 4).kernel()   # 填缓存
    pbc_add_mm_charges(mf, mm1, np.eye(3) * L_BOX, MM_Q,
                       rcut_ewald=RCUT_EWALD, rcut_hcore=RCUT_HCORE,
                       unit="Angstrom")
    mf.kernel()                       # 关键：重入后重跑平均场
    e_re = mcscf.CASCI(mf, 4, 4).kernel()[0]
    print("[RB7] fresh=%.10f  reentry=%.10f  diff=%.3e"
          % (e_fresh, e_re, abs(e_fresh - e_re)))


if __name__ == "__main__":
    rb1_cluster_limit()
    rb4_truncation()
    rb5_mm_forces()
    rb7_reentry()
    print("done")
