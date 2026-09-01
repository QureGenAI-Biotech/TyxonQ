"""用例 12：周期性固体 QM/MM（``pyscf.qmmm.pbc`` Ewald 嵌入）——阶段 B。

验证 ``qc_scanner`` 的 ``mm_lattice`` 分支（``MD_INTEGRATION_PLAN.md`` §6.1
验证门 RB1-RB10 的实现验收）。全程 ``method="uccsd"``（本项目目标算法）：
UCCSD 数值求解器跨初始猜测有 ~1e-8 Ha 能量抖动，属正常量子数值误差，
容差按测试规范放宽而非换经典求解器。

MM 反作用力（``mm_gradient``）的已知近似：上游解析式缺 post-HF 轨道响应项，
基准偏置 ~4.3e-5 Ha/Bohr（复核结论见 ``examples/qmmm/md_lammps_qmmm_pbc/
VALIDATION.md`` §4；2026-09-01 降级收回，仍交付）。因此有限差分断言用
绝对容差 2e-4 Ha/Bohr + 5% 相对（覆盖该偏置与求解器抖动），
步长取 0.1 Å 使嵌入信号（~2.5e-4 Ha）远高于抖动。
"""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import gto, qmmm

from tyxonq.applications.chem.interfaces import qc_scanner

try:
    import pyscf  # noqa: F401

    _HAS_PYSCF = True
except Exception:  # pragma: no cover
    _HAS_PYSCF = False

needs_pyscf = pytest.mark.skipif(not _HAS_PYSCF, reason="pyscf not installed.")

BOHR_TO_ANGSTROM = 0.52917721092

# 与阶段 A / 验证门同款的基准体系：水分子（Å）+ 两个 ±0.5 e 点电荷。
H2O = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
H2O_POS_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])
MM_POS_ANG = np.array([(3.0, 1.0, 0.5), (-3.0, -1.0, 0.0)])
MM_Q = np.array([0.5, -0.5])

L_20 = np.eye(3) * 20.0   # 验证门基准盒（Å）
L_40 = np.eye(3) * 40.0   # 大盒：周期镜像影响趋于零，用于簇/周期一致性
RCUT_EWALD = 12.0
RCUT_HCORE = 9.0

POS_BOHR = H2O_POS_ANG / BOHR_TO_ANGSTROM


def _make_pbc_scan(mm_pos_ang=MM_POS_ANG, lattice=L_20, rcut_hcore=RCUT_HCORE):
    return qc_scanner(
        H2O,
        basis="sto-3g",
        active_space=(4, 4),
        method="uccsd",
        mm_charges=(np.asarray(mm_pos_ang, dtype=float), MM_Q),
        mm_lattice=lattice,
        rcut_ewald=RCUT_EWALD,
        rcut_hcore=rcut_hcore,
    )


@needs_pyscf
def test_pbc_matches_cluster_embedding_large_box():
    """RB1：大盒极限下周期嵌入收敛到分子版簇嵌入（能量与 MM 梯度）。"""
    scan_pbc = _make_pbc_scan(lattice=L_40, rcut_hcore=17.0)
    e_pbc, _ = scan_pbc(POS_BOHR)
    g_pbc = scan_pbc.mm_gradient()

    scan_cl = qc_scanner(
        H2O,
        basis="sto-3g",
        active_space=(4, 4),
        method="uccsd",
        mm_charges=(MM_POS_ANG, MM_Q),
    )
    e_cl, _ = scan_cl(POS_BOHR)
    g_cl = scan_cl.mm_gradient()

    # 能量：UCCSD 跨初始猜测抖动 ~1e-8 Ha，容差 5e-8 已远大于抖动。
    assert e_pbc == pytest.approx(e_cl, abs=5e-8)
    # MM 梯度：抖动 ~1e-4 Ha/Bohr 量级，15% 相对 + 绝对兜底。
    np.testing.assert_allclose(g_pbc, g_cl, rtol=0.15, atol=5e-5)


@needs_pyscf
def test_pbc_set_mm_charges_matches_fresh_build():
    """RB7：pbc 重入（重置 5 个缓存 + 重跑平均场）后 == 新鲜构建。"""
    scan = _make_pbc_scan()
    e0, _ = scan(POS_BOHR)
    mm_new = MM_POS_ANG + np.array([0.2, -0.1, 0.3])
    scan.set_mm_charges(mm_new)  # 电荷沿用初始值
    e1, _ = scan(POS_BOHR)

    ref = _make_pbc_scan(mm_pos_ang=mm_new)
    e_ref, _ = ref(POS_BOHR)

    assert e1 != pytest.approx(e0, abs=1e-8)  # 嵌入确实变了（不是静默忽略）
    assert e1 == pytest.approx(e_ref, abs=5e-8)


def test_pbc_guards():
    """RB2/RB3/RB4/RB10：参数守卫报带源码位置的 ValueError。"""
    kw = dict(basis="sto-3g", active_space=(4, 4), method="uccsd")

    with pytest.raises(ValueError, match="requires mm_charges"):
        qc_scanner(H2O, mm_lattice=L_20, rcut_ewald=12.0, rcut_hcore=9.0, **kw)

    with pytest.raises(ValueError, match="rcut_ewald and rcut_hcore"):
        qc_scanner(
            H2O, mm_charges=(MM_POS_ANG, MM_Q), mm_lattice=L_20, **kw
        )

    with pytest.raises(ValueError, match="must be diagonal"):
        offdiag = L_20.copy()
        offdiag[0, 1] = 1.0
        qc_scanner(
            H2O, mm_charges=(MM_POS_ANG, MM_Q), mm_lattice=offdiag,
            rcut_ewald=12.0, rcut_hcore=9.0, **kw,
        )

    with pytest.raises(ValueError, match="rcut_ewald=25.0 must be < min box edge"):
        qc_scanner(
            H2O, mm_charges=(MM_POS_ANG, MM_Q), mm_lattice=L_20,
            rcut_ewald=25.0, rcut_hcore=9.0, **kw,
        )

    with pytest.raises(ValueError, match="rcut_hcore=12.0 must be < half"):
        qc_scanner(
            H2O, mm_charges=(MM_POS_ANG, MM_Q), mm_lattice=L_20,
            rcut_ewald=12.0, rcut_hcore=12.0, **kw,
        )


@needs_pyscf
def test_pbc_rcut_hcore_too_small_for_qm_region():
    """RB3/RB10：rcut_hcore 小于 QM 区半径时构建期报错（get_hcore L183）。"""
    scan = _make_pbc_scan(rcut_hcore=0.5)  # 水分子半径约 1.06 Å
    with pytest.raises(ValueError, match="must exceed the QM region"):
        scan(POS_BOHR)


@needs_pyscf
def test_pbc_mm_gradient_finite_difference():
    """RB5：mm_gradient（含 Ewald 项）与总能量的中心差分一致（含已知近似）。

    有限差分路径做完整重优化（含轨道响应），解析路径缺轨道响应项
    （基准偏置 ~4.3e-5 Ha/Bohr），故容差 = 绝对 2e-4 + 相对 5%。
    """
    scan = _make_pbc_scan()
    scan(POS_BOHR)
    g_analytic = scan.mm_gradient()
    assert g_analytic.shape == MM_POS_ANG.shape

    h_ang = 0.1  # 嵌入信号 ~2.5e-4 Ha，远高于 UCCSD 抖动
    fd = np.zeros_like(MM_POS_ANG)
    for i in range(len(MM_POS_ANG)):
        for x in range(3):
            mp = MM_POS_ANG.copy()
            mm = MM_POS_ANG.copy()
            mp[i, x] += h_ang
            mm[i, x] -= h_ang
            e_p = _make_pbc_scan(mm_pos_ang=mp)(POS_BOHR)[0]
            e_m = _make_pbc_scan(mm_pos_ang=mm)(POS_BOHR)[0]
            # Å 位移换算成 Bohr 后才是 Hartree/Bohr。
            fd[i, x] = (e_p - e_m) / (2 * h_ang / BOHR_TO_ANGSTROM)

    np.testing.assert_allclose(g_analytic, fd, rtol=5e-2, atol=2e-4)
