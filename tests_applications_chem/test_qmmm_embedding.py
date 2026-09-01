"""QM/MM 静电嵌入测试（用例 11，E8 阶段 A）。

依据 ``MD_INTEGRATION_PLAN.md`` §4 用例表 / §5 E8-A：

1. ``QCScanner.set_mm_charges`` 每步更新与新鲜构建一致（上游 ``qmmm_for_scf``
   重入路径，``pyscf/qmmm/itrf.py`` L99-101）；
2. ``QCScanner.mm_gradient``（上游 ``QMMMGrad.grad_hcore_mm`` + ``grad_nuc_mm``）
   与 MM 坐标的中心差分一致——这是 MM 反作用力的数值验证；
3. ``TyxonQCalculator(qm_indices=..., atom_charges=...)`` 区域划分模式：
   全原子力 = QM 子集梯度 ⊕ MM 反作用力，且与手搭 ``add_mm_charges`` 链路一致。

嵌入层验证全程用 ``method="uccsd"``（本项目的目标算法）：UCCSD 数值求解器
跨初始猜测有 ~1e-8 Ha 能量抖动，属正常量子数值误差，因此有限差分用较大步长
delta=0.05 Å（嵌入信号 ~5e-5 Ha，信噪比 > 1000）并按 5% 相对容差断言。
"""

from __future__ import annotations

import numpy as np
import pytest


def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


needs_pyscf = pytest.mark.skipif(not _has("pyscf"), reason="pyscf not installed.")
needs_ase = pytest.mark.skipif(
    not (_has("pyscf") and _has("ase")), reason="pyscf/ase not all installed."
)

BOHR_TO_ANGSTROM = 0.52917721092
HARTREE_TO_EV = 27.211386245988

# 水分子几何（Å）+ 两个 MM 点电荷（±0.5 e，在分子外侧约 3 Å）
H2O = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"
H2O_POS_ANG = np.array([(0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692)])
MM_POS_ANG = np.array([(3.0, 1.0, 0.5), (-3.0, -1.0, 0.0)])
MM_Q = np.array([0.5, -0.5])


def _make_scan(mm_pos_ang=None):
    from tyxonq.applications.chem.interfaces import qc_scanner

    mm = None if mm_pos_ang is None else (np.asarray(mm_pos_ang), MM_Q)
    return qc_scanner(
        H2O, basis="sto-3g", active_space=(4, 4), method="uccsd", mm_charges=mm
    )


@needs_pyscf
def test_set_mm_charges_matches_fresh_build():
    """每步更新 MM 环境后的结果 == 用新环境新鲜构建的结果。"""
    scan = _make_scan(MM_POS_ANG)
    pos_bohr = H2O_POS_ANG / BOHR_TO_ANGSTROM
    e0, de0 = scan(pos_bohr)

    # 把两个点电荷整体平移 0.3 Å
    mm_new = MM_POS_ANG + np.array([0.2, -0.1, 0.3])
    scan.set_mm_charges(mm_new)
    e1, de1 = scan(pos_bohr)

    ref = _make_scan(mm_new)
    e_ref, de_ref = ref(pos_bohr)

    assert e1 != pytest.approx(e0, abs=1e-8)  # 嵌入确实变了（不是静默忽略）
    assert e1 == pytest.approx(e_ref, abs=1e-8)
    # UCCSD 数值求解器跨初始猜测有 ~1e-8 Ha / ~1e-4 Ha·Bohr⁻¹ 级抖动，
    # 属正常量子数值误差；两条链路能量已一致到 1e-8，差异全来自求解器抖动。
    np.testing.assert_allclose(de1, de_ref, rtol=1e-3, atol=1e-5)


@needs_pyscf
def test_mm_gradient_matches_finite_difference():
    """mm_gradient（反作用力梯度）与 MM 坐标中心差分一致。"""
    scan = _make_scan(MM_POS_ANG)
    pos_bohr = H2O_POS_ANG / BOHR_TO_ANGSTROM
    scan(pos_bohr)
    g_mm = scan.mm_gradient()
    assert g_mm.shape == MM_POS_ANG.shape
    assert np.max(np.abs(g_mm)) > 1e-6  # 反作用力非零

    delta = 0.05  # Å：步长足够大使嵌入信号（~5e-5 Ha）远超求解器抖动（~1e-8 Ha）
    g_fd = np.zeros_like(g_mm)
    for i in range(len(MM_Q)):
        for x in range(3):
            mm_p = MM_POS_ANG.copy()
            mm_m = MM_POS_ANG.copy()
            mm_p[i, x] += delta
            mm_m[i, x] -= delta
            scan.set_mm_charges(mm_p)
            e_p, _ = scan(pos_bohr)
            scan.set_mm_charges(mm_m)
            e_m, _ = scan(pos_bohr)
            # dE/dR 中 R 以 Bohr 度量，坐标位移按 Å→Bohr 换算
            g_fd[i, x] = (e_p - e_m) / (2 * delta / BOHR_TO_ANGSTROM)
    # 5% 相对容差：覆盖 UCCSD 求解器抖动（~1% 噪声/信号）与差分截断误差（~0.5%）。
    np.testing.assert_allclose(g_mm, g_fd, rtol=5e-2, atol=1e-4)


@needs_ase
def test_region_partition_forces_assemble_qm_plus_mm():
    """区域划分模式：全原子力 = QM 梯度 ⊕ MM 反作用力，与手搭链路一致。"""
    from ase import Atoms
    from tyxonq.applications.chem.interfaces.ase_calculator import TyxonQCalculator

    # 全体系 = 水（QM，下标 0-2）+ 两个氢形占位原子作 MM 点电荷位点
    mm_sites = MM_POS_ANG
    atoms = Atoms(
        "OHHHH",
        positions=np.vstack([H2O_POS_ANG, mm_sites]),
    )
    atom_charges = np.array([0.0, 0.0, 0.0, MM_Q[0], MM_Q[1]])
    calc = TyxonQCalculator(
        active_space=(4, 4), method="uccsd",
        qm_indices=[0, 1, 2], atom_charges=atom_charges,
    )
    atoms.calc = calc
    e_ev = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # ---- 手搭参考：qc_scanner(水, mm_charges) ----
    ref = _make_scan(mm_sites)
    e_ref, de_qm_ref = ref(H2O_POS_ANG / BOHR_TO_ANGSTROM)
    de_mm_ref = ref.mm_gradient()

    assert e_ev == pytest.approx(e_ref * HARTREE_TO_EV, rel=1e-10)
    # QM 梯度与 MM 反作用力都走同一套上游代码，仅存在求解器抖动级差异。
    np.testing.assert_allclose(
        forces[:3], -np.asarray(de_qm_ref) * HARTREE_TO_EV / BOHR_TO_ANGSTROM,
        rtol=1e-3, atol=1e-5,
    )
    np.testing.assert_allclose(
        forces[3:], -np.asarray(de_mm_ref) * HARTREE_TO_EV / BOHR_TO_ANGSTROM,
        rtol=1e-3, atol=1e-5,
    )
    assert np.max(np.abs(forces[3:])) > 1e-6  # MM 位点受力非零

    # ---- 第二步：MM 位点移动后重算，验证每步更新生效 ----
    atoms.positions[3:] = mm_sites + np.array([0.0, 0.25, -0.15])
    e2 = atoms.get_potential_energy()
    ref2 = _make_scan(mm_sites + np.array([0.0, 0.25, -0.15]))
    e2_ref, _ = ref2(H2O_POS_ANG / BOHR_TO_ANGSTROM)
    assert e2 != pytest.approx(e_ev, abs=1e-8)
    # rel=1e-8 对应 ~1e-8 Ha，与 UCCSD 求解器抖动同量级；第一步已按同样量级验证过。
    assert e2 == pytest.approx(e2_ref * HARTREE_TO_EV, rel=1e-8)


@needs_ase
def test_region_partition_guards():
    """参数守卫：qm_indices 与 mm_charges 互斥；qm_indices 必须配 atom_charges。"""
    from tyxonq.applications.chem.interfaces.ase_calculator import TyxonQCalculator

    with pytest.raises(ValueError, match="do not pass static mm_charges"):
        TyxonQCalculator(
            active_space=(4, 4), method="uccsd",
            qm_indices=[0, 1, 2], atom_charges=[0.0] * 5,
            mm_charges=(MM_POS_ANG, MM_Q),
        )
    with pytest.raises(ValueError, match="requires atom_charges"):
        TyxonQCalculator(active_space=(4, 4), method="uccsd", qm_indices=[0, 1, 2])
