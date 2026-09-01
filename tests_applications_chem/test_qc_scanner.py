"""qc_scanner 门面测试（用例 5-6）。

依据 ``MD_INTEGRATION_PLAN.md`` §4；数值基准见 ``MD_INTEGRATION_RESEARCH.md`` §4.1。
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from pyscf import gto, scf, mcscf

from tyxonq.applications.chem.interfaces import qc_scanner


def _has_pyscf():
    try:
        import pyscf  # noqa: F401
        return True
    except Exception:
        return False


H2O_ATOM = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
ALL_STRS = [sum(1 << o for o in occ) for occ in combinations(range(4), 2)]


def _reference(mol):
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, 4, (2, 2))
    e_tot = mc.kernel()[0]
    # 注意：不能用 as_scanner()(mol.atom_coords())——裸数组会按 mol.unit（Å）
    # 解读而 atom_coords() 返回 Bohr，几何被缩小 1.889 倍。直接 kernel() 无此问题。
    de = mc.nuc_grad_method().kernel()
    return e_tot, np.asarray(de)


@pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping qc_scanner tests.")
def test_uccsd_path_matches_stock_casci():
    """用例 5：VQE 族（UCCSD）路径回归。

    实测基准（§4.1）：max|dg| = 2.91e-06（残余误差为 ansatz 误差）。
    """
    scan = qc_scanner(H2O_ATOM, basis="sto-3g", active_space=(4, 4), method="uccsd")
    mol = gto.M(atom=H2O_ATOM, basis="sto-3g", verbose=0)
    e, de = scan(mol.atom_coords())
    e_ref, de_ref = _reference(mol)

    assert abs(e - e_ref) < 1e-4  # ansatz 能量误差量级
    assert np.abs(np.asarray(de) - de_ref).max() < 1e-4


@pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping qc_scanner tests.")
def test_cross_geometry_consistency():
    """用例 6a：同一 scanner 连续吃两个几何，均与 stock CASCI 一致。"""
    scan = qc_scanner(
        H2O_ATOM,
        basis="sto-3g",
        active_space=(4, 4),
        solver_kwargs={"ci_strs": (ALL_STRS, ALL_STRS)},
    )
    mol = gto.M(atom=H2O_ATOM, basis="sto-3g", verbose=0)
    coords0 = mol.atom_coords()
    for dz in (0.0, 0.05):
        coords = coords0.copy()
        coords[1, 2] += dz
        e, de = scan(coords)

        m2 = gto.M(
            atom=[(sym, c.tolist()) for sym, c in zip(("O", "H", "H"), coords)],
            basis="sto-3g",
            unit="Bohr",
            verbose=0,
        )
        e_ref, de_ref = _reference(m2)
        assert abs(e - e_ref) < 1e-7
        assert np.abs(np.asarray(de) - de_ref).max() < 1e-6


@pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping qc_scanner tests.")
def test_mm_charges_consistency():
    """用例 6b：mm_charges 与手工 qmmm.add_mm_charges 链路一致，且电荷真进哈密顿量。

    实测基准（§4.1）：与手工链路 dE = 2.96e-07；MM 电荷带来 ~1e-2 Ha 量级位移。
    """
    from pyscf import qmmm

    mol = gto.M(atom=H2O_ATOM, basis="sto-3g", verbose=0)
    mm_coords = np.array([[0.0, 3.0, 0.0]])
    mm_q = np.array([-1.0])

    # 手工链路
    mf = scf.RHF(mol)
    mf = qmmm.add_mm_charges(mf, mm_coords, mm_q)
    mf.run()
    mc = mcscf.CASCI(mf, 4, (2, 2))
    mc.fcisolver = _make_full_cas_solver()
    e_ref = mc.kernel()[0]
    de_ref = mc.nuc_grad_method().kernel()

    scan = qc_scanner(
        H2O_ATOM,
        basis="sto-3g",
        active_space=(4, 4),
        mm_charges=(mm_coords, mm_q),
        solver_kwargs={"ci_strs": (ALL_STRS, ALL_STRS)},
    )
    e, de = scan(mol.atom_coords())
    # 两条链各自跑 DIIS-SCF，收敛差异 ~1e-8，容差取 1e-7
    assert abs(e - e_ref) < 1e-7
    assert np.abs(np.asarray(de) - np.asarray(de_ref)).max() < 1e-5

    # MM 电荷必须真实影响结果（对照无 MM 的能量）
    scan_bare = qc_scanner(
        H2O_ATOM,
        basis="sto-3g",
        active_space=(4, 4),
        solver_kwargs={"ci_strs": (ALL_STRS, ALL_STRS)},
    )
    e_bare, _ = scan_bare(mol.atom_coords())
    assert abs(e - e_bare) > 1e-3


def _make_full_cas_solver():
    from tyxonq.applications.chem.algorithms.sqd import as_pyscf_solver

    return as_pyscf_solver(ci_strs=(ALL_STRS, ALL_STRS))
