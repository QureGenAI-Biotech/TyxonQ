"""SQD → PySCF fcisolver 适配器测试（用例 1-4）。

依据 ``MD_INTEGRATION_PLAN.md`` §4；数值基准见 ``MD_INTEGRATION_RESEARCH.md`` §4.4。
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from pyscf import gto, scf, mcscf

from tyxonq.applications.chem.algorithms.sqd import (
    as_pyscf_solver,
    lucj_sampler,
    reverse_bitstring_halves,
)


def _has_pyscf():
    try:
        import pyscf  # noqa: F401
        return True
    except Exception:
        return False


H2O_ATOM = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
NCAS, NELECAS = 4, (2, 2)


def _all_cas_strings(ncas: int, n_elec: int) -> list[int]:
    """全 CAS 空间的行列式串（n 选 k 枚举）。"""
    return [sum(1 << o for o in occ) for occ in combinations(range(ncas), n_elec)]


def _build_mol():
    return gto.M(atom=H2O_ATOM, basis="sto-3g", verbose=0)


def _reference_gradients(mol):
    """stock CASCI 的能量与解析梯度。"""
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, NCAS, NELECAS)
    e_tot = mc.kernel()[0]
    _, de = mc.nuc_grad_method().as_scanner()(mol.atom_coords())
    return e_tot, np.asarray(de), mc


def _sqd_gradients(mol, ci_strs):
    """冻结子空间 SQD 的能量与解析梯度。"""
    mf = scf.RHF(mol).run()
    mc = mcscf.CASCI(mf, NCAS, NELECAS)
    mc.fcisolver = as_pyscf_solver(ci_strs=ci_strs)
    e_tot = mc.kernel()[0]
    _, de = mc.nuc_grad_method().as_scanner()(mol.atom_coords())
    return e_tot, np.asarray(de), mc


@pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping SQD solver tests.")
def test_sqd_full_cas_matches_stock_casci():
    """用例 1：冻结 = 全 CAS 时，SQD 与 stock CASCI 精确相等（无 ansatz 误差）。

    实测基准（§4.4 A）：dE = 1.563e-13、max|dGrad| = 1.892e-08。
    """
    mol = _build_mol()
    e_ref, de_ref, _ = _reference_gradients(mol)

    all_strs = _all_cas_strings(NCAS, 2)
    e_sqd, de_sqd, _ = _sqd_gradients(mol, (all_strs, all_strs))

    assert abs(e_sqd - e_ref) < 1e-9
    assert np.abs(de_sqd - de_ref).max() < 1e-6


@pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping SQD solver tests.")
def test_sqd_truncated_subspace_gradient_matches_finite_difference():
    """用例 2：冻结截断子空间内，解析梯度应与有限差分一致（Hellmann-Feynman）。

    实测基准（§4.4 B）：两个方向差 2.820e-07 / 5.767e-07。
    """
    mol = _build_mol()
    _, _, mc_ref = _reference_gradients(mol)

    # 全 CAS 的 CI 矩阵是 (nstr_a, nstr_b)；单重态下取对角幅值最大的 3 个串。
    all_strs = _all_cas_strings(NCAS, 2)
    ci_matrix = np.asarray(mc_ref.ci)
    diag_amps = np.abs(np.diagonal(ci_matrix))
    keep = np.argsort(diag_amps)[::-1][:3]
    trunc_strs = sorted(all_strs[i] for i in keep)
    assert len(trunc_strs) == 3

    coords = mol.atom_coords()
    scan_ref = _sqd_scanner_factory(trunc_strs)

    def energy_at(dz_atom: int, dz_axis: int, h_bohr: float) -> float:
        c = coords.copy()
        c[dz_atom, dz_axis] += h_bohr
        e, _ = scan_ref(c)
        return float(e)

    h = 1.89e-3  # ≈ 1e-3 Angstrom
    e0, de0 = scan_ref(coords)
    for atom_idx, axis in ((1, 2), (2, 1)):
        fd = (energy_at(atom_idx, axis, h) - energy_at(atom_idx, axis, -h)) / (2 * h)
        assert abs(de0[atom_idx, axis] - fd) < 1e-5


def _sqd_scanner_factory(trunc_strs):
    """构造冻结截断子空间的 scanner（独立于 interfaces 层，直接验证求解器）。"""
    from tyxonq.applications.chem.interfaces import qc_scanner

    return qc_scanner(
        H2O_ATOM,
        basis="sto-3g",
        active_space=(sum(NELECAS), NCAS),
        solver_kwargs={"ci_strs": (trunc_strs, trunc_strs)},
    )


@pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping SQD solver tests.")
def test_sqd_truncated_subspace_is_variational_upper_bound():
    """用例 3：截断子空间能量必须高于全 CAS（变分上界，符号正确）。

    实测基准（§4.4 B）：+3.037e-03 Hartree。
    """
    mol = _build_mol()
    _, _, mc_ref = _reference_gradients(mol)
    e_full = float(mc_ref.e_tot)

    all_strs = _all_cas_strings(NCAS, 2)
    diag_amps = np.abs(np.diagonal(np.asarray(mc_ref.ci)))
    keep = np.argsort(diag_amps)[::-1][:3]
    trunc_strs = sorted(all_strs[i] for i in keep)

    e_trunc, _, _ = _sqd_gradients(mol, (trunc_strs, trunc_strs))
    assert e_trunc > e_full
    assert e_trunc - e_full < 0.1  # 截断误差应在合理量级


@pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping SQD solver tests.")
def test_stochastic_subspace_rejected_by_scanner_guard():
    """用例 4：非冻结子空间模式必须被 qc_scanner 拒绝，除非显式放行。"""
    from tyxonq.applications.chem.interfaces import qc_scanner

    with pytest.raises(ValueError, match="non-smooth"):
        qc_scanner("H 0 0 0; H 0 0 1", active_space=(2, 2), subspace="refresh")

    # 显式放行时不应抛错（仅验证构造通过，不做随机采样）
    scan = qc_scanner(
        "H 0 0 0; H 0 0 1",
        active_space=(2, 2),
        subspace="adaptive",
        allow_discontinuous=True,
    )
    assert scan.subspace == "adaptive"


@pytest.mark.skipif(not _has_pyscf(), reason="PySCF not installed; skipping SQD solver tests.")
def test_lucj_sampler_bitstring_order_freezes_correct_hf_determinants():
    """用例 5：lucj_sampler 采样路径必须冻结到正确的低轨道 HF 行列式。

    回归防护（SQD_KNOWN_ISSUES.md）：TyxonQ/LUCJ raw order 位串 [alpha0..|beta0..]
    与 PySCF selected-CI 的 LSB 约定相反。修复前 lucj_sampler 直接返回 raw counts，
    经 bitstring_matrix_to_integers（MSB 优先）把 HF 串 0b0011=3 读成 0b1100=12，
    冻结到错误子空间，能量比全 CAS 高 ~3.02 Ha。修复后每个自旋半区先经
    reverse_bitstring_halves 反转（与 examples/h2o_sqd.py 一致）。本用例直连
    lucj_sampler→run_sqd_fermion 采样分支（用例 1-4 只走显式 ci_strs 冻结路径）。

    实测基准（noise_p=0, shots=2048, seed=7，确定性）：
      冻结串 alpha=beta=[3]；SQD 能量 - 全 CAS = +7.44e-03 Ha。
    """
    mol = _build_mol()
    mf = scf.RHF(mol).run()
    e_full = float(mcscf.CASCI(mf, NCAS, NELECAS).kernel()[0])

    sampler = lucj_sampler(mf, n_layers=1, shots=2048, noise_p=0.0, seed=7)

    # 采样器直接输出必须是 SQD/PySCF order（反转后的 HF 串），而非 raw order。
    counts = sampler(None, None, NCAS, NELECAS)
    top_bitstring = max(counts, key=counts.get)
    assert top_bitstring == reverse_bitstring_halves("11001100") == "00110011"

    solver = as_pyscf_solver(
        subspace="frozen",
        sampler=sampler,
        sampler_kwargs={"samples_per_batch": 8, "num_batches": 4, "max_iterations": 5, "seed": 7},
    )
    mc = mcscf.CASCI(mf, NCAS, NELECAS)
    mc.fcisolver = solver
    e_sqd = float(mc.kernel()[0])

    # 冻结子空间必须含正确的 HF 行列式 3（0b0011），而非位序读反的 12（0b1100）。
    strs_a, strs_b = solver.frozen_strings
    assert 3 in {int(s) for s in strs_a}
    assert 3 in {int(s) for s in strs_b}
    assert 12 not in {int(s) for s in strs_a}

    # 能量接近全 CAS（+7.4e-03，变分上界）；位序读反会高出 ~1.4-3.0 Ha。
    assert 0.0 < e_sqd - e_full < 0.05
