"""闭壳层密度组装、线性能量和近似梯度的聚焦验证。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pyscf import ao2mo, gto, mcscf, scf

from tyxonq.applications.chem.algorithms.ewf import EWF
from tyxonq.applications.chem.algorithms.ewf.density import (
    DensityResult,
    assemble_density,
    effective_rdm2,
)
from tyxonq.applications.chem.algorithms.ewf.gradient import nuclear_gradient


def _rhf(atom, unit="Angstrom"):
    """小体系参考态收紧 SCF 精度，避免有限差分放大求解误差。"""

    mol = gto.M(atom=atom, basis="sto-3g", unit=unit, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    return mf.run()


def _inputs(mf):
    emb = EWF(mf, solver="FCI")
    emb.kernel()
    return tuple((r.cluster, r.solver, f.c_proj) for r, f in zip(emb.result.fragments, emb.fragments))


@pytest.fixture(scope="module")
def lih_data():
    mf = _rhf("Li 0 0 0; H 0 0 1.6")
    return mf, _inputs(mf)


@pytest.mark.parametrize("assembly", ["rdm_t", "rdm_t_lambda"])
def test_density_energy_is_linear_and_has_correct_particle_number(lih_data, assembly):
    mf, inputs = lih_data
    density = assemble_density(mf, inputs, assembly=assembly)
    coeff = mf.mo_coeff
    nmo = coeff.shape[1]
    eri = ao2mo.kernel(mf.mol, coeff, compact=False).reshape((nmo,) * 4)
    dm2 = effective_rdm2(density.rdm1, density.dm2_corr, mf.mo_occ)
    contracted = (mf.energy_nuc() + np.einsum("pq,pq", coeff.T @ mf.get_hcore() @ coeff, density.rdm1)
                  + 0.5 * np.einsum("pqrs,pqrs", eri, dm2))
    np.testing.assert_allclose(density.energy, contracted, atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.trace(density.rdm1), mf.mol.nelectron, atol=1e-10, rtol=0)
    # 错误地加入 Δγ⊗Δγ 会改变能量；避免只检查同一公式的自洽性。
    quadratic = (density.dm2_corr + np.einsum("ij,kl->ijkl", density.rdm1, density.rdm1)
                 - np.einsum("ij,kl->iklj", density.rdm1, density.rdm1) / 2)
    assert abs(0.5 * np.einsum("pqrs,pqrs", eri, quadratic - dm2)) > 1e-8
    assert density.diagnostics["amplitudes_optimized"] is False
    assert density.diagnostics["bath_projector_response"] is False


def test_full_space_fci_density_gradient_matches_pyscf():
    """用真实 FCI 密度验证积分/响应引擎，不声称有效 T 重构等于 FCI。"""

    mf = _rhf("Li 0 0 0; H 0 0 1.6")
    nmo = mf.mo_coeff.shape[1]
    mc = mcscf.CASCI(mf, nmo, mf.mol.nelec).run()
    dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci, nmo, mf.mol.nelec)
    corr = dm2 - effective_rdm2(dm1, np.zeros_like(dm2), mf.mo_occ)
    density = DensityResult(float(mc.e_tot), dm1, corr, np.empty(0), np.empty(0), {})
    np.testing.assert_allclose(nuclear_gradient(mf, density), mc.nuc_grad_method().kernel(), atol=1e-8, rtol=0)


def test_no_virtual_orbitals_reduces_to_hf():
    mf = _rhf("He 0 0 0")
    density = assemble_density(mf, _inputs(mf))
    assert density.energy == pytest.approx(mf.e_tot, abs=1e-12)
    np.testing.assert_allclose(density.dm2_corr, 0, atol=1e-12)
    np.testing.assert_allclose(nuclear_gradient(mf, density), 0, atol=1e-12)


def test_unconverged_lambda_stops_before_returning_density(lih_data, monkeypatch):
    mf, inputs = lih_data
    from pyscf.cc.ccsd import CCSD

    def fail_lambda(self, t1, t2, **kwargs):
        self.converged_lambda = False
        return t1, t2

    monkeypatch.setattr(CCSD, "solve_lambda", fail_lambda)
    with pytest.raises(RuntimeError, match="Lambda equations did not converge"):
        assemble_density(mf, inputs)


def test_reject_invalid_particle_number_and_cluster_convergence(lih_data):
    mf, inputs = lih_data
    c, solution, p = inputs[0]
    with pytest.raises(ValueError, match="particle number"):
        assemble_density(mf, [(c, replace(solution, rdm1=solution.rdm1 * 0.5), p)])
    with pytest.raises(RuntimeError, match="cluster solver did not converge"):
        assemble_density(mf, [(c, replace(solution, converged=False), p)])


def test_multistep_fixed_density_and_rebuilt_energy_finite_difference(record_property):
    """区分固定密度的公式导数与完整 EWF 重建产生的近似误差。"""

    bond = 1.4
    mf = _rhf(f"H 0 0 0; H 0 0 {bond}", unit="Bohr")
    density = assemble_density(mf, _inputs(mf))
    analytic = nuclear_gradient(mf, density)[1, 2]
    frozen_fd, rebuilt_fd = [], []
    for step in (1e-3, 5e-4, 2.5e-4):
        frozen, rebuilt = [], []
        for sign in (-1, 1):
            displaced = _rhf(f"H 0 0 0; H 0 0 {bond + sign * step}", unit="Bohr")
            coeff = displaced.mo_coeff
            eri = ao2mo.kernel(displaced.mol, coeff, compact=False).reshape((2,) * 4)
            # H2 对称伸缩中 MO 奇偶性固定，无占据/虚轨道内部规范歧义。
            frozen.append(displaced.e_tot + np.einsum("pq,pq", coeff.T @ displaced.get_fock() @ coeff,
                          density.rdm1 - np.diag(displaced.mo_occ))
                          + 0.5 * np.einsum("pqrs,pqrs", eri, density.dm2_corr))
            rebuilt.append(assemble_density(displaced, _inputs(displaced)).energy)
        frozen_fd.append((frozen[1] - frozen[0]) / (2 * step))
        rebuilt_fd.append((rebuilt[1] - rebuilt[0]) / (2 * step))
    record_property("analytic_Hartree_per_Bohr", analytic)
    record_property("fixed_density_finite_differences", frozen_fd)
    record_property("rebuilt_energy_finite_differences", rebuilt_fd)
    record_property("rebuilt_gradient_error", rebuilt_fd[-1] - analytic)
    np.testing.assert_allclose(frozen_fd[-1], analytic, atol=1e-7, rtol=0)
    # 完整重建差分应随步长稳定；不把真实近似误差写成数值实现失败。
    assert abs(rebuilt_fd[-1] - rebuilt_fd[-2]) < 1e-6
    assert np.all(np.isfinite(rebuilt_fd))
