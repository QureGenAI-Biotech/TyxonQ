"""分子 UHF/ROHF-EWF 的开壳层回归测试。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import cc, fci, gto, scf

from tyxonq.applications.chem.algorithms.ewf import (
    EWF,
    build_bath,
    build_cluster,
    solve_cluster,
)
from tyxonq.applications.chem.algorithms.ewf import bath as bath_module


@pytest.fixture(scope="module")
def h2o_cation_uhf():
    mol = gto.M(
        atom="O 0 0 0; H 0 0 0.96; H 0.9 0 -0.24",
        basis="sto-3g",
        charge=1,
        spin=1,
        verbose=0,
    )
    return scf.UHF(mol).run()


@pytest.mark.parametrize("method", ["FCI", "CCSD"])
def test_full_fragment_uhf_matches_direct_solver(h2o_cation_uhf, method):
    mf = h2o_cation_uhf
    if method == "FCI":
        reference = float(fci.FCI(mf).kernel()[0])
        solver_options = None
    else:
        direct = cc.UCCSD(mf)
        direct.conv_tol = 1e-9
        reference = float(direct.run().e_tot)
        solver_options = {"conv_tol": 1e-9}

    emb = EWF(
        mf,
        solver=method,
        bath_options={"bathtype": "full"},
        solver_options=solver_options,
    )
    with emb.iao_fragmentation() as fragmentation:
        fragmentation.add_atomic_fragment([0, 1, 2], name="H2O+")

    np.testing.assert_allclose(emb.kernel(), reference, atol=1e-7, rtol=0)
    np.testing.assert_allclose(emb.get_dmet_energy(), reference, atol=1e-7, rtol=0)
    result = emb.result.fragments[0]
    cluster = result.cluster
    assert cluster.is_unrestricted
    assert cluster.nelec == mf.mol.nelec == (5, 4)
    assert cluster.physical_norb == (7, 7)
    assert cluster.dummy_orbitals == ((), ())
    assert cluster.spin_orbital_overlap.shape == (cluster.norb, cluster.norb)
    np.testing.assert_allclose(
        [np.trace(value) for value in result.solver.rdm1],
        cluster.nelec,
        atol=1e-9,
        rtol=0,
    )


@pytest.mark.parametrize("bathtype", ["dmet", "mp2", "full"])
@pytest.mark.parametrize("method", ["FCI", "CCSD"])
@pytest.mark.parametrize("energy_functional", ["wf", "dmet"])
def test_uhf_all_bath_solver_energy_combinations(
    h2o_cation_uhf,
    bathtype,
    method,
    energy_functional,
):
    emb = EWF(
        h2o_cation_uhf,
        solver=method,
        bath_options={"bathtype": bathtype},
        solver_options={"conv_tol": 1e-8} if method == "CCSD" else None,
        energy_functional=energy_functional,
    )
    energy = emb.kernel()
    assert np.isfinite(energy)
    assert emb.converged
    for fragment in emb.result.fragments:
        assert isinstance(fragment.bath.c_dmet, tuple)
        assert isinstance(fragment.cluster.h_eff, tuple)
        assert len(fragment.cluster.eri) == 3
        assert isinstance(fragment.solver.rdm1, tuple)
        assert len(fragment.solver.rdm2) == 3


def test_unequal_spin_cluster_adds_and_freezes_dummy_virtual():
    mol = gto.M(
        atom="O 0 0 0; H 0 0 1.0",
        basis="6-31g",
        spin=1,
        verbose=0,
    )
    mf = scf.UHF(mol).run()
    emb = EWF(mf, solver="CCSD")
    with emb.iao_fragmentation() as fragmentation:
        oxygen = fragmentation.add_atomic_fragment(0)
    bath = build_bath(oxygen, bathtype="mp2", threshold=1e-3)
    cluster = build_cluster(mf, bath)
    assert cluster.physical_norb == (9, 8)
    assert cluster.norb == 9
    assert cluster.dummy_orbitals == ((), (8,))
    dummy_energy = 10.0 + max(np.max(np.diag(value)[:n]) for value, n in zip(cluster.fock, cluster.physical_norb))
    np.testing.assert_allclose(cluster.fock[1][8, 8], dummy_energy, atol=1e-10, rtol=0)
    _, gab, gbb = cluster.eri
    np.testing.assert_allclose(gab[:, :, 8, :], 0.0, atol=0, rtol=0)
    np.testing.assert_allclose(gab[:, :, :, 8], 0.0, atol=0, rtol=0)
    for axis in range(4):
        np.testing.assert_allclose(np.take(gbb, 8, axis=axis), 0.0, atol=0, rtol=0)

    result = solve_cluster(cluster, mf, method="CCSD", options={"conv_tol": 1e-8})
    assert result.converged
    np.testing.assert_allclose(
        [np.trace(value) for value in result.rdm1],
        cluster.nelec,
        atol=1e-9,
        rtol=0,
    )
    np.testing.assert_allclose(result.rdm1[1][8], 0.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.rdm1[1][:, 8], 0.0, atol=1e-12, rtol=0)


def test_uhf_mp2_validation_accepts_unequal_dmet_occupation_lengths(monkeypatch):
    """UHF 的 alpha/beta DMET bath 数目不等时也应逐 spin 验证。"""

    fragment = SimpleNamespace(
        c_frag=(np.zeros((2, 1)), np.zeros((2, 1))),
    )
    dmet = SimpleNamespace(
        occupations=(np.array([0.25, 0.75]), np.array([0.50])),
    )
    sentinel = object()

    def fake_build_uhf(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(bath_module, "_build_mp2_bno_uhf", fake_build_uhf)
    result = bath_module.build_mp2_bno(
        fragment,
        dmet,
        occtype="occupied",
        project_dmet_mode="squared-entropy",
    )
    assert result is sentinel


@pytest.mark.parametrize("method", ["FCI", "CCSD"])
def test_rohf_warns_and_matches_explicit_uhf_conversion(method):
    mol = gto.M(
        atom="O 0 0 0; H 0 0 1.0",
        basis="sto-3g",
        spin=1,
        verbose=0,
    )
    rohf = scf.ROHF(mol).run()
    explicit_uhf = rohf.to_uhf()
    solver_options = {"conv_tol": 1e-9} if method == "CCSD" else None
    with pytest.warns(RuntimeWarning, match="converted to UHF"):
        converted = EWF(
            rohf,
            solver=method,
            bath_options={"bathtype": "full"},
            solver_options=solver_options,
        )
    direct = EWF(
        explicit_uhf,
        solver=method,
        bath_options={"bathtype": "full"},
        solver_options=solver_options,
    )
    with converted.iao_fragmentation() as fragmentation:
        fragmentation.add_atomic_fragment([0, 1], name="OH")
    with direct.iao_fragmentation() as fragmentation:
        fragmentation.add_atomic_fragment([0, 1], name="OH")
    np.testing.assert_allclose(converted.kernel(), direct.kernel(), atol=1e-10, rtol=0)
