"""Vayesta 1.0.1 RHF-EWF 最小迁移的聚焦测试。"""

from __future__ import annotations


import numpy as np
import pytest
from pyscf import cc, fci, gto, scf

from tyxonq.applications.chem.algorithms.ewf import (
    EWF,
    SolverResult,
    build_bath,
    build_cluster,
    ccsd_to_cisd,
    fci_to_cisd,
)


def _rhf(atom: str, basis: str = "sto-3g"):
    mol = gto.M(
        atom=atom,
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    return scf.RHF(mol).run()


def _ao_projector(coeff: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """返回 AO 协变表示的 metric projector，避免比较任意轨道列。"""

    return overlap @ coeff @ coeff.T @ overlap


def test_iao_basis_and_atomic_owner_partition_are_complete():
    mf = _rhf("H 0 0 0; H 0 0 0.74", basis="cc-pvdz")
    emb = EWF(mf, solver="FCI", bath_options={"bathtype": "dmet"})
    with emb.iao_fragmentation() as fragmentation:
        fragments = fragmentation.add_all_atomic_fragments()

    overlap = mf.get_ovlp()
    np.testing.assert_allclose(
        fragmentation.coeff.T @ overlap @ fragmentation.coeff,
        np.eye(mf.mo_coeff.shape[1]),
        atol=1e-8,
        rtol=0,
    )
    assert fragmentation.coeff.shape == mf.mo_coeff.shape
    assert fragmentation.owner_coeff.shape[1] == len(fragmentation.labels) == 2
    assert sorted(index for fragment in fragments for index in fragment.full_orbital_indices) == [0, 1]
    assert fragmentation.fragmentation_complete
    assert fragmentation.owner_projector_error < 1e-8


def test_h2_ccpvdz_mp2_bno_matches_vayesta_golden():
    mf = _rhf("H 0 0 0; H 0 0 0.74", basis="cc-pvdz")
    emb = EWF(mf, solver="FCI")
    with emb.iao_fragmentation() as fragmentation:
        fragment = fragmentation.add_atomic_fragment(0)

    bath = build_bath(
        fragment,
        bathtype="mp2",
        threshold=1e-4,
        dmet_threshold=1e-8,
        canonicalize=True,
        project_dmet_order=2,
        project_dmet_mode="squared-entropy",
    )
    golden_virtual = np.array(
        [
            1.839768550927812e-3,
            1.0584592451269908e-3,
            1.0584592451269891e-3,
            3.0506730887349255e-4,
            1.4440994525722668e-4,
            1.1191632227293694e-4,
            1.1191632227293684e-4,
            3.837943478692128e-5,
        ]
    )
    np.testing.assert_allclose(bath.dmet_occupations, [0.5], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        bath.mp2_vir_occupations,
        golden_virtual,
        atol=1e-8,
        rtol=0,
    )
    assert bath.c_mp2_occ.shape[1] == 0
    assert bath.c_mp2_vir.shape[1] == 7
    assert bath.n_mp2_occ_kept == 0
    assert bath.n_mp2_vir_kept == 7
    assert bath.diagnostics["n_mp2_vir_candidates"] == 8
    assert bath.diagnostics["assembled_orthonormality_error"] < 1e-8

    canonical_bath = build_bath(
        fragment,
        bathtype="mp2",
        threshold=1e-3,
        canonicalize=True,
    )
    noncanonical_bath = build_bath(
        fragment,
        bathtype="mp2",
        threshold=1e-3,
        canonicalize=False,
    )
    np.testing.assert_allclose(
        noncanonical_bath.mp2_vir_occupations,
        canonical_bath.mp2_vir_occupations,
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        _ao_projector(noncanonical_bath.c_mp2_vir, mf.get_ovlp()),
        _ao_projector(canonical_bath.c_mp2_vir, mf.get_ovlp()),
        atol=1e-8,
        rtol=0,
    )

    # Vayesta 的 global canonicalize=False 只关闭最终 cluster 旋转，
    # MP2-BNO 内部仍使用 canonical 轨道构造振幅与 occupation。
    noncanonical = EWF(
        mf,
        solver="FCI",
        bath_options={
            "bathtype": "mp2",
            "threshold": 1e-3,
            "canonicalize": False,
        },
    )
    np.testing.assert_allclose(
        noncanonical.kernel(),
        -1.1605792467895906,
        atol=1e-9,
        rtol=0,
    )
    for result in noncanonical.result.fragments:
        assert result.bath.c_mp2_vir.shape[1] == 3
        assert result.cluster.norb == 5


def test_lih_sto3g_dmet_and_mp2_match_vayesta_golden():
    """固定 LiH 的 bath occupation、分片能量和总能量参考值。"""

    mf = _rhf("Li 0 0 0; H 0 0 1.6")

    mp2_emb = EWF(mf, solver="FCI", bath_options={"bathtype": "mp2"})
    np.testing.assert_allclose(mp2_emb.kernel(), -7.882324378874305, atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        [result.e_corr_wf for result in mp2_emb.result.fragments],
        [-0.010457380333820166, -0.010002228731838168],
        atol=1e-9,
        rtol=0,
    )
    lithium_bath, hydrogen_bath = [result.bath for result in mp2_emb.result.fragments]
    np.testing.assert_allclose(lithium_bath.dmet_occupations, [0.49355796109092753], atol=1e-8, rtol=0)
    np.testing.assert_allclose(hydrogen_bath.dmet_occupations, [0.5064420389090738], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        hydrogen_bath.mp2_occ_occupations,
        [1.6055050067978153e-5],
        atol=1e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        hydrogen_bath.mp2_vir_occupations,
        [7.036797399028275e-4, 7.036797399028274e-4, 3.4155621900603994e-4],
        atol=1e-8,
        rtol=0,
    )

    dmet_emb = EWF(mf, solver="FCI", bath_options={"bathtype": "dmet"})
    np.testing.assert_allclose(dmet_emb.kernel(), -7.8807989921187, atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        [result.e_corr_wf for result in dmet_emb.result.fragments],
        [-0.010457380333842256, -0.008476841976210986],
        atol=1e-9,
        rtol=0,
    )

    pathological = EWF(
        mf,
        solver="FCI",
        bath_options={"bathtype": "dmet", "dmet_threshold": 0.499},
    )
    with pytest.raises(RuntimeError, match="density eigenvalues"):
        pathological.kernel()

    # 非平凡多分片 CCSD 同时覆盖 first-index WF 与 DMET cumulant 能量。
    ccsd_emb = EWF(mf, solver="CCSD", energy_functional="dmet")
    np.testing.assert_allclose(ccsd_emb.kernel(), -7.882313820333443, atol=1e-9, rtol=0)
    np.testing.assert_allclose(ccsd_emb.get_wf_energy(), -7.882313821664061, atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        [result.e_dmet for result in ccsd_emb.result.fragments],
        [-10.915417428289823, -1.5520482718228596],
        atol=1e-9,
        rtol=0,
    )


def test_full_fragment_fci_wf_and_dmet_recover_direct_fci():
    mf = _rhf("H 0 0 0; H 0 0 0.735")
    reference = float(fci.FCI(mf).kernel()[0])
    emb = EWF(mf, solver="FCI", bath_options={"bathtype": "full"})
    with emb.iao_fragmentation() as fragmentation:
        fragmentation.add_atomic_fragment([0, 1], name="H2")

    np.testing.assert_allclose(emb.kernel(), reference, atol=1e-9, rtol=0)
    np.testing.assert_allclose(emb.get_wf_energy(), reference, atol=1e-9, rtol=0)
    fragment_result = emb.result.fragments[0]
    c0, c1, c2 = fci_to_cisd(fragment_result.solver, fragment_result.cluster)
    assert c0 == 1.0
    assert c1.shape == (fragment_result.cluster.nocc, fragment_result.cluster.nvir)
    assert c2.shape == (
        fragment_result.cluster.nocc,
        fragment_result.cluster.nocc,
        fragment_result.cluster.nvir,
        fragment_result.cluster.nvir,
    )
    np.testing.assert_allclose(
        fragment_result.solver.reference_amplitude,
        fragment_result.solver.ci[0, 0],
        atol=0,
        rtol=0,
    )
    for part_cumulant in (False, True):
        for approx_cumulant in (False, True):
            np.testing.assert_allclose(
                emb.get_dmet_energy(
                    part_cumulant=part_cumulant,
                    approx_cumulant=approx_cumulant,
                ),
                reference,
                atol=1e-9,
                rtol=0,
            )

    # 同时覆盖把 DMET 直接选作 kernel() 默认能量泛函的入口。
    dmet_emb = EWF(
        mf,
        solver="FCI",
        bath_options={"bathtype": "full"},
        energy_functional="dmet",
    )
    with dmet_emb.iao_fragmentation() as fragmentation:
        fragmentation.add_atomic_fragment([0, 1], name="H2")
    np.testing.assert_allclose(dmet_emb.kernel(), reference, atol=1e-9, rtol=0)


def test_full_fragment_ccsd_wf_recovers_direct_ccsd():
    mf = _rhf("H 0 0 0; H 0 0 0.735", basis="cc-pvdz")
    reference_solver = cc.CCSD(mf)
    reference_solver.conv_tol = 1e-11
    reference = float(reference_solver.run().e_tot)

    emb = EWF(
        mf,
        solver="CCSD",
        bath_options={"bathtype": "full"},
        solver_options={"conv_tol": 1e-11},
    )
    with emb.iao_fragmentation() as fragmentation:
        fragmentation.add_atomic_fragment([0, 1], name="H2")
    np.testing.assert_allclose(emb.kernel(), reference, atol=1e-9, rtol=0)

    result = emb.result.fragments[0]
    cluster = result.cluster
    occupied = slice(0, cluster.nocc)
    active_potential = 2 * np.einsum(
        "iipq->pq", cluster.eri[occupied, occupied, :, :], optimize=True
    ) - np.einsum(
        "iqpi->pq", cluster.eri[occupied, :, :, occupied], optimize=True
    )
    np.testing.assert_allclose(
        cluster.h_eff + active_potential,
        cluster.fock,
        atol=1e-10,
        rtol=0,
    )
    overlap = mf.get_ovlp()
    np.testing.assert_allclose(
        cluster.c_total.T @ overlap @ cluster.c_total,
        np.eye(mf.mo_coeff.shape[1]),
        atol=1e-8,
        rtol=0,
    )
    assert cluster.nelec == (cluster.nocc, cluster.nocc)
    np.testing.assert_allclose(
        np.trace(result.solver.rdm1),
        sum(cluster.nelec),
        atol=1e-10,
        rtol=0,
    )
    assert result.solver.t1 is not None and result.solver.l1 is not None
    assert result.solver.t2 is not None and result.solver.l2 is not None
    assert result.solver.t1.shape == result.solver.l1.shape
    assert result.solver.t2.shape == result.solver.l2.shape


def test_ccsd_to_cisd_uses_intermediate_normalization():
    """直接固定 CCSD→CISD 的 disconnected singles 约定。"""

    t1 = np.array([[0.1, -0.2], [0.3, 0.4]])
    t2 = np.arange(16, dtype=float).reshape(2, 2, 2, 2) / 100
    result = SolverResult(
        method="CCSD",
        converged=True,
        energy=0.0,
        rdm1=np.zeros((4, 4)),
        rdm2=np.zeros((4, 4, 4, 4)),
        t1=t1,
        t2=t2,
    )
    c0, c1, c2 = ccsd_to_cisd(result)
    assert c0 == 1.0
    np.testing.assert_allclose(c1, t1, atol=0, rtol=0)
    np.testing.assert_allclose(
        c2,
        t2 + np.einsum("ia,jb->ijab", t1, t1),
        atol=0,
        rtol=0,
    )


def test_active_all_is_identity_and_active_subset_rebuilds_environment():
    mf = _rhf("H 0 0 0; H 0 0 0.735")

    baseline = EWF(mf, solver="FCI", bath_options={"bathtype": "dmet"})
    baseline_energy = baseline.kernel()

    all_active = EWF(mf, solver="FCI", bath_options={"bathtype": "dmet"})
    with all_active.iao_fragmentation() as fragmentation:
        for fragment in fragmentation.add_all_atomic_fragments():
            fragment.set_active_orbitals(list(range(fragment.n_full_orbitals)))
    np.testing.assert_allclose(all_active.kernel(), baseline_energy, atol=1e-10, rtol=0)
    assert all_active.result.fragmentation_complete
    overlap = mf.get_ovlp()
    for baseline_result, active_result in zip(
        baseline.result.fragments,
        all_active.result.fragments,
    ):
        np.testing.assert_allclose(
            active_result.bath.dmet_occupations,
            baseline_result.bath.dmet_occupations,
            atol=1e-8,
            rtol=0,
        )
        np.testing.assert_allclose(
            _ao_projector(active_result.cluster.c_active, overlap),
            _ao_projector(baseline_result.cluster.c_active, overlap),
            atol=1e-8,
            rtol=0,
        )
        np.testing.assert_allclose(
            active_result.e_corr_wf,
            baseline_result.e_corr_wf,
            atol=1e-10,
            rtol=0,
        )

    lih_mf = _rhf("Li 0 0 0; H 0 0 1.6")
    active = EWF(lih_mf, solver="FCI", bath_options={"bathtype": "dmet"})
    with active.iao_fragmentation() as fragmentation:
        lithium = fragmentation.add_atomic_fragment(0)
        fragmentation.add_atomic_fragment(1)
        lih_overlap = lih_mf.get_ovlp()
        unselected_coeff = lithium.c_full[:, [0, -1]]
        full_bath = build_bath(lithium, bathtype="dmet")
        full_cluster = build_cluster(lih_mf, full_bath)
        lithium.set_active_orbitals([1, 2, 3])
        environment_overlap = unselected_coeff.T @ lih_overlap @ lithium.c_env
        np.testing.assert_allclose(
            environment_overlap @ environment_overlap.T,
            np.eye(2),
            atol=1e-8,
            rtol=0,
        )

    with pytest.warns(RuntimeWarning, match="fragmentation is incomplete"):
        active.kernel()
    assert not active.result.fragmentation_complete
    assert active.result.n_unassigned_owner_orbitals == 2
    assert active.result.owner_projector_error > 1
    assert lithium.c_env.shape[1] == lih_mf.mo_coeff.shape[1] - 3
    assert lithium.cluster is not None
    assert np.linalg.norm(
        _ao_projector(full_cluster.c_active, lih_overlap)
        - _ao_projector(lithium.cluster.c_active, lih_overlap)
    ) > 1e-6


def test_active_selector_validation_and_locking():
    mf = _rhf("Li 0 0 0; H 0 0 1.6")
    emb = EWF(mf, solver="FCI", bath_options={"bathtype": "dmet"})
    with emb.iao_fragmentation() as fragmentation:
        lithium = fragmentation.add_atomic_fragment(0)
        fragmentation.add_atomic_fragment(1)
        with pytest.raises(ValueError, match="不能为空"):
            lithium.set_active_orbitals([])
        with pytest.raises(TypeError, match="布尔"):
            lithium.set_active_orbitals([True])
        with pytest.raises(ValueError, match="空字符串"):
            lithium.set_active_orbitals([""])
        with pytest.raises(TypeError, match="一维"):
            lithium.set_active_orbitals(np.array([[0, 1]]))
        with pytest.raises(TypeError, match="不能混合"):
            lithium.set_active_orbitals([0, "Li 2s"])
        with pytest.raises(ValueError, match="不能重复"):
            lithium.set_active_orbitals([0, 0])
        with pytest.raises(ValueError, match="越界"):
            lithium.set_active_orbitals([99])
        with pytest.raises(ValueError, match="没有匹配"):
            lithium.set_active_orbitals(["does-not-exist"])

        # 重叠标签按原始 IAO 顺序稳定去重，且不做能量重排。
        expected_indices = tuple(
            index
            for index, label in zip(lithium.full_orbital_indices, lithium.full_labels)
            if label[2] == "2p"
        )
        expected_labels = tuple(
            label for label in lithium.full_labels if label[2] == "2p"
        )
        lithium.set_active_orbitals(["Li 2p", "Li 2px"])
        assert lithium.active_orbital_indices == expected_indices
        assert lithium.active_labels == expected_labels

    with pytest.warns(RuntimeWarning, match="fragmentation is incomplete"):
        emb.kernel()
    with pytest.raises(RuntimeError, match="不能再修改"):
        lithium.set_active_orbitals([0])


def test_unknown_options_are_rejected():
    mf = _rhf("H 0 0 0; H 0 0 0.735")
    with pytest.raises(ValueError, match="solver must"):
        EWF(mf, solver="MP2")
    with pytest.raises(ValueError, match="energy_functional"):
        EWF(mf, energy_functional="unknown")
    with pytest.raises(ValueError, match="Unsupported bathtype"):
        EWF(mf, bath_options={"bathtype": "rpa"})
    with pytest.raises(TypeError, match="project_dmet_order"):
        EWF(mf, bath_options={"project_dmet_order": 1.5})
    with pytest.raises(TypeError, match="Unsupported bath option"):
        EWF(mf, bath_options={"screening": "rpa"})
    with pytest.raises(TypeError, match="Unsupported CCSD option"):
        EWF(mf, solver="CCSD", solver_options={"tailoring": True})
    with pytest.raises(TypeError, match="Unsupported FCI option"):
        EWF(mf, solver="FCI", solver_options={"init_guess": "CISD"})

    unconverged = scf.RHF(mf.mol)
    with pytest.raises(ValueError, match="must be converged"):
        EWF(unconverged)


def test_density_fitted_rhf_uses_consistent_cluster_eris():
    """DF Fock、MP2 bath 与 cluster ERI 必须使用同一 auxiliary basis。"""

    mf = _rhf("H 0 0 0; H 0 0 0.735")
    density_fitted = scf.RHF(mf.mol).density_fit().run()
    density_fitted_emb = EWF(density_fitted, solver="FCI")
    np.testing.assert_allclose(
        density_fitted_emb.kernel(),
        -1.1373326833980497,
        atol=1e-9,
        rtol=0,
    )
