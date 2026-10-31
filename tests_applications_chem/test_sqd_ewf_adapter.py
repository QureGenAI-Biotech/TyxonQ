"""selected-CI 稀疏振幅到 EWF wavefunction 能量的回归测试。"""

from __future__ import annotations

from math import comb
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
from pyscf import gto, scf
from pyscf.fci import cistring

from tyxonq.applications.chem.algorithms.ewf import (
    EWF,
    SolverResult,
    fragment_wf_correlation_energy,
    intermediate_normalized_cisd,
)
from tyxonq.applications.chem.algorithms.ewf.types import Cluster
from tyxonq.applications.chem.algorithms.sqd import (
    SCIState,
    SCIResult,
    sci_result_to_solver_result,
    sci_state_to_cisd,
    solve_sci,
)


def _assert_spin_blocks_close(actual, expected, *, atol: float) -> None:
    """逐个比较 alpha/beta 或 aa/ab/bb 张量，避免不同形状被转成 object array。"""

    assert len(actual) == len(expected)
    for actual_block, expected_block in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_block, expected_block, atol=atol, rtol=0)


@pytest.mark.parametrize(
    ("atom", "charge", "spin", "name"),
    [
        ("H 0 0 0; H 0 0 0.8; H 0.1 0 1.7", 0, 1, "H3"),
        ("O 0 0 0; H 0 0 0.96; H 0.9 0 -0.24", 1, 1, "H2O+"),
    ],
)
def test_sparse_sci_adapter_matches_dense_fci_conversion_and_fragment_energy(
    atom: str,
    charge: int,
    spin: int,
    name: str,
):
    """H3/H2O+ 的符号、spin block 与 EWF 分片能量应和 dense 旧路径一致。"""

    mol = gto.M(atom=atom, basis="sto-3g", charge=charge, spin=spin, verbose=0)
    mf = scf.UHF(mol).run()
    emb = EWF(
        mf,
        solver="FCI",
        bath_options={"bathtype": "full"},
        solver_options={"conv_tol": 1e-14},
    )
    with emb.iao_fragmentation() as fragmentation:
        fragment_definition = fragmentation.add_atomic_fragment(
            list(range(mol.natm)), name=name
        )
    emb.kernel()
    fragment = emb.result.fragments[0]
    cluster = fragment.cluster

    full_strings = tuple(
        cistring.make_strings(range(cluster.norb), electron_count)
        for electron_count in cluster.nelec
    )
    sci_result = solve_sci(
        full_strings,
        cluster.h_eff,
        cluster.eri,
        cluster.norb,
        cluster.nelec,
        tol=1e-14,
        max_cycle=200,
    )
    sparse_result = sci_result_to_solver_result(sci_result, cluster)

    # 这里只在小体系测试中构造 dense 矩阵，作为现有 FCI 转换的数值基准。
    dense_result = SolverResult(
        method="FCI",
        converged=True,
        energy=sci_result.energy,
        rdm1=tuple(np.asarray(value) for value in sci_result.rdm1s),
        rdm2=tuple(np.asarray(value) for value in sci_result.rdm2s),
        ci=np.asarray(sci_result.sci_state.amplitudes),
    )
    _, dense_c1, dense_c2 = intermediate_normalized_cisd(dense_result, cluster)
    _, sparse_c1, sparse_c2 = intermediate_normalized_cisd(sparse_result, cluster)
    _assert_spin_blocks_close(sparse_c1, dense_c1, atol=1e-13)
    _assert_spin_blocks_close(sparse_c2, dense_c2, atol=1e-13)

    dense_energy = fragment_wf_correlation_energy(
        mf,
        cluster,
        dense_result,
        fragment_definition.c_proj,
    )
    sparse_energy = fragment_wf_correlation_energy(
        mf,
        cluster,
        sparse_result,
        fragment_definition.c_proj,
    )
    direct_energy = fragment_wf_correlation_energy(
        mf,
        cluster,
        fragment.solver,
        fragment_definition.c_proj,
    )
    np.testing.assert_allclose(sparse_energy, dense_energy, atol=1e-12, rtol=0)
    np.testing.assert_allclose(sparse_energy, direct_energy, atol=1e-10, rtol=0)


def _synthetic_uhf_cluster(
    norb: int,
    nelec: tuple[int, int],
    physical_norb: tuple[int, int] | None = None,
) -> Cluster:
    """只构造振幅提取所需的轨道形状，不分配四指标积分。"""

    physical_norb = physical_norb or (norb, norb)
    empty_rows = 0
    c_occ = tuple(np.empty((empty_rows, count)) for count in nelec)
    c_vir = tuple(
        np.empty((empty_rows, total - count))
        for total, count in zip(physical_norb, nelec, strict=True)
    )
    empty_pair = (np.empty((empty_rows, 0)), np.empty((empty_rows, 0)))
    return Cluster(
        c_active_occ=c_occ,
        c_active_vir=c_vir,
        c_frozen_occ=empty_pair,
        c_frozen_vir=empty_pair,
        fock=(np.empty((0, 0)), np.empty((0, 0))),
        h_eff=(np.empty((0, 0)), np.empty((0, 0))),
        eri=(np.empty(0), np.empty(0), np.empty(0)),
    )


def test_sparse_sci_adapter_handles_dummy_virtuals_and_rejects_dummy_occupation():
    """短 spin channel 的末尾 dummy virtual 必须保留为零，并禁止被占据。"""

    cluster = _synthetic_uhf_cluster(4, (2, 1), physical_norb=(4, 3))
    reference_a = (1 << 2) - 1
    reference_b = 1
    state = SCIState(
        amplitudes=np.ones((1, 1)),
        ci_strs_a=np.asarray([reference_a]),
        ci_strs_b=np.asarray([reference_b]),
        norb=4,
        nelec=(2, 1),
    )
    _, c1, c2 = sci_state_to_cisd(state, cluster)
    assert cluster.dummy_orbitals == ((), (3,))
    np.testing.assert_array_equal(c1[1][:, -1], 0.0)
    np.testing.assert_array_equal(c2[1][:, :, :, -1], 0.0)
    np.testing.assert_array_equal(c2[2][:, :, -1, :], 0.0)
    np.testing.assert_array_equal(c2[2][:, :, :, -1], 0.0)

    invalid_state = SCIState(
        amplitudes=np.ones((1, 1)),
        ci_strs_a=np.asarray([reference_a]),
        ci_strs_b=np.asarray([1 << 3]),
        norb=4,
        nelec=(2, 1),
    )
    with pytest.raises(ValueError, match="occupies a dummy orbital"):
        sci_state_to_cisd(invalid_state, cluster)


def test_sparse_sci_adapter_requires_nonzero_reference_amplitude():
    """参考组态缺失或其系数过小时必须硬失败，不能静默改变归一化参考。"""

    cluster = _synthetic_uhf_cluster(4, (2, 1))
    reference_a = (1 << 2) - 1
    reference_b = 1
    missing_reference = SCIState(
        amplitudes=np.ones((1, 1)),
        ci_strs_a=np.asarray([reference_a ^ 1 ^ (1 << 2)]),
        ci_strs_b=np.asarray([reference_b]),
        norb=4,
        nelec=(2, 1),
    )
    with pytest.raises(ValueError, match="does not contain the EWF reference"):
        sci_state_to_cisd(missing_reference, cluster)

    zero_reference = SCIState(
        amplitudes=np.asarray([[0.0], [1.0]]),
        ci_strs_a=np.asarray([reference_a, reference_a ^ 1 ^ (1 << 2)]),
        ci_strs_b=np.asarray([reference_b]),
        norb=4,
        nelec=(2, 1),
    )
    with pytest.raises(ValueError, match="reference amplitude is too small"):
        sci_state_to_cisd(zero_reference, cluster)


def test_sparse_sci_adapter_never_constructs_the_full_ci_shape():
    """大组合维数下只分配 C1/C2；完整 alpha×beta FCI shape 被守卫禁止。"""

    # 48 个空间轨道正好对应本任务最大 96 qubit 线路。
    norb = 48
    nelec = (24, 20)
    cluster = _synthetic_uhf_cluster(norb, nelec)
    state = SCIState(
        amplitudes=np.ones((1, 1)),
        ci_strs_a=np.asarray([(1 << nelec[0]) - 1]),
        ci_strs_b=np.asarray([(1 << nelec[1]) - 1]),
        norb=norb,
        nelec=nelec,
    )
    full_ci_shape = (comb(norb, nelec[0]), comb(norb, nelec[1]))
    real_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        normalized_shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if normalized_shape == full_ci_shape:
            raise AssertionError("完整 FCI 矩阵不应在稀疏适配器中分配。")
        return real_zeros(shape, *args, **kwargs)

    with (
        patch(
            "tyxonq.applications.chem.algorithms.sqd.ewf.np.zeros",
            side_effect=guarded_zeros,
        ),
        patch(
            "pyscf.fci.cistring.num_strings",
            side_effect=AssertionError("稀疏适配器不应计算完整 FCI shape。"),
        ),
    ):
        _, c1, c2 = sci_state_to_cisd(state, cluster)

    assert min(full_ci_shape) > 2_000_000_000
    assert c1[0].shape == (24, 24)
    assert c1[1].shape == (20, 28)
    assert c2[1].shape == (24, 20, 24, 28)


def test_rdm_only_adapter_needs_no_reference_and_preserves_convergence():
    """无 HF 行列式仍可传递密度；原 wf 入口继续拒绝，失败状态不被改写。"""
    cluster = _synthetic_uhf_cluster(4, (2, 1))
    state = SCIState(np.ones((1, 1)), np.array([6]), np.array([2]), 4, (2, 1))
    dm1s = state.rdm(1)
    dm2s = state.rdm(2)
    sci = SCIResult(
        energy=0.0, sci_state=state,
        orbital_occupancies=tuple(np.diag(x) for x in dm1s),
        rdm1s=dm1s, rdm2s=dm2s, converged=False,
    )
    adapted = sci_result_to_solver_result(sci, cluster, include_cisd=False)
    assert not adapted.converged
    assert adapted.reference_amplitude is None
    assert adapted.cisd_c1 is None and adapted.cisd_c2 is None
    _assert_spin_blocks_close(adapted.rdm1, dm1s, atol=0)
    with pytest.raises(ValueError, match="does not contain the EWF reference"):
        sci_result_to_solver_result(sci, cluster)
    with pytest.raises(TypeError, match="C1/C2"):
        intermediate_normalized_cisd(adapted, cluster)


def test_ewf_callback_full_space_rdm_only_and_lazy_wf():
    """同一闭壳层 FCI 基上，RDM-only 回调复现 DMET 能量且不调用 CISD。"""
    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    reference = EWF(mf, solver="FCI", bath_options={"bathtype": "full"})
    reference.kernel()
    calls = []

    def solve_fragment(fragment, cluster, reference_mf):
        calls.append(fragment.id)
        strings = tuple(cistring.make_strings(range(cluster.norb), n) for n in cluster.nelec)
        sci = solve_sci(strings, cluster.h_eff, cluster.eri, cluster.norb,
                        cluster.nelec, tol=1e-13, max_cycle=200)
        assert sci.converged
        return sci_result_to_solver_result(sci, cluster, include_cisd=False)

    emb = EWF(mf, bath_options={"bathtype": "full"},
              energy_functional="dmet", cluster_solver=solve_fragment)
    with patch("tyxonq.applications.chem.algorithms.ewf.driver.fragment_wf_correlation_energy",
               side_effect=AssertionError("RDM-only 不应计算 wf 能量")):
        emb.kernel()
    assert len(calls) == mol.natm
    assert emb.e_tot == pytest.approx(reference.get_dmet_energy(), abs=1e-9)
    assert all(x.e_corr_wf is None for x in emb.result.fragments)
    with pytest.raises(TypeError, match="C1/C2"):
        emb.get_wf_energy()

    # 原 FCI 路径在 dmet 后仍可按需计算 wf，保持公开 getter 的兼容。
    legacy = EWF(mf, solver="FCI", bath_options={"bathtype": "full"}, energy_functional="dmet")
    legacy.kernel()
    assert legacy.get_wf_energy() == pytest.approx(reference.e_tot, abs=1e-10)


def test_ewf_density_stops_before_assembly_on_failed_fragment():
    """未收敛片段不能进入密度优化流程。"""
    from tyxonq.applications.chem.algorithms.ewf.solver import solve_cluster

    mol = gto.M(atom="H 0 0 0; H 0 0 0.8", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()

    def failed(fragment, cluster, mf):
        return replace(solve_cluster(cluster, mf, method="FCI"), converged=False)

    emb = EWF(mf, cluster_solver=failed, energy_functional="density")
    with pytest.raises(RuntimeError, match="converged fragment"):
        emb.kernel()
    assert not emb.converged and emb.result is None
