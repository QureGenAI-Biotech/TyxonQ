"""针对开壳层 EWF+LUCJ+SQD 流程中已识别风险的补充测试。

覆盖:
1. _pad_uccsd_amplitudes 在 dummy 轨道下的维度正确性
2. MP2 bath 触发非对称 physical_norb 的端到端流程 (EWF→LUCJ→SQD)
3. ext-SQD 开壳层单激发扩展验证
"""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import fci, gto, scf
from pyscf.fci import cistring

from tyxonq.applications.chem.algorithms.ewf import (
    EWF,
    build_bath,
    build_cluster,
    solve_cluster,
)
from tyxonq.applications.chem.algorithms.ewf.solver import (
    _pad_uccsd_amplitudes,
    _pad_matrix,
)
from tyxonq.applications.chem.algorithms.ewf.types import Cluster
from tyxonq.applications.chem.algorithms.lucj import (
    LUCJ,
    initialize_lucj_parameters_from_ccsd,
)
from tyxonq.applications.chem.algorithms.sqd import (
    bitstring_matrix_to_integers,
    reverse_bitstring_halves,
    run_sqd_fermion,
    samples_to_arrays,
)
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine


# ---------------------------------------------------------------------------
# 辅助函数: 构造不对称 physical_norb 的 UHF cluster
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def oh_uhf_631g():
    """OH 自由基 UHF/6-31G — 用 mp2 bath 产生 physical_norb=(9,8) 的不对称 cluster。"""
    mol = gto.M(
        atom="O 0 0 0; H 0 0 1.0",
        basis="6-31g",
        spin=1,
        verbose=0,
    )
    return scf.UHF(mol).run()


@pytest.fixture(scope="module")
def asymmetric_cluster(oh_uhf_631g):
    """返回 physical_norb=(9,8), dummy_orbitals=((), (8,)) 的 cluster。"""
    mf = oh_uhf_631g
    emb = EWF(mf, solver="CCSD")
    with emb.iao_fragmentation() as fragmentation:
        oxygen = fragmentation.add_atomic_fragment(0)
    bath = build_bath(oxygen, bathtype="mp2", threshold=1e-3)
    cluster = build_cluster(mf, bath)
    assert cluster.physical_norb == (9, 8)
    assert cluster.norb == 9
    assert cluster.dummy_orbitals == ((), (8,))
    return cluster


# ---------------------------------------------------------------------------
# 测试 1: _pad_uccsd_amplitudes 维度与 dummy 零值验证
# ---------------------------------------------------------------------------

def test_pad_uccsd_amplitudes_dimensions_with_dummy(asymmetric_cluster, oh_uhf_631g):
    """验证 UCCSD 求解后振幅补齐到公共 norb 且 dummy 维度的值为零。"""
    cluster = asymmetric_cluster
    mf = oh_uhf_631g

    # 用 CCSD 求解
    result = solve_cluster(cluster, mf, method="CCSD", options={"conv_tol": 1e-8})
    assert result.converged
    assert result.t1 is not None
    assert result.t2 is not None

    # 验证 t1 形状
    t1a, t1b = result.t1
    nocca, noccb = cluster.nocc
    assert t1a.shape == (nocca, cluster.norb - nocca), (
        f"t1a shape {t1a.shape} != ({nocca}, {cluster.norb - nocca})"
    )
    assert t1b.shape == (noccb, cluster.norb - noccb), (
        f"t1b shape {t1b.shape} != ({noccb}, {cluster.norb - noccb})"
    )

    # 验证 t2 形状
    t2aa, t2ab, t2bb = result.t2
    assert t2aa.shape == (nocca, nocca, cluster.norb - nocca, cluster.norb - nocca)
    assert t2ab.shape == (nocca, noccb, cluster.norb - nocca, cluster.norb - noccb)
    assert t2bb.shape == (noccb, noccb, cluster.norb - noccb, cluster.norb - noccb)

    # 验证 dummy beta 轨道 (index 8) 上的振幅为零
    dummy_b = list(cluster.dummy_orbitals[1])  # (8,)
    assert dummy_b == [8]
    # t1b: dummy 是 virtual index = 8 - noccb (物理 virtual 末尾之后)
    # 在补齐后的 t1b 中，dummy virtual 对应的列应该全部为零
    # 由于 noccb=4, norb=9, nvirb=5, 物理 nvirb=4, dummy 占 index 4 (即第5列)
    dummy_vir_col_b = cluster.norb - noccb - 1  # 最后一列
    np.testing.assert_allclose(t1b[:, dummy_vir_col_b], 0.0, atol=1e-15, rtol=0,
                               err_msg="t1b dummy virtual column must be zero")
    np.testing.assert_allclose(t2ab[:, :, :, dummy_vir_col_b], 0.0, atol=1e-15, rtol=0,
                               err_msg="t2ab dummy beta virtual column must be zero")
    np.testing.assert_allclose(t2bb[:, :, :, dummy_vir_col_b], 0.0, atol=1e-15, rtol=0,
                               err_msg="t2bb dummy beta virtual column must be zero")
    np.testing.assert_allclose(t2bb[:, :, dummy_vir_col_b, :], 0.0, atol=1e-15, rtol=0,
                               err_msg="t2bb dummy beta virtual row must be zero")

    # 验证 l1/l2 同样补齐正确
    assert result.l1 is not None
    l1a, l1b = result.l1
    assert l1a.shape == t1a.shape
    assert l1b.shape == t1b.shape

    # 验证 RDM1 在 dummy 轨道上为零
    np.testing.assert_allclose(result.rdm1[1][8], 0.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.rdm1[1][:, 8], 0.0, atol=1e-12, rtol=0)


def test_pad_matrix_unit():
    """直接测试 _pad_matrix 在典型不对称场景下的行为。"""
    # 模拟 alpha 有 3 物理 virtual + 2 dummy virtual 的情况
    # 物理 t1a 形状: (nocc, 3), 补齐到 (nocc, 5)
    t1_physical = np.ones((2, 3))
    result = _pad_matrix(t1_physical, (2, 5))
    assert result.shape == (2, 5)
    np.testing.assert_array_equal(result[:, :3], t1_physical)
    np.testing.assert_array_equal(result[:, 3:], np.zeros((2, 2)))

    # t2 测试: 物理 (2,2,3,3) → (2,2,5,5)
    t2_physical = np.ones((2, 2, 3, 3))
    result = _pad_matrix(t2_physical, (2, 2, 5, 5))
    assert result.shape == (2, 2, 5, 5)
    np.testing.assert_array_equal(result[:, :, :3, :3], t2_physical)
    np.testing.assert_array_equal(result[:, :, 3:, :], np.zeros((2, 2, 2, 5)))
    np.testing.assert_array_equal(result[:, :, :, 3:], np.zeros((2, 2, 5, 2)))


def test_pad_uccsd_amplitudes_preserves_physical_block():
    """验证补齐不改变原始物理块的数值。"""
    # 模拟 UCCSD 返回的物理空间振幅
    physical_t1 = (np.full((2, 3), 1.5), np.full((3, 4), 2.5))
    physical_t2 = (
        np.full((2, 2, 3, 3), 0.1),
        np.full((2, 3, 3, 4), 0.2),
        np.full((3, 3, 4, 4), 0.3),
    )

    # 模拟 cluster: nocc=(2,3), norb=6
    # 物理: alpha (2+3=5), beta (3+4=7) — 但 norb=max=7...
    # 等等，physical_t2[2] 是 (3,3,4,4) 对应 bb: nocc=3, nvir=4, norb_b=7
    # 但是我们传入 norb=7，所以 nvira = 7-2 = 5, nvirb = 7-3 = 4
    # t1a: (2, 3) → (2, 5) 补 2 列
    # t1b: (3, 4) → (3, 4) 不需要补
    # t2aa: (2,2,3,3) → (2,2,5,5) 补行列
    # t2ab: (2,3,3,4) → (2,3,5,4) 补 alpha virtual 列
    # t2bb: (3,3,4,4) → (3,3,4,4) 不需要补

    class MockCluster:
        nocc = (2, 3)
        norb = 7

    t1_padded, t2_padded = _pad_uccsd_amplitudes(MockCluster, physical_t1, physical_t2)

    # t1a: physical 部分不变
    np.testing.assert_array_equal(t1_padded[0][:, :3], physical_t1[0])
    np.testing.assert_array_equal(t1_padded[1], physical_t1[1])

    # t2aa: physical 部分不变
    np.testing.assert_array_equal(t2_padded[0][:, :, :3, :3], physical_t2[0])
    # t2ab: physical 部分不变
    np.testing.assert_array_equal(t2_padded[1][:, :, :3, :4], physical_t2[1])
    # t2bb: 不变
    np.testing.assert_array_equal(t2_padded[2], physical_t2[2])


# ---------------------------------------------------------------------------
# 测试 2: MP2 bath 触发 dummy 的完整 EWF→LUCJ→SQD 流程
# ---------------------------------------------------------------------------

def test_mp2_bath_dummy_ewf_lucj_sqd_end_to_end(oh_uhf_631g):
    """OH/6-31G + mp2 bath → asymmetric cluster → LUCJ → SQD → compare FCI。"""
    mf = oh_uhf_631g

    emb = EWF(
        mf,
        solver="CCSD",
        bath_options={"bathtype": "mp2", "threshold": 1e-3},
        solver_options={"conv_tol": 1e-8},
    )
    with emb.iao_fragmentation() as fragmentation:
        fragmentation.add_atomic_fragment(0, name="O")  # 仅氧原子，触发不对称 physical_norb
    emb.kernel()

    fragment = emb.result.fragments[0]
    cluster = fragment.cluster
    solver = fragment.solver

    # 确认不对称场景已触发
    assert cluster.physical_norb[0] != cluster.physical_norb[1], (
        f"Expected unequal physical_norb, got {cluster.physical_norb}"
    )
    assert any(len(d) > 0 for d in cluster.dummy_orbitals), (
        f"Expected non-empty dummy orbitals, got {cluster.dummy_orbitals}"
    )
    assert cluster.norb == max(cluster.physical_norb)
    assert solver.converged

    # 验证补齐后的振幅形状
    t1a, t1b = solver.t1
    nocca, noccb = cluster.nocc
    assert t1a.shape == (nocca, cluster.norb - nocca)
    assert t1b.shape == (noccb, cluster.norb - noccb)

    # LUCJ 参数初始化（排除 dummy）
    params = initialize_lucj_parameters_from_ccsd(
        solver.t2,
        t1=solver.t1,
        n_spatial_orbitals=cluster.norb,
        n_layers=1,
        topology="linear",
        optimize=False,
        excluded_orbitals=cluster.dummy_orbitals,
    )

    # 验证参数形状
    assert params["orbital_rotations"].shape == (1, 2, cluster.norb, cluster.norb)
    assert params["diag_coulomb_mats"].shape == (1, 3, cluster.norb, cluster.norb)

    # 验证 dummy 轨道上的库仑耦合已清零
    # 对于 aa 矩阵 (mat=0): 仅 alpha dummy 的行/列需清零
    # 对于 ab 矩阵 (mat=1): alpha dummy 清行, beta dummy 清列
    # 对于 bb 矩阵 (mat=2): 仅 beta dummy 的行/列需清零
    dummy_a, dummy_b = cluster.dummy_orbitals
    diag = params["diag_coulomb_mats"]
    for orb in dummy_a:
        # aa: alpha-alpha, both indices are alpha spatial
        np.testing.assert_allclose(diag[0, 0, orb, :], 0.0, atol=1e-15, rtol=0)
        np.testing.assert_allclose(diag[0, 0, :, orb], 0.0, atol=1e-15, rtol=0)
        # ab: alpha rows (p index is alpha)
        np.testing.assert_allclose(diag[0, 1, orb, :], 0.0, atol=1e-15, rtol=0)
    for orb in dummy_b:
        # ab: beta cols (q index is beta)
        np.testing.assert_allclose(diag[0, 1, :, orb], 0.0, atol=1e-15, rtol=0)
        # bb: beta-beta, both indices are beta spatial
        np.testing.assert_allclose(diag[0, 2, orb, :], 0.0, atol=1e-15, rtol=0)
        np.testing.assert_allclose(diag[0, 2, :, orb], 0.0, atol=1e-15, rtol=0)

    # LUCJ 线路
    circuit = LUCJ(
        cluster.norb,
        cluster.nelec,
        1,
        "linear",
        excluded_orbitals=cluster.dummy_orbitals,
    ).get_circuit(params, with_final_orbital_rotation=True)

    # 状态向量采样
    probs = np.asarray(StatevectorEngine().probability(circuit), dtype=float).reshape(-1)
    sample_idx = int(np.argmax(probs))
    raw = "".join(
        "1" if (sample_idx >> (circuit.num_qubits - 1 - qb)) & 1 else "0"
        for qb in range(circuit.num_qubits)
    )
    sqd_sample = reverse_bitstring_halves(raw)

    # 验证 dominant 样本的粒子数和 dummy 占据
    assert sqd_sample[:cluster.norb].count("1") == cluster.nelec[0]
    assert sqd_sample[cluster.norb:].count("1") == cluster.nelec[1]
    for spin_idx, excluded in enumerate(cluster.dummy_orbitals):
        offset = spin_idx * cluster.norb
        for orb in excluded:
            assert sqd_sample[offset + cluster.norb - 1 - orb] == "0", (
                f"Dummy orbital {orb} (spin {spin_idx}) is occupied in dominant sample"
            )

    # SQD: 完整 determinant 池对角化，对比 direct_uhf FCI
    full_strings = tuple(
        [int(v) for v in cistring.make_strings(range(cluster.norb), ec)]
        for ec in cluster.nelec
    )

    result = run_sqd_fermion(
        cluster.h_eff,
        cluster.eri,
        {sqd_sample: 1},
        samples_per_batch=1,
        norb=cluster.norb,
        nelec=cluster.nelec,
        num_batches=1,
        max_iterations=1,
        symmetrize_spin=False,
        include_configurations=full_strings,
        excluded_orbitals=cluster.dummy_orbitals,
        seed=7,
    )

    # 基准: direct_uhf FCI
    direct_energy, _ = fci.direct_uhf.kernel(
        cluster.h_eff,
        cluster.eri,
        cluster.norb,
        cluster.nelec,
        tol=1e-12,
    )

    np.testing.assert_allclose(result.energy, direct_energy, atol=1e-8, rtol=0,
                               err_msg="SQD energy with dummy orbitals must match direct FCI")

    # RDM 迹检查
    assert result.rdm1s is not None
    for spin_idx, (dm1, ne) in enumerate(zip(result.rdm1s, cluster.nelec)):
        np.testing.assert_allclose(np.trace(dm1), ne, atol=1e-9, rtol=0,
                                   err_msg=f"RDM1 trace spin {spin_idx} incorrect")

    # Dummy 占据必须严格为零
    for spin_idx, excluded in enumerate(cluster.dummy_orbitals):
        if excluded:
            dm_block = result.rdm1s[spin_idx][np.ix_(list(excluded), list(excluded))]
            np.testing.assert_allclose(dm_block, 0.0, atol=1e-12, rtol=0,
                                       err_msg=f"Dummy occupation spin {spin_idx} not zero")

    # 自旋平方 — 预期接近 pure doublet，但因单原子 UHF fragment 的自旋
    # 污染 (IAO 分片不完整，见 RuntimeWarning)，允许 0.01 的容差。
    s2 = result.sci_state.spin_square(cluster.spin_orbital_overlap)
    expected_s2 = 0.75  # doublet
    np.testing.assert_allclose(s2, expected_s2, atol=1e-2, rtol=0,
                               err_msg=f"<S^2>={s2} deviates too far from expected {expected_s2}")


# ---------------------------------------------------------------------------
# 测试 3: 非对称 cluster + FCI solver 的 dummy 处理
# ---------------------------------------------------------------------------

def test_fci_solver_with_dummy_orbitals(asymmetric_cluster, oh_uhf_631g):
    """FCI 求解器在存在 dummy 轨道时应给出正确的能量和 RDM。"""
    cluster = asymmetric_cluster
    mf = oh_uhf_631g

    result = solve_cluster(cluster, mf, method="FCI")
    assert result.converged
    assert result.rdm1 is not None

    # RDM1 迹
    traces = tuple(np.trace(np.asarray(dm)) for dm in result.rdm1)
    np.testing.assert_allclose(traces, cluster.nelec, atol=1e-9, rtol=0)

    # dummy 轨道占据为零
    np.testing.assert_allclose(np.asarray(result.rdm1[1])[8], 0.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.asarray(result.rdm1[1])[:, 8], 0.0, atol=1e-12, rtol=0)

    # 与 direct_uhf 对比
    direct_energy, _ = fci.direct_uhf.kernel(
        cluster.h_eff,
        cluster.eri,
        cluster.norb,
        cluster.nelec,
        tol=1e-12,
    )
    np.testing.assert_allclose(result.energy, direct_energy, atol=1e-10, rtol=0)


# ---------------------------------------------------------------------------
# 测试 4: ext-SQD 开壳层验证
# ---------------------------------------------------------------------------

def test_ext_sqd_open_shell_energy_with_dummy(oh_uhf_631g):
    """ext-SQD 在开壳层 + dummy 的场景下，不破坏粒子数并保持 dummy 为零占据。"""
    mf = oh_uhf_631g

    emb = EWF(
        mf,
        solver="CCSD",
        bath_options={"bathtype": "mp2", "threshold": 1e-3},
        solver_options={"conv_tol": 1e-8},
    )
    with emb.iao_fragmentation() as fragmentation:
        fragmentation.add_atomic_fragment(0, name="O")
    emb.kernel()

    fragment = emb.result.fragments[0]
    cluster = fragment.cluster

    # 生成 LUCJ 样本
    params = initialize_lucj_parameters_from_ccsd(
        fragment.solver.t2,
        t1=fragment.solver.t1,
        n_spatial_orbitals=cluster.norb,
        n_layers=1,
        topology="linear",
        optimize=False,
        excluded_orbitals=cluster.dummy_orbitals,
    )
    circuit = LUCJ(
        cluster.norb,
        cluster.nelec,
        1,
        "linear",
        excluded_orbitals=cluster.dummy_orbitals,
    ).get_circuit(params, with_final_orbital_rotation=True)

    probs = np.asarray(StatevectorEngine().probability(circuit), dtype=float).reshape(-1)

    # 取所有非零概率样本作为 counts
    counts = {}
    for idx in np.flatnonzero(probs > 1e-14):
        raw = "".join(
            "1" if (idx >> (circuit.num_qubits - 1 - qb)) & 1 else "0"
            for qb in range(circuit.num_qubits)
        )
        counts[reverse_bitstring_halves(raw)] = float(probs[idx])

    # FCI 基准
    full_strings = tuple(
        [int(v) for v in cistring.make_strings(range(cluster.norb), ec)]
        for ec in cluster.nelec
    )

    # 运行普通 SQD
    result_sqd = run_sqd_fermion(
        cluster.h_eff,
        cluster.eri,
        counts,
        samples_per_batch=300,
        norb=cluster.norb,
        nelec=cluster.nelec,
        num_batches=1,
        max_iterations=5,
        symmetrize_spin=False,
        max_dim=20,
        include_configurations=full_strings,
        excluded_orbitals=cluster.dummy_orbitals,
        seed=42,
        ext_sqd=False,
    )

    # 运行 ext-SQD
    result_ext = run_sqd_fermion(
        cluster.h_eff,
        cluster.eri,
        counts,
        samples_per_batch=300,
        norb=cluster.norb,
        nelec=cluster.nelec,
        num_batches=1,
        max_iterations=5,
        symmetrize_spin=False,
        max_dim=20,
        include_configurations=full_strings,
        excluded_orbitals=cluster.dummy_orbitals,
        seed=42,
        ext_sqd=True,
    )

    # ext-SQD 能量应该不差于普通 SQD
    assert result_ext.energy <= result_sqd.energy + 1e-10, (
        f"ext-SQD energy {result_ext.energy} should not be worse than SQD {result_sqd.energy}"
    )

    # RDM 迹检查
    for spin_idx, (dm1, ne) in enumerate(
        zip(result_ext.rdm1s, cluster.nelec, strict=True)
    ):
        np.testing.assert_allclose(np.trace(dm1), ne, atol=1e-9, rtol=0,
                                   err_msg=f"ext-SQD RDM1 trace spin {spin_idx} incorrect")

    # ext-SQD 不应占据 dummy 轨道
    for spin_idx, excluded in enumerate(cluster.dummy_orbitals):
        if excluded:
            dm_block = result_ext.rdm1s[spin_idx][np.ix_(list(excluded), list(excluded))]
            np.testing.assert_allclose(dm_block, 0.0, atol=1e-12, rtol=0,
                                       err_msg=f"ext-SQD dummy occupation spin {spin_idx} not zero")

    # ext-SQD 扩展后的 CI 空间应至少包含原始空间
    assert len(result_ext.sci_state.ci_strs_a) >= len(result_sqd.sci_state.ci_strs_a), (
        "ext-SQD should have at least as many alpha strings as SQD"
    )


# ---------------------------------------------------------------------------
# 测试 5: ROHF→UHF → MP2 bath → SQD 完整链路
# ---------------------------------------------------------------------------

def test_rohf_to_uhf_mp2_bath_sqd_end_to_end():
    """ROHF→UHF 转换后，经过 mp2 bath → SQD 仍应与 FCI 一致。"""
    mol = gto.M(
        atom="O 0 0 0; H 0 0 1.0",
        basis="sto-3g",
        spin=1,
        verbose=0,
    )
    mf_rohf = scf.ROHF(mol).run()

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        emb = EWF(
            mf_rohf,
            solver="CCSD",
            bath_options={"bathtype": "mp2", "threshold": 1e-3},
            solver_options={"conv_tol": 1e-8},
        )
        with emb.iao_fragmentation() as fragmentation:
            fragmentation.add_atomic_fragment([0, 1], name="OH")
        emb.kernel()

    # 确认 ROHF 转换警告已发出
    rohf_warnings = [x for x in w if "ROHF" in str(x.message)]
    assert len(rohf_warnings) >= 1, "ROHF→UHF conversion warning must be emitted"

    fragment = emb.result.fragments[0]
    cluster = fragment.cluster
    solver = fragment.solver
    assert cluster.is_unrestricted
    assert solver.converged

    # LUCJ + SQD
    params = initialize_lucj_parameters_from_ccsd(
        solver.t2,
        t1=solver.t1,
        n_spatial_orbitals=cluster.norb,
        n_layers=1,
        topology="linear",
        optimize=False,
        excluded_orbitals=cluster.dummy_orbitals,
    )
    circuit = LUCJ(
        cluster.norb,
        cluster.nelec,
        1,
        "linear",
        excluded_orbitals=cluster.dummy_orbitals,
    ).get_circuit(params, with_final_orbital_rotation=True)

    probs = np.asarray(StatevectorEngine().probability(circuit), dtype=float).reshape(-1)
    idx = int(np.argmax(probs))
    raw = "".join(
        "1" if (idx >> (circuit.num_qubits - 1 - qb)) & 1 else "0"
        for qb in range(circuit.num_qubits)
    )
    sqd_sample = reverse_bitstring_halves(raw)

    full_strings = tuple(
        [int(v) for v in cistring.make_strings(range(cluster.norb), ec)]
        for ec in cluster.nelec
    )
    result = run_sqd_fermion(
        cluster.h_eff,
        cluster.eri,
        {sqd_sample: 1},
        samples_per_batch=1,
        norb=cluster.norb,
        nelec=cluster.nelec,
        num_batches=1,
        max_iterations=1,
        symmetrize_spin=False,
        include_configurations=full_strings,
        excluded_orbitals=cluster.dummy_orbitals,
        seed=7,
    )

    direct_energy, _ = fci.direct_uhf.kernel(
        cluster.h_eff,
        cluster.eri,
        cluster.norb,
        cluster.nelec,
        tol=1e-12,
    )
    np.testing.assert_allclose(result.energy, direct_energy, atol=1e-8, rtol=0,
                               err_msg="ROHF→UHF→EWF→LUCJ→SQD energy must match FCI")
    np.testing.assert_allclose(
        result.sci_state.spin_square(cluster.spin_orbital_overlap),
        0.75, atol=1e-4, rtol=0,
        err_msg="ROHF→UHF <S^2> must be 0.75 (pure doublet)"
    )
