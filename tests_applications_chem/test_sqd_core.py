from __future__ import annotations


import numpy as np
import pytest
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.fci import cistring, direct_spin1, direct_uhf, selected_ci, spin_op

from tyxonq.applications.chem.algorithms.sqd import (
    SCIResult,
    SCIState,
    bitstring_matrix_to_integers,
    counts_to_arrays,
    enlarge_batch_from_single_excitations,
    postselect_by_hamming_right_and_left,
    recover_configurations,
    reverse_bitstring_halves,
    run_sqd_fermion,
    samples_to_arrays,
    solve_sci,
    spin_square_from_rdms,
    subsample,
)
from tyxonq.applications.chem.algorithms.sqd.selected_ci_uhf import (
    contract_2e_uhf,
    make_hdiag_uhf,
)


def _h2_problem():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.735",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    hf = scf.RHF(mol)
    hf.kernel()
    norb = hf.mo_coeff.shape[1]
    nelec = (mol.nelectron // 2, mol.nelectron // 2)
    cas = mcscf.CASCI(hf, norb, nelec)
    one_body_tensor, nuclear_repulsion_energy = cas.get_h1cas(hf.mo_coeff)
    two_body_tensor = ao2mo.restore(1, cas.get_h2cas(hf.mo_coeff), norb)
    fci_electronic_energy, _ = direct_spin1.kernel(
        one_body_tensor, two_body_tensor, norb, nelec
    )
    return (
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
        float(nuclear_repulsion_energy),
        float(fci_electronic_energy + nuclear_repulsion_energy),
        float(hf.e_tot),
    )


def _hf_counts(norb: int, nelec: tuple[int, int]) -> dict[str, int]:
    n_alpha, n_beta = nelec
    raw_hf = (
        "1" * n_alpha + "0" * (norb - n_alpha) + "1" * n_beta + "0" * (norb - n_beta)
    )
    return {reverse_bitstring_halves(raw_hf): 1}


def _random_uhf_problem(norb: int, seed: int = 12):
    """生成满足 UHF 积分置换对称性的实数小 Hamiltonian。"""
    rng = np.random.default_rng(seed)
    npair = norb * (norb + 1) // 2

    def symmetric_one_body():
        matrix = rng.normal(size=(norb, norb))
        return matrix + matrix.T

    def same_spin_eri():
        pair_matrix = rng.normal(size=(npair, npair))
        pair_matrix = pair_matrix + pair_matrix.T
        return ao2mo.restore(1, pair_matrix, norb)

    h1e = (symmetric_one_body(), symmetric_one_body())
    eri = (
        same_spin_eri(),
        ao2mo.restore(1, rng.normal(size=(npair, npair)), norb),
        same_spin_eri(),
    )
    return h1e, eri


def _assert_ci_strings_have_electron_count(
    strings: np.ndarray, electron_count: int
) -> None:
    """检查每个 selected-CI determinant 都保持指定电子数。"""
    assert all(int(string).bit_count() == electron_count for string in strings)


def test_reverse_bitstring_halves_round_trip_and_invalid_input():
    raw = "100011"
    sqd_order = reverse_bitstring_halves(raw)

    assert sqd_order == "001110"
    assert reverse_bitstring_halves(sqd_order) == raw

    with pytest.raises(ValueError, match="even"):
        reverse_bitstring_halves("101")
    with pytest.raises(ValueError, match="only '0' and '1'"):
        reverse_bitstring_halves("10x0")


def test_counts_and_samples_to_arrays_normalize_and_reorder():
    bitstrings, probabilities = counts_to_arrays({"1001": 3, "0110": 1})

    assert bitstrings.shape == (2, 4)
    np.testing.assert_allclose(probabilities, [0.75, 0.25])

    internal_bitstrings, internal_probabilities = samples_to_arrays({"1001": 2})

    # 用户输入是 [alpha | beta] = 10|01；SQD 内部应变成 [beta | alpha] = 01|10。
    np.testing.assert_array_equal(internal_bitstrings, [[False, True, True, False]])
    np.testing.assert_allclose(internal_probabilities, [1.0])


def test_postselect_recover_and_subsample_run():
    bitstrings = np.array(
        [
            [False, True, False, True],
            [False, True, True, False],
            [True, False, False, True],
            [True, True, False, False],
        ],
        dtype=bool,
    )
    probabilities = np.array([0.4, 0.2, 0.3, 0.1])

    selected, selected_probs = postselect_by_hamming_right_and_left(
        bitstrings,
        probabilities,
        hamming_right=1,
        hamming_left=1,
    )

    assert selected.shape == (3, 4)
    np.testing.assert_allclose(np.sum(selected_probs), 1.0)

    recovered, recovered_probs = recover_configurations(
        bitstrings,
        probabilities,
        avg_occupancies=(np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        num_elec_a=1,
        num_elec_b=1,
        rand_seed=123,
    )
    assert recovered.shape[1] == 4
    np.testing.assert_allclose(np.sum(recovered_probs), 1.0)
    np.testing.assert_array_equal(
        np.sum(recovered[:, :2], axis=1), np.ones(recovered.shape[0])
    )
    np.testing.assert_array_equal(
        np.sum(recovered[:, 2:], axis=1), np.ones(recovered.shape[0])
    )

    batches = subsample(
        selected, selected_probs, samples_per_batch=2, num_batches=2, rand_seed=123
    )
    assert len(batches) == 2
    assert all(batch.shape == (2, 4) for batch in batches)


def test_recovery_handles_exact_hamming_zero_weights_and_excluded_orbitals():
    """严格 HF 初始占据和 dummy 轨道都不能让 recovery 产生 NaN。"""
    probabilities = np.array([1.0])

    # 电子数已正确时直接保留输入，即使轨道占据给出严格的 0/1。
    already_valid = np.array([[True, False, True, False]], dtype=bool)
    recovered, recovered_probs = recover_configurations(
        already_valid,
        probabilities,
        avg_occupancies=(np.array([1.0, 0.0]), np.array([1.0, 0.0])),
        num_elec_a=1,
        num_elec_b=1,
        rand_seed=1,
    )
    np.testing.assert_array_equal(recovered, already_valid)
    np.testing.assert_allclose(recovered_probs, probabilities)

    # 过满时，严格占据给所有 1->0 候选的权重都是零，需均匀回退。
    overfilled = np.array([[True, True, True, True]], dtype=bool)
    recovered, recovered_probs = recover_configurations(
        overfilled,
        probabilities,
        avg_occupancies=(np.ones(2), np.ones(2)),
        num_elec_a=1,
        num_elec_b=1,
        rand_seed=2,
    )
    np.testing.assert_array_equal(np.sum(recovered[:, :2], axis=1), [1])
    np.testing.assert_array_equal(np.sum(recovered[:, 2:], axis=1), [1])
    assert np.all(np.isfinite(recovered_probs))
    np.testing.assert_allclose(np.sum(recovered_probs), 1.0)

    # 欠满时，严格空占据给所有 0->1 候选的权重都是零，需均匀回退。
    underfilled = np.zeros((1, 4), dtype=bool)
    recovered, recovered_probs = recover_configurations(
        underfilled,
        probabilities,
        avg_occupancies=(np.zeros(2), np.zeros(2)),
        num_elec_a=1,
        num_elec_b=1,
        rand_seed=3,
    )
    np.testing.assert_array_equal(np.sum(recovered[:, :2], axis=1), [1])
    np.testing.assert_array_equal(np.sum(recovered[:, 2:], axis=1), [1])
    assert np.all(np.isfinite(recovered_probs))
    np.testing.assert_allclose(np.sum(recovered_probs), 1.0)

    # excluded 轨道即使原始 bitstring 中为 1，也必须清零且不可被重新填回。
    recovered, _ = recover_configurations(
        np.array([[True, False, True, False]], dtype=bool),
        probabilities,
        avg_occupancies=(np.zeros(2), np.zeros(2)),
        num_elec_a=1,
        num_elec_b=1,
        rand_seed=4,
        excluded_orbitals_a=[1],
        excluded_orbitals_b=[1],
    )
    assert not np.any(recovered[:, 0])
    assert not np.any(recovered[:, 2])
    np.testing.assert_array_equal(np.sum(recovered[:, :2], axis=1), [1])
    np.testing.assert_array_equal(np.sum(recovered[:, 2:], axis=1), [1])


def test_enlarge_batch_from_single_excitations_preserves_spin_halves():
    bitstrings = np.array([[False, True, False, True]], dtype=bool)

    expanded = enlarge_batch_from_single_excitations(bitstrings)

    assert expanded.shape[0] > bitstrings.shape[0]
    assert expanded.shape[1] == bitstrings.shape[1]
    np.testing.assert_array_equal(
        np.sum(expanded[:, :2], axis=1), np.ones(expanded.shape[0])
    )
    np.testing.assert_array_equal(
        np.sum(expanded[:, 2:], axis=1), np.ones(expanded.shape[0])
    )
    assert np.unique(expanded, axis=0).shape[0] == expanded.shape[0]


def test_run_sqd_fermion_h2_matches_direct_spin1_fci():
    one_body, two_body, norb, nelec, nuclear_repulsion, fci_total_energy, _ = (
        _h2_problem()
    )
    counts = {
        "0101": 40,
        "0110": 10,
        "1001": 10,
        "1010": 40,
    }

    result = run_sqd_fermion(
        one_body,
        two_body,
        counts,
        samples_per_batch=4,
        norb=norb,
        nelec=nelec,
        nuclear_repulsion_energy=nuclear_repulsion,
        max_iterations=3,
        symmetrize_spin=True,
        seed=7,
    )

    assert np.isfinite(result.energy)
    np.testing.assert_allclose(result.total_energy, fci_total_energy, atol=1e-8)


def test_run_sqd_fermion_can_average_batch_occupancies(monkeypatch):
    """PCM 模式应使用所有 batch 的平均占据数驱动下一轮恢复。"""
    from tyxonq.applications.chem.algorithms.sqd import fermion

    recovered_with: list[tuple[np.ndarray, np.ndarray]] = []
    original_recover = fermion.recover_configurations

    def capture_recovery(bitstrings, probabilities, avg_occupancies, *args, **kwargs):
        recovered_with.append(
            tuple(np.asarray(block, dtype=float).copy() for block in avg_occupancies)
        )
        return original_recover(
            bitstrings,
            probabilities,
            avg_occupancies,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(fermion, "recover_configurations", capture_recovery)

    state_a = SCIState(
        amplitudes=np.ones((1, 1)),
        ci_strs_a=np.asarray([1]),
        ci_strs_b=np.asarray([1]),
        norb=2,
        nelec=(1, 1),
    )
    state_b = SCIState(
        amplitudes=np.ones((1, 1)),
        ci_strs_a=np.asarray([2]),
        ci_strs_b=np.asarray([2]),
        norb=2,
        nelec=(1, 1),
    )

    def two_batch_solver(*_args):
        return [
            SCIResult(-1.0, state_a, (np.asarray([1.0, 0.0]),) * 2),
            SCIResult(-0.9, state_b, (np.asarray([0.0, 1.0]),) * 2),
        ]

    run_sqd_fermion(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        {"0101": 1, "0110": 1, "1001": 1, "1010": 1},
        samples_per_batch=2,
        norb=2,
        nelec=(1, 1),
        num_batches=2,
        max_iterations=2,
        sci_solver=two_batch_solver,
        initial_occupancies=(np.asarray([0.8, 0.2]),) * 2,
        average_batch_occupancies=True,
        energy_tol=0.0,
        occupancies_tol=0.0,
        seed=7,
    )

    assert len(recovered_with) == 2
    np.testing.assert_allclose(recovered_with[1][0], [0.5, 0.5])
    np.testing.assert_allclose(recovered_with[1][1], [0.5, 0.5])


def test_run_sqd_fermion_spin_sq_selects_o2_singlet_and_triplet():
    """顶层 spin_sq 应在同一 M_S=0 空间中选择目标总自旋。"""

    mol = gto.M(
        atom="O 0 0 0; O 0 0 1.21",
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    mean_field = scf.ROHF(mol).run()
    assert mean_field.converged
    casci = mcscf.CASCI(mean_field, 2, (1, 1))
    h1e, _ = casci.get_h1eff(mean_field.mo_coeff)
    active = mean_field.mo_coeff[:, casci.ncore : casci.ncore + 2]
    eri = ao2mo.restore(1, ao2mo.full(mol, active, compact=False), 2)
    strings = np.asarray(cistring.make_strings(range(2), 1))
    counts = {"0101": 1, "0110": 1, "1001": 1, "1010": 1}
    common = {
        "samples_per_batch": 4,
        "norb": 2,
        "nelec": (1, 1),
        "num_batches": 1,
        "max_iterations": 1,
        "include_configurations": (strings.tolist(), strings.tolist()),
        "seed": 7,
    }

    singlet = run_sqd_fermion(h1e, eri, counts, spin_sq=0.0, **common)
    triplet = run_sqd_fermion(h1e, eri, counts, spin_sq=2.0, **common)

    assert singlet.sci_state.spin_square() == pytest.approx(0.0, abs=1e-10)
    assert triplet.sci_state.spin_square() == pytest.approx(2.0, abs=1e-10)
    assert singlet.energy > triplet.energy + 1e-4

    def custom_solver(*_args):
        raise AssertionError("冲突参数应在调用自定义求解器前被拒绝。")

    with pytest.raises(ValueError, match="custom sci_solver"):
        run_sqd_fermion(
            h1e,
            eri,
            counts,
            spin_sq=0.0,
            sci_solver=custom_solver,
            **common,
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        run_sqd_fermion(h1e, eri, counts, spin_sq=-1.0, **common)


def test_run_sqd_fermion_h2_hf_only_counts_match_rhf_energy():
    one_body, two_body, norb, nelec, nuclear_repulsion, _, hf_total_energy = (
        _h2_problem()
    )
    counts = _hf_counts(norb, nelec)
    sqd_kwargs = dict(
        samples_per_batch=1,
        norb=norb,
        nelec=nelec,
        nuclear_repulsion_energy=nuclear_repulsion,
        num_batches=1,
        max_iterations=1,
        symmetrize_spin=False,
        max_dim=1,
        seed=7,
    )

    result = run_sqd_fermion(
        one_body,
        two_body,
        counts,
        **sqd_kwargs,
    )
    explicit_default = run_sqd_fermion(
        one_body,
        two_body,
        counts,
        ext_sqd=False,
        **sqd_kwargs,
    )

    np.testing.assert_allclose(result.total_energy, hf_total_energy, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        explicit_default.total_energy, result.total_energy, atol=0, rtol=0
    )
    np.testing.assert_array_equal(
        explicit_default.sci_state.ci_strs_a, result.sci_state.ci_strs_a
    )
    np.testing.assert_array_equal(
        explicit_default.sci_state.ci_strs_b, result.sci_state.ci_strs_b
    )


def test_run_sqd_fermion_h2_ext_sqd_expands_hf_only_subspace():
    one_body, two_body, norb, nelec, nuclear_repulsion, _, _ = _h2_problem()
    counts = _hf_counts(norb, nelec)
    sqd_kwargs = dict(
        samples_per_batch=1,
        norb=norb,
        nelec=nelec,
        nuclear_repulsion_energy=nuclear_repulsion,
        num_batches=1,
        max_iterations=1,
        symmetrize_spin=False,
        max_dim=1,
        seed=7,
    )

    plain = run_sqd_fermion(one_body, two_body, counts, **sqd_kwargs)
    extended = run_sqd_fermion(one_body, two_body, counts, ext_sqd=True, **sqd_kwargs)
    n_alpha, n_beta = nelec

    assert isinstance(extended, SCIResult)
    assert isinstance(extended.sci_state, SCIState)
    assert np.isfinite(extended.energy)
    assert len(extended.sci_state.ci_strs_a) > len(plain.sci_state.ci_strs_a)
    assert len(extended.sci_state.ci_strs_b) > len(plain.sci_state.ci_strs_b)
    assert len(extended.sci_state.ci_strs_a) > sqd_kwargs["max_dim"]
    assert len(extended.sci_state.ci_strs_b) > sqd_kwargs["max_dim"]
    _assert_ci_strings_have_electron_count(plain.sci_state.ci_strs_a, n_alpha)
    _assert_ci_strings_have_electron_count(plain.sci_state.ci_strs_b, n_beta)
    _assert_ci_strings_have_electron_count(extended.sci_state.ci_strs_a, n_alpha)
    _assert_ci_strings_have_electron_count(extended.sci_state.ci_strs_b, n_beta)
    assert extended.sci_state.amplitudes.shape == (
        len(extended.sci_state.ci_strs_a),
        len(extended.sci_state.ci_strs_b),
    )
    assert extended.rdm1 is not None
    assert extended.rdm2 is not None
    assert extended.rdm1.shape == (norb, norb)
    assert extended.rdm2.shape == (norb, norb, norb, norb)


def test_unrestricted_selected_ci_hop_and_hdiag_match_projected_direct_uhf():
    norb = 4
    nelec = (2, 1)
    h1e, eri = _random_uhf_problem(norb, seed=31)
    rng = np.random.default_rng(32)
    all_strings_a = cistring.make_strings(range(norb), nelec[0])
    all_strings_b = cistring.make_strings(range(norb), nelec[1])
    strings_a = all_strings_a[[0, 2, 5]]
    strings_b = all_strings_b[[0, 3]]
    coefficients = rng.normal(size=(len(strings_a), len(strings_b)))

    sci_vector = selected_ci._as_SCIvector(coefficients, (strings_a, strings_b))
    absorbed_eri = direct_uhf.absorb_h1e(h1e, eri, norb, nelec, 0.5)
    contracted = contract_2e_uhf(absorbed_eri, sci_vector, norb, nelec)

    # 把 selected 系数嵌入完整 FCI 空间，直接计算后再投影回同一子空间。
    index_a = {int(string): i for i, string in enumerate(all_strings_a)}
    index_b = {int(string): i for i, string in enumerate(all_strings_b)}
    full_coefficients = np.zeros((len(all_strings_a), len(all_strings_b)))
    for i, string_a in enumerate(strings_a):
        for j, string_b in enumerate(strings_b):
            full_coefficients[index_a[int(string_a)], index_b[int(string_b)]] = (
                coefficients[i, j]
            )
    links_a = cistring.gen_linkstr_index(range(norb), nelec[0], tril=True)
    links_b = cistring.gen_linkstr_index(range(norb), nelec[1], tril=True)
    full_contracted = direct_uhf.contract_2e(
        absorbed_eri, full_coefficients, norb, nelec, (links_a, links_b)
    )
    projected = np.array(
        [
            [full_contracted[index_a[int(a)], index_b[int(b)]] for b in strings_b]
            for a in strings_a
        ]
    )
    np.testing.assert_allclose(contracted, projected, atol=1e-12, rtol=0)

    full_hdiag = direct_uhf.make_hdiag(h1e, eri, norb, nelec).reshape(
        len(all_strings_a), len(all_strings_b)
    )
    projected_hdiag = np.array(
        [
            full_hdiag[index_a[int(a)], index_b[int(b)]]
            for a in strings_a
            for b in strings_b
        ]
    )
    np.testing.assert_allclose(
        make_hdiag_uhf(h1e, eri, (strings_a, strings_b), norb, nelec),
        projected_hdiag,
        atol=1e-12,
        rtol=0,
    )


def test_solve_sci_unrestricted_matches_direct_uhf_energy_rdms_and_spin_square():
    norb = 3
    nelec = (2, 1)
    h1e, eri = _random_uhf_problem(norb, seed=41)
    strings = (
        cistring.make_strings(range(norb), nelec[0]),
        cistring.make_strings(range(norb), nelec[1]),
    )

    direct_energy, _ = direct_uhf.kernel(h1e, eri, norb, nelec, tol=1e-12)
    result = solve_sci(strings, h1e, eri, norb, nelec, tol=1e-12)

    np.testing.assert_allclose(result.energy, direct_energy, atol=1e-10, rtol=0)
    assert result.rdm1s is not None
    assert result.rdm2s is not None
    np.testing.assert_allclose(np.trace(result.rdm1s[0]), nelec[0], atol=1e-10)
    np.testing.assert_allclose(np.trace(result.rdm1s[1]), nelec[1], atol=1e-10)

    # 用两套正交轨道制造非平凡交叉重叠，和 PySCF general 公式直接比对。
    rng = np.random.default_rng(42)
    coeff_a, _ = np.linalg.qr(rng.normal(size=(norb, norb)))
    coeff_b, _ = np.linalg.qr(rng.normal(size=(norb, norb)))
    overlap_ab = coeff_a.T @ coeff_b
    reference_spin, _ = spin_op.spin_square_general(
        result.rdm1s[0],
        result.rdm1s[1],
        result.rdm2s[0],
        result.rdm2s[1],
        result.rdm2s[2],
        (coeff_a, coeff_b),
        np.eye(norb),
    )
    np.testing.assert_allclose(
        result.sci_state.spin_square(overlap_ab), reference_spin, atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        result.spin_square(overlap_ab), reference_spin, atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        spin_square_from_rdms(result.rdm1s, result.rdm2s, overlap_ab),
        reference_spin,
        atol=1e-10,
        rtol=0,
    )


def test_unrestricted_input_validation_and_excluded_orbitals():
    norb = 3
    nelec = (1, 1)
    h1e, eri = _random_uhf_problem(norb, seed=51)
    strings = (
        cistring.make_strings(range(norb), nelec[0]),
        cistring.make_strings(range(norb), nelec[1]),
    )

    with pytest.raises(ValueError, match="must both use"):
        solve_sci(strings, h1e, eri[0], norb, nelec)
    with pytest.raises(ValueError, match="spin_sq"):
        solve_sci(strings, h1e, eri, norb, nelec, spin_sq=0.0)

    # 内部顺序为 [beta | alpha]，每半区列顺序为 [orbital 2, 1, 0]。
    samples = []
    for beta_orbital in (0, 1):
        for alpha_orbital in (0, 1):
            beta = np.zeros(norb, dtype=bool)
            alpha = np.zeros(norb, dtype=bool)
            beta[norb - 1 - beta_orbital] = True
            alpha[norb - 1 - alpha_orbital] = True
            samples.append(np.concatenate((beta, alpha)))
    samples.append(np.array([True, False, False, True, False, False]))

    recovered, _ = recover_configurations(
        np.asarray(samples[-1:]),
        np.array([1.0]),
        avg_occupancies=(np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.5, 0.0])),
        num_elec_a=1,
        num_elec_b=1,
        rand_seed=7,
        excluded_orbitals_a=[2],
        excluded_orbitals_b=[2],
    )
    assert not np.any(recovered[:, 0])
    assert not np.any(recovered[:, norb])

    result = run_sqd_fermion(
        h1e,
        eri,
        np.asarray(samples),
        samples_per_batch=4,
        norb=norb,
        nelec=nelec,
        sample_order="beta_alpha",
        excluded_orbitals=[2],
        max_iterations=1,
        seed=7,
    )
    assert all((int(string) & (1 << 2)) == 0 for string in result.sci_state.ci_strs_a)
    assert all((int(string) & (1 << 2)) == 0 for string in result.sci_state.ci_strs_b)

    physical = [0, 1]
    h1e_physical = tuple(block[np.ix_(physical, physical)] for block in h1e)
    eri_indices = np.ix_(physical, physical, physical, physical)
    eri_physical = tuple(block[eri_indices] for block in eri)
    direct_energy, _ = direct_uhf.kernel(
        h1e_physical, eri_physical, len(physical), nelec, tol=1e-12
    )
    np.testing.assert_allclose(result.energy, direct_energy, atol=1e-10, rtol=0)

    with pytest.raises(ValueError, match="symmetrization"):
        run_sqd_fermion(
            h1e,
            eri,
            np.asarray(samples),
            samples_per_batch=4,
            norb=norb,
            nelec=nelec,
            sample_order="beta_alpha",
            symmetrize_spin=True,
            max_iterations=1,
        )


def test_ext_sqd_amplitude_probability_boundaries_and_bit_order():
    """概率使用严格大于；不对称电子数揭示 alpha/beta 位序颠倒。"""
    from tyxonq.applications.chem.algorithms.sqd.fermion import _dominant_product_bitstrings

    state = SCIState(
        np.array([[0.5, 0.25], [0.25, np.sqrt(0.625)]]),
        np.array([3, 12]), np.array([1, 8]), 4, (2, 1),
    )
    result = SCIResult(-1.0, state, (np.zeros(4), np.zeros(4)))
    amplitude = _dominant_product_bitstrings(result, 0.5)
    probability = _dominant_product_bitstrings(result, 0.25, threshold_mode="probability")
    np.testing.assert_array_equal(bitstring_matrix_to_integers(amplitude[:, :4]), [8, 1])
    np.testing.assert_array_equal(bitstring_matrix_to_integers(amplitude[:, 4:]), [12, 3])
    np.testing.assert_array_equal(probability, amplitude[:1])
    fallback = _dominant_product_bitstrings(result, 1.0, threshold_mode="probability")
    np.testing.assert_array_equal(fallback, probability)


def test_ext_sqd_uses_global_best_and_keeps_original_space():
    """最后一轮较差时仍从历轮最优态扩展，并保留所有原 strings。"""
    recorded = []
    state = SCIState(np.array([[0.5, 0.25], [0.25, np.sqrt(0.625)]]), np.array([3, 12]), np.array([1, 8]), 4, (2, 1))

    def solver(batches, *_):
        recorded.append(batches)
        if len(recorded) == 2:
            other = SCIState(np.ones((1, 1)), np.array([5]), np.array([2]), 4, (2, 1))
            return [SCIResult(-1.0, other, (np.full(4, 0.5), np.full(4, 0.25)))]
        return [SCIResult(-2.0, state, (np.full(4, 0.5), np.full(4, 0.25)))]

    run_sqd_fermion(
        np.eye(4), np.zeros((4,) * 4), {"00110001": 1}, 1, 4, (2, 1),
        max_iterations=2, carryover_threshold=0.9, ext_sqd=True,
        ext_sqd_threshold=0.25, ext_sqd_threshold_mode="probability", sci_solver=solver, seed=1,
    )
    assert len(recorded) == 3
    alpha, beta = recorded[-1][0]
    assert set(state.ci_strs_a) <= set(alpha)
    assert set(state.ci_strs_b) <= set(beta)
    assert set(alpha) == {3, 5, 6, 9, 10, 12}
    assert set(beta) == {1, 2, 4, 8}


@pytest.mark.parametrize("extra", [{"ext_sqd_threshold": -1}, {"ext_sqd_threshold": np.nan}, {"ext_sqd_threshold_mode": "weight"}])
def test_ext_sqd_rejects_invalid_threshold_settings(extra):
    with pytest.raises(ValueError, match="ext_sqd_threshold"):
        run_sqd_fermion(np.eye(2), np.zeros((2,) * 4), {"0101": 1}, 1, 2, (1, 1), **extra)
