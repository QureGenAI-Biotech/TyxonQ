import builtins

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from tyxonq.applications.chem.algorithms.lucj import double_factorization as df_module

from tyxonq.applications.chem.algorithms.lucj import (
    LUCJ,
    initialize_lucj_parameters_from_ccsd,
    normalize_lucj_params,
)
from tyxonq.applications.chem.algorithms.lucj.conversion import factors_to_lucj_parameters
from tyxonq.applications.chem.algorithms.lucj.double_factorization import (
    DoubleFactorizationResult,
    double_factorize_t2,
    double_factorized_t2,
    double_factorized_t2_alpha_beta,
    orbital_rotation_from_t1_amplitudes,
    reconstruct_t2_alpha_beta_from_factors,
    reconstruct_t2_from_factors,
    t2_reconstruction_loss,
)


def _small_t2(nocc=2, nvir=2):
    """生成一个稳定的实数 t2 测试张量。"""
    rng = np.random.default_rng(1234)
    return 0.01 * rng.normal(size=(nocc, nocc, nvir, nvir))


def _small_unrestricted_t2():
    """生成 `norb=3, nalpha=2, nbeta=1` 的 UCCSD block。"""
    rng = np.random.default_rng(5678)
    return (
        0.01 * rng.normal(size=(2, 2, 1, 1)),
        0.01 * rng.normal(size=(2, 1, 1, 2)),
        0.01 * rng.normal(size=(1, 1, 2, 2)),
    )


@pytest.mark.parametrize(
    ("topology", "opposite_count"),
    [
        ("square", 4),
        ("hex", 2),
        ("linear", 1),
        ("paired7", 4),
        ("paired9", 4),
        ("paired10", 4),
        ("paired12", 4),
    ],
)
def test_initialize_lucj_parameters_from_ccsd_shapes(topology, opposite_count):
    params = initialize_lucj_parameters_from_ccsd(
        _small_t2(),
        n_spatial_orbitals=4,
        n_layers=2,
        topology=topology,
        optimize=False,
    )
    normalized = normalize_lucj_params(params, 4, 2, topology)

    assert normalized["orbital_rotations"].shape == (2, 4, 4)
    assert normalized["diag_coulomb_mats"].shape == (2, 2, 4, 4)
    assert normalized["final_orbital_rotation"] is None
    jab = normalized["diag_coulomb_mats"][:, 1]
    assert np.count_nonzero(np.any(np.abs(jab) > 0, axis=0)) <= opposite_count


def test_initialize_lucj_parameters_defaults_to_linear():
    params = initialize_lucj_parameters_from_ccsd(
        _small_t2(),
        n_spatial_orbitals=4,
        n_layers=1,
        optimize=False,
    )
    circuit = LUCJ(4, 4, 1).get_circuit(params)

    assert circuit.metadata["lucj"]["topology"] == "linear"
    opposite_ops = [
        op
        for op in circuit.metadata["lucj"]["logical_ops"]
        if op["kind"] == "opposite_spin_cphase"
    ]
    assert all(op["orbitals"] == (0, 0) for op in opposite_ops)


def test_initialize_lucj_parameters_from_uccsd_shapes_and_final_rotation():
    params = initialize_lucj_parameters_from_ccsd(
        _small_unrestricted_t2(),
        t1=(np.zeros((2, 1)), np.zeros((1, 2))),
        n_spatial_orbitals=3,
        n_layers=2,
        topology="linear",
        optimize=False,
    )

    assert params["orbital_rotations"].shape == (2, 2, 3, 3)
    assert params["diag_coulomb_mats"].shape == (2, 3, 3, 3)
    assert params["final_orbital_rotation"].shape == (2, 3, 3)
    assert np.allclose(
        params["final_orbital_rotation"].conj().transpose(0, 2, 1)
        @ params["final_orbital_rotation"],
        np.tile(np.eye(3), (2, 1, 1)),
    )


def test_uccsd_initialization_excludes_dummy_orbitals():
    params = initialize_lucj_parameters_from_ccsd(
        _small_unrestricted_t2(),
        n_spatial_orbitals=3,
        n_layers=1,
        topology="linear",
        optimize=False,
        excluded_orbitals=((), (2,)),
    )
    rotations_beta = params["orbital_rotations"][:, 1]
    expected_row = np.array([0.0, 0.0, 1.0])

    assert np.allclose(rotations_beta[:, 2, :], expected_row)
    assert np.allclose(rotations_beta[:, :, 2], expected_row)
    assert np.allclose(params["diag_coulomb_mats"][:, 1, :, 2], 0.0)
    assert np.allclose(params["diag_coulomb_mats"][:, 2, 2, :], 0.0)


def test_alpha_beta_explicit_factorization_reconstructs_t2():
    t2ab = _small_unrestricted_t2()[1]
    diag, rotations = double_factorized_t2_alpha_beta(t2ab)
    reconstructed = reconstruct_t2_alpha_beta_from_factors(rotations, diag, 2, 1)

    assert np.allclose(reconstructed, t2ab, atol=1e-10)


def test_alpha_beta_pair_validation_checks_pairs_ab_not_pairs_bb_twice():
    t2ab = _small_unrestricted_t2()[1]
    duplicate_pairs_ab = [(0, 0), (0, 0)]

    with pytest.raises(ValueError, match="Duplicate interaction pairs"):
        double_factorized_t2_alpha_beta(
            t2ab,
            diag_coulomb_indices=(None, duplicate_pairs_ab, None),
        )


def test_double_factorized_t2_reconstruction_loss_is_finite():
    t2 = _small_t2()
    result = double_factorize_t2(
        t2,
        n_spatial_orbitals=4,
        n_layers=2,
        topology="hex",
        optimize=False,
    )
    reconstructed = reconstruct_t2_from_factors(
        result.orbital_rotations,
        result.diagonal_coulomb_matrices,
        result.nocc,
        result.nvir,
    )
    loss = t2_reconstruction_loss(t2, result.orbital_rotations, result.diagonal_coulomb_matrices)

    assert reconstructed.shape == t2.shape
    assert np.isfinite(loss)


def test_optimized_factorization_does_not_increase_loss():
    pytest.importorskip("torch")
    t2 = _small_t2(nocc=1, nvir=2)
    unoptimized = double_factorize_t2(
        t2,
        n_spatial_orbitals=3,
        n_layers=1,
        topology="square",
        optimize=False,
    )
    optimized = double_factorize_t2(
        t2,
        n_spatial_orbitals=3,
        n_layers=1,
        topology="square",
        optimize=True,
        maxiter=5,
    )

    assert optimized.loss <= unoptimized.loss + 1e-12


def test_optimized_alpha_beta_factorization_returns_trace_and_lowers_loss():
    pytest.importorskip("torch")
    t2ab = _small_unrestricted_t2()[1]
    unoptimized_diag, unoptimized_rotations = double_factorized_t2_alpha_beta(
        t2ab,
        max_terms=1,
        optimize=False,
    )
    unoptimized_reconstruction = reconstruct_t2_alpha_beta_from_factors(
        unoptimized_rotations,
        unoptimized_diag,
        nocc_a=2,
        nocc_b=1,
    )
    unoptimized_loss = 0.5 * float(
        np.vdot(unoptimized_reconstruction - t2ab, unoptimized_reconstruction - t2ab).real
    )

    loss_trace = []

    def record_loss(intermediate_result):
        loss_trace.append(float(intermediate_result.fun))

    optimized_diag, optimized_rotations, optimize_result = double_factorized_t2_alpha_beta(
        t2ab,
        max_terms=1,
        optimize=True,
        options={"maxiter": 3},
        callback=record_loss,
        return_optimize_result=True,
    )
    optimized_reconstruction = reconstruct_t2_alpha_beta_from_factors(
        optimized_rotations,
        optimized_diag,
        nocc_a=2,
        nocc_b=1,
    )
    optimized_loss = 0.5 * float(
        np.vdot(optimized_reconstruction - t2ab, optimized_reconstruction - t2ab).real
    )

    assert isinstance(optimize_result, OptimizeResult)
    assert 0 < optimize_result.nit <= 3
    assert len(loss_trace) == optimize_result.nit
    assert len(optimize_result.loss_history) == optimize_result.nit + 1
    np.testing.assert_allclose(
        optimize_result.loss_history[1:], loss_trace, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(loss_trace[-1], optimize_result.fun, atol=1e-12, rtol=0)
    np.testing.assert_allclose(optimized_loss, optimize_result.fun, atol=1e-12, rtol=0)
    assert optimized_loss < unoptimized_loss


def test_unrestricted_factorization_keeps_alpha_beta_optimize_result():
    pytest.importorskip("torch")
    result = double_factorize_t2(
        _small_unrestricted_t2(),
        n_spatial_orbitals=3,
        n_layers=1,
        topology="square",
        optimize=True,
        maxiter=2,
    )

    assert isinstance(result.optimize_result, tuple)
    assert len(result.optimize_result) == 1
    assert isinstance(result.optimize_result[0], OptimizeResult)
    assert 0 < result.optimize_result[0].nit <= 2


def test_public_double_factorization_forwards_optimizer_options(monkeypatch):
    """maxiter 由公共参数固定，gtol/ftol 原样进入 SciPy options。"""

    captured = {}

    def fake_factorization(amplitudes, **kwargs):
        captured.update(kwargs["options"])
        norb = amplitudes.shape[0] + amplitudes.shape[2]
        diagonal = np.zeros((1, norb, norb))
        rotations = np.eye(norb, dtype=complex)[None, :, :]
        result = OptimizeResult(
            success=True,
            status=0,
            message="test",
            nit=0,
            nfev=1,
            fun=0.0,
            jac=np.zeros(1),
        )
        result.loss_history = np.array([0.0])
        return diagonal, rotations, result

    monkeypatch.setattr(df_module, "double_factorized_t2", fake_factorization)
    df_module.double_factorize_t2(
        _small_t2(nocc=1, nvir=2),
        n_spatial_orbitals=3,
        n_layers=1,
        optimize=True,
        maxiter=5000,
        optimizer_options={"gtol": 1e-5, "ftol": 2.220446049250313e-9},
    )

    assert captured == {
        "gtol": 1e-5,
        "ftol": 2.220446049250313e-9,
        "maxiter": 5000,
    }


def test_public_initializer_can_return_factorization_audit_result():
    pytest.importorskip("torch")
    params, factors = initialize_lucj_parameters_from_ccsd(
        _small_unrestricted_t2(),
        n_spatial_orbitals=3,
        n_layers=1,
        topology="paired10",
        optimize=True,
        maxiter=2,
        return_factorization_result=True,
    )

    assert params["orbital_rotations"].shape == (1, 2, 3, 3)
    assert isinstance(factors, DoubleFactorizationResult)
    assert isinstance(factors.optimize_result, tuple)
    assert factors.optimize_result[0].loss_history.shape == (
        factors.optimize_result[0].nit + 1,
    )


def test_optimized_alpha_beta_factorization_has_no_ffsim_runtime_import(monkeypatch):
    pytest.importorskip("torch")
    original_import = builtins.__import__

    def reject_ffsim(name, *args, **kwargs):
        if name == "ffsim" or name.startswith("ffsim."):
            raise AssertionError("LUCJ initialization must not import vendored ffsim at runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_ffsim)
    _, _, result = double_factorized_t2_alpha_beta(
        _small_unrestricted_t2()[1],
        max_terms=1,
        optimize=True,
        options={"maxiter": 1},
        return_optimize_result=True,
    )

    assert isinstance(result, OptimizeResult)


def test_topology_mask_is_applied_to_diag_coulomb_mats():
    result = double_factorize_t2(
        _small_t2(),
        n_spatial_orbitals=4,
        n_layers=1,
        topology="linear",
        optimize=False,
    )
    jaa = result.diag_coulomb_mats[0, 0]
    jab = result.diag_coulomb_mats[0, 1]

    assert np.count_nonzero(np.abs(jaa) > 0) <= 6
    assert np.count_nonzero(np.abs(jab) > 0) <= 1
    assert np.allclose(jab[1:, 1:], 0.0)


def test_final_orbital_rotation_from_t1_is_unitary():
    t1 = np.array([[0.01, -0.02], [0.03, 0.04]])
    final = orbital_rotation_from_t1_amplitudes(t1)

    assert final.shape == (4, 4)
    assert np.allclose(final.conj().T @ final, np.eye(4), atol=1e-10)


def test_conversion_preserves_factor_arrays():
    rotations = np.tile(np.eye(4, dtype=complex), (1, 1, 1))
    diag = np.zeros((1, 2, 4, 4), dtype=float)
    diag[0, 0, 0, 1] = diag[0, 0, 1, 0] = 0.2
    diag[0, 1, 0, 0] = 0.3
    factors = DoubleFactorizationResult(
        orbital_rotations=rotations,
        diagonal_coulomb_matrices=diag[:, 0],
        diag_coulomb_mats=diag,
        final_orbital_rotation=None,
        nocc=2,
        nvir=2,
        n_spatial_orbitals=4,
        topology="square",
        loss=0.0,
    )

    params = factors_to_lucj_parameters(factors, 4, 1, "square")

    assert np.allclose(params["orbital_rotations"], rotations)
    assert np.allclose(params["diag_coulomb_mats"], diag)


@pytest.mark.parametrize(
    "topology",
    ["square", "hex", "linear", "paired7", "paired9", "paired10", "paired12"],
)
def test_initialized_parameters_build_lucj_circuit(topology):
    params = initialize_lucj_parameters_from_ccsd(
        _small_t2(),
        n_spatial_orbitals=4,
        n_layers=2,
        topology=topology,
        optimize=False,
    )
    circuit = LUCJ(4, 4, 2, topology).get_circuit(params)

    assert circuit.num_qubits == 8
    assert circuit.metadata["lucj"]["n_orbitals"] == 4
    assert circuit.metadata["lucj"]["n_electrons"] == 4


def test_double_factorized_t2_no_ffsim_runtime_import_path():
    diag, rotations = double_factorized_t2(_small_t2(nocc=1, nvir=1), max_terms=1)

    assert diag.shape == (1, 2, 2)
    assert rotations.shape == (1, 2, 2)


def test_explicit_interaction_pairs_propagate_through_initialization():
    """压缩拟合、参数转换和线路共用同一掩码。"""
    from tyxonq.applications.chem.algorithms.lucj import LUCJ
    from tyxonq.applications.chem.algorithms.lucj.conversion import factors_to_lucj_parameters

    t2 = np.array([[[[0.1, 0.03], [0.03, -0.04]]]])
    pairs = ([(0, 2)], [(1, 1)])
    params, factors = initialize_lucj_parameters_from_ccsd(
        t2, n_spatial_orbitals=3, n_layers=1,
        interaction_pairs=pairs, maxiter=3, return_factorization_result=True,
    )
    assert factors.interaction_pairs == pairs
    original_mats = factors.diag_coulomb_mats.copy()
    restored = factors_to_lucj_parameters(factors, 3, 1)
    np.testing.assert_array_equal(factors.diag_coulomb_mats, original_mats)
    assert not np.shares_memory(restored["diag_coulomb_mats"], factors.diag_coulomb_mats)
    np.testing.assert_allclose(restored["diag_coulomb_mats"], params["diag_coulomb_mats"])
    aa, ab = params["diag_coulomb_mats"][0]
    assert np.allclose(aa[np.array([[True, True, False], [True, True, True], [False, True, True]])], 0)
    ab_without_pair = ab.copy()
    ab_without_pair[1, 1] = 0
    assert np.allclose(ab_without_pair, 0)
    circuit = LUCJ(3, 2, 1, interaction_pairs=pairs).get_circuit(params)
    assert {op["orbitals"] for op in circuit.metadata["lucj"]["logical_ops"] if op["kind"] == "opposite_spin_cphase"} <= {(1, 1)}


@pytest.mark.parametrize("unrestricted", [False, True])
def test_empty_explicit_interactions_are_noop(unrestricted):
    """空耦合集合表示无 Jastrow 相互作用，不应在参数打包时失败。"""
    t2 = np.ones((1, 1, 1, 1)) * 0.1
    if unrestricted:
        t2 = (np.zeros_like(t2), t2, np.zeros_like(t2))
    params = initialize_lucj_parameters_from_ccsd(
        t2, n_spatial_orbitals=2, n_layers=1, interaction_pairs=([], []), maxiter=1
    )
    np.testing.assert_array_equal(params["diag_coulomb_mats"], 0)


@pytest.mark.parametrize("topology", ["linear", "square"])
def test_explicit_topology_pairs_preserve_default_initialization(topology):
    """把既有 topology 展开成显式 pair 后，初始化参数保持相同。"""
    from tyxonq.applications.chem.algorithms.lucj.topology import interaction_pairs_spin_balanced

    t2 = np.array([[[[0.1, 0.03], [0.03, -0.04]]]])
    options = dict(n_spatial_orbitals=3, n_layers=1, topology=topology, optimize=False)
    default = initialize_lucj_parameters_from_ccsd(t2, **options)
    explicit = initialize_lucj_parameters_from_ccsd(
        t2, interaction_pairs=interaction_pairs_spin_balanced(3, topology), **options
    )
    np.testing.assert_allclose(explicit["orbital_rotations"], default["orbital_rotations"])
    np.testing.assert_allclose(explicit["diag_coulomb_mats"], default["diag_coulomb_mats"])
