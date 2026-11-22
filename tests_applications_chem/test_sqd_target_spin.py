import numpy as np
import pytest
from pyscf.fci import cistring, spin_op

from tyxonq.applications.chem.algorithms.sqd.gpu.spin_target import (
    _spin_blocks, full_spin_residual,
)


def test_full_spin_residual_includes_amplitudes_outside_sampled_space():
    """嵌入完整 FCI 后用独立 PySCF 算符复核带符号的自旋翻转。"""
    n, electrons = 5, (3, 2)
    full = tuple(cistring.make_strings(range(n), e) for e in electrons)
    labels = (full[0][::2], full[1][1::3])
    coefficients = np.random.default_rng(47).normal(size=tuple(map(len, labels)))
    coefficients /= np.linalg.norm(coefficients)
    ix = np.ix_(*(np.searchsorted(f, s) for f, s in zip(full, labels)))
    embedded = np.zeros(tuple(map(len, full)))
    embedded[ix] = coefficients
    target = 0.75
    residual = spin_op.contract_ss(embedded, n, electrons) - target * embedded
    actual = full_spin_residual(coefficients.ravel(), _spin_blocks(labels), 0.25, target)
    assert actual == pytest.approx(np.linalg.norm(residual), abs=1e-12)
    assert actual > np.linalg.norm(residual[ix]) + 0.01


@pytest.mark.parametrize("n,strategy", [(2, "dense_target_subspace"), (9, "multistart_spin_filter")])
def test_target_triplet_matches_exact_lowest_energy(n, strategy):
    """非相互作用三重态至少在两个轨道上占据，默认初猜不能返回双占单重态。"""
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal GPU unavailable")
    from tyxonq.applications.chem.algorithms.sqd.gpu.api import solve_sci_gpu

    labels = tuple(cistring.make_strings(range(n), 1) for _ in range(2))
    h = np.diag(0.2 * np.arange(n))
    result, report = solve_sci_gpu(labels, h, np.zeros((n,) * 4), n, (1, 1),
                                   backend="metal", spin_sq=2., spin_shift=10.)
    assert report["resolved_backend"] == "metal"
    assert report["target_search_strategy"] == strategy
    assert report["target_spin_verified"]
    assert report["target_spin_residual"] <= 1e-4
    assert result.energy == pytest.approx(0.2, abs=1e-5)
    assert result.spin_square() == pytest.approx(2.0, abs=1e-5)


def test_metal_rejects_space_without_pure_triplet():
    """只有 a0 b1，缺少 a1 b0，不能用它表示纯三重态。"""
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal GPU unavailable")
    from tyxonq.applications.chem.algorithms.sqd.gpu.api import solve_sci_gpu

    with pytest.raises(RuntimeError, match="no state with the requested target spin"):
        solve_sci_gpu((np.array([1]), np.array([2])), np.diag([0., .2]),
                      np.zeros((2,) * 4), 2, (1, 1), backend="metal", spin_sq=2.)


def test_correct_spin_mean_can_hide_mixed_total_spins():
    """单重态与五重态混合的期望可恰等于三重态，但物理残差不为零。"""
    n, electrons = 4, (2, 2)
    labels = tuple(cistring.make_strings(range(n), e) for e in electrons)
    size = len(labels[0]) * len(labels[1])
    s2 = np.column_stack([spin_op.contract_ss(c.reshape(6, 6), n, electrons).ravel()
                          for c in np.eye(size)])
    _, eigenvectors = np.linalg.eigh(s2)
    mixed = np.sqrt(2/3) * eigenvectors[:, 0] + np.sqrt(1/3) * eigenvectors[:, -1]
    assert mixed @ s2 @ mixed == pytest.approx(2.0)
    assert full_spin_residual(mixed, _spin_blocks(labels), 0., 2.) == pytest.approx(np.sqrt(8.))


def test_degenerate_target_roots_are_identified_by_their_joint_subspace():
    """两个最低三重态简并时，检验对简并子空间的总重叠。"""
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal GPU unavailable")
    from tyxonq.applications.chem.algorithms.sqd.gpu.api import solve_sci_gpu

    n, electrons = 3, (1, 1)
    labels = tuple(cistring.make_strings(range(n), 1) for _ in range(2))
    result, _ = solve_sci_gpu(labels, np.diag([0., .2, .2]), np.zeros((n,) * 4),
                              n, electrons, backend="metal", spin_sq=2.)
    v = result.sci_state.amplitudes
    overlap = sum((v[0, p] - v[p, 0])**2 / 2 for p in (1, 2))
    assert result.energy == pytest.approx(.2)
    assert overlap == pytest.approx(1., abs=1e-6)


@pytest.mark.parametrize("electrons", [(1, 1), (2, 0)])
def test_metal_hamiltonian_scalar_shift_covariance(electrons):
    """固定 N 下 h→h+kI 只令能量移动 Nk，系数、密度和原 Hamiltonian 残差应保持。"""
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal GPU unavailable")
    from tyxonq.applications.chem.algorithms.sqd.gpu.api import solve_sci_gpu

    n = 3
    labels = tuple(cistring.make_strings(range(n), e) for e in electrons)
    h = np.array([[-1.02, .08, .03], [.08, -.32, -.06], [.03, -.06, .2]])
    eri = np.zeros((n,) * 4)
    kwargs = dict(backend="metal", tolerance=1e-6, operator_residual_tolerance=4e-6)
    base, ref = solve_sci_gpu(labels, h, eri, n, electrons, **kwargs)
    shift = 1.0e4
    translated, measured = solve_sci_gpu(labels, h + shift*np.eye(n), eri, n, electrons, **kwargs)
    delta = sum(electrons) * shift
    assert translated.energy - base.energy == pytest.approx(delta, abs=1e-7)
    assert measured["physical_rayleigh_energy_hartree"] - ref["physical_rayleigh_energy_hartree"] == pytest.approx(delta, abs=2e-6)
    assert measured["hamiltonian_energy_offset_hartree"] - ref["hamiltonian_energy_offset_hartree"] == pytest.approx(delta)
    assert measured["pyscf_operator_residual_norm"] <= 4e-6
    np.testing.assert_allclose(translated.rdm1, base.rdm1, atol=2e-6, rtol=0)
    np.testing.assert_allclose(translated.rdm2, base.rdm2, atol=2e-6, rtol=0)


@pytest.mark.parametrize("high", [False, True])
def test_unconstrained_root_obeys_determinant_variational_bound(high):
    """两个互不耦合的行列式都是精确根；双占高能轨道必须触发定向重试。"""
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal GPU unavailable")
    from tyxonq.applications.chem.algorithms.sqd.fermion import SCIState
    from tyxonq.applications.chem.algorithms.sqd.gpu.api import solve_sci_gpu

    labels = np.array([1, 2]), np.array([1, 2])
    v = np.zeros((2, 2))
    v[int(high), int(high)] = 1.
    initial = SCIState(v, *labels, 2, (1,1))
    # 同时检查报告使用原始物理能量，判断在平移后数值尺度内进行。
    result, report = solve_sci_gpu(labels, np.diag([10., 11.]), np.zeros((2,)*4),
                                   2, (1,1), backend="metal", initial_state=initial)
    assert result.energy == pytest.approx(20., abs=1e-7)
    assert report["determinant_upper_bound_hartree"] == pytest.approx(20.)
    assert report["first_candidate_rayleigh_hartree"] == pytest.approx(22. if high else 20.)
    assert report["determinant_bound_labels"] == [1, 1]
    assert report["initial_guess_kind"] == "projected_explicit"
    assert report["determinant_bound_retries"] == int(high)
    assert report["retry_davidson_h_matvecs"] == int(high)
    assert report["retry_davidson_iterations"] == int(high)
    assert (report["retry_seconds"] > 0) == high
    if high:
        assert report["initial_guess_dominant_labels"] == [2, 2]
        assert report["initial_guess_dominant_weight"] == pytest.approx(1.)


def test_unconstrained_root_rejects_second_bound_violation(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal GPU unavailable")
    from tyxonq.applications.chem.algorithms.sqd.fermion import SCIState
    from tyxonq.applications.chem.algorithms.sqd.gpu.api import solve_sci_gpu
    from tyxonq.numerics.sbd import metal

    davidson = metal.davidson_lowest
    def always_high(hc, diagonal, **kwargs):
        kwargs["initial_vector"] = mx.array([0., 0., 0., 1.])
        return davidson(hc, diagonal, **kwargs)

    monkeypatch.setattr(metal, "davidson_lowest", always_high)
    labels = np.array([1,2]), np.array([1,2])
    initial = SCIState(np.array([[0., 0.], [0., 1.]]), *labels, 2, (1,1))
    with pytest.raises(RuntimeError, match="variational upper bound.*retries=1"):
        solve_sci_gpu(labels, np.diag([0., 1.]), np.zeros((2,)*4), 2, (1,1),
                      backend="metal", initial_state=initial)
