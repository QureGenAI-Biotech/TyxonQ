from __future__ import annotations

from functools import wraps

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from pyscf.fci import cistring, direct_spin1, selected_ci, spin_op

from tyxonq.numerics.sbd.metal import (
    MetalSelectedCIContractor, MetalSelectedS2Contractor, absorb_h1e,
)

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU is unavailable")


def _cpu_default(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            return function(*args, **kwargs)
        finally:
            mx.set_default_device(previous)
    return wrapped


def _assert_gpu_execution(execution):
    assert execution["device"] == "Device(gpu, 0)"
    assert "Device(gpu, 0)" in execution["stream"]
    assert execution["completed_evaluations"] > 0


def _problem(nelec, partial=False, seed=123):
    rng = np.random.default_rng(seed)
    norb = 4
    h = rng.normal(size=(norb, norb))
    h = (h + h.T) / 2
    factors = rng.normal(size=(5, norb, norb))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eri = np.einsum("lpq,lrs->pqrs", factors, factors)
    strings = tuple(cistring.make_strings(range(norb), n) for n in nelec)
    if partial:
        strings = tuple(s[::2] for s in strings)
    coefficients = rng.normal(size=tuple(map(len, strings)))
    coefficients /= np.linalg.norm(coefficients)
    return h, eri, strings, coefficients


@pytest.mark.parametrize("nelec", [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1)])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("strategy", ["pair", "connected"])
@_cpu_default
def test_native_metal_hamiltonian_and_spin_match_cpu_fp64(nelec, partial, strategy):
    h, eri, strings, coefficients = _problem(nelec, partial)
    norb = h.shape[0]
    # PySCF 的 C 接口按连续内存解释标签，切片必须在跨库边界时转为连续。
    links = selected_ci._all_linkstr_index(tuple(np.ascontiguousarray(x) for x in strings), norb, nelec)
    contractor = MetalSelectedCIContractor(absorb_h1e(h, eri, nelec), links, nelec, chunk_size=2, strategy=strategy)
    spin = MetalSelectedS2Contractor(strings, norb, nelec)
    actual = contractor.contract(mx.array(coefficients.astype(np.float32)))
    actual_spin = spin.contract(mx.array(coefficients.astype(np.float32)))
    mx.eval(actual, actual_spin)
    full = tuple(cistring.make_strings(range(norb), n) for n in nelec)
    selected = np.ix_(*(np.searchsorted(all_labels, subset) for all_labels, subset in zip(full, strings)))
    vector = np.zeros(tuple(map(len, full)))
    vector[selected] = coefficients
    expected = direct_spin1.contract_2e(
        direct_spin1.absorb_h1e(h, eri, norb, nelec, 0.5), vector, norb, nelec)[selected]
    expected_spin = spin_op.contract_ss(vector, norb, nelec)[selected]
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(actual_spin), expected_spin, atol=2e-6, rtol=2e-6)
    if strategy == "pair":
        assert contractor.native_calls["metal_pair_contract"] > 0
    else:
        assert contractor.native_calls["metal_connected_rows"] > 0
    assert spin.native_calls["metal_spin_exchange"] > 0
    _assert_gpu_execution(contractor.execution)
    _assert_gpu_execution(spin.execution)
    assert mx.default_device() == mx.cpu


@_cpu_default
def test_end_to_end_metal_sbd_energy_residual_and_native_calls():
    from tyxonq.applications.chem.algorithms.sqd.gpu.api import solve_sci_gpu

    h, eri, strings, _ = _problem((2, 2), seed=4)
    result, report = solve_sci_gpu(
        strings, h, eri, h.shape[0], (2, 2), backend="metal",
        max_space=12, tolerance=2e-5, operator_residual_tolerance=1e-4,
        rdm_energy_tolerance=1e-4,
    )
    assert np.isfinite(result.energy)
    assert report["resolved_backend"] == "metal"
    assert report["precision_mode"] == "mixed"
    assert report["rdm_backend"] == "cpu"
    assert report["gpu_fresh_residual_norm_device"] < 1e-4
    assert report["operator_image_max_abs_error"] < 1e-4
    assert report["spin_square_rdm_abs_error"] < 2e-5
    assert report["native_metal_calls"]["metal_connected_rows"] > 0
    assert report["native_metal_calls"]["spin_metal_spin_exchange"] > 0
    _assert_gpu_execution(report["native_metal_execution"]["hamiltonian"])
    _assert_gpu_execution(report["native_metal_execution"]["spin"])
    assert mx.default_device() == mx.cpu


def test_metal_spin_constraint_and_empty_spin_sector():
    from tyxonq.applications.chem.algorithms.sqd.gpu.api import solve_sci_gpu

    h, eri, strings, _ = _problem((1, 1), seed=9)
    constrained, spin_report = solve_sci_gpu(
        strings, h, eri, h.shape[0], (1, 1), backend="metal", spin_sq=0.0,
        spin_shift=10.0, max_space=16, tolerance=2e-5,
        operator_residual_tolerance=1e-4, rdm_energy_tolerance=1e-4,
    )
    assert constrained.spin_square() == pytest.approx(0.0, abs=1e-3)
    assert spin_report["spin_constraint_applied"]
    assert spin_report["spin_backend"] == "metal"

    h, eri, strings, _ = _problem((2, 0), seed=10)
    empty, empty_report = solve_sci_gpu(
        strings, h, eri, h.shape[0], (2, 0), backend="metal",
        max_space=8, tolerance=2e-5, operator_residual_tolerance=1e-4,
        rdm_energy_tolerance=1e-4,
    )
    assert empty.spin_square() == pytest.approx(2.0, abs=2e-5)
    assert empty_report["hamiltonian_uses_zero_coupling_spectator"]


def test_metal_single_determinant():
    """只有 HF 行列式时，一次矩阵作用即可返回解析本征对。"""
    from tyxonq.applications.chem.algorithms.sqd.gpu.api import solve_sci_gpu

    labels = (np.array([1]), np.array([1]))
    result, report = solve_sci_gpu(labels, np.diag([-1., 1.]), np.zeros((2,) * 4),
                                   2, (1, 1), backend="metal")
    assert result.energy == pytest.approx(-2.)
    assert report["gpu_iterations"] == 1


@_cpu_default
def test_connected_integral_update_matches_independent_projection():
    nelec = (2, 1)
    h, eri, labels, ci = _problem(nelec, partial=True)
    norb = h.shape[0]
    links = selected_ci._all_linkstr_index(tuple(np.ascontiguousarray(s) for s in labels), norb, nelec)
    op = MetalSelectedCIContractor(absorb_h1e(h, eri, nelec), links, nelec, strategy="connected")
    h, eri, _, _ = _problem(nelec, partial=True, seed=7)
    op.update_integrals(absorb_h1e(h, eri, nelec))
    full = tuple(cistring.make_strings(range(norb), n) for n in nelec)
    subset = np.ix_(*(np.searchsorted(all_s, kept) for all_s, kept in zip(full, labels)))
    extended = np.zeros(tuple(map(len, full)))
    extended[subset] = ci
    expected = direct_spin1.contract_2e(direct_spin1.absorb_h1e(h, eri, norb, nelec, .5),
                                       extended, norb, nelec)[subset]
    actual = op.contract(mx.array(ci.astype(np.float32)))
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-5, rtol=2e-5)


@_cpu_default
def test_large_davidson_after_thick_restart_matches_secular_equation():
    """用可独立求解的负秩一更新检验长向量归约及厚重启。"""
    from scipy.optimize import brentq
    from tyxonq.numerics.sbd.metal.solver import davidson_lowest

    n = 32771
    d = np.linspace(0.5, 2., n)
    u = np.random.default_rng(713).normal(size=n)
    u /= np.linalg.norm(u)
    weight = 2.
    # H=D−wuuᵀ 的最低特征值在 D 的最小对角元之下。
    expected = brentq(lambda e: 1 - weight * np.sum(u*u/(d-e)), -4., .499999)
    diag = mx.array(d.astype(np.float32))
    move = mx.array(u.astype(np.float32))
    op = lambda x: diag*x - weight*move*mx.sum(move*x)
    initial = move + 0.3 * mx.ones_like(move) / np.sqrt(n)
    result = davidson_lowest(op, diag-weight*move**2, initial_vector=initial,
                             max_cycles=100, max_space=3, tolerance=2e-6)
    assert result.converged and result.iterations > 3
    assert result.energy == pytest.approx(expected, abs=3e-6)
    v = np.asarray(result.coefficients, dtype=np.float64)
    assert np.linalg.norm(d*v-weight*u*(u @ v)-result.energy*v) < 3e-6


@pytest.mark.parametrize("nelec", [(2,2), (2,1), (2,0), (0,2)])
@pytest.mark.parametrize("validation", ["gpu", "full_cpu"])
def test_public_metal_q60_rdm_preserves_all_spin_blocks(nelec, validation, monkeypatch):
    from tyxonq.applications.chem.algorithms.sqd.gpu import sbd_solver
    from tyxonq.applications.chem.algorithms.sqd.gpu.api import solve_sci_gpu

    if validation == "gpu":
        def unexpected_cpu_reference(*args, **kwargs):
            pytest.fail("GPU validation unexpectedly requested the CPU mixed density/operator")
        monkeypatch.setattr(sbd_solver, "independent_cpu_validation", unexpected_cpu_reference)
    h, eri, labels, _ = _problem(nelec, partial=True, seed=417)
    n = len(h)
    result, report = solve_sci_gpu(labels, h, eri, n, nelec, backend="metal",
                                   validation=validation, rdm_backend="metal", tolerance=2e-5)
    full_labels = tuple(cistring.make_strings(range(n), count) for count in nelec)
    vec = np.zeros(tuple(map(len,full_labels)))
    selected = np.ix_(*(np.searchsorted(all_s, s) for all_s,s in zip(full_labels,labels)))
    vec[selected] = result.sci_state.amplitudes
    ref1, ref2 = direct_spin1.make_rdm12s(vec,n,nelec)
    for value, expected in zip((*result.rdm1s,*result.rdm2s), (*ref1,*ref2)):
        np.testing.assert_allclose(value, expected, rtol=0, atol=1e-11)
    assert report["rdm_backend"] == "metal"
    assert report["rdm_ab_kernel_calls"] == int(min(nelec) > 0)
    assert report["rdm_ab_cpu_reference"] == (validation == "full_cpu")
    if validation == "full_cpu":
        assert report["rdm2_ab_cpu_max_abs_error"] < 1e-11
    else:
        assert report["cpu_subprocess_seconds"] == 0


def test_q60_limits_and_independent_rejection(monkeypatch):
    from tyxonq.numerics.sbd.metal import rdm
    from tyxonq.applications.chem.algorithms.sqd.gpu.api import solve_sci_gpu

    with pytest.raises(ValueError, match="4096 strings"):
        rdm.check_mixed_rdm_size((4097, 1))
    with pytest.raises(ValueError, match=r"2\*\*23"):
        rdm.check_mixed_rdm_size((4096, 2049))
    with pytest.raises(ValueError, match="normalized"):
        rdm.mixed_rdm_q60(np.full((1,1),2.), 1, (1,1), ())
    h, eri, labels, _ = _problem((1,1), partial=True)
    original = rdm.mixed_rdm_q60

    def corrupted(*args):
        block = original(*args)
        block[0,0,0,0] += 1e-9
        return block

    monkeypatch.setattr(rdm, "mixed_rdm_q60", corrupted)
    with pytest.raises(RuntimeError, match="mixed RDM disagrees"):
        solve_sci_gpu(labels,h,eri,len(h),(1,1),backend="metal",rdm_backend="metal")


@pytest.mark.parametrize("option,value", [("tolerance", np.inf), ("operator_residual_tolerance", np.nan),
                                         ("rdm_energy_tolerance", np.inf)])
def test_metal_rejects_nonfinite_accuracy_limits(option, value):
    from tyxonq.applications.chem.algorithms.sqd.gpu.sbd_solver import solve_sci_metal

    with pytest.raises(ValueError, match="finite and positive"):
        solve_sci_metal((np.array([1]), np.array([1])), np.zeros((1, 1)),
                        np.zeros((1,)*4), 1, (1, 1), **{option: value})
