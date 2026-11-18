from __future__ import annotations

from functools import wraps

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from pyscf.fci import cistring, direct_spin1, selected_ci

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
@_cpu_default
def test_native_metal_hamiltonian_and_spin_match_cpu_fp64(nelec, partial):
    h, eri, strings, coefficients = _problem(nelec, partial)
    norb = h.shape[0]
    links = selected_ci._all_linkstr_index(strings, norb, nelec)
    contractor = MetalSelectedCIContractor(absorb_h1e(h, eri, nelec), links, nelec, chunk_size=2)
    spin = MetalSelectedS2Contractor(strings, norb, nelec)
    actual = contractor.contract(mx.array(coefficients.astype(np.float32)))
    actual_spin = spin.contract(mx.array(coefficients.astype(np.float32)))
    mx.eval(actual, actual_spin)
    vector = selected_ci._as_SCIvector(coefficients, strings)
    expected = selected_ci.contract_2e(
        direct_spin1.absorb_h1e(h, eri, norb, nelec, 0.5), vector, norb, nelec)
    expected_spin = selected_ci.contract_ss(vector, norb, nelec)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(np.asarray(actual_spin), expected_spin, atol=2e-6, rtol=2e-6)
    assert contractor.native_calls["metal_gather"] > 0
    assert contractor.native_calls["metal_pair_contract"] > 0
    if nelec[0] > 1:
        assert contractor.native_calls["same_spin"] > 0
    assert spin.native_calls["metal_gather"] > 0
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
    assert report["native_metal_calls"]["metal_pair_contract"] > 0
    assert report["native_metal_calls"]["spin_metal_gather"] > 0
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
