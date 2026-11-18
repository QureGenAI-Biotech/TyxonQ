from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from tyxonq.applications.chem.algorithms.sqd.gpu import api
from tyxonq.numerics.accelerator import BackendResolution


def _resolution(backend):
    return BackendResolution("auto", backend, api._default_device(backend), "test", "fp64", "cpu")


def _fake_solve(backend):
    def solve(ci, h1e, eri, norb, nelec, **options):
        return "result", {"chosen": backend, "options": options}
    return solve


def _solve_with(monkeypatch, backend, target):
    monkeypatch.setattr(api, "resolve_sqd_backend", lambda request="auto", device=None: _resolution(backend))
    monkeypatch.setattr(target, _fake_solve(backend))
    return api.solve_sci_gpu(
        (np.array([1]), np.array([1])), np.zeros((1, 1)),
        np.zeros((1, 1, 1, 1)), 1, (1, 1),
    )


def test_solve_sci_gpu_dispatches_to_sbd(monkeypatch):
    result, report = _solve_with(
        monkeypatch, "sbd",
        "tyxonq.applications.chem.algorithms.sqd.gpu.sbd_solver.solve_sci_sbd",
    )
    assert result == "result"
    assert report["backend"] == "sbd"
    assert report["options"]["device_text"] == "cuda:0"
    assert report["resolved_backend"] == "sbd"


def test_solve_sci_gpu_dispatches_to_metal(monkeypatch):
    result, report = _solve_with(
        monkeypatch, "metal",
        "tyxonq.applications.chem.algorithms.sqd.gpu.sbd_solver.solve_sci_metal",
    )
    assert result == "result"
    assert report["backend"] == "metal"
    assert report["options"]["device_text"] == "metal:0"
    assert "metal_hot_kernels" not in report


def test_solve_sci_gpu_dispatches_to_pytorch_cpu(monkeypatch):
    result, report = _solve_with(
        monkeypatch, "pytorch",
        "tyxonq.applications.chem.algorithms.sqd.gpu.solver.solve_sci_cuda",
    )
    assert result == "result"
    assert report["backend"] == "pytorch"
    assert report["options"]["device_text"] == "cpu"


def test_metal_failure_never_dispatches_to_pytorch(monkeypatch):
    monkeypatch.setattr(api, "resolve_sqd_backend", lambda request="auto", device=None: _resolution("metal"))

    def fail(*args, **kwargs):
        raise RuntimeError("native Metal failure")

    def forbidden(*args, **kwargs):
        raise AssertionError("PyTorch must not handle Metal failures")

    monkeypatch.setattr("tyxonq.applications.chem.algorithms.sqd.gpu.sbd_solver.solve_sci_metal", fail)
    monkeypatch.setattr("tyxonq.applications.chem.algorithms.sqd.gpu.solver.solve_sci_cuda", forbidden)
    with pytest.raises(RuntimeError, match="native Metal failure"):
        api.solve_sci_gpu(
            (np.array([1]), np.array([1])), np.zeros((1, 1)),
            np.zeros((1, 1, 1, 1)), 1, (1, 1),
        )


def test_unavailable_hardware_cpu_fallback_is_explicit_in_report(monkeypatch):
    resolution = BackendResolution("cuda", "pytorch", "cpu", "test fallback", "fp64", "cpu", True)
    monkeypatch.setattr(api, "resolve_sqd_backend", lambda request="auto", device=None: resolution)
    monkeypatch.setattr(
        "tyxonq.applications.chem.algorithms.sqd.gpu.solver.solve_sci_cuda", _fake_solve("pytorch"))
    _, report = api.solve_sci_gpu(
        (np.array([1]), np.array([1])), np.zeros((1, 1)),
        np.zeros((1, 1, 1, 1)), 1, (1, 1), backend="cuda",
    )
    assert report["requested_backend"] == "cuda"
    assert report["resolved_backend"] == "pytorch"
    assert report["used_cpu_fallback"]


def test_pytorch_cpu_path_runs_real_end_to_end_solve():
    from pyscf.fci import cistring

    norb, nelec = 2, (1, 1)
    strings = tuple(cistring.make_strings(range(norb), n) for n in nelec)
    h = np.array([[0.0, 0.4], [0.4, -0.2]])
    factors = np.array([[[1.0, 0.2], [0.2, 0.7]]])
    eri = np.einsum("lpq,lrs->pqrs", factors, factors)
    result, report = api.solve_sci_gpu(strings, h, eri, norb, nelec, backend="pytorch")
    assert np.isfinite(result.energy)
    assert report["resolved_backend"] == "pytorch"
    assert report["device"] == "cpu"
    assert report["solver_backend"] == "pytorch_cpu_fp64_davidson_cpu_validation"
    assert report["gpu_fresh_residual_norm_device"] < 1e-7
    assert report["available_device_bytes_at_setup"] == 0

