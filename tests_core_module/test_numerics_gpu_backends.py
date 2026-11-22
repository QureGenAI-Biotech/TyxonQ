from __future__ import annotations

import pytest

from tyxonq.numerics import DeviceSpec, get_backend, select_backend
from tyxonq.numerics.backends.base import BackendUnavailableError
from tyxonq.numerics import accelerator as device_module


def test_device_spec_string():
    assert str(DeviceSpec("cpu")) == "cpu"
    assert str(DeviceSpec("cuda", 3, "sbd")) == "cuda:3"


def test_select_backend_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(device_module, "sbd_available", lambda: True)
    monkeypatch.setattr(device_module, "metal_available", lambda: True)
    resolution = device_module.resolve_sqd_backend("auto")
    assert resolution.resolved == "sbd"
    assert resolution.device == "cuda:0"
    assert select_backend("auto") == "sbd"


def test_select_backend_auto_uses_metal_without_cuda(monkeypatch):
    monkeypatch.setattr(device_module, "cuda_available", lambda: False)
    monkeypatch.setattr(device_module, "metal_available", lambda: True)
    resolution = device_module.resolve_sqd_backend("auto")
    assert resolution.resolved == "metal"
    assert resolution.device == "metal:0"
    assert resolution.precision_mode == "mixed"


def test_select_backend_auto_uses_pytorch_cpu_without_accelerators(monkeypatch):
    monkeypatch.setattr(device_module, "cuda_available", lambda: False)
    monkeypatch.setattr(device_module, "metal_available", lambda: False)
    resolution = device_module.resolve_sqd_backend("auto")
    assert resolution.resolved == "pytorch"
    assert resolution.device == "cpu"
    assert resolution.precision_mode == "fp64"


def test_cuda_alias_and_explicit_unavailable_fallback(monkeypatch):
    monkeypatch.setattr(device_module, "sbd_available", lambda: True)
    assert select_backend("cuda") == "sbd"
    monkeypatch.setattr(device_module, "sbd_available", lambda: False)
    monkeypatch.setattr(device_module, "metal_available", lambda: False)
    for backend in ("sbd", "metal"):
        with pytest.warns(RuntimeWarning, match="falling back to PyTorch CPU"):
            resolution = device_module.resolve_sqd_backend(backend)
        assert resolution.resolved == "pytorch"
        assert resolution.used_cpu_fallback


def test_pytorch_sqd_device_is_cpu_only(monkeypatch):
    with pytest.raises(BackendUnavailableError, match="only the CPU"):
        device_module.resolve_sqd_backend("pytorch", "cuda:0")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        select_backend("cupynumeric")
    with pytest.raises(ValueError):
        get_backend("cupynumeric")


def test_general_auto_array_uses_metal_without_cuda(monkeypatch):
    pytest.importorskip("mlx.core")
    monkeypatch.setattr(device_module, "cuda_available", lambda: False)
    monkeypatch.setattr(device_module, "metal_available", lambda: True)
    backend = get_backend("auto")
    assert backend.name == "metal"
    assert backend.device_name == "metal:0"


def test_general_auto_array_uses_pytorch_cpu_without_accelerators(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(device_module, "cuda_available", lambda: False)
    monkeypatch.setattr(device_module, "metal_available", lambda: False)
    backend = get_backend("auto")
    assert backend.name == "pytorch"
    assert str(backend.device) == "cpu"


def test_general_auto_cuda_dependency_error_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(device_module, "cuda_available", lambda: True)
    monkeypatch.setattr(device_module, "metal_available", lambda: True)
    with pytest.raises(RuntimeError, match="cupy"):
        get_backend("auto")
