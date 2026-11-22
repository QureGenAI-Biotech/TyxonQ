from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from tyxonq.applications.chem.algorithms.sqd.gpu.api import _default_device
from tyxonq.applications.chem.algorithms.sqd.gpu.solver import (
    davidson_space_for_memory, wait_for_davidson_memory,
)
from tyxonq.numerics.accelerator import DeviceSpec, resolve_device, resolve_sqd_backend, select_backend


def test_default_device_follows_backend(monkeypatch):
    from tyxonq.numerics import accelerator as device_module

    monkeypatch.setattr(device_module, "cuda_available", lambda: False)
    monkeypatch.setattr(device_module, "mps_available", lambda: False)
    assert _default_device("sbd") == "cuda:0"
    assert _default_device("metal") == "metal:0"
    assert _default_device("pytorch") == "cpu"


def test_resolve_device_uses_selected_backend(monkeypatch):
    from tyxonq.numerics import accelerator as device_module

    monkeypatch.setattr(device_module, "cuda_available", lambda: True)
    monkeypatch.setattr(device_module, "sbd_available", lambda: True)
    monkeypatch.setattr(device_module, "mps_available", lambda: False)
    assert resolve_device("cuda:0", "sbd").kind == "cuda"
    assert select_backend("cuda") == "sbd"


def test_device_spec_is_hashable():
    assert {DeviceSpec("cuda", 0), DeviceSpec("cuda", 0)} == {DeviceSpec("cuda", 0)}


def test_cpu_skips_gpu_memory_wait_and_dtype_budget_is_explicit():
    import torch

    assert wait_for_davidson_memory(torch.device("cpu"), 12, 96, None) == (12, 0)
    assert davidson_space_for_memory(80, 400, 2**40, element_size=4) == 80
    assert davidson_space_for_memory(80, 400, 2**40, element_size=8) == 50


def test_gpu_cpu_aliases_follow_runtime_environment(monkeypatch):
    from tyxonq.numerics import accelerator

    monkeypatch.setattr(accelerator, "sbd_available", lambda: False)
    monkeypatch.setattr(accelerator, "metal_available", lambda: False)
    assert select_backend("gpu") == "pytorch"
    assert select_backend("cpu") == "pytorch"
    assert not resolve_sqd_backend("gpu").used_cpu_fallback

    monkeypatch.setattr(accelerator, "metal_available", lambda: True)
    assert select_backend("gpu") == "metal"
    monkeypatch.setattr(accelerator, "sbd_available", lambda: True)
    assert select_backend("gpu") == "sbd"


@pytest.mark.parametrize("backend", ["metal", "cuda"])
def test_explicit_unavailable_accelerator_visibly_falls_back_to_cpu(monkeypatch, backend):
    from tyxonq.numerics import accelerator

    monkeypatch.setattr(accelerator, "sbd_available", lambda: False)
    monkeypatch.setattr(accelerator, "metal_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="falling back to PyTorch CPU"):
        resolution = resolve_sqd_backend(backend)
    assert resolution.resolved == "pytorch"
    assert resolution.device == "cpu"
    assert resolution.used_cpu_fallback

