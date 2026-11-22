"""Local accelerator discovery, synchronization and deterministic selection.

This module concerns the machine's CUDA, Metal and CPU execution capabilities;
external hardware-platform integrations remain outside ``tyxonq.numerics``.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

from .backends.base import BackendUnavailableError, MemoryInfo

DeviceKind = str  # "cuda" | "mps" | "metal" | "cpu"


@dataclass(frozen=True)
class BackendResolution:
    """Environment-aware SQD selection with visible unavailable-hardware fallback."""

    requested: str
    resolved: str
    device: str
    selection_reason: str
    precision_mode: str
    rdm_backend: str
    used_cpu_fallback: bool = False


@dataclass(frozen=True)
class DeviceSpec:
    kind: DeviceKind
    index: int | None = None
    backend: str = "pytorch"

    def __str__(self) -> str:
        if self.kind == "cpu":
            return "cpu"
        return f"{self.kind}:{self.index if self.index is not None else 0}"


def _torch() -> Any:
    return importlib.import_module("torch")


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def cuda_available() -> bool:
    try:
        torch = _torch()
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def mps_available() -> bool:
    try:
        torch = _torch()
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def metal_available() -> bool:
    """Return whether Apple Metal and the required MLX runtime are usable."""
    if not is_apple_silicon():
        return False
    try:
        mx = importlib.import_module("mlx.core")
        return bool(mx.metal.is_available())
    except Exception:
        return False


def sbd_available() -> bool:
    """Return whether CUDA hardware and the explicitly built SBD extension are usable."""
    if not cuda_available():
        return False
    try:
        from tyxonq.numerics.sbd.cuda import is_available
        return bool(is_available())
    except Exception:
        return False


def _cpu_fallback(request: str, reason: str) -> BackendResolution:
    try:
        _torch()
    except Exception as error:
        raise BackendUnavailableError("PyTorch CPU fallback requires torch") from error
    message = f"{reason}; falling back to PyTorch CPU"
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    return BackendResolution(request, "pytorch", "cpu", message, "fp64", "cpu", True)


def resolve_sqd_backend(request: str = "auto", device: str | None = None) -> BackendResolution:
    """Resolve ``gpu``/``auto`` by CUDA SBD, Metal SBD, then PyTorch CPU.

    An explicitly requested accelerator that is unavailable falls back visibly to
    PyTorch CPU. Solver and numerical failures after dispatch are never caught here.
    """
    requested = (request or "auto").strip().lower()
    normalized = {"cpu": "pytorch", "cuda": "sbd", "gpu": "auto"}.get(requested, requested)
    if normalized == "auto":
        if sbd_available():
            normalized = "sbd"
            reason = "CUDA hardware and native SBD extension are available (highest priority)"
        elif metal_available():
            normalized = "metal"
            reason = "CUDA SBD is unavailable and the MLX Metal runtime is available"
        else:
            normalized = "pytorch"
            reason = "CUDA SBD and MLX Metal are unavailable"
    else:
        reason = f"explicit backend request: {normalized}"
    if normalized == "sbd":
        if not sbd_available() or (device is not None and not device.startswith("cuda")):
            return _cpu_fallback(request or "auto", "requested CUDA SBD/device is unavailable")
        return BackendResolution(request or "auto", "sbd", device or "cuda:0", reason, "fp64", "sbd")
    if normalized == "metal":
        if not metal_available() or (device is not None and not device.startswith("metal")):
            return _cpu_fallback(request or "auto", "requested MLX Metal/device is unavailable")
        return BackendResolution(request or "auto", "metal", device or "metal:0", reason, "mixed", "cpu")
    if normalized == "pytorch":
        try:
            _torch()
        except Exception as error:
            raise BackendUnavailableError("backend 'pytorch' requires torch") from error
        if device not in (None, "cpu"):
            raise BackendUnavailableError("SQD PyTorch fallback supports only the CPU device")
        return BackendResolution(request or "auto", "pytorch", "cpu", reason, "fp64", "cpu")
    raise ValueError(f"Unknown SQD backend: {request!r}")


def select_backend(request: str | None = None) -> str:
    """Compatibility wrapper returning only the resolved backend name."""
    return resolve_sqd_backend(request or "auto").resolved


class DeviceAdapter(Protocol):
    """Uniform device operations used by solver and validation code."""

    def resolve(self, device_text: str | None, backend: str) -> DeviceSpec: ...
    def synchronize(self, spec: DeviceSpec) -> None: ...
    def memory_info(self, spec: DeviceSpec) -> MemoryInfo: ...
    def empty_cache(self, spec: DeviceSpec) -> None: ...
    def visible_devices_env(self, spec: DeviceSpec) -> dict[str, str]: ...


class TorchDeviceAdapter:
    """Adapter for torch CUDA/MPS/CPU devices."""

    def __init__(self, kind: DeviceKind) -> None:
        self.kind = kind

    def resolve(self, device_text: str | None, backend: str) -> DeviceSpec:
        torch = _torch()
        if self.kind == "cuda":
            if not cuda_available():
                raise BackendUnavailableError("CUDA device requested but unavailable")
            index = 0 if not device_text or device_text == "cuda" else torch.device(device_text).index
            return DeviceSpec("cuda", index, backend)
        if self.kind == "mps":
            if not mps_available():
                raise BackendUnavailableError("MPS device requested but unavailable")
            return DeviceSpec("mps", 0, backend)
        return DeviceSpec("cpu", None, backend)

    def synchronize(self, spec: DeviceSpec) -> None:
        torch = _torch()
        if spec.kind == "cuda":
            torch.cuda.synchronize(spec.index)
        elif spec.kind == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def memory_info(self, spec: DeviceSpec) -> MemoryInfo:
        torch = _torch()
        if spec.kind == "cuda":
            free, total = torch.cuda.mem_get_info(spec.index)
            return MemoryInfo(
                int(free), int(total), int(torch.cuda.memory_allocated(spec.index)),
                int(torch.cuda.memory_reserved(spec.index)),
            )
        if spec.kind == "mps" and hasattr(torch.mps, "recommended_max_memory"):
            total = int(torch.mps.recommended_max_memory())
            allocated = int(torch.mps.current_allocated_memory())
            return MemoryInfo(total - allocated, total, allocated, allocated)
        return MemoryInfo(0, 0)

    def empty_cache(self, spec: DeviceSpec) -> None:
        torch = _torch()
        if spec.kind == "cuda":
            torch.cuda.empty_cache()
        elif spec.kind == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def visible_devices_env(self, spec: DeviceSpec) -> dict[str, str]:
        if spec.kind == "cuda":
            return {"CUDA_VISIBLE_DEVICES": str(spec.index if spec.index is not None else 0)}
        return {}


class MetalDeviceAdapter(TorchDeviceAdapter):
    """Metal native kernels use the MPS allocator view for host-side tensors."""

    def __init__(self) -> None:
        super().__init__("mps")

    def resolve(self, device_text: str | None, backend: str) -> DeviceSpec:
        return DeviceSpec("metal", 0, backend)

    def visible_devices_env(self, spec: DeviceSpec) -> dict[str, str]:
        return {}


class CpuDeviceAdapter:
    def resolve(self, device_text: str | None, backend: str) -> DeviceSpec:
        return DeviceSpec("cpu", None, backend)

    def synchronize(self, spec: DeviceSpec) -> None:
        return None

    def memory_info(self, spec: DeviceSpec) -> MemoryInfo:
        return MemoryInfo(0, 0)

    def empty_cache(self, spec: DeviceSpec) -> None:
        return None

    def visible_devices_env(self, spec: DeviceSpec) -> dict[str, str]:
        return {}


def get_adapter(backend: str) -> DeviceAdapter:
    if backend == "sbd":
        return TorchDeviceAdapter("cuda")
    if backend == "metal":
        return MetalDeviceAdapter()
    if backend == "cupy":
        return TorchDeviceAdapter("cuda")
    if backend == "pytorch":
        return CpuDeviceAdapter()
    raise ValueError(f"Unknown backend for device adapter: {backend!r}")


def resolve_device(device_text: str | None, backend: str) -> DeviceSpec:
    return get_adapter(backend).resolve(device_text, backend)


def synchronize(spec: DeviceSpec) -> None:
    get_adapter(spec.backend).synchronize(spec)


def memory_info(spec: DeviceSpec) -> MemoryInfo:
    return get_adapter(spec.backend).memory_info(spec)


def empty_cache(spec: DeviceSpec) -> None:
    get_adapter(spec.backend).empty_cache(spec)


def visible_devices_env(spec: DeviceSpec) -> dict[str, str]:
    return get_adapter(spec.backend).visible_devices_env(spec)


__all__ = [
    "BackendResolution",
    "CpuDeviceAdapter",
    "DeviceAdapter",
    "DeviceKind",
    "DeviceSpec",
    "MetalDeviceAdapter",
    "TorchDeviceAdapter",
    "cuda_available",
    "empty_cache",
    "get_adapter",
    "is_apple_silicon",
    "memory_info",
    "mps_available",
    "metal_available",
    "resolve_device",
    "resolve_sqd_backend",
    "sbd_available",
    "select_backend",
    "synchronize",
    "visible_devices_env",
]
