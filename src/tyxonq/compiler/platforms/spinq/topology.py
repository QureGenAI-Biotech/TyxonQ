"""SpinQ 平台能力与拓扑的只读发现，不在本模块执行路由。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ....devices.base import DeviceRule

_PLATFORM_QUBITS = {
    "gemini_vp": 2,
    "triangulum_vp": 3,
    "superconductor_vp": 8,
    "sqc_25_vp": 25,
}


def _normalize_coupling_map(raw: Any) -> List[List[int]]:
    if not isinstance(raw, (list, tuple)):
        return []
    edges = set()
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            q0, q1 = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if q0 == q1 or q0 < 0 or q1 < 0:
            continue
        edges.add((min(q0, q1), max(q0, q1)))
    normalized: List[List[int]] = []
    for q0, q1 in sorted(edges):
        normalized.append([q0, q1])
        normalized.append([q1, q0])
    return normalized


def _platform_config(platform: Any) -> Dict[str, Any]:
    config = getattr(platform, "config", {})
    if callable(config):
        config = config()
    return config if isinstance(config, dict) else {}


def get_topology(
    device: str, token: Optional[str] = None, **opts: Any
) -> Optional["DeviceRule"]:
    """读取 SpinQ 平台能力；官方未声明 coupling_map 时返回空列表。"""
    from ....devices.hardware.spinq.driver import (
        _create_backend,
        _device_name,
        _resolve_credentials,
    )

    name = _device_name(device)
    if name not in _PLATFORM_QUBITS:
        raise ValueError(f"未知 SpinQ 平台: {name}")
    credentials = _resolve_credentials(dict(opts))
    backend = _create_backend(**credentials)
    platform = backend.get_platform(name)
    available = platform.available() if callable(getattr(platform, "available", None)) else None
    config = _platform_config(platform)
    topology = config.get("topology") if isinstance(config.get("topology"), dict) else {}
    coupling_map = _normalize_coupling_map(
        getattr(platform, "coupling_map", None) or topology.get("coupling_map")
    )
    num_qubits = (
        getattr(platform, "max_bitnum", None)
        or config.get("qubits")
        or config.get("num_qubits")
        or _PLATFORM_QUBITS[name]
    )
    native_gates = config.get("native_gates") or config.get("basis_gates")
    active_qubits = getattr(platform, "active_qubits", None)
    return {
        "provider": "spinq",
        "device": name,
        "num_qubits": int(num_qubits),
        "coupling_map": coupling_map,
        "native_gates": list(native_gates) if isinstance(native_gates, (list, tuple)) else None,
        "status": "active" if available else "unavailable",
        "raw": {
            "machine_count": getattr(platform, "machine_count", None),
            "available": available,
            "active_qubits": sorted(active_qubits) if active_qubits is not None else None,
            "display_name": getattr(platform, "name", None),
        },
    }


def supports_runtime_layout(device: str) -> bool:
    """SQC-25 云端支持 keep_layout；Gemini/Triangulum 不支持自定义映射。"""
    from ....devices.hardware.spinq.driver import _device_name

    return _device_name(device) == "sqc_25_vp"


__all__ = ["get_topology", "supports_runtime_layout"]
