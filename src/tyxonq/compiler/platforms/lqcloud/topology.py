"""LQCloud 设备拓扑抓取（只读发现，不做路由）。

LQCloud SDK（``lqcloud==0.4.2``）的 backend 只暴露 ``config`` / ``run`` 等方法，
**没有** ``transpile``/``route``，因此布局与 SWAP 插入必须由 TyxonQ 编译器完成；
而编译所需的耦合图来自 ``backend.config['topology']['coupling_map']``。

本模块只负责「问一次设备、拿回拓扑」，缓存与 TTL 策略归
``devices.hardware.topology``（跨 provider 统一），路由归
``gate_compiler.stages.layout.routing``（设备无关核心）。

实测 config 结构（AGate-100，2026-09）::

    {"name": "AGate-100", "qubits": 101, "status": "active", "native_gates": None,
     "topology": {"type": ..., "rows": ..., "cols": ...,
                  "coupling_map": [[0, 1], [1, 2], ...],   # 157 条边
                  "qubit_coordinates": {...}},              # 101 项，体积大
     "properties": {"qubit_metrics": ..., "coupler_metrics": ...},  # 体积大
     ...}

其中 ``native_gates`` 现网设备均为 ``None``（服务端未声明），``qubits`` 是比特总数。
落盘的 ``raw`` 会剔除 ``properties`` 与 ``topology.qubit_coordinates`` 两个大字段，
把单设备缓存控制在几 KB。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ....devices.base import DeviceRule

# 落盘 raw 时剔除的大字段（校准指标 / 比特坐标：体积大且路由用不到）。
_RAW_DROP_KEYS = ("properties",)
_RAW_TOPOLOGY_DROP_KEYS = ("qubit_coordinates",)


def _backend_name(device: str) -> str:
    """``"lqcloud::AGate-100"`` / ``"AGate-100"`` 归一为 SDK backend 名。"""
    name = str(device or "")
    return name.split("::")[-1] if "::" in name else name


def _normalize_coupling_map(raw: Any) -> List[List[int]]:
    """把 SDK 的耦合图归一为**双向**去重的 ``[[i, j], ...]``。

    服务端返回的边列表不保证对称（AGate-100 是 157 条，奇数），而物理耦合本身是
    无向的；补全双向后 qiskit 可自由选择门方向，减少无谓的 SWAP / 方向修正门。
    自环与非法项被丢弃。
    """
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
    out: List[List[int]] = []
    for q0, q1 in sorted(edges):
        out.append([q0, q1])
        out.append([q1, q0])
    return out


def _pick_num_qubits(config: Dict[str, Any], coupling_map: List[List[int]]) -> Optional[int]:
    """从 config 里取比特总数；字段缺失时退化为耦合图最大下标 + 1。"""
    for key in ("qubits", "num_qubits", "n_qubits", "qubit_count"):
        value = config.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    if coupling_map:
        return max(max(int(a), int(b)) for a, b in coupling_map) + 1
    return None


def _pick_native_gates(config: Dict[str, Any], topology: Dict[str, Any]) -> Optional[List[str]]:
    """取设备原生门集合；现网设备多为 ``None``（服务端未声明）。"""
    for source in (config, topology):
        for key in ("native_gates", "basis_gates", "gates"):
            value = source.get(key)
            if isinstance(value, (list, tuple)) and value:
                return [str(g) for g in value]
            if isinstance(value, dict) and value:
                return [str(g) for g in value.keys()]
    return None


def _trim_raw(config: Dict[str, Any]) -> Dict[str, Any]:
    """裁掉大字段后的 config 快照（供缓存文件排查用）。"""
    raw = {k: v for k, v in config.items() if k not in _RAW_DROP_KEYS}
    topology = raw.get("topology")
    if isinstance(topology, dict):
        raw["topology"] = {
            k: v for k, v in topology.items() if k not in _RAW_TOPOLOGY_DROP_KEYS
        }
    return raw


def get_topology(
    device: str, token: Optional[str] = None, **opts: Any
) -> Optional["DeviceRule"]:
    """抓取 LQCloud 设备拓扑（一次只读网络请求，不提交任何作业）。

    Args:
        device: 设备名，``"AGate-100"`` 或 ``"lqcloud::AGate-100"`` 均可。
        token: API key；缺省时复用驱动的解析链（``api_key=`` → token →
            ``tq.set_token(..., provider="lqcloud")`` → 环境变量）。
        **opts: 透传给 ``_resolve_api_key``（可含 ``api_key=``）。

    Returns:
        归一后的拓扑描述：``{"provider", "device", "num_qubits", "coupling_map",
        "native_gates", "status", "raw"}``；设备未声明耦合图时 ``coupling_map`` 为
        空列表（调用方据此判定「全连接 / 无需路由」）。

    Raises:
        RuntimeError: 缺少 API key、SDK 未安装或设备不存在（由上层拓扑缓存统一兜底）。
    """
    from ....devices.hardware.lqcloud.driver import _resolve_api_key, _sdk_classes

    name = _backend_name(device)
    if not name:
        raise ValueError("LQCloud 拓扑发现需要设备名（如 'AGate-100'）。")

    api_key = _resolve_api_key(token, dict(opts))
    LQCloudProvider, _, _ = _sdk_classes()
    provider = LQCloudProvider(api_key=api_key, interactive=False)
    # backend.config 是 SDK 侧带缓存的只读属性（provider.get_backend_config(name)）。
    config = provider.get_backend(name).config
    if not isinstance(config, dict):
        raise RuntimeError(f"LQCloud 设备 {name} 返回了非预期的 config 类型: {type(config).__name__}")

    topology = config.get("topology") if isinstance(config.get("topology"), dict) else {}
    coupling_map = _normalize_coupling_map(topology.get("coupling_map"))
    return {
        "provider": "lqcloud",
        "device": name,
        "num_qubits": _pick_num_qubits(config, coupling_map),
        "coupling_map": coupling_map,
        "native_gates": _pick_native_gates(config, topology),
        "status": config.get("status"),
        "raw": _trim_raw(config),
    }


__all__ = ["get_topology"]
