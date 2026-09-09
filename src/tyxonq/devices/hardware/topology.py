"""设备拓扑发现 + 持久化 TTL 缓存（跨 provider 统一入口）。

编译器需要设备耦合图才能做布局 / 路由，但耦合图只能运行期向 provider 询问（一次
网络请求）。本模块把「问一次、存一天」的策略集中在这里，各 provider 只负责实现
「怎么问」（``fetch_topology`` 分派到 ``compiler.platforms.<provider>.get_topology``）。

缓存分两级，**两级都遵守 ``max_age_days``**：
  1. **会话内内存缓存**：按 ``(provider, device)`` 命中，同一进程内不重复读盘 / 联网；
     条目仍带 ``fetched_at``，故调用方收紧 TTL 时不会被内存缓存绕过。
  2. **磁盘持久化缓存**：``devices/hardware/<provider>/.topology_cache.json``（已
     gitignore），结构::

         {"<device>": {"num_qubits": 101,
                       "coupling_map": [[0, 1], [1, 0], ...],
                       "native_gates": null,
                       "fetched_at": "2026-09-09T10:00:00+00:00",
                       "raw": {...}}}

     TTL 规则：文件缺失 / 无该设备 → 联网抓取并写盘；``fetched_at`` 距今超过
     ``max_age_days``（默认 1 天）→ 刷新；否则直接用磁盘值。``force_refresh=True``
     可强制刷新。

离线韧性：抓取失败（无网络 / 无凭证 / SDK 未装）时，若存在**过期**缓存则降级沿用并
告警；若连缓存都没有，返回空 ``DeviceRule``（编译器据此跳过路由，行为与拓扑发现引入
之前一致），绝不因拓扑发现失败而中断编译 / 提交。

缓存目录可用环境变量 ``TYXONQ_TOPOLOGY_CACHE_DIR`` 重定向（测试、以及包目录只读的
安装场景）。
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..base import DeviceRule

logger = logging.getLogger(__name__)

# 缓存文件名 / 目录环境变量 / 默认 TTL（天）。
CACHE_FILENAME = ".topology_cache.json"
CACHE_DIR_ENV = "TYXONQ_TOPOLOGY_CACHE_DIR"
DEFAULT_MAX_AGE_DAYS = 1.0

# 会话内内存缓存：{(provider, device): DeviceRule}
_MEMORY_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _provider_dir(provider: str) -> Path:
    """返回某 provider 的缓存目录（默认在包内，可被环境变量重定向）。"""
    root = os.environ.get(CACHE_DIR_ENV)
    if root:
        return Path(root).expanduser() / str(provider)
    return Path(__file__).resolve().parent / str(provider)


def cache_path(provider: str) -> Path:
    """某 provider 拓扑缓存文件的绝对路径。"""
    return _provider_dir(provider) / CACHE_FILENAME


def _device_key(device: str) -> str:
    """``"lqcloud::AGate-100"`` / ``"AGate-100"`` 归一为缓存键。"""
    name = str(device or "")
    return name.split("::")[-1] if "::" in name else name


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _age_days(entry: Dict[str, Any]) -> Optional[float]:
    """缓存条目的年龄（天）；``fetched_at`` 缺失或不可解析时返回 None（视为过期）。"""
    fetched_at = entry.get("fetched_at")
    if not isinstance(fetched_at, str) or not fetched_at:
        return None
    try:
        stamp = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - stamp).total_seconds() / 86400.0


def _load_cache(provider: str) -> Dict[str, Any]:
    """读磁盘缓存；文件缺失 / 损坏都退化为空字典（并告警损坏情形）。"""
    path = cache_path(provider)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("拓扑缓存 %s 读取失败（%s），按无缓存处理。", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(provider: str, data: Dict[str, Any]) -> bool:
    """写磁盘缓存；目录不可写（如只读 site-packages）时只告警，不影响本次编译。"""
    path = cache_path(provider)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8",
        )
        return True
    except OSError as exc:
        logger.warning("拓扑缓存写入 %s 失败（%s）；本次仅使用内存缓存。", path, exc)
        return False


def _to_entry(topology: Dict[str, Any]) -> Dict[str, Any]:
    """把抓取结果归一为缓存条目（打上 ``fetched_at`` 时间戳）。"""
    return {
        "num_qubits": topology.get("num_qubits"),
        "coupling_map": list(topology.get("coupling_map") or []),
        "native_gates": topology.get("native_gates"),
        "status": topology.get("status"),
        "fetched_at": _now_iso(),
        "raw": topology.get("raw") or {},
    }


def _to_device_rule(provider: str, device: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """缓存条目 → ``DeviceRule``（不带 ``raw``，避免编译 metadata 被大字段撑爆）。"""
    rule: Dict[str, Any] = {
        "provider": provider,
        "device": device,
        "coupling_map": [list(pair) for pair in (entry.get("coupling_map") or [])],
        "fetched_at": entry.get("fetched_at"),
    }
    num_qubits = entry.get("num_qubits")
    if isinstance(num_qubits, int) and num_qubits > 0:
        rule["num_qubits"] = int(num_qubits)
        rule["max_qubits"] = int(num_qubits)
    native_gates = entry.get("native_gates")
    if native_gates:
        rule["native_gates"] = list(native_gates)
    return rule


def _copy_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    """``DeviceRule`` 的安全副本：连同 ``coupling_map`` / ``native_gates`` 列表一起复制。

    调用方会往返回的字典里回填 ``initial_layout``（``Routing`` 步骤），甚至可能原地
    改动 ``coupling_map``；若直接返回缓存里的同一个列表对象，会污染会话内存缓存。
    """
    out = dict(rule)
    if isinstance(out.get("coupling_map"), list):
        out["coupling_map"] = [list(pair) for pair in out["coupling_map"]]
    if isinstance(out.get("native_gates"), list):
        out["native_gates"] = list(out["native_gates"])
    return out


def fetch_topology(
    provider: str, device: str, token: Optional[str] = None, **opts: Any
) -> Optional[Dict[str, Any]]:
    """按 provider 分派拓扑抓取；未实现的 provider 返回 ``None``（表示不路由）。

    目前只有 lqcloud 落地（``compiler.platforms.lqcloud.get_topology``）；guodun /
    quafu / qcos / tyxonq / ibm 的抓取列为后续增量。
    """
    prov = str(provider or "").lower()
    if prov == "lqcloud":
        from ...compiler.platforms.lqcloud import get_topology

        return get_topology(device, token=token, **opts)
    return None


def get_device_rule(
    provider: str,
    device: str,
    token: Optional[str] = None,
    *,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    force_refresh: bool = False,
) -> "DeviceRule":
    """取得设备拓扑规则（内存缓存 → 磁盘 TTL 缓存 → 联网抓取）。

    Args:
        provider: provider 名（如 ``"lqcloud"``）。
        device: 设备名（``"AGate-100"`` 或 ``"lqcloud::AGate-100"``）。
        token: provider 凭证；缺省时由各 provider 的抓取实现自行解析。
        max_age_days: 缓存有效期（天），默认 1 天；内存与磁盘两级缓存都遵守，超期则刷新。
        force_refresh: 为真时跳过所有缓存，强制联网抓取。

    Returns:
        ``DeviceRule``：含 ``coupling_map`` / ``num_qubits`` / ``native_gates`` 等；
        拿不到拓扑时返回空字典（编译器据此跳过路由，不中断流程）。
    """
    prov = str(provider or "").lower()
    name = _device_key(device)
    if not prov or not name:
        return {}

    key = (prov, name)
    if not force_refresh and key in _MEMORY_CACHE:
        # 内存缓存同样受 TTL 约束：否则调用方收紧 max_age_days 会被静默绕过。
        cached = _MEMORY_CACHE[key]
        age = _age_days(cached)
        if age is not None and age <= float(max_age_days):
            return _copy_rule(cached)

    cache = _load_cache(prov)
    entry = cache.get(name) if isinstance(cache, dict) else None
    if isinstance(entry, dict) and not force_refresh:
        age = _age_days(entry)
        if age is not None and age <= float(max_age_days):
            rule = _to_device_rule(prov, name, entry)
            _MEMORY_CACHE[key] = rule
            return _copy_rule(rule)

    # 缓存缺失 / 过期 / 强制刷新：联网抓取一次（只读发现，不提交作业）。
    error: Optional[BaseException] = None
    topology: Optional[Dict[str, Any]] = None
    try:
        topology = fetch_topology(prov, name, token=token)
    except Exception as exc:  # 无网络 / 无凭证 / SDK 缺失 / 设备不存在
        error = exc
        logger.warning("%s::%s 拓扑抓取失败：%s", prov, name, exc)

    if isinstance(topology, dict):
        entry = _to_entry(topology)
        cache[name] = entry
        _save_cache(prov, cache)
        rule = _to_device_rule(prov, name, entry)
        _MEMORY_CACHE[key] = rule
        return _copy_rule(rule)

    if isinstance(entry, dict):
        # 抓取失败但有陈旧缓存：降级沿用（离线 / 凭证过期时仍可路由）。
        warnings.warn(
            f"{prov}::{name} 拓扑刷新失败（{error}），沿用过期缓存"
            f"（fetched_at={entry.get('fetched_at')}）。",
            RuntimeWarning,
            stacklevel=2,
        )
        rule = _to_device_rule(prov, name, entry)
        _MEMORY_CACHE[key] = rule
        return _copy_rule(rule)

    warnings.warn(
        f"无法获取 {prov}::{name} 的设备拓扑（{error or '该 provider 暂未实现拓扑发现'}）；"
        f"本次编译跳过路由，若线路含非相邻两比特门可能被设备拒绝。",
        RuntimeWarning,
        stacklevel=2,
    )
    return {}


def clear_topology_cache(
    *, provider: Optional[str] = None, device: Optional[str] = None, disk: bool = True
) -> None:
    """清空拓扑缓存（内存必清；``disk=True`` 时同时删缓存文件里的对应条目）。

    主要给测试与「设备拓扑刚被运维改动」的场景用。
    """
    if provider is None and device is None:
        _MEMORY_CACHE.clear()
        if disk:
            override = os.environ.get(CACHE_DIR_ENV)
            root = Path(override).expanduser() if override else Path(__file__).resolve().parent
            if root.is_dir():
                for path in root.glob(f"*/{CACHE_FILENAME}"):
                    try:
                        path.unlink()
                    except OSError as exc:  # pragma: no cover - 只读安装
                        logger.warning("删除拓扑缓存 %s 失败：%s", path, exc)
        return

    prov = str(provider).lower() if provider else None
    name = _device_key(device) if device else None
    for key in [k for k in _MEMORY_CACHE if (prov is None or k[0] == prov) and (name is None or k[1] == name)]:
        _MEMORY_CACHE.pop(key, None)
    if disk and prov:
        cache = _load_cache(prov)
        if name is None:
            cache = {}
        else:
            cache.pop(name, None)
        _save_cache(prov, cache)


__all__ = [
    "CACHE_FILENAME",
    "DEFAULT_MAX_AGE_DAYS",
    "cache_path",
    "clear_topology_cache",
    "fetch_topology",
    "get_device_rule",
]
