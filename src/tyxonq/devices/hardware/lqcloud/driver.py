"""TyxonQ 对 LQCloud 官方 SDK 的最小门线路适配。"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from ..config import get_token as _hw_get_token


logger = logging.getLogger(__name__)

_KEY_HELP = (
    "LQCloud API key required. Pass api_key=..., call "
    "tq.set_token(..., provider='lqcloud'), or set "
    "TYXONQ_LQCLOUD_API_KEY / LQCLOUD_API_KEY."
)
_INSTALL_HINT = "请执行 `pip install 'tyxonq[lqcloud]'`。"


def _sdk_classes():
    """延迟导入官方 SDK。"""
    try:
        from lqcloud import LQCloudProvider, QuantumCircuit, Result
    except ImportError as exc:
        raise RuntimeError(f"LQCloud provider 需要 lqcloud==0.4.2；{_INSTALL_HINT}") from exc
    return LQCloudProvider, QuantumCircuit, Result


def _resolve_api_key(token: Optional[str], opts: Dict[str, Any]) -> str:
    """解析 API key；绝不回退到 TyxonQ 平台自己的全局密钥。"""
    key = opts.pop("api_key", None) or token
    if not key:
        key = _hw_get_token(provider="lqcloud", env_fallback=False)
    if not key:
        key = os.getenv("TYXONQ_LQCLOUD_API_KEY") or os.getenv("LQCLOUD_API_KEY")
    if not key:
        raise RuntimeError(_KEY_HELP)
    return str(key)


@dataclass
class LQCloudTask:
    """保存官方 Job，使统一设备层能够继续查询和取消。"""

    id: str
    device: str
    status: str = "submitted"
    job: Any = field(default=None, repr=False)
    async_result: bool = True


def run(
    device: str,
    token: Optional[str] = None,
    *,
    source: Any = None,
    shots: Union[int, Sequence[int]] = 1024,
    **opts: Any,
) -> List[LQCloudTask]:
    """提交一条官方 LQCloud QuantumCircuit；首版不开放批量任务。"""
    LQCloudProvider, QuantumCircuit, _ = _sdk_classes()
    if not isinstance(source, QuantumCircuit):
        raise TypeError("LQCloud source 必须是 lqcloud.QuantumCircuit。")
    if isinstance(shots, (list, tuple)):
        raise ValueError("LQCloud 首版一次只支持一条线路和一个 shots 值。")

    shot_count = int(shots)
    if shot_count <= 0:
        raise ValueError(f"shots 必须大于 0，当前为 {shot_count}。")

    driver_opts = dict(opts)
    api_key = _resolve_api_key(token, driver_opts)
    physical_qubits = driver_opts.pop("physical_qubits", None)
    if driver_opts:
        names = ", ".join(sorted(driver_opts))
        raise TypeError(f"LQCloud 首版不支持这些运行参数: {names}")

    backend_name = device.split("::")[-1] if "::" in device else device
    provider = LQCloudProvider(api_key=api_key, interactive=False)
    backend = provider.get_backend(backend_name)
    run_opts: Dict[str, Any] = {}
    if physical_qubits is not None:
        run_opts["initial_layout"] = list(physical_qubits)
    job = backend.run(source, shots=shot_count, **run_opts)
    job_id = getattr(job, "job_id", None)
    if not job_id:
        raise RuntimeError("LQCloud submission did not return a job_id.")
    return [LQCloudTask(id=str(job_id), device=backend_name, job=job)]


def _map_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in {"queued", "pending", "submitted", "retrying"}:
        return "queued"
    if status == "running":
        return "running"
    if status in {"completed", "done", "success", "finished"}:
        return "completed"
    if status in {"failed", "error"}:
        return "failed"
    if status in {"cancelled", "canceled"}:
        return "cancelled"
    return "unknown"


def _decode_result(raw_result: Any) -> Any:
    if isinstance(raw_result, str):
        try:
            return json.loads(raw_result)
        except json.JSONDecodeError:
            return raw_result
    return raw_result


def get_task_details(
    task: LQCloudTask, token: Optional[str] = None
) -> Dict[str, Any]:
    """只查询已有 job_id，并转换为 TyxonQ 统一结果结构。"""
    if task.job is None:
        raise RuntimeError("LQCloudTask 缺少官方 Job，无法查询。")
    _, _, Result = _sdk_classes()
    try:
        raw = task.job.status_info()
    except Exception as exc:
        return {
            "result": {},
            "result_meta": {"job_id": task.id, "device": task.device},
            "uni_status": "error",
            "error": str(exc),
        }

    if not isinstance(raw, dict):
        return {
            "result": {},
            "result_meta": {"job_id": task.id, "device": task.device, "raw": raw},
            "uni_status": "unknown",
            "error": f"unexpected response: {raw!r}",
        }

    status = _map_status(raw.get("status"))
    counts: Dict[str, int] = {}
    probabilities: Dict[str, float] = {}
    result_format: Optional[str] = None
    error = raw.get("error", "") or ""
    decoded = _decode_result(raw.get("result"))

    if status == "completed" and isinstance(decoded, dict):
        try:
            result = Result(
                backend_name=task.device,
                job_id=task.id,
                status=str(raw.get("status", "completed")),
                data=decoded,
                metadata=raw,
            )
            counts = result.get_counts() or {}
            probabilities = result.get_probabilities() or {}
            result_format = result.result_format
        except Exception as exc:
            error = str(exc)
    elif status in {"failed", "cancelled"} and isinstance(decoded, dict):
        error = decoded.get("error", error) or error

    return {
        "result": counts,
        "result_meta": {
            "job_id": task.id,
            "device": task.device,
            "shots": sum(counts.values()) if counts else None,
            "probability": probabilities,
            "result_format": result_format,
            "raw": raw,
        },
        "uni_status": status,
        "error": error,
    }


def remove_task(task: LQCloudTask, token: Optional[str] = None) -> bool:
    """取消排队中的官方 LQCloud Job。"""
    if task.job is None:
        raise RuntimeError("LQCloudTask 缺少官方 Job，无法取消。")
    return bool(task.job.cancel())


def list_devices(token: Optional[str] = None, **kws: Any) -> List[str]:
    """动态列出当前 API key 可见的 LQCloud 设备。"""
    opts = dict(kws)
    try:
        api_key = _resolve_api_key(token, opts)
    except RuntimeError:
        return []
    if opts:
        return []

    try:
        LQCloudProvider, _, _ = _sdk_classes()
        provider = LQCloudProvider(api_key=api_key, interactive=False)
        rows = provider.get_backends()
    except Exception as exc:
        logger.warning("LQCloud list_devices failed: %s", exc)
        return []

    names = []
    for row in rows if isinstance(rows, list) else []:
        name = row.get("name") if isinstance(row, dict) else None
        if isinstance(name, str) and name:
            names.append(f"lqcloud::{name}")
    return names


__all__ = [
    "LQCloudTask",
    "get_task_details",
    "list_devices",
    "remove_task",
    "run",
]
