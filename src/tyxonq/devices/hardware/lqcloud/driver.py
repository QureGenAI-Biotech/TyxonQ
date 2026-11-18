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


def _looks_like_waves(source: Any) -> bool:
    """判断 source 是否为 lqcloud ``run_wave`` 的 waves dict（轻量内联判据）。

    判据：非空 dict，且所有键都是形如 ``"<qname>.<channel>"`` 的字符串，
    通道名落在 ``{"xy","z","readin","readout"}``。刻意内联（不 import
    compiler 层），保持设备层只依赖 SDK；waves 的构造方为
    ``compiler/platforms/lqcloud/lqwave_adapter.py``（硬件方言归
    platforms 目录）。
    """
    if not isinstance(source, dict) or not source:
        return False
    for key in source:
        if not isinstance(key, str) or "." not in key:
            return False
        if key.rsplit(".", 1)[-1].lower() not in ("xy", "z", "readin", "readout"):
            return False
    return True


def _run_wave(
    device: str,
    token: Optional[str],
    *,
    waves: Dict[str, Any],
    shots: Union[int, Sequence[int]],
    **opts: Any,
) -> List[LQCloudTask]:
    """提交 raw-waveform 实验（``backend.run_wave``，绕过线路编译器直传 QPU）。

    waves 由 ``compiler/platforms/lqcloud/lqwave_adapter.program_to_waves``
    产出（参数化波形 → IQ 采样数组 + 通道映射）；本函数只透传：

    - ``result_format``：``"counts"``（默认）/``"memory"``/``"raw"``（原始 IQ）；
    - ``block_intervals``：可选 ``[[start_ns, end_ns], ...]`` block 边界；
    - ``readout_correction``：读出校正开关（opt-in）。
    """
    LQCloudProvider, _, _ = _sdk_classes()
    if isinstance(shots, (list, tuple)):
        raise ValueError("LQCloud 首版一次只支持一条波形实验和一个 shots 值。")
    shot_count = int(shots)
    if shot_count <= 0:
        raise ValueError(f"shots 必须大于 0，当前为 {shot_count}。")

    driver_opts = dict(opts)
    api_key = _resolve_api_key(token, driver_opts)
    result_format = driver_opts.pop("result_format", "counts")
    block_intervals = driver_opts.pop("block_intervals", None)
    readout_correction = driver_opts.pop("readout_correction", False)
    if driver_opts:
        names = ", ".join(sorted(driver_opts))
        raise TypeError(f"LQCloud run_wave 首版不支持这些运行参数: {names}")

    backend_name = device.split("::")[-1] if "::" in device else device
    provider = LQCloudProvider(api_key=api_key, interactive=False)
    backend = provider.get_backend(backend_name)
    job = backend.run_wave(
        waves,
        shots=shot_count,
        block_intervals=block_intervals,
        result_format=str(result_format),
        readout_correction=bool(readout_correction),
    )
    job_id = getattr(job, "job_id", None)
    if not job_id:
        raise RuntimeError("LQCloud run_wave submission did not return a job_id.")
    return [LQCloudTask(id=str(job_id), device=backend_name, job=job)]


def run(
    device: str,
    token: Optional[str] = None,
    *,
    source: Any = None,
    shots: Union[int, Sequence[int]] = 1024,
    **opts: Any,
) -> List[LQCloudTask]:
    """提交一条官方 LQCloud QuantumCircuit 或 run_wave 波形实验。

    ``source`` 为 ``lqcloud.QuantumCircuit``（门线路）或 waves dict
    （原始波形，键形如 ``"q0.xy"``，由 platforms/lqcloud/lqwave_adapter 产出）。
    首版不开放批量任务。
    """
    LQCloudProvider, QuantumCircuit, _ = _sdk_classes()
    if _looks_like_waves(source):
        return _run_wave(device, token, waves=source, shots=shots, **opts)
    if not isinstance(source, QuantumCircuit):
        raise TypeError("LQCloud source 必须是 lqcloud.QuantumCircuit 或 run_wave waves dict。")
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
    memory: Optional[List[str]] = None
    iq_data: Optional[Any] = None
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
        # run_wave 专属结果透出：per-shot memory（counts 格式下 SDK 会抛
        # ValueError，忽略）与原始 IQ 数据（仅 run_wave 且 result_format="raw"）
        try:
            memory = result.get_memory()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 —— 非 memory 格式属正常
            memory = None
        try:
            iq_data = result.get_iq_data()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            iq_data = None
    elif status in {"failed", "cancelled"} and isinstance(decoded, dict):
        error = decoded.get("error", error) or error

    meta: Dict[str, Any] = {
        "job_id": task.id,
        "device": task.device,
        "shots": sum(counts.values()) if counts else None,
        "probability": probabilities,
        "result_format": result_format,
        "raw": raw,
    }
    if memory:
        meta["memory"] = memory
    if iq_data:
        meta["iq_data"] = iq_data

    return {
        "result": counts,
        "result_meta": meta,
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
