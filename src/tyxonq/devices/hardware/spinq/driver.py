"""SpinQit 云后端的 TyxonQ 设备驱动。"""

from __future__ import annotations

import configparser
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_DEVICE_QUBITS = {
    "gemini_vp": 2,
    "triangulum_vp": 3,
    "superconductor_vp": 8,
    "sqc_25_vp": 25,
}
_DEVICE_ALIASES = {
    "triangulum-pro-1": "triangulum_vp",
    "triangulum": "triangulum_vp",
    "sqc-25": "sqc_25_vp",
    "sqc_25": "sqc_25_vp",
}
_INSTALL_HINT = "请在隔离环境安装 spinqit==0.3.0，不要降级 TyxonQ 主环境 numpy。"


def _sdk_api() -> Any:
    try:
        from spinqit import SpinQCloudConfig, get_spinq_cloud
    except ImportError as exc:
        raise RuntimeError(f"SpinQ provider 需要 SpinQit SDK。{_INSTALL_HINT}") from exc
    return get_spinq_cloud, SpinQCloudConfig


def _read_pypirc_credentials() -> Dict[str, Optional[str]]:
    """只读 .pypirc [spinqcloud]，不输出账号、host 或私钥内容。"""
    configured = os.getenv("TYXONQ_SPINQ_CONFIG")
    candidates = (
        [Path(configured).expanduser()]
        if configured
        else [
            Path.cwd() / ".pypirc",
            Path(__file__).resolve().parents[5] / ".pypirc",
        ]
    )
    for path in candidates:
        if not path.is_file():
            continue
        parser = configparser.ConfigParser()
        parser.read(path)
        if not parser.has_section("spinqcloud"):
            return {}
        section = parser["spinqcloud"]
        keyfile = section.get("keyfile")
        if keyfile:
            key_path = Path(keyfile).expanduser()
            if not key_path.is_absolute():
                key_path = path.parent / key_path
            keyfile = str(key_path.resolve())
        return {
            "username": section.get("username"),
            "keyfile": keyfile,
            "host": section.get("host") or None,
        }
    return {}


def _resolve_credentials(opts: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """显式参数 > 环境变量 > .pypirc；不回退全局 TYXONQ_API_KEY。"""
    file_credentials = _read_pypirc_credentials()
    username = (
        opts.pop("username", None)
        or os.getenv("TYXONQ_SPINQ_USERNAME")
        or file_credentials.get("username")
    )
    keyfile = (
        opts.pop("keyfile", None)
        or os.getenv("TYXONQ_SPINQ_KEYFILE")
        or file_credentials.get("keyfile")
    )
    host = (
        opts.pop("host", None)
        or os.getenv("TYXONQ_SPINQ_HOST")
        or file_credentials.get("host")
    )
    if not username or not keyfile:
        raise RuntimeError(
            "SpinQ 凭证不足：请传 username/keyfile、设置 TYXONQ_SPINQ_USERNAME/KEYFILE，"
            "或配置 .pypirc 的 [spinqcloud] username/keyfile。"
        )
    return {
        "username": str(username),
        "keyfile": str(keyfile),
        "host": str(host) if host else None,
    }


def _create_backend(username: str, keyfile: str, host: Optional[str] = None) -> Any:
    get_spinq_cloud, _ = _sdk_api()
    if host:
        return get_spinq_cloud(username, keyfile, host=host)
    return get_spinq_cloud(username, keyfile)


@dataclass
class SpinQTask:
    """包装已完成的 SpinQit 同步结果，并保留云任务号用于审计。"""

    id: str
    device: str
    status: str = "completed"
    result: Any = field(default=None, repr=False)
    shots: int = 1024
    task_name: str = ""
    compile_metadata: Dict[str, Any] = field(default_factory=dict)
    async_result: bool = False


def _device_name(device: str) -> str:
    name = device.split("::")[-1] if "::" in device else device
    return _DEVICE_ALIASES.get(name.casefold(), name)


def run(
    device: str,
    token: Optional[str] = None,
    *,
    source: Any = None,
    shots: int | Sequence[int] = 1024,
    **opts: Any,
) -> List[SpinQTask]:
    """提交一条 SpinQit native 可执行对象，且只执行一次。"""
    if source is None:
        raise TypeError("SpinQ source 必须是 SpinQit native compiler 生成的可执行对象。")
    if isinstance(shots, (list, tuple)):
        raise ValueError("SpinQ 首版一次只支持一条线路和一个 shots 值。")
    if isinstance(shots, bool) or int(shots) <= 0 or float(shots) != int(shots):
        raise ValueError("SpinQ 真机 shots 必须是正整数，不支持 shots=0。")

    driver_opts = dict(opts)
    credentials = _resolve_credentials(driver_opts)
    task_name = str(driver_opts.pop("task_name", None) or f"tyxonq-{uuid.uuid4().hex[:12]}")
    task_description = str(driver_opts.pop("task_description", "TyxonQ native IR circuit"))
    compile_metadata = driver_opts.pop("compile_metadata", {}) or {}
    if driver_opts:
        names = ", ".join(sorted(driver_opts))
        raise TypeError(f"SpinQ 首版不支持这些运行参数: {names}")

    name = _device_name(device)
    if name not in _DEVICE_QUBITS:
        raise ValueError(f"未知 SpinQ 平台: {name}")
    backend = _create_backend(**credentials)
    platform = backend.get_platform(name)
    if callable(getattr(platform, "available", None)) and not platform.available():
        raise RuntimeError(f"SpinQ 平台 {name} 当前不可用。")

    _, SpinQCloudConfig = _sdk_api()
    config = SpinQCloudConfig()
    config.configure_platform(name)
    config.configure_shots(int(shots))
    config.configure_task(task_name, task_description)
    keep_layout = compile_metadata.get("runtime_keep_layout")
    if keep_layout is not None:
        config.configure_keep_layout([int(q) for q in keep_layout])
    measure_qubits = compile_metadata.get("cloud_measure_qubits")
    if measure_qubits is not None:
        config.configure_measure_qubits([int(q) for q in measure_qubits])
    result = backend.execute(source, config)
    remote_id = next(
        (
            getattr(result, attr)
            for attr in ("task_code", "task_id", "job_id", "id")
            if getattr(result, attr, None)
        ),
        None,
    )
    return [
        SpinQTask(
            id=str(remote_id or uuid.uuid4().hex),
            device=name,
            result=result,
            shots=int(shots),
            task_name=task_name,
            compile_metadata=dict(compile_metadata),
        )
    ]


def _map_status(value: Any) -> str:
    status = str(value or "completed").strip().lower()
    if status in {"completed", "done", "success", "finished"}:
        return "completed"
    if status in {"failed", "error"}:
        return "failed"
    if status in {"cancelled", "canceled"}:
        return "cancelled"
    if status in {"queued", "pending", "submitted"}:
        return "queued"
    if status == "running":
        return "running"
    return "unknown"


def _normalize_counts(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, dict):
        raise RuntimeError("SpinQit 完成结果缺少 counts 字典；不会用 probabilities 伪造采样。")
    counts: Dict[str, int] = {}
    for bitstring, value in raw.items():
        count = int(value)
        if count < 0:
            raise RuntimeError("SpinQit counts 不能包含负数。")
        counts[str(bitstring)] = count
    if not counts or sum(counts.values()) <= 0:
        raise RuntimeError("SpinQit 完成结果没有有效 counts；不会用 probabilities 伪造采样。")
    return counts


def get_task_details(task: SpinQTask, token: Optional[str] = None) -> Dict[str, Any]:
    """读取已保存的同步结果，不重新提交实验。"""
    meta: Dict[str, Any] = {
        "task_id": task.id,
        "device": task.device,
        "shots": task.shots,
        "task_name": task.task_name,
        "bit_order": "little_endian",
        "compile": dict(task.compile_metadata),
    }
    try:
        if task.result is None:
            raise RuntimeError("SpinQTask 缺少 SpinQit 结果对象。")
        status = _map_status(getattr(task.result, "status", task.status))
        counts = _normalize_counts(getattr(task.result, "counts", None)) if status == "completed" else {}
        raw_probabilities = getattr(task.result, "probabilities", {}) or {}
        probabilities = (
            {str(key): float(value) for key, value in raw_probabilities.items()}
            if isinstance(raw_probabilities, dict)
            else {}
        )
        meta["counts_shots"] = sum(counts.values()) if counts else 0
        meta["counts_shots_delta"] = meta["counts_shots"] - task.shots
        return {
            "result": counts,
            "probabilities": probabilities,
            "result_meta": meta,
            "uni_status": status,
            "error": "",
        }
    except Exception as exc:
        return {
            "result": {},
            "probabilities": {},
            "result_meta": meta,
            "uni_status": "error",
            "error": str(exc),
        }


def remove_task(task: SpinQTask, token: Optional[str] = None) -> bool:
    raise NotImplementedError("SpinQit execute 为同步接口，当前不支持取消已完成调用。")


def list_devices(token: Optional[str] = None, **kws: Any) -> List[str]:
    """返回官方文档确认的 SpinQ 平台目录，不为列表查询发起登录。"""
    return [f"spinq::{name}" for name in _DEVICE_QUBITS]


def expval(*args: Any, **kwargs: Any) -> float:
    raise NotImplementedError("SpinQ 真机不支持 shots=0 的解析期望值。")


__all__ = ["SpinQTask", "expval", "get_task_details", "list_devices", "remove_task", "run"]
