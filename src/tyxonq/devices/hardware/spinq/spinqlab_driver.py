"""SpinQLab Link TCP 实验驱动；当前仅支持官方原生矩形 Pulse 参数。"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_FIELDS = ("path", "phase", "amplitude", "width", "detuning")
_PARAMETER_FIELDS = ("freq_h", "freq_p", "makePps", "samplePath", "custom_freq")


def _sdk_api() -> Any:
    try:
        from spinqlablink import ExperimentType, Pulse, SpinQLabLink
    except ImportError as exc:
        raise RuntimeError("SpinQLab provider 需要可选 SDK spinqlablink==1.0.2。") from exc
    return SpinQLabLink, ExperimentType, Pulse


def _resolve_connection(opts: Dict[str, Any]) -> Dict[str, Any]:
    host = opts.pop("host", None) or os.getenv("TYXONQ_SPINQLAB_HOST")
    port = opts.pop("port", None) or os.getenv("TYXONQ_SPINQLAB_PORT", "8181")
    account = opts.pop("account", None) or os.getenv("TYXONQ_SPINQLAB_ACCOUNT")
    password = opts.pop("password", None) or os.getenv("TYXONQ_SPINQLAB_PASSWORD")
    if not host or not account or not password:
        raise RuntimeError(
            "SpinQLab 连接参数不足：请设置 host/account/password，或对应 "
            "TYXONQ_SPINQLAB_HOST/ACCOUNT/PASSWORD。"
        )
    return {"host": str(host), "port": int(port), "account": str(account), "password": str(password)}


@dataclass
class SpinQLabTask:
    id: str
    device: str
    status: str = "completed"
    result: Dict[str, Any] = field(default_factory=dict)
    experiment_type: str = "NMR_PHENOMENON_AND_SIGNAL"
    async_result: bool = False


def _native_pulse_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    if not entry.get("native"):
        raise ValueError("非矩形采样草案不能作为 SpinQLab 原生 Pulse 提交。")
    return {field: entry[field] for field in _FIELDS}


def run(
    device: str,
    token: Optional[str] = None,
    *,
    source: Any = None,
    shots: int | Sequence[int] = 1,
    **opts: Any,
) -> List[SpinQLabTask]:
    """执行一次 NMR 原生脉冲实验；不把 shots 隐式映射为 repeat。"""
    if isinstance(shots, (list, tuple, bool)) or int(shots) != 1:
        raise ValueError("SpinQLab NMR 实验不使用 shots；请保持 shots=1 并显式配置实验参数。")
    if not isinstance(source, dict) or source.get("schema") != "tyxonq.spinqlab.parameters.v1":
        raise TypeError("SpinQLab source 必须是 waveform_adapter 生成的参数草案。")
    pulses = source.get("pulses")
    if not isinstance(pulses, list) or not pulses:
        raise ValueError("SpinQLab 参数草案没有 pulses。")
    native_pulses = [_native_pulse_fields(dict(entry)) for entry in pulses]

    driver_opts = dict(opts)
    connection = _resolve_connection(driver_opts)
    if driver_opts:
        raise TypeError(f"SpinQLab 首版不支持这些运行参数: {sorted(driver_opts)}")
    parameters_data = source.get("parameters", {})
    unknown = set(parameters_data) - set(_PARAMETER_FIELDS)
    if unknown:
        raise ValueError(f"未确认的 SpinQLab 实验参数: {sorted(unknown)}")

    SpinQLabLink, ExperimentType, Pulse = _sdk_api()
    client = SpinQLabLink(**connection)
    registered = False
    try:
        if client.connect() is False or not client.wait_for_login():
            raise RuntimeError("SpinQLab 连接或登录失败。")
        experiment, parameters = client.register_experiment(
            ExperimentType.NMR_PHENOMENON_AND_SIGNAL
        )
        registered = True
        parameters.pulses = [Pulse(**fields) for fields in native_pulses]
        for key, value in parameters_data.items():
            setattr(parameters, key, value)
        client.run_experiment()
        if client.wait_for_experiment_completion() is False:
            raise RuntimeError("SpinQLab 实验未成功完成。")
        raw_result = client.get_experiment_result()
        if not isinstance(raw_result, dict):
            raise RuntimeError("SpinQLab 返回了非字典实验结果。")
        experiment_id = getattr(experiment, "id", None) or uuid.uuid4().hex
        task = SpinQLabTask(id=str(experiment_id), device=str(device), result=raw_result)
    finally:
        if registered:
            client.deregister_experiment()
        client.disconnect()
    return [task]


def get_task_details(task: SpinQLabTask, token: Optional[str] = None) -> Dict[str, Any]:
    """返回已保存的实验结果，不重新连接或重提。"""
    payload = task.result.get("result", {})
    probabilities = payload.get("probability", {}) if isinstance(payload, dict) else {}
    return {
        "result": payload,
        "probabilities": probabilities,
        "result_meta": {
            "experiment_id": task.id,
            "device": task.device,
            "experiment_type": task.experiment_type,
            "result_kind": "nmr_experiment",
            "result_keys": sorted(task.result),
        },
        "uni_status": task.status,
        "error": "",
    }


def remove_task(task: SpinQLabTask, token: Optional[str] = None) -> bool:
    raise NotImplementedError("SpinQLab 同步实验完成后不支持取消。")


def list_devices(token: Optional[str] = None, **kws: Any) -> List[str]:
    return []


def expval(*args: Any, **kwargs: Any) -> float:
    raise NotImplementedError("SpinQLab NMR 实验不提供统一解析期望值接口。")


__all__ = ["SpinQLabTask", "expval", "get_task_details", "list_devices", "remove_task", "run"]
