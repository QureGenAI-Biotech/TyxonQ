"""PulseProgram 到 SpinQLab Link 实验参数的离线可行性适配。

SpinQLab Link 1.0.2 的 ``Pulse(path, phase, amplitude, width, detuning)`` 是
矩形脉冲参数，不是任意 IQ 采样数组接口。因此 Constant 波形可生成原生参数；
Gaussian/DRAG 等只保留采样草案并标记为不可直接提交，等待 SHAPE_PULSE 或
设备专用接口验证。纯函数、无 SDK、无网络。
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List

import numpy as np

from ....libs.quantum_library.pulse_simulation import (
    SAMPLING_RATE,
    sample_waveform,
)

_PULSE_FIELDS = ("path", "phase", "amplitude", "width", "detuning")
_PARAMETER_FIELDS = ("freq_h", "freq_p", "makePps", "samplePath", "custom_freq")


def _samples(waveform: Any) -> np.ndarray:
    duration = int(getattr(waveform, "duration", 0))
    if duration <= 0:
        raise ValueError("波形 duration 必须是正整数样本数。")
    return np.asarray(
        [complex(sample_waveform(waveform, i / float(SAMPLING_RATE))) for i in range(duration)],
        dtype=np.complex128,
    )


def _pairs(values: np.ndarray) -> List[List[float]]:
    return [[float(value.real), float(value.imag)] for value in values]


def program_to_spinqlab_parameters(prog: Any) -> Dict[str, Any]:
    """生成 JSON-friendly SpinQLab 参数草案，并显式标记原生可提交性。"""
    ops = list(getattr(prog, "ops", []) or [])
    if not ops:
        raise ValueError("PulseProgram 为空，无法构造 SpinQLab 参数。")

    pulses: List[Dict[str, Any]] = []
    for qubit, waveform, raw_params in ops:
        params = dict(raw_params or {})
        channel = str(params.get("channel", "xy")).lower()
        if channel != "xy":
            raise ValueError("SpinQLab NMR Pulse 只支持 xy 驱动通道，不支持 z/readout。")
        path = int(params.get("path", qubit))
        duration = int(getattr(waveform, "duration", 0))
        samples = _samples(waveform)
        native = waveform.__class__.__name__.lower() == "constant"
        amplitude = float(getattr(waveform, "amp", 0.0)) * 100.0
        entry: Dict[str, Any] = {
            "path": path,
            "phase": math.degrees(float(params.get("phase", 0.0))) % 360.0,
            "amplitude": amplitude,
            "width": duration * 1e6 / float(SAMPLING_RATE),
            "detuning": float(params.get("detuning", 0.0)),
            "native": native,
            "source": {
                "qubit": int(qubit),
                "waveform": waveform.__class__.__name__,
                "sample_count": duration,
            },
        }
        if native:
            entry["sample_preview"] = _pairs(samples[:16])
        else:
            entry["samples"] = _pairs(samples)
            entry["unsupported_reason"] = (
                "SpinQLab Pulse 仅定义矩形参数；任意采样需另行验证 SHAPE_PULSE/设备接口。"
            )
        pulses.append(entry)

    metadata = getattr(prog, "metadata", {}) or {}
    spinqlab_meta = metadata.get("spinqlab", {}) if isinstance(metadata, dict) else {}
    parameters = {
        "makePps": bool(spinqlab_meta.get("makePps", True)),
        "samplePath": int(spinqlab_meta.get("samplePath", pulses[0]["path"])),
        "custom_freq": bool(spinqlab_meta.get("custom_freq", False)),
    }
    for key in ("freq_h", "freq_p"):
        if spinqlab_meta.get(key) is not None:
            parameters[key] = float(spinqlab_meta[key])

    draft = {
        "schema": "tyxonq.spinqlab.parameters.v1",
        "experiment_type": "NMR_PHENOMENON_AND_SIGNAL",
        "sampling_rate_hz": float(SAMPLING_RATE),
        "pulses": pulses,
        "parameters": parameters,
        "feasibility": {
            "native_pulse_only": all(entry["native"] for entry in pulses),
            "requires_shape_pulse_validation": any(not entry["native"] for entry in pulses),
        },
    }
    validate_spinqlab_parameters(draft)
    return draft


def validate_spinqlab_parameters(draft: Dict[str, Any]) -> Dict[str, Any]:
    """按 SpinQLab Link 1.0.2 文档范围校验数值；允许非原生草案离线存在。"""
    if not isinstance(draft, dict) or draft.get("schema") != "tyxonq.spinqlab.parameters.v1":
        raise ValueError("不是有效的 SpinQLab 参数草案。")
    pulses = draft.get("pulses")
    if not isinstance(pulses, list) or not pulses:
        raise ValueError("SpinQLab 参数草案必须包含至少一个 pulse。")
    if float(draft.get("sampling_rate_hz", 0.0)) <= 0:
        raise ValueError("sampling_rate_hz 必须大于 0。")
    for pulse in pulses:
        for field in _PULSE_FIELDS:
            value = float(pulse[field])
            if not math.isfinite(value):
                raise ValueError(f"SpinQLab {field} 必须是有限数值。")
        if int(pulse["path"]) not in (0, 1):
            raise ValueError("SpinQLab path 只能是 0（氢）或 1（磷）。")
        if not 0.0 < float(pulse["width"]) <= 200000.0:
            raise ValueError("SpinQLab width 必须在 (0, 200000] 微秒。")
        if not 0.0 <= float(pulse["amplitude"]) <= 100.0:
            raise ValueError("SpinQLab amplitude 必须在 [0, 100] 百分比。")
        if not 0.0 <= float(pulse["phase"]) <= 360.0:
            raise ValueError("SpinQLab phase 必须在 [0, 360] 度。")
        if not 0.0 <= float(pulse["detuning"]) <= 10000.0:
            raise ValueError("SpinQLab detuning 必须在 [0, 10000] Hz。")
    unknown = set(draft.get("parameters", {})) - set(_PARAMETER_FIELDS)
    if unknown:
        raise ValueError(f"未确认的 SpinQLab 实验参数: {sorted(unknown)}")
    return draft


def native_pulse_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """提取可交给官方 Pulse 的五个字段；非原生草案禁止提交。"""
    if not entry.get("native"):
        raise ValueError(entry.get("unsupported_reason") or "该波形不能作为原生 SpinQLab Pulse 提交。")
    return {field: entry[field] for field in _PULSE_FIELDS}


def spinqlab_parameters_to_jsonable(draft: Dict[str, Any]) -> Dict[str, Any]:
    validate_spinqlab_parameters(draft)
    return json.loads(json.dumps(draft, ensure_ascii=False, allow_nan=False))


def spinqlab_parameters_from_jsonable(data: Dict[str, Any]) -> Dict[str, Any]:
    return validate_spinqlab_parameters(dict(data))


__all__ = [
    "native_pulse_fields",
    "program_to_spinqlab_parameters",
    "spinqlab_parameters_from_jsonable",
    "spinqlab_parameters_to_jsonable",
    "validate_spinqlab_parameters",
]
