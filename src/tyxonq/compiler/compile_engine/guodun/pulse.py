"""国盾原生门脉冲 QCIS 的生成和安全校验。"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from typing import Any, Sequence

import numpy as np


_NATIVE_GATES = {"X2P", "X2M", "Y2P", "Y2M", "CZ"}
_QUBIT = re.compile(r"^Q\d+$")
_COUPLER = re.compile(r"^G\d+$")
_QAGENT = re.compile(r"^[QG]\d+$")
_MAX_PULSE_LENGTH_NS = 49_984
_PLATFORM_PHASE_LIMIT = 3.1415926


def _finite_number(value: Any, field_name: str) -> float:
    """把 QCIS 数值字段转成有限浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是数值，实际为 {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} 必须是有限数值")
    return number


def _normalize_native_phase(value: Any) -> float:
    """把 SDK 模板中的数学 ±pi 对齐到服务端固定小数边界。"""
    phase = _finite_number(value, "PXY phase")
    if math.isclose(abs(phase), math.pi, rel_tol=0.0, abs_tol=1e-12):
        return math.copysign(_PLATFORM_PHASE_LIMIT, phase)
    return phase


def _pulse_length(value: Any) -> int:
    """校验脉冲时长；平台单位为 ns。"""
    number = _finite_number(value, "length")
    length = int(number)
    if number != length or not 1 <= length <= _MAX_PULSE_LENGTH_NS:
        raise ValueError(
            f"length 必须是 1 到 {_MAX_PULSE_LENGTH_NS} 的整数 ns"
        )
    return length


def _delay_length(value: Any) -> int:
    """校验 I 指令时长；同步模板允许使用 0 ns 占位。"""
    number = _finite_number(value, "I length")
    length = int(number)
    if number != length or not 0 <= length <= _MAX_PULSE_LENGTH_NS:
        raise ValueError(
            f"I length 必须是 0 到 {_MAX_PULSE_LENGTH_NS} 的整数 ns"
        )
    return length


def _extra_numbers(text: str | None) -> list[float]:
    """解析模板中的波形附加参数列表。"""
    if text is None or not text.strip() or text.strip() in {"[]", "None"}:
        return []
    try:
        value = ast.literal_eval(text.strip())
    except (SyntaxError, ValueError):
        # cqlib 的 CZ 模板会把 Slepian 参数写成空格分隔文本，
        # 例如 ``0.864 0.05 -0.1875 0.04166``，并不带方括号。
        normalized = text.strip().strip("[]()")
        values = [
            item for item in re.split(r"[\s,]+", normalized) if item
        ]
    else:
        values = list(value) if isinstance(value, (list, tuple)) else [value]
    return [_finite_number(item, "wave_params") for item in values]


def _validate_wave_parameters(
    wave_type: int,
    length_ns: int,
    wave_params: Sequence[float],
) -> None:
    """按 cqlib 1.3.11 的波形类型校验附加参数。"""
    if wave_type not in (-1, 0, 1, 2):
        raise ValueError("wave_type 只支持 -1、0、1、2")
    if wave_type == -1:
        if not wave_params:
            raise ValueError("numeric 波形必须提供逐点数据")
        if any(abs(value) > 1.0 for value in wave_params):
            raise ValueError("numeric 波形采样点必须位于 [-1, 1]")
    elif wave_type == 1:
        if not wave_params:
            raise ValueError("flattop 波形必须提供 edge")
        edge_ns = wave_params[0]
        if edge_ns < 0 or 2 * edge_ns > length_ns:
            raise ValueError("flattop edge 必须满足 0 <= 2 * edge <= length")
    elif wave_type == 2:
        if len(wave_params) < 4:
            raise ValueError("slepian 波形必须提供 thf、thi、lam2、lam3")
        if any(abs(value) > 1.0 for value in wave_params[:4]):
            raise ValueError("slepian 参数必须位于 [-1, 1]")


def _sample_envelope(
    get_pulse: Any,
    wave_type: int,
    sample_count: int,
    wave_params: Sequence[float],
) -> np.ndarray:
    """用锁定版本 SDK 的包络函数生成单位、无符号波形。"""
    if sample_count <= 1:
        raise ValueError("当前采样率下脉冲采样点不足")
    if wave_type == 0:
        samples = get_pulse._cosine_samples(amplitude=1.0, length=sample_count)
    elif wave_type == 1:
        edge_samples = wave_params[0] * get_pulse.z_sample_rate * 1e-9
        samples = get_pulse._flattop_samples(
            amplitude=1.0,
            length=sample_count,
            edge=edge_samples,
        )
    elif wave_type == 2:
        samples = get_pulse._slepian_samples(
            amplitude=1.0,
            length=sample_count,
            thf=wave_params[0],
            thi=wave_params[1],
            lam2=wave_params[2],
            lam3=wave_params[3],
        )
    else:
        raise ValueError("numeric 波形暂不进入原生门真机安全链路")
    envelope = np.asarray(samples, dtype=float)
    if envelope.size != sample_count or not np.all(np.isfinite(envelope)):
        raise ValueError("SDK 生成的波形包含缺失值或非有限值")
    return envelope


def _validate_xy_samples(
    get_pulse: Any,
    qubit: str,
    wave_type: int,
    length_ns: int,
    amplitude: float,
    phase: float,
    drag_alpha: float,
    wave_params: Sequence[float],
) -> dict[str, Any]:
    """生成并检查原生 XY 包络的每一个归一化采样点。"""
    sample_count = int(length_ns * get_pulse.xy_sample_rate * 1e-9)
    if wave_type != 0:
        envelope = _sample_envelope(
            get_pulse, wave_type, sample_count, wave_params
        )
        samples = np.abs(envelope * amplitude)
    else:
        samples = np.asarray(
            get_pulse._cosine_samples(
                amplitude=amplitude,
                length=sample_count,
                carrier_phase=phase,
                drag_alpha=drag_alpha,
            ),
            dtype=float,
        )
    if samples.size != sample_count or not np.all(np.isfinite(samples)):
        raise ValueError(f"{qubit} XY 波形包含缺失值或非有限值")
    sample_min = float(np.min(samples))
    sample_max = float(np.max(samples))
    if sample_min < -1e-12 or sample_max > 1.0 + 1e-12:
        raise ValueError(
            f"{qubit} XY 归一化采样范围 [{sample_min}, {sample_max}] 超出 [0, 1]"
        )
    return {
        "qagent": qubit,
        "channel": "XY",
        "sample_count": sample_count,
        "mapped_min": sample_min,
        "mapped_max": sample_max,
        "safe_min": 0.0,
        "safe_max": 1.0,
    }


def _validate_z_samples(
    get_pulse: Any,
    qagent: str,
    wave_type: int,
    length_ns: int,
    amplitude: float,
    call_mapper: int,
    wave_params: Sequence[float],
) -> dict[str, Any]:
    """保留幅度正负号，并逐点检查 Z 波形映射后的 AWG 码值。"""
    sample_count = int(length_ns * get_pulse.z_sample_rate * 1e-9)
    envelope = _sample_envelope(get_pulse, wave_type, sample_count, wave_params)
    physical_samples = envelope * amplitude
    if call_mapper == 1:
        if qagent.startswith("Q"):
            mapped = get_pulse.f01_shift_2_detune(qagent, physical_samples)
        else:
            mapped = get_pulse.coupler_strength_2_zpulse_amp(
                qagent, physical_samples
            )
    else:
        mapped = physical_samples
    mapped_samples = np.asarray(mapped, dtype=float)
    if mapped_samples.size != sample_count or not np.all(np.isfinite(mapped_samples)):
        raise ValueError(f"{qagent} Z 波形包含缺失值或非有限值")
    safe_min, safe_max = map(float, get_pulse.get_qagent_zbias_amp_range(qagent))
    sample_min = float(np.min(mapped_samples))
    sample_max = float(np.max(mapped_samples))
    if sample_min < safe_min - 1e-12 or sample_max > safe_max + 1e-12:
        raise ValueError(
            f"{qagent} Z 码值范围 [{sample_min}, {sample_max}] "
            f"超出当前安全范围 [{safe_min}, {safe_max}]"
        )
    return {
        "qagent": qagent,
        "channel": "Z",
        "sample_count": sample_count,
        "mapped_min": sample_min,
        "mapped_max": sample_max,
        "safe_min": safe_min,
        "safe_max": safe_max,
        "call_mapper": call_mapper,
    }


def validate_pulse_qcis(
    qcis: str,
    *,
    get_pulse: Any | None = None,
    require_measurement: bool = False,
) -> dict[str, Any]:
    """校验 pulse-QCIS 语法，并在有当前标定时逐点检查波形。"""
    if not isinstance(qcis, str) or not qcis.strip():
        raise ValueError("pulse-QCIS 不能为空")
    stats: Counter[str] = Counter()
    channels: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(qcis.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        command = line.split(maxsplit=1)[0]
        stats[command] += 1
        try:
            if command == "PXY":
                parts = line.split(maxsplit=8)
                if len(parts) not in (8, 9) or not _QUBIT.fullmatch(parts[1]):
                    raise ValueError("PXY 字段或目标比特格式错误")
                wave_type = int(parts[2])
                length_ns = _pulse_length(parts[3])
                amplitude = _finite_number(parts[4], "PXY amplitude")
                frequency = _finite_number(parts[5], "PXY frequency")
                phase = _finite_number(parts[6], "PXY phase")
                drag_alpha = _finite_number(parts[7], "PXY dragalpha")
                params = _extra_numbers(parts[8] if len(parts) == 9 else None)
                _validate_wave_parameters(wave_type, length_ns, params)
                if not 4e9 <= frequency <= 6e9:
                    raise ValueError("PXY frequency 必须位于 [4e9, 6e9] Hz")
                if not -_PLATFORM_PHASE_LIMIT <= phase <= _PLATFORM_PHASE_LIMIT:
                    raise ValueError(
                        "PXY phase 必须位于 "
                        f"[-{_PLATFORM_PHASE_LIMIT}, {_PLATFORM_PHASE_LIMIT}]"
                    )
                if not -10 <= drag_alpha <= 10:
                    raise ValueError("PXY dragalpha 必须位于 [-10, 10]")
                if get_pulse is not None:
                    channels.append(
                        _validate_xy_samples(
                            get_pulse,
                            parts[1],
                            wave_type,
                            length_ns,
                            amplitude,
                            phase,
                            drag_alpha,
                            params,
                        )
                    )
            elif command in {"PZ", "PZ0"}:
                parts = line.split(maxsplit=6)
                if len(parts) not in (6, 7) or not _QAGENT.fullmatch(parts[1]):
                    raise ValueError(f"{command} 字段或 qagent 格式错误")
                wave_type = int(parts[2])
                length_ns = _pulse_length(parts[3])
                amplitude = _finite_number(parts[4], f"{command} amplitude")
                call_mapper = int(parts[5])
                if call_mapper not in (0, 1):
                    raise ValueError("call_mapper 只允许 0 或 1")
                params = _extra_numbers(parts[6] if len(parts) == 7 else None)
                _validate_wave_parameters(wave_type, length_ns, params)
                if get_pulse is not None:
                    channels.append(
                        _validate_z_samples(
                            get_pulse,
                            parts[1],
                            wave_type,
                            length_ns,
                            amplitude,
                            call_mapper,
                            params,
                        )
                    )
            elif command == "G":
                parts = line.split()
                if len(parts) != 4 or not _COUPLER.fullmatch(parts[1]):
                    raise ValueError("G 字段或 coupler 格式错误")
                length_ns = _pulse_length(parts[2])
                coupling = _finite_number(parts[3], "coupling_strength")
                if get_pulse is not None:
                    mapped = np.asarray(
                        get_pulse.coupler_strength_2_zpulse_amp(
                            parts[1],
                            np.full(
                                int(length_ns * get_pulse.z_sample_rate * 1e-9),
                                coupling,
                            ),
                        ),
                        dtype=float,
                    )
                    safe_min, safe_max = get_pulse.get_qagent_zbias_amp_range(parts[1])
                    if (
                        not np.all(np.isfinite(mapped))
                        or float(np.min(mapped)) < float(safe_min) - 1e-12
                        or float(np.max(mapped)) > float(safe_max) + 1e-12
                    ):
                        raise ValueError(f"{parts[1]} G 波形超出当前安全范围")
            elif command == "I":
                parts = line.split()
                if len(parts) != 3 or not _QAGENT.fullmatch(parts[1]):
                    raise ValueError("I 字段或 qagent 格式错误")
                _delay_length(parts[2])
            elif command == "B":
                parts = line.split()
                if (
                    len(parts) != 4
                    or not _QUBIT.fullmatch(parts[1])
                    or not _QUBIT.fullmatch(parts[2])
                    or not _COUPLER.fullmatch(parts[3])
                ):
                    raise ValueError("B 必须是两个比特和一个 coupler")
            elif command == "M":
                parts = line.split()
                if len(parts) != 2 or not _QUBIT.fullmatch(parts[1]):
                    raise ValueError("M 必须测量一个物理比特")
            elif command == "RZ":
                parts = line.split()
                if len(parts) != 3 or not _QUBIT.fullmatch(parts[1]):
                    raise ValueError("RZ 字段或目标比特格式错误")
                _finite_number(parts[2], "RZ angle")
            else:
                raise ValueError(f"不支持的 pulse-QCIS 指令 {command!r}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"pulse-QCIS 第 {line_number} 行无效：{exc}") from exc

    if require_measurement and stats["M"] == 0:
        raise ValueError("真机 pulse-QCIS 必须包含显式 M 测量指令")
    return {"gate_stats": dict(stats), "channels": channels}


def compile_native_gate_pulse(
    get_pulse: Any,
    gate: str,
    qagent: str,
    *,
    measure_qubits: Sequence[str] = (),
) -> dict[str, Any]:
    """读取当前标定模板，生成一个国盾原生门的 pulse-QCIS。"""
    gate_name = str(gate).upper()
    target = str(qagent).upper()
    if gate_name not in _NATIVE_GATES:
        raise ValueError(f"原生门只支持 {sorted(_NATIVE_GATES)}")
    expected = _COUPLER if gate_name == "CZ" else _QUBIT
    if not expected.fullmatch(target):
        kind = "coupler，例如 G0" if gate_name == "CZ" else "比特，例如 Q0"
        raise ValueError(f"{gate_name} 的 qagent 必须是{kind}")

    parameters = dict(
        get_pulse.get_gate_pulse_parameter(
            gate=gate_name,
            qagent=target,
        )
    )
    if "phase" in parameters:
        parameters["phase"] = _normalize_native_phase(parameters["phase"])
    template = get_pulse.get_gate_pulse_qcis_template(gate=gate_name)
    try:
        pulse_qcis = template.format(**parameters).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"当前 {gate_name} 标定模板无法填充：{exc}") from exc

    measurements: list[str] = []
    for qubit in measure_qubits:
        normalized = str(qubit).upper()
        if not _QUBIT.fullmatch(normalized):
            raise ValueError("measure_qubits 必须使用 Q0 这类物理比特名")
        if normalized not in measurements:
            measurements.append(normalized)
    qcis = "\n".join(
        [pulse_qcis, *(f"M {qubit}" for qubit in measurements)]
    )
    safety = validate_pulse_qcis(
        qcis,
        get_pulse=get_pulse,
        require_measurement=bool(measurements),
    )
    return {
        "qcis": qcis,
        "gate": gate_name,
        "qagent": target,
        "measure_qubits": measurements,
        "parameters": parameters,
        "safety": safety,
    }


__all__ = [
    "compile_native_gate_pulse",
    "validate_pulse_qcis",
]
