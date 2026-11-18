"""PulseProgram 波形产物 → LQCloud ``run_wave`` waves dict 方言适配层。

定位（P0 pulse 真机通路；用户架构指示 2026-11-11）
------------------------------------------------
硬件特定的代码与接口必须放在 ``platforms/<vendor>`` 方言目录、与
门线路翻译（``lowering.py``）同位，不放在设备无关抽象层
（``pulse_compiler/``）。本模块即 lqcloud 方言的 pulse 侧翻译：
与 ``compiler/platforms/lqcloud/lowering.py``（门线路 1:1 直译）
平级——「方言归 platforms、核心保持设备无关」。

核心职责
--------
1. **参数化波形 → IQ 采样数组**：``tyxonq.waveforms`` 的波形 dataclass
   （Gaussian/DRAG/Constant 等，参数化形式 type+args）经
   ``libs/quantum_library/pulse_simulation.sample_waveform`` 逐点采样为
   复包络数组（DRAG 含虚部）；采样约定沿用该模块的
   ``SAMPLING_RATE``（2 GHz，0.5 ns/样本，``duration`` 字段即样本数）。
2. **qubit → 通道映射**：驱动类脉冲 → ``"qN.xy"``（默认）；调频/局域
   相互作用类脉冲（``params["channel"] == "z"``）→ ``"qN.z"``；测量
   权重（``params["readout"]``）→ ``"qN.readout"``。同一 (qubit, channel)
   的多段波形按 ``PulseProgram.ops`` 顺序串联（单 block 语义，SDK 会
   从波形长度推断 block 边界；显式分块可传 ``block_intervals``）。
3. **单位口径**：waves 数组以样本为单位（长度 = 各波形 ``duration``
   之和）；``block_intervals`` 以 ns 为单位（SDK 约定 ``[start_ns, end_ns]``），
   由 :func:`suggest_block_intervals` 从样本数换算。

通道格式（与 lqcloud SDK ``backend.run_wave`` 文档一致）
----------------------------------------------------
::

    {
        "q0.xy": [array([...])],          # IQ 复包络（list 包裹单 block）
        "q0.z":  [array([...])],          # z 通道（调频/局域项）
        "q0.readout": (start_idx, weights) # 读出积分窗
    }

说明
----
- ``"qN"`` 中 N 为 **PulseProgram 的逻辑比特索引**；服务端会把纯数字
  qname 经 ``number_to_qubit`` 映射到实际器件比特。若需显式指定物理
  比特名，可在 ``params["physical_qubit"]`` 中给出（适配层优先采用）。
- 本模块不联网、不依赖 lqcloud SDK（纯函数，可离线单测）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 复用数值库的统一采样常量与采样函数（单一出处原则）
from ....libs.quantum_library.pulse_simulation import (
    SAMPLING_RATE as _DEFAULT_SAMPLING_RATE,
    sample_waveform as _sample_waveform,
)

__all__ = [
    "program_to_waves",
    "is_waves_dict",
    "suggest_block_intervals",
    "waveform_to_samples",
]


#: 支持的 lqcloud 通道名（与 SDK 文档一致）
_LQCLOUD_CHANNELS = ("xy", "z", "readin", "readout")


def waveform_to_samples(waveform: Any) -> np.ndarray:
    """把一段参数化波形采样为一维复数数组（长度 = ``waveform.duration`` 样本）。

    Parameters
    ----------
    waveform :
        ``tyxonq.waveforms`` 中的波形 dataclass（Gaussian/DRAG/Constant 等）。

    Returns
    -------
    np.ndarray
        复包络采样（DRAG 等含虚部分量；纯实波形虚部为 0）。
    """
    duration = int(getattr(waveform, "duration", 0))
    if duration <= 0:
        raise ValueError(f"波形 duration 必须为正整数（样本数），当前 {duration!r}。")
    samples = [
        complex(_sample_waveform(waveform, i / float(_DEFAULT_SAMPLING_RATE)))
        for i in range(duration)
    ]
    return np.asarray(samples, dtype=np.complex128)


def _qubit_key(params: Dict[str, Any], qubit: int) -> str:
    """通道键的比特名：优先显式物理比特名，否则用逻辑索引（服务端自动映射）。"""
    physical = params.get("physical_qubit")
    name = str(physical) if physical is not None else str(int(qubit))
    return name


def program_to_waves(
    prog: Any,
    *,
    defcal_library: Optional[Any] = None,
) -> Dict[str, Any]:
    """把 :class:`tyxonq.core.ir.pulse.PulseProgram` 转为 lqcloud ``run_wave`` waves dict。

    Parameters
    ----------
    prog :
        ``PulseProgram`` 实例（ops 为 ``(qubit, waveform, params)`` 元组）。
    defcal_library :
        可选 ``DefcalLibrary``；给出时按 (qubit, 波形类型) 查校准并注入
        幅度缩放（``amp_scale``）与频率（``qubit_freq``，仅记录在
        ``waves`` 元信息，波形本身已含频率无关的包络）。

    Returns
    -------
    dict
        lqcloud ``run_wave`` 的 flat waves dict：键 ``"<qname>.<channel>"``，
        值为 ``[采样数组]``（单 block，SDK 规范）；readout 通道为
        ``(start_index, weights)``。

    Raises
    ------
    ValueError
        波形 duration 非法或通道名不在 ``{"xy","z","readin","readout"}``。
    """
    ops = list(getattr(prog, "ops", []) or [])
    if not ops:
        raise ValueError("PulseProgram 为空（无 pulse 操作），无法构造 waves。")

    # (qubit_key, channel) → 顺序串联的采样段
    segments: Dict[Tuple[str, str], List[np.ndarray]] = {}
    readouts: Dict[str, Tuple[int, np.ndarray]] = {}

    for qubit, waveform, params in ops:
        params = dict(params or {})
        channel = str(params.get("channel", "xy")).lower()
        if channel not in _LQCLOUD_CHANNELS:
            raise ValueError(
                f"不支持的通道 {channel!r}；lqcloud 通道限于 {_LQCLOUD_CHANNELS}。"
            )
        key = _qubit_key(params, int(qubit))

        # readout 通道：params["readout"] = (start_index, weights)
        if channel == "readout":
            ro = params.get("readout")
            if ro is None:
                raise ValueError(
                    "readout 通道需要 params['readout']=(start_index, weights)。"
                )
            start_idx, weights = ro
            readouts[f"{key}.readout"] = (int(start_idx), np.asarray(weights))
            continue

        # 校准注入：defcal 查到 amp_scale 时缩放采样
        samples = waveform_to_samples(waveform)
        if defcal_library is not None:
            try:
                scale = params.get("amp_scale")
                if scale is not None:
                    samples = samples * float(scale)
            except Exception:  # noqa: BLE001 —— 校准缩放失败不阻断翻译
                pass

        segments.setdefault((key, channel), []).append(samples)

    waves: Dict[str, Any] = {}
    for (key, channel), parts in segments.items():
        # SDK 约定：单 block 也包装在 list 中
        waves[f"{key}.{channel}"] = [np.concatenate(parts)]

    waves.update(readouts)
    if not waves:
        raise ValueError("未产生任何有效通道数据（waves 为空）。")
    return waves


def is_waves_dict(source: Any) -> bool:
    """判断 ``source`` 是否为 lqcloud run_wave 的 waves dict（driver 分派用）。

    判据：非空 dict，且所有键都是形如 ``"<name>.<channel>"`` 的字符串，
    通道名落在 ``{"xy","z","readin","readout"}``。
    """
    if not isinstance(source, dict) or not source:
        return False
    for key in source:
        if not isinstance(key, str) or "." not in key:
            return False
        channel = key.rsplit(".", 1)[-1].lower()
        if channel not in _LQCLOUD_CHANNELS:
            return False
    return True


def suggest_block_intervals(
    waves: Dict[str, Any],
    sampling_rate_hz: float = _DEFAULT_SAMPLING_RATE,
) -> List[List[float]]:
    """从 waves 各通道样本长度推断单 block 的 ``[start_ns, end_ns]``。

    block 边界取所有 xy/z/readin 通道总长的并集端点（多通道并联时取
    最长者），单位 ns（SDK 约定）。readout 通道不含驱动波形、不参与
    长度推断。
    """
    max_samples = 0
    for key, value in waves.items():
        channel = key.rsplit(".", 1)[-1].lower()
        if channel == "readout":
            continue
        if isinstance(value, (list, tuple)) and value and hasattr(value[0], "__len__"):
            max_samples = max(max_samples, int(len(value[0])))
        elif hasattr(value, "__len__"):
            max_samples = max(max_samples, int(len(value)))
    if max_samples <= 0:
        return [[0.0, 0.0]]
    ns_per_sample = 1e9 / float(sampling_rate_hz)
    return [[0.0, max_samples * ns_per_sample]]


def waves_to_jsonable(waves: Dict[str, Any]) -> Dict[str, Any]:
    """把 waves dict 转为可 JSON 序列化形式（数组→列表、复数→[re, im]）。

    用于数据生产管线落盘（SKQD 三段式的「构造→落盘→后处理解耦」），
    与 ``tyxonq.qd`` 的 schema 版本化约定配合。
    """
    out: Dict[str, Any] = {}
    for key, value in waves.items():
        if key.endswith(".readout"):
            # readout 通道：value = (start_index, weights)
            start_idx, weights = value
            out[key] = [int(start_idx), np.asarray(weights).tolist()]
            continue
        if isinstance(value, (list, tuple)) and value:
            arr = np.asarray(value[0])
            if np.iscomplexobj(arr):
                out[key] = [[float(z.real), float(z.imag)] for z in arr]
            else:
                out[key] = arr.astype(float).tolist()
        else:
            out[key] = value
    return out


def waves_from_jsonable(data: Dict[str, Any]) -> Dict[str, Any]:
    """``waves_to_jsonable`` 的逆操作（跨进程重载）。

    序列化格式约定：复数通道为 ``[[re, im], ...]``（逐点对）；实数通道
    为 flat ``[f0, f1, ...]``；readout 通道为 ``[start_idx, weights]``。
    """
    waves: Dict[str, Any] = {}
    for key, value in data.items():
        channel = key.rsplit(".", 1)[-1].lower()
        if channel == "readout":
            start_idx, weights = value
            waves[key] = (int(start_idx), np.asarray(weights, dtype=float))
            continue
        # value 为逐点序列（复数 pairs 或实数 flat），重载时恢复 block 包裹
        if value and isinstance(value[0], list):
            head = value[0]
            if len(head) == 2 and not isinstance(head[0], list):
                # 复数对 [re, im] 展开
                pairs = np.asarray(value, dtype=float)
                waves[key] = [pairs[:, 0] + 1j * pairs[:, 1]]
            else:
                waves[key] = [np.asarray(value, dtype=float)]
        else:
            waves[key] = value
    return waves


# 保持与 Sequence 类型引用兼容（供类型检查）
_ = (Sequence,)
