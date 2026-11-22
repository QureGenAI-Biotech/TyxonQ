"""波形档（pulse/analog）时间演化编排——站在 ``PulseProgram`` 链式 IR 上。

分层（用户架构指示 2026-11-11）：

- 本模块（chem/dynamics 应用层）只做**物理编排**：把 ``PulseProgram``
  送到正确的执行链；
- **数值档**：``prog.state(backend=...)``（既有 ``libs/quantum_library/
  pulse_simulation`` 数值模拟链，含 T1/T2/anharmonicity）；
- **真机档**：``compiler/platforms/lqcloud/lqwave_adapter.program_to_waves``
  做方言翻译（参数化波形 → IQ 采样数组），再经 ``devices.base.run`` 的
  统一提交入口走 lqcloud ``run_wave``（driver 侧 SDK 1:1 透传）；
- 哈密顿量 → 波形序列的物理映射归模型层（``models/spin_lattice`` /
  ``models/exciton`` 的 ``to_pulse_program``）。

真机提交消耗云资源：调用方须逐次向用户确认（HANDOFF 纪律）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tyxonq.core.ir.pulse import PulseProgram


#: 落盘 schema 版本（agent/AI 友好：JSON 可序列化 + 版本化）
PULSE_RESULT_SCHEMA = "tyxonq.qd.pulse.v1"


def evolve_with_pulses(
    prog: PulseProgram,
    *,
    provider: Optional[str] = None,
    device: Optional[str] = None,
    shots: int = 1024,
    result_format: str = "counts",
    readout_correction: bool = False,
    block_intervals=None,
    backend: str = "numpy",
    **device_kwargs,
) -> Dict[str, Any]:
    """波形档演化（长时间窗高保真路线；provider 直切数值/真机档）。

    Parameters
    ----------
    prog :
        ``PulseProgram``（``.drag/.gaussian/...`` 链式构造的波形序列）。
    provider :
        ``None``（默认）= 数值模拟档（``prog.state(backend=...)``）；
        ``"lqcloud"`` = 真机 ``run_wave`` 提交（消耗云资源，逐次确认）。
    device :
        真机档的设备名（如 ``"AGate-100"``）；数值档忽略。
    shots / result_format / readout_correction / block_intervals :
        真机档 ``run_wave`` 选项（``result_format`` ∈ counts/memory/raw）。
    backend :
        数值档的数组后端（numpy/pytorch/cupy，见 ``prog.state``）。

    Returns
    -------
    dict
        数值档：``{"schema", "mode": "numeric", "state", "n_qubits"}``；
        真机档：``{"schema", "mode": "run_wave", "task", "device"}``
        （统一 task 结构经 ``devices.base.get_task_details`` 轮询取回）。
    """
    if not isinstance(prog, PulseProgram):
        raise TypeError("prog 必须是 tyxonq.core.ir.pulse.PulseProgram。")
    if provider is None:
        state = prog.state(backend=backend)
        return {
            "schema": PULSE_RESULT_SCHEMA,
            "mode": "numeric",
            "state": state,
            "n_qubits": int(prog.num_qubits),
        }
    if str(provider) != "lqcloud":
        raise ValueError(
            f"波形档当前仅支持 provider=None（数值）或 'lqcloud'（真机），"
            f"收到 {provider!r}。"
        )
    # 真机档：方言翻译归 platforms/lqcloud；提交走 devices 统一入口
    from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import (
        program_to_waves,
        suggest_block_intervals,
    )
    from tyxonq.devices import base as device_base

    waves = program_to_waves(prog)
    if block_intervals is None:
        block_intervals = suggest_block_intervals(waves)
    task = device_base.run(
        provider="lqcloud",
        device=device,
        source=waves,
        shots=int(shots),
        result_format=str(result_format),
        readout_correction=bool(readout_correction),
        block_intervals=block_intervals,
        **device_kwargs,
    )
    return {
        "schema": PULSE_RESULT_SCHEMA,
        "mode": "run_wave",
        "task": task,
        "device": str(device),
    }


def pulse_counts(result_details: Dict[str, Any]) -> Dict[str, int]:
    """从 ``devices.base.get_task_details`` 的返回取 counts（真机档便捷读取）。"""
    counts = (result_details or {}).get("result") or {}
    if not isinstance(counts, dict):
        raise TypeError("result_details['result'] 应为 counts dict。")
    return {str(k): int(v) for k, v in counts.items()}


__all__ = ["evolve_with_pulses", "pulse_counts", "PULSE_RESULT_SCHEMA"]
