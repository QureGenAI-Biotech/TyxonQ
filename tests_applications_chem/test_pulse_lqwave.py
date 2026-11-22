"""P0：pulse → lqcloud ``run_wave`` 真机通路（lqwave_adapter + driver 分支）离线测试。

三层测试策略（仿 ``test_lqcloud_chem_realhw.py`` 惯例）：

1. **adapter 纯函数层**（总是跑，零依赖零联网）：参数化波形 → IQ 采样
   数组的解析对照（Gaussian 峰值/对称性）、通道映射（xy/z/readout）、
   多段串联、block_intervals 的 ns 换算、JSON 序列化跨进程往返。

2. **driver 分支层**（总是跑，fake SDK 替身）：``run(source=waves)``
  正确派发到 ``backend.run_wave``（而非 ``backend.run``）、非法 opts
  被 TypeError 拒绝、门线路（QuantumCircuit）仍走 ``backend.run``；
  ``get_task_details`` 把 memory/iq_data 透出到 result_meta。

3. **真机 run_wave 冒烟**（skipif 无 token/SDK/设备；显式 opt-in）：
   提交一条最小 XY 波形并取回 counts。**会消耗云资源**，仅在环境变量
   齐备（``TYXONQ_LQCLOUD_API_KEY`` + ``TYXONQ_QD_LQCLOUD_DEVICE``）时执行。

本文件属 ``tests_applications_chem``：不带 KMP 标志。
token 只从环境变量读取、绝不回显、绝不写入任何持久化位置。
"""

from __future__ import annotations

import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# 环境探测（只读）
# --------------------------------------------------------------------------- #
def _has_lqcloud() -> bool:
    return importlib.util.find_spec("lqcloud") is not None


def _live_wave_env() -> bool:
    return bool(os.getenv("TYXONQ_LQCLOUD_API_KEY")) and bool(
        os.getenv("TYXONQ_QD_LQCLOUD_DEVICE")
    )


# --------------------------------------------------------------------------- #
# 1. adapter 纯函数层
# --------------------------------------------------------------------------- #
def test_gaussian_sampling_analytic():
    """Gaussian 采样解析对照：峰值=amp、以中点对称、界外为零。"""
    from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import waveform_to_samples
    from tyxonq.waveforms import Gaussian

    wf = Gaussian(amp=0.5, duration=100, sigma=20.0)
    arr = waveform_to_samples(wf)
    assert arr.shape == (100,)
    assert arr.dtype == np.complex128
    # 峰值在正中、幅值 ≈ amp（离散采样有限偏差）
    assert abs(abs(arr[50]) - 0.5) < 0.05
    # Gaussian 无虚部；离散 τ 采样中心 0.5 处最大，向两端单调衰减
    assert np.allclose(arr.imag, 0.0)
    assert abs(arr[50]) >= abs(arr[49]) >= abs(arr[40]) >= abs(arr[20])
    assert abs(arr[0]) < 0.1 * abs(arr[50])
    assert abs(arr[-1]) < 0.1 * abs(arr[50])


def test_drag_sampling_has_quadrature():
    """DRAG 含虚部（正交分量）：包络复数化。"""
    from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import waveform_to_samples
    from tyxonq.waveforms import Drag

    wf = Drag(amp=0.5, duration=100, sigma=20.0, beta=0.3)
    arr = waveform_to_samples(wf)
    assert np.any(np.abs(arr.imag) > 1e-6), "DRAG 应携带虚部分量"


def test_program_to_waves_channels_and_concat():
    """通道映射（xy/z）+ 同通道多段按 ops 顺序串联 + physical_qubit 键名。"""
    from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import program_to_waves
    from tyxonq.core.ir.pulse import PulseProgram

    prog = PulseProgram(2)
    prog.constant(0, amp=0.1, duration=100, channel="z")
    prog.gaussian(0, amp=0.4, duration=80, sigma=20.0)
    prog.drag(1, amp=0.2, duration=60, sigma=15.0, beta=0.1, physical_qubit="q13")

    waves = program_to_waves(prog)
    assert set(waves.keys()) == {"0.z", "0.xy", "q13.xy"}
    assert len(np.asarray(waves["0.z"][0])) == 100
    assert len(np.asarray(waves["0.xy"][0])) == 80
    assert len(np.asarray(waves["q13.xy"][0])) == 60


def test_program_to_waves_readout_channel():
    """readout 通道映射为 (start_index, weights)。"""
    from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import program_to_waves
    from tyxonq.core.ir.pulse import PulseProgram
    from tyxonq.waveforms import Constant

    prog = PulseProgram(1)
    prog.add_pulse(0, Constant(amp=0.1, duration=50), channel="readout",
                   readout=(20, [0.1, 0.2, 0.3]))
    waves = program_to_waves(prog)
    assert set(waves.keys()) == {"0.readout"}
    start_idx, weights = waves["0.readout"]
    assert start_idx == 20
    assert np.allclose(weights, [0.1, 0.2, 0.3])


def test_program_to_waves_empty_rejected():
    from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import program_to_waves
    from tyxonq.core.ir.pulse import PulseProgram

    with pytest.raises(ValueError):
        program_to_waves(PulseProgram(1))


def test_is_waves_dict():
    from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import is_waves_dict

    assert is_waves_dict({"0.xy": [np.zeros(4)]})
    assert is_waves_dict({"q1.z": [1.0, 2.0], "c1_2.z": [1.0]})
    assert not is_waves_dict({})
    assert not is_waves_dict({"foo": [1.0]})  # 缺通道名
    assert not is_waves_dict({"0.gauss": [1.0]})  # 非法通道
    assert not is_waves_dict("0.xy")
    assert not is_waves_dict(None)


def test_suggest_block_intervals_ns():
    """样本 → ns 换算：160 样本 @ 2 GHz = 80 ns。"""
    from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import suggest_block_intervals

    waves = {"0.xy": [np.zeros(160)], "1.z": [np.zeros(60)]}
    bi = suggest_block_intervals(waves)
    assert bi == [[0.0, 80.0]]


def test_waves_json_roundtrip():
    """JSON 序列化跨进程往返（复数 pairs / 实数 flat / readout）。"""
    from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import (
        program_to_waves,
        waves_from_jsonable,
        waves_to_jsonable,
    )
    from tyxonq.core.ir.pulse import PulseProgram
    from tyxonq.waveforms import Constant

    prog = PulseProgram(2)
    prog.drag(0, amp=0.5, duration=64, sigma=16.0, beta=0.2)
    prog.add_pulse(1, Constant(amp=0.1, duration=32), channel="z")
    prog.add_pulse(0, Constant(amp=0.1, duration=8), channel="readout",
                  readout=(4, [0.5, 0.5]))

    waves = program_to_waves(prog)
    blob = json.dumps(waves_to_jsonable(waves))
    reloaded = waves_from_jsonable(json.loads(blob))
    assert set(reloaded.keys()) == set(waves.keys())
    a1 = np.asarray(waves["0.xy"][0])
    a2 = np.asarray(reloaded["0.xy"][0])
    assert a1.shape == a2.shape
    assert np.allclose(a1, a2)
    z2 = np.asarray(reloaded["1.z"][0])
    assert z2.shape == (32,) and np.allclose(z2.real, np.asarray(waves["1.z"][0]).real)
    assert reloaded["0.readout"][0] == 4


# --------------------------------------------------------------------------- #
# 2. driver 分支层（fake SDK 替身，不联网）
# --------------------------------------------------------------------------- #
class _FakeCircuit:
    """替身 QuantumCircuit（与 _make_fake_driver 提供的同一个类）。"""


class _FakeJob:
    def __init__(self, job_id: str):
        self.job_id = job_id

    def status_info(self):
        return {
            "status": "completed",
            "result": {
                "counts": {"00": 500, "11": 524},
                "shots": 1024,
                "result_format": "counts",
            },
        }


class _FakeBackend:
    def __init__(self):
        self.wave_calls: list = []
        self.run_calls: list = []

    def run(self, source, shots=1024, **kwargs):
        self.run_calls.append((source, shots, kwargs))
        return _FakeJob(f"fake-run-{len(self.run_calls)}")

    def run_wave(self, waves, shots=1024, block_intervals=None,
                 result_format="counts", readout_correction=False):
        self.wave_calls.append((waves, shots, block_intervals,
                                result_format, readout_correction))
        return _FakeJob(f"fake-wave-{len(self.wave_calls)}")


def _make_fake_driver(monkeypatch, backend: _FakeBackend):
    """把 lqcloud driver 的 SDK 类与 provider 换成替身（不联网）。"""
    from tyxonq.devices.hardware.lqcloud import driver as lqdrv

    class _FakeProvider:
        def __init__(self, api_key=None, interactive=False):
            self.api_key = api_key

        def get_backend(self, name):
            return backend

    monkeypatch.setattr(
        lqdrv, "_sdk_classes",
        lambda: (_FakeProvider, _FakeCircuit, _get_real_result_class()),
    )
    return lqdrv


def _get_real_result_class():
    """Result 用 SDK 真类（get_counts/get_memory/get_iq_data 解析逻辑要真）。"""
    from lqcloud import Result

    return Result


def test_driver_run_wave_dispatch(monkeypatch):
    """waves dict → backend.run_wave（而非 backend.run）；参数正确透传。"""
    backend = _FakeBackend()
    lqdrv = _make_fake_driver(monkeypatch, backend)

    waves = {"0.xy": [np.asarray([0.1 + 0.0j] * 8)]}
    tasks = lqdrv.run(
        "AGate-100",
        source=waves,
        shots=512,
        api_key="fake-key",
        result_format="counts",
        readout_correction=False,
    )
    assert len(tasks) == 1 and tasks[0].device == "AGate-100"
    assert backend.wave_calls and not backend.run_calls
    got_waves, got_shots, got_bi, got_rf, got_rc = backend.wave_calls[0]
    assert got_waves is waves and got_shots == 512 and got_rf == "counts"
    assert got_bi is None and got_rc is False


def test_driver_run_wave_rejects_unknown_opts(monkeypatch):
    backend = _FakeBackend()
    lqdrv = _make_fake_driver(monkeypatch, backend)
    waves = {"0.xy": [np.zeros(4)]}
    with pytest.raises(TypeError):
        lqdrv.run("AGate-100", source=waves, shots=8, api_key="fake-key", noise=0.1)


def test_driver_gate_circuit_still_backend_run(monkeypatch):
    """门线路（lqcloud.QuantumCircuit）仍走 backend.run 原路径。"""
    backend = _FakeBackend()
    lqdrv = _make_fake_driver(monkeypatch, backend)

    qc = _FakeCircuit()
    tasks = lqdrv.run("MQ02", source=qc, shots=64, api_key="fake-key")
    assert backend.run_calls and not backend.wave_calls
    assert backend.run_calls[0][0] is qc


def test_driver_get_task_details_exposes_memory_and_iq(monkeypatch):
    """get_task_details 透出 memory 与 iq_data（run_wave 专属结果）。"""
    backend = _FakeBackend()
    lqdrv = _make_fake_driver(monkeypatch, backend)

    job = SimpleNamespace(
        job_id="wave-1",
        status_info=lambda: {
            "status": "completed",
            "result": {
                "memory": ["00", "11", "00"],
                "corrected_counts": None,
                "result_format": "memory",
                "iq_data": {"q0": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
                "shots": 3,
            },
        },
    )
    task = lqdrv.LQCloudTask(id="wave-1", device="AGate-100", job=job)
    info = lqdrv.get_task_details(task)
    assert info["uni_status"] == "completed"
    meta = info["result_meta"]
    assert meta["memory"] == ["00", "11", "00"]
    assert meta["iq_data"]["q0"][0] == [0.1, 0.2, 0.3]
    # memory 格式下 counts 由聚合得出
    assert info["result"] == {"00": 2, "11": 1}


def test_driver_get_task_details_counts_only_has_no_memory(monkeypatch):
    """counts 格式（无 memory/iq_data）不产生多余键。"""
    backend = _FakeBackend()
    lqdrv = _make_fake_driver(monkeypatch, backend)
    job = SimpleNamespace(
        job_id="run-1",
        status_info=lambda: {
            "status": "completed",
            "result": {"counts": {"0": 512, "1": 512}, "shots": 1024,
                       "result_format": "counts"},
        },
    )
    task = lqdrv.LQCloudTask(id="run-1", device="MQ02", job=job)
    info = lqdrv.get_task_details(task)
    assert "memory" not in info["result_meta"]
    assert "iq_data" not in info["result_meta"]
    assert info["result"] == {"0": 512, "1": 512}


# --------------------------------------------------------------------------- #
# 3. 真机 run_wave 冒烟（显式 opt-in）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (_has_lqcloud() and _live_wave_env()),
    reason="真机 run_wave 冒烟需 TYXONQ_LQCLOUD_API_KEY 与 TYXONQ_QD_LQCLOUD_DEVICE（消耗云资源）",
)
def test_lqcloud_run_wave_live_smoke():
    """真机最小 XY 波形提交（Rabi 类驱动）→ counts 回读。

    注意：波形参数为占位值，仅验证「waves 构造 → run_wave 提交 → 轮询 →
    counts 返回」链路贯通，不代表物理正确性。
    """
    from tyxonq.compiler.platforms.lqcloud.lqwave_adapter import (
        program_to_waves,
        suggest_block_intervals,
    )
    from tyxonq.core.ir.pulse import PulseProgram
    from tyxonq.devices import base as device_base

    device = os.getenv("TYXONQ_QD_LQCLOUD_DEVICE")
    prog = PulseProgram(1)
    prog.gaussian(0, amp=0.2, duration=100, sigma=25.0)
    waves = program_to_waves(prog)

    tasks = device_base.run(
        provider="lqcloud",
        device=device,
        source=waves,
        shots=256,
        result_format="counts",
    )
    details = device_base.get_task_details(tasks[0], wait=True, timeout=120.0)
    counts = details.get("result") or {}
    assert details.get("uni_status") == "completed", details
    assert sum(counts.values()) > 0, "真机 run_wave 应返回非空计数"
