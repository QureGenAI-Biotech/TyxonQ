"""国盾原生门脉冲离线测试；所有标定和平台对象均为 mock。"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tyxonq.compiler.compile_engine.guodun.pulse import (
    compile_native_gate_pulse,
    validate_pulse_qcis,
)
from tyxonq.devices.hardware.guodun import driver


def _config(disabled_qubits="", disabled_couplers=""):
    return {
        "disabledQubits": disabled_qubits,
        "disabledCouplers": disabled_couplers,
        "overview": {"coupler_map": {"G0": ["Q0", "Q7"]}},
    }


class FakeGetPulse:
    """只实现本组测试需要的 cqlib 1.3.11 接口。"""

    xy_sample_rate = 5e8
    z_sample_rate = 1e9

    def __init__(self, platform=None):
        self.platform = platform
        self.config = (
            platform.download_config() if platform is not None else _config()
        )

    def get_gate_pulse_parameter(self, gate, qagent):
        assert gate in {"X2P", "X2M", "Y2P", "Y2M"}
        phases = {
            "X2P": 0.0,
            "X2M": np.pi,
            "Y2P": np.pi / 2,
            "Y2M": -np.pi / 2,
        }
        return {
            "qubit": qagent,
            "wave_class": "0",
            "length": 20,
            "amplitude": 0.2,
            "frequency": 5e9,
            "phase": phases[gate],
            "drag_alpha": 0.0,
            "extra_param": [],
        }

    @staticmethod
    def get_gate_pulse_qcis_template(gate):
        assert gate in {"X2P", "X2M", "Y2P", "Y2M"}
        return (
            "PXY {qubit} {wave_class} {length} {amplitude} "
            "{frequency} {phase} {drag_alpha} {extra_param}"
        )

    @staticmethod
    def _cosine_samples(amplitude, length, **kwargs):
        return np.linspace(0.0, abs(float(amplitude)), int(length)).tolist()

    @staticmethod
    def _flattop_samples(amplitude, length, edge, **kwargs):
        samples = np.ones(int(length), dtype=float) * abs(float(amplitude))
        samples[0] = 0.0
        samples[-1] = 0.0
        return samples.tolist()

    @staticmethod
    def _slepian_samples(amplitude, length, **kwargs):
        return np.linspace(0.0, abs(float(amplitude)), int(length)).tolist()

    @staticmethod
    def get_qagent_zbias_amp_range(qagent):
        return (-1.0, 1.0)

    @staticmethod
    def f01_shift_2_detune(qubit, samples):
        return np.asarray(samples, dtype=float) / 10.0

    @staticmethod
    def coupler_strength_2_zpulse_amp(coupler, samples):
        return np.asarray(samples, dtype=float) / 10.0


def test_compile_native_single_qubit_pulse_from_current_template():
    compiled = compile_native_gate_pulse(
        FakeGetPulse(),
        "x2p",
        "q0",
        measure_qubits=["q0"],
    )

    assert compiled["qcis"] == "PXY Q0 0 20 0.2 5000000000.0 0.0 0.0 []\nM Q0"
    assert compiled["safety"]["gate_stats"] == {"PXY": 1, "M": 1}
    channel = compiled["safety"]["channels"][0]
    assert channel["qagent"] == "Q0"
    assert channel["sample_count"] == 10
    assert channel["mapped_max"] == pytest.approx(0.2)


def test_native_pi_phase_is_clamped_to_server_decimal_limit():
    compiled = compile_native_gate_pulse(
        FakeGetPulse(),
        "X2M",
        "Q0",
        measure_qubits=["Q0"],
    )

    assert " 3.1415926 " in compiled["qcis"]
    validate_pulse_qcis(compiled["qcis"], get_pulse=FakeGetPulse())
    with pytest.raises(ValueError, match="phase"):
        validate_pulse_qcis(
            "PXY Q0 0 20 0.2 5e9 3.141592653589793 0 []"
        )


def test_pulse_qcis_requires_measurement_only_for_hardware_mode():
    qcis = "PXY Q0 0 20 0.2 5e9 0 0 []"
    validate_pulse_qcis(qcis, get_pulse=FakeGetPulse())
    with pytest.raises(ValueError, match="必须包含显式 M"):
        validate_pulse_qcis(
            qcis,
            get_pulse=FakeGetPulse(),
            require_measurement=True,
        )


def test_z_waveform_preserves_negative_sign_and_checks_every_sample():
    safe = validate_pulse_qcis(
        "PZ Q0 1 10 -2.0 1 [2]",
        get_pulse=FakeGetPulse(),
    )
    channel = safe["channels"][0]
    assert channel["mapped_min"] == pytest.approx(-0.2)
    assert channel["mapped_max"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="超出当前安全范围"):
        validate_pulse_qcis(
            "PZ Q0 1 10 -2.0 0 [2]",
            get_pulse=FakeGetPulse(),
        )


def test_slepian_parameters_accept_cqlib_space_separated_format():
    safe = validate_pulse_qcis(
        "PZ G0 2 20 -2.0 1 0.864 0.05 -0.1875 0.04166666666666667",
        get_pulse=FakeGetPulse(),
    )
    channel = safe["channels"][0]
    assert channel["qagent"] == "G0"
    assert channel["sample_count"] == 20


def test_invalid_frequency_and_numeric_sample_are_rejected():
    with pytest.raises(ValueError, match="frequency"):
        validate_pulse_qcis("PXY Q0 0 20 0.2 3e9 0 0 []")
    with pytest.raises(ValueError, match="numeric 波形采样点"):
        validate_pulse_qcis("PZ Q0 -1 10 0 0 [0.0, 1.1]")


def test_topology_checks_pulse_coupler_and_barrier_pair():
    driver._validate_topology(
        ["B Q0 Q7 G0\nPZ G0 1 10 -2 1 [2]\nB Q0 Q7 G0"],
        _config(),
    )
    with pytest.raises(RuntimeError, match="禁用或未知耦合器"):
        driver._validate_topology(
            ["B Q0 Q7 G0"],
            _config(disabled_couplers="G0"),
        )
    with pytest.raises(RuntimeError, match="不连接"):
        driver._validate_topology(["B Q0 Q6 G0"], _config())


def test_open_pulse_context_downloads_current_config_without_submission(monkeypatch):
    platform = MagicMock()
    platform.download_config.return_value = _config()
    monkeypatch.setattr(driver, "_create_platform", lambda token, device: platform)
    monkeypatch.setattr(driver, "_load_get_pulse_class", lambda: FakeGetPulse)

    context = driver.open_pulse_context("gd_qc1", token="secret")

    assert context.device == "gd_qc1"
    assert context.platform is platform
    platform.download_config.assert_called_once_with()
    platform.submit_job.assert_not_called()
    platform.create_waveform_data.assert_not_called()


def test_waveform_create_and_query_are_strictly_separated():
    platform = MagicMock()
    platform.create_waveform_data.return_value = 901
    context = driver.GuodunPulseContext(
        device="gd_qc1",
        platform=platform,
        get_pulse=FakeGetPulse(),
    )
    source = "PXY Q0 0 20 0.2 5e9 0 0 []"

    task = driver.create_waveform(context, source, circuit_name="offline-test")

    assert task.id == 901
    platform.create_waveform_data.assert_called_once_with(
        circuit=source,
        circuit_name="offline-test",
        is_verify=True,
    )
    platform.query_waveform_data.assert_not_called()

    platform.query_waveform_data.side_effect = [None, "https://example/waveform"]
    assert driver.get_waveform(task)["status"] == "pending"
    completed = driver.get_waveform(task)
    assert completed["status"] == "completed"
    assert completed["url"] == "https://example/waveform"
    platform.create_waveform_data.assert_called_once()
    assert platform.query_waveform_data.call_count == 2


def test_waveform_invalid_query_id_stops_without_recreation():
    platform = MagicMock()
    platform.create_waveform_data.return_value = None
    context = driver.GuodunPulseContext(
        device="gd_qc1",
        platform=platform,
        get_pulse=FakeGetPulse(),
    )

    with pytest.raises(RuntimeError, match="不重建"):
        driver.create_waveform(
            context,
            "PXY Q0 0 20 0.2 5e9 0 0 []",
        )
    platform.create_waveform_data.assert_called_once()
    platform.query_waveform_data.assert_not_called()


def test_pulse_submit_refreshes_calibration_and_validates_before_submit(monkeypatch):
    platform = MagicMock()
    platform.download_config.return_value = _config()
    platform.submit_job.return_value = [902]
    monkeypatch.setattr(driver, "_create_platform", lambda token, device: platform)
    monkeypatch.setattr(driver, "_load_get_pulse_class", lambda: FakeGetPulse)

    tasks = driver.run(
        "gd_qc1",
        token="secret",
        source="I Q0 20\nM Q0",
        shots=20,
        exp_name="pulse-permission-check",
    )

    assert [task.id for task in tasks] == [902]
    platform.download_config.assert_called_once_with()
    platform.submit_job.assert_called_once_with(
        circuit=["I Q0 20\nM Q0"],
        exp_name="pulse-permission-check",
        num_shots=20,
        is_verify=True,
    )


def test_unsafe_pulse_stops_before_submit(monkeypatch):
    platform = MagicMock()
    platform.download_config.return_value = _config()
    monkeypatch.setattr(driver, "_create_platform", lambda token, device: platform)
    monkeypatch.setattr(driver, "_load_get_pulse_class", lambda: FakeGetPulse)

    with pytest.raises(ValueError, match="frequency"):
        driver.run(
            "gd_qc1",
            token="secret",
            source="PXY Q0 0 20 0.2 3e9 0 0 []\nM Q0",
            shots=20,
        )
    platform.submit_job.assert_not_called()
