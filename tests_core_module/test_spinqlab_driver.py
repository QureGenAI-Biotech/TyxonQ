"""SpinQLab Link fake SDK 驱动测试，不连接 TCP 设备。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tyxonq.compiler.platforms.spinq.waveform_adapter import program_to_spinqlab_parameters
from tyxonq.core.ir.pulse import PulseProgram
from tyxonq.devices import base
from tyxonq.devices.hardware.spinq import spinqlab_driver


class _FakePulse:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeExperimentType:
    NMR_PHENOMENON_AND_SIGNAL = "NMR_PHENOMENON_AND_SIGNAL"


class _FakeClient:
    instances = []
    login_ok = True

    def __init__(self, **connection):
        self.connection = connection
        self.calls = []
        self.parameters = SimpleNamespace()
        self.experiment = SimpleNamespace(id="exp-mock")
        self.__class__.instances.append(self)

    def connect(self):
        self.calls.append("connect")
        return True

    def wait_for_login(self):
        self.calls.append("login")
        return self.login_ok

    def register_experiment(self, kind):
        self.calls.append(("register", kind))
        return self.experiment, self.parameters

    def run_experiment(self):
        self.calls.append("run")

    def wait_for_experiment_completion(self):
        self.calls.append("wait")
        return True

    def get_experiment_result(self):
        self.calls.append("result")
        return {"result": {"probability": {"0": 1.0}, "graph": []}}

    def deregister_experiment(self):
        self.calls.append("deregister")

    def disconnect(self):
        self.calls.append("disconnect")


@pytest.fixture(autouse=True)
def clean_fake(monkeypatch):
    _FakeClient.instances.clear()
    _FakeClient.login_ok = True
    for name in (
        "TYXONQ_SPINQLAB_HOST", "TYXONQ_SPINQLAB_PORT",
        "TYXONQ_SPINQLAB_ACCOUNT", "TYXONQ_SPINQLAB_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_sdk(monkeypatch):
    monkeypatch.setattr(
        spinqlab_driver, "_sdk_api", lambda: (_FakeClient, _FakeExperimentType, _FakePulse)
    )


def _native_draft():
    return program_to_spinqlab_parameters(PulseProgram(1).constant(0, amp=0.5, duration=80))


def test_native_pulse_lifecycle_and_cleanup(fake_sdk):
    tasks = spinqlab_driver.run(
        "fake-device", source=_native_draft(), shots=1,
        host="127.0.0.1", port=8181, account="user", password="pass",
    )
    client = _FakeClient.instances[0]
    assert client.calls[:2] == ["connect", "login"]
    assert client.calls[2] == ("register", "NMR_PHENOMENON_AND_SIGNAL")
    assert client.calls[3:] == ["run", "wait", "result", "deregister", "disconnect"]
    assert client.parameters.pulses[0].kwargs == {
        "path": 0, "phase": 0.0, "amplitude": 50.0, "width": 0.04, "detuning": 0.0
    }
    result = spinqlab_driver.get_task_details(tasks[0])
    assert result["uni_status"] == "completed"
    assert result["probabilities"] == {"0": 1.0}
    assert result["result_meta"]["result_kind"] == "nmr_experiment"


def test_login_failure_stops_before_registration(fake_sdk):
    _FakeClient.login_ok = False
    with pytest.raises(RuntimeError, match="登录失败"):
        spinqlab_driver.run(
            "fake", source=_native_draft(), shots=1,
            host="h", account="a", password="p",
        )
    calls = _FakeClient.instances[0].calls
    assert ("register", "NMR_PHENOMENON_AND_SIGNAL") not in calls
    assert calls[-1] == "disconnect"


def test_non_native_draft_is_rejected_before_connection(fake_sdk):
    draft = program_to_spinqlab_parameters(
        PulseProgram(1).gaussian(0, amp=0.2, duration=16, sigma=4.0)
    )
    with pytest.raises(ValueError, match="非矩形"):
        spinqlab_driver.run(
            "fake", source=draft, shots=1, host="h", account="a", password="p"
        )
    assert _FakeClient.instances == []


def test_shots_are_not_silently_mapped_to_repeat(fake_sdk):
    with pytest.raises(ValueError, match="shots=1"):
        spinqlab_driver.run(
            "fake", source=_native_draft(), shots=1024,
            host="h", account="a", password="p",
        )


def test_environment_connection_and_provider_registration(fake_sdk, monkeypatch):
    monkeypatch.setenv("TYXONQ_SPINQLAB_HOST", "env-host")
    monkeypatch.setenv("TYXONQ_SPINQLAB_ACCOUNT", "env-account")
    monkeypatch.setenv("TYXONQ_SPINQLAB_PASSWORD", "env-password")
    spinqlab_driver.run("fake", source=_native_draft(), shots=1)
    assert _FakeClient.instances[0].connection == {
        "host": "env-host", "port": 8181, "account": "env-account", "password": "env-password"
    }
    assert base.resolve_driver("spinqlab", "fake") is spinqlab_driver
