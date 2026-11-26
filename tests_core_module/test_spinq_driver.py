"""SpinQit 云驱动离线测试。"""

from __future__ import annotations

import pytest

from tyxonq.devices import base
from tyxonq.devices.hardware.spinq import driver


class _FakeConfig:
    instances = []

    def __init__(self):
        self.calls = []
        self.__class__.instances.append(self)

    def configure_platform(self, value):
        self.calls.append(("platform", value))

    def configure_shots(self, value):
        self.calls.append(("shots", value))

    def configure_task(self, name, description):
        self.calls.append(("task", name, description))

    def configure_measure_qubits(self, qubits):
        self.calls.append(("measure_qubits", list(qubits)))

    def configure_keep_layout(self, layout):
        self.calls.append(("keep_layout", list(layout)))


class _FakeResult:
    def __init__(self, counts=None, probabilities=None, status="completed"):
        self.counts = counts
        self.probabilities = probabilities or {}
        self.status = status


class _FakePlatform:
    def __init__(self, available=True):
        self.machine_count = 1
        self._available = available

    def available(self):
        return self._available


class _FakeBackend:
    def __init__(self, result=None, available=True):
        self.result = result or _FakeResult({"00": 3, "11": 1}, {"00": 0.75, "11": 0.25})
        self.platform = _FakePlatform(available)
        self.execute_calls = []

    def get_platform(self, name):
        self.platform_name = name
        return self.platform

    def execute(self, source, config):
        self.execute_calls.append((source, config))
        return self.result


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    _FakeConfig.instances.clear()
    for name in (
        "TYXONQ_API_KEY",
        "TYXONQ_SPINQ_USERNAME",
        "TYXONQ_SPINQ_KEYFILE",
        "TYXONQ_SPINQ_HOST",
        "TYXONQ_SPINQ_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_sdk(monkeypatch):
    backend = _FakeBackend()
    auth_calls = []

    def fake_get_cloud(username, keyfile, host=None):
        auth_calls.append((username, keyfile, host))
        return backend

    monkeypatch.setattr(driver, "_sdk_api", lambda: (fake_get_cloud, _FakeConfig))
    return backend, auth_calls


def test_credentials_precedence_and_no_global_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("TYXONQ_SPINQ_USERNAME", "env-user")
    monkeypatch.setenv("TYXONQ_SPINQ_KEYFILE", "env-key")
    monkeypatch.setenv("TYXONQ_SPINQ_HOST", "env-host")
    opts = {"username": "explicit-user", "keyfile": "explicit-key", "host": "explicit-host"}
    assert driver._resolve_credentials(opts) == {
        "username": "explicit-user",
        "keyfile": "explicit-key",
        "host": "explicit-host",
    }
    assert opts == {}
    assert driver._resolve_credentials({})["username"] == "env-user"

    monkeypatch.delenv("TYXONQ_SPINQ_USERNAME")
    monkeypatch.setenv("TYXONQ_SPINQ_CONFIG", str(tmp_path / "missing.pypirc"))
    monkeypatch.setenv("TYXONQ_API_KEY", "wrong-provider")
    with pytest.raises(RuntimeError, match="SpinQ 凭证不足"):
        driver._resolve_credentials({})


def test_pypirc_credentials_and_relative_keyfile(monkeypatch, tmp_path):
    config = tmp_path / ".pypirc"
    config.write_text(
        "[spinqcloud]\nusername=file-user\nkeyfile=keys/test-key\nhost=http://cloud.example\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TYXONQ_SPINQ_CONFIG", str(config))
    resolved = driver._resolve_credentials({})
    assert resolved["username"] == "file-user"
    assert resolved["keyfile"] == str((tmp_path / "keys" / "test-key").resolve())
    assert resolved["host"] == "http://cloud.example"

    monkeypatch.setenv("TYXONQ_SPINQ_USERNAME", "env-user")
    assert driver._resolve_credentials({})["username"] == "env-user"


def test_run_executes_once_with_native_config(fake_sdk):
    backend, auth_calls = fake_sdk
    source = object()
    tasks = driver.run(
        "spinq::gemini_vp",
        source=source,
        shots=4,
        username="user",
        keyfile="/tmp/test-key",
        host="https://example.invalid",
        task_name="offline-mock",
        task_description="fake",
        compile_metadata={"physical_qubits": [0, 1], "cloud_measure_qubits": [1, 0]},
    )

    assert auth_calls == [("user", "/tmp/test-key", "https://example.invalid")]
    assert len(backend.execute_calls) == 1
    assert backend.execute_calls[0][0] is source
    config = _FakeConfig.instances[-1]
    assert ("platform", "gemini_vp") in config.calls
    assert ("shots", 4) in config.calls
    assert ("task", "offline-mock", "fake") in config.calls
    assert ("measure_qubits", [1, 0]) in config.calls
    assert len(tasks) == 1 and tasks[0].async_result is False


def test_sqc25_runtime_layout_and_compact_measurements(fake_sdk):
    backend, _ = fake_sdk
    driver.run(
        "SQC-25",
        source=object(),
        shots=8,
        username="u",
        keyfile="k",
        compile_metadata={
            "runtime_keep_layout": [15, 24],
            "cloud_measure_qubits": [0, 1],
        },
    )
    config = _FakeConfig.instances[-1]
    assert backend.platform_name == "sqc_25_vp"
    assert ("keep_layout", [15, 24]) in config.calls
    assert ("measure_qubits", [0, 1]) in config.calls


def test_run_rejects_invalid_inputs(fake_sdk):
    with pytest.raises(ValueError, match="shots=0"):
        driver.run("gemini_vp", source=object(), shots=0, username="u", keyfile="k")
    with pytest.raises(ValueError, match="一次只支持"):
        driver.run("gemini_vp", source=object(), shots=[1], username="u", keyfile="k")
    with pytest.raises(TypeError, match="physical_qubits"):
        driver.run(
            "gemini_vp", source=object(), shots=1,
            username="u", keyfile="k", physical_qubits=[0],
        )


def test_unavailable_platform_is_not_submitted(monkeypatch):
    backend = _FakeBackend(available=False)
    monkeypatch.setattr(driver, "_sdk_api", lambda: (lambda *a, **k: backend, _FakeConfig))
    with pytest.raises(RuntimeError, match="当前不可用"):
        driver.run("gemini_vp", source=object(), shots=1, username="u", keyfile="k")
    assert backend.execute_calls == []


def test_completed_counts_probabilities_and_little_endian_metadata(fake_sdk):
    tasks = driver.run(
        "gemini_vp", source=object(), shots=4,
        username="u", keyfile="k", compile_metadata={"mapping_stage": "compile_time_lowering"},
    )
    result = driver.get_task_details(tasks[0])
    assert result["uni_status"] == "completed"
    assert result["result"] == {"00": 3, "11": 1}
    assert result["probabilities"] == {"00": 0.75, "11": 0.25}
    assert result["result_meta"]["bit_order"] == "little_endian"
    assert result["result_meta"]["counts_shots"] == 4
    assert result["result_meta"]["compile"]["mapping_stage"] == "compile_time_lowering"


def test_probabilities_are_never_converted_to_fake_counts():
    task = driver.SpinQTask(
        id="local", device="gemini_vp", result=_FakeResult(None, {"1": 1.0}), shots=8
    )
    result = driver.get_task_details(task)
    assert result["uni_status"] == "error"
    assert result["result"] == {}
    assert "不会用 probabilities" in result["error"]


def test_failed_status_and_interfaces(fake_sdk):
    task = driver.SpinQTask(
        id="task", device="gemini_vp", result=_FakeResult(status="failed"), status="failed"
    )
    assert driver.get_task_details(task)["uni_status"] == "failed"
    with pytest.raises(NotImplementedError):
        driver.remove_task(task)
    with pytest.raises(NotImplementedError):
        driver.expval()
    assert driver.list_devices() == [
        "spinq::gemini_vp", "spinq::triangulum_vp", "spinq::superconductor_vp", "spinq::sqc_25_vp"
    ]
    assert driver._device_name("Triangulum-pro-1") == "triangulum_vp"
    assert driver._device_name("spinq::SQC-25") == "sqc_25_vp"
    assert base.resolve_driver("spinq", "gemini_vp") is driver
