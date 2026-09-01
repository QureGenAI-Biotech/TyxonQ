"""LQCloud driver 的纯离线测试。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tyxonq.devices import base
from tyxonq.devices.hardware import config as hwcfg
from tyxonq.devices.hardware.lqcloud import driver


class _FakeCircuit:
    pass


class _FakeResult:
    def __init__(self, backend_name, job_id, status, data, metadata=None):
        self.data = data
        self.result_format = data.get("result_format", "counts")

    def get_counts(self):
        if "counts" in self.data:
            return {key: int(value) for key, value in self.data["counts"].items()}
        counts = {}
        for bitstring in self.data.get("memory", []):
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts

    def get_probabilities(self):
        counts = self.get_counts()
        total = sum(counts.values())
        return {key: value / total for key, value in counts.items()} if total else {}


class _FakeJob:
    def __init__(self, raw=None):
        self.job_id = "job-123"
        self.raw = raw or {"status": "queued", "result": None}
        self.cancel_calls = 0

    def status_info(self):
        return self.raw

    def cancel(self):
        self.cancel_calls += 1
        return True


class _FakeBackend:
    def __init__(self, job):
        self.job = job
        self.run_calls = []

    def run(self, source, shots, **opts):
        self.run_calls.append((source, shots, opts))
        return self.job


class _FakeProvider:
    instances = []
    rows = [{"name": "QZ02"}, {"name": "LQ-Test"}]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.job = _FakeJob()
        self.backend = _FakeBackend(self.job)
        self.__class__.instances.append(self)

    def get_backend(self, name):
        self.backend_name = name
        return self.backend

    def get_backends(self):
        return list(self.rows)


@pytest.fixture(autouse=True)
def clean_credentials(monkeypatch):
    hwcfg._TOKENS.clear()
    _FakeProvider.instances.clear()
    for name in (
        "TYXONQ_API_KEY",
        "TYXONQ_LQCLOUD_API_KEY",
        "LQCLOUD_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    hwcfg._TOKENS.clear()


@pytest.fixture
def fake_sdk(monkeypatch):
    monkeypatch.setattr(
        driver,
        "_sdk_classes",
        lambda: (_FakeProvider, _FakeCircuit, _FakeResult),
    )


def test_api_key_precedence(monkeypatch):
    monkeypatch.setenv("TYXONQ_LQCLOUD_API_KEY", "tyxonq-env")
    monkeypatch.setenv("LQCLOUD_API_KEY", "upstream-env")
    hwcfg.set_token("stored", provider="lqcloud")
    opts = {"api_key": "explicit"}
    assert driver._resolve_api_key("token", opts) == "explicit"
    assert opts == {}


def test_set_token_then_namespaced_and_upstream_env(monkeypatch):
    hwcfg.set_token("stored", provider="lqcloud")
    assert driver._resolve_api_key(None, {}) == "stored"
    hwcfg._TOKENS.clear()
    monkeypatch.setenv("TYXONQ_LQCLOUD_API_KEY", "tyxonq-env")
    assert driver._resolve_api_key(None, {}) == "tyxonq-env"
    monkeypatch.delenv("TYXONQ_LQCLOUD_API_KEY")
    monkeypatch.setenv("LQCLOUD_API_KEY", "upstream-env")
    assert driver._resolve_api_key(None, {}) == "upstream-env"


def test_global_tyxonq_key_is_ignored(monkeypatch):
    monkeypatch.setenv("TYXONQ_API_KEY", "wrong-provider-key")
    with pytest.raises(RuntimeError, match="LQCloud API key required"):
        driver._resolve_api_key(None, {})


def test_run_submits_exactly_once_and_passes_layout(fake_sdk):
    source = _FakeCircuit()
    tasks = driver.run(
        "lqcloud::QZ02",
        token="key",
        source=source,
        shots=256,
        physical_qubits=[3, 4, 5, 6],
    )
    provider = _FakeProvider.instances[0]
    assert provider.kwargs == {"api_key": "key", "interactive": False}
    assert provider.backend_name == "QZ02"
    assert provider.backend.run_calls == [
        (source, 256, {"initial_layout": [3, 4, 5, 6]})
    ]
    assert len(tasks) == 1 and tasks[0].id == "job-123"


def test_run_rejects_batch_and_unknown_options(fake_sdk):
    with pytest.raises(ValueError, match="一次只支持一条线路"):
        driver.run("QZ02", token="key", source=_FakeCircuit(), shots=[10])
    with pytest.raises(TypeError, match="readout_correction"):
        driver.run(
            "QZ02",
            token="key",
            source=_FakeCircuit(),
            shots=10,
            readout_correction=True,
        )


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("pending", "queued"),
        ("running", "running"),
        ("completed", "completed"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("new-state", "unknown"),
    ],
)
def test_status_mapping(raw_status, expected):
    assert driver._map_status(raw_status) == expected


def test_completed_counts_and_memory_are_normalized(fake_sdk):
    for payload, expected in (
        ({"result_format": "counts", "counts": {"00": "3", "11": 2}}, {"00": 3, "11": 2}),
        ({"result_format": "memory", "memory": ["0", "1", "1"]}, {"0": 1, "1": 2}),
    ):
        job = _FakeJob({"status": "completed", "result": payload})
        task = driver.LQCloudTask("job-123", "QZ02", job=job)
        result = driver.get_task_details(task)
        assert result["uni_status"] == "completed"
        assert result["result"] == expected
        assert result["result_meta"]["shots"] == sum(expected.values())
        assert result["result_meta"]["raw"] == job.raw


def test_failure_and_cancel(fake_sdk):
    job = _FakeJob({"status": "failed", "result": {"error": "calibration"}})
    task = driver.LQCloudTask("job-123", "QZ02", job=job)
    result = driver.get_task_details(task)
    assert result["uni_status"] == "failed"
    assert result["error"] == "calibration"
    assert driver.remove_task(task) is True
    assert job.cancel_calls == 1


def test_device_discovery_is_dynamic(fake_sdk):
    assert driver.list_devices(token="key") == ["lqcloud::QZ02", "lqcloud::LQ-Test"]
    assert driver.list_devices() == []


def test_provider_is_registered():
    assert base.resolve_driver("lqcloud", "QZ02") is driver


def test_device_layer_prefers_explicit_token_and_ignores_global(monkeypatch):
    fake_task = SimpleNamespace(async_result=True)
    fake_driver = SimpleNamespace(run=MagicMock(return_value=[fake_task]))
    monkeypatch.setenv("TYXONQ_API_KEY", "wrong-provider-key")
    monkeypatch.setattr(base, "resolve_driver", lambda provider, device: fake_driver)

    base.run(
        provider="lqcloud",
        device="QZ02",
        source=object(),
        shots=1,
        token="explicit",
    )
    assert fake_driver.run.call_args.args[1] == "explicit"

    fake_driver.run.reset_mock()
    base.run(provider="lqcloud", device="QZ02", source=object(), shots=1)
    assert fake_driver.run.call_args.args[1] is None


def test_failed_status_stops_unified_polling(monkeypatch):
    handle = SimpleNamespace(async_result=True)
    fake_driver = SimpleNamespace(
        get_task_details=MagicMock(
            return_value={"result": {}, "uni_status": "failed", "error": "boom"}
        )
    )
    monkeypatch.setattr(base, "resolve_driver", lambda provider, device: fake_driver)
    task = base.DeviceTask("lqcloud", "QZ02", handle, async_result=True)
    result = base.get_task_details(task, wait=True, poll_interval=0.01, timeout=1)
    assert result["uni_status"] == "failed"
    assert fake_driver.get_task_details.call_count == 1


def test_cloud_api_forwards_explicit_token(monkeypatch):
    from tyxonq.cloud import api

    mocked = MagicMock(return_value=[])
    monkeypatch.setattr(base, "run", mocked)
    api.run(provider="lqcloud", device="QZ02", source=object(), token="key")
    assert mocked.call_args.kwargs["token"] == "key"
