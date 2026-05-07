"""Unit tests for the quafu driver. All HTTPS is mocked — these run on any
Python in TyxonQ's supported range without a live token."""
from __future__ import annotations

import json as _json
import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _reset_quafu_state(monkeypatch):
    """Clear singleton + env vars + in-memory tokens around each test."""
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task
    from tyxonq.devices.hardware import config as hwcfg

    def _cleanup():
        if hasattr(Task, "instance"):
            del Task.instance
        hwcfg._TOKENS.clear()

    _cleanup()
    for var in ("TYXONQ_API_KEY", "TYXONQ_QUAFU_TOKEN", "QPU_API_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    yield
    _cleanup()


# ---------- Token resolution ----------

def test_resolve_token_prefers_explicit_kwarg(monkeypatch):
    monkeypatch.setenv("TYXONQ_QUAFU_TOKEN", "from-env")
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    assert _resolve_token("from-kwarg") == "from-kwarg"


def test_resolve_token_falls_back_to_set_token():
    import tyxonq as tq
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    tq.set_token("from-set-token", provider="quafu")
    assert _resolve_token(None) == "from-set-token"


def test_resolve_token_falls_back_to_tyxonq_quafu_env(monkeypatch):
    monkeypatch.setenv("TYXONQ_QUAFU_TOKEN", "from-tq-env")
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    assert _resolve_token(None) == "from-tq-env"


def test_resolve_token_falls_back_to_qpu_api_token(monkeypatch):
    monkeypatch.setenv("QPU_API_TOKEN", "from-qpu-env")
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    assert _resolve_token(None) == "from-qpu-env"


def test_resolve_token_prefers_tyxonq_env_over_upstream_env(monkeypatch):
    monkeypatch.setenv("TYXONQ_QUAFU_TOKEN", "tq-wins")
    monkeypatch.setenv("QPU_API_TOKEN", "qpu-loses")
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    assert _resolve_token(None) == "tq-wins"


def test_resolve_token_ignores_global_tyxonq_api_key(monkeypatch):
    """Regression: TYXONQ_API_KEY must NOT be sent to Quafu."""
    monkeypatch.setenv("TYXONQ_API_KEY", "tyxonq-cloud-token")
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    with pytest.raises(RuntimeError, match="Quafu token required"):
        _resolve_token(None)


def test_resolve_token_raises_with_helpful_message_when_missing():
    from tyxonq.devices.hardware.quafu.driver import _resolve_token

    with pytest.raises(RuntimeError) as exc_info:
        _resolve_token(None)
    msg = str(exc_info.value)
    assert "quafu-sqc.baqis.ac.cn" in msg
    assert "TYXONQ_QUAFU_TOKEN" in msg
    assert "tq.set_token" in msg


# ---------- Submission ----------

_BELL_QASM = (
    "OPENQASM 2.0;\n"
    'include "qelib1.inc";\n'
    "qreg q[2];\ncreg c[2];\n"
    "h q[0];\ncx q[0],q[1];\n"
    "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"
)


def _mock_session(verify_response="ok", run_response=12345, get_response=None):
    """Build a mock requests.Session that returns the given JSON-encoded
    payloads for the verify (init) and run (submit) calls.
    """
    import json

    def _bytes(obj):
        return MagicMock(content=json.dumps(obj).encode())

    sess = MagicMock()
    sess.get.return_value = _bytes(verify_response if get_response is None else get_response)
    sess.post.return_value = _bytes(run_response)
    return sess


def test_run_submits_returns_quafu_task():
    from tyxonq.devices.hardware.quafu import driver
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    with patch.object(Task, "session", _mock_session(run_response=99001)):
        tasks = driver.run(
            device="Dongling", token="t", source=_BELL_QASM, shots=1024,
        )

    assert isinstance(tasks, list) and len(tasks) == 1
    assert tasks[0].id == 99001
    assert tasks[0].device == "Dongling"
    assert tasks[0].status == "submitted"


def test_run_strips_provider_prefix_from_device():
    from tyxonq.devices.hardware.quafu import driver
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    with patch.object(Task, "session", _mock_session()):
        tasks = driver.run(
            device="quafu::Dongling", token="t", source=_BELL_QASM, shots=1024,
        )
    assert tasks[0].device == "Dongling"


def test_run_sends_correct_payload():
    from tyxonq.devices.hardware.quafu import driver
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    sess = _mock_session()
    with patch.object(Task, "session", sess):
        driver.run(
            device="Dongling", token="t", source=_BELL_QASM, shots=2048,
            task_name="MyJob",
        )

    # Examine the POST that was made for /task/run
    post_url = sess.post.call_args[0][0]
    assert "/task/run/" in post_url
    assert "chip=Dongling" in post_url
    assert "shots=2048" in post_url
    assert "name=MyJob" in post_url

    import json
    body = json.loads(sess.post.call_args[1]["data"])
    assert body["circuit"] == _BELL_QASM
    # By default, TyxonQ asks the server NOT to recompile
    # (the user already compiled to QASM via Circuit.compile()).
    assert body["options"]["compiler"] is None


def test_run_passes_compiler_option_through():
    from tyxonq.devices.hardware.quafu import driver
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    sess = _mock_session()
    with patch.object(Task, "session", sess):
        driver.run(
            device="Dongling", token="t", source=_BELL_QASM, shots=1024,
            compiler="qsteed",
        )

    import json
    body = json.loads(sess.post.call_args[1]["data"])
    assert body["options"]["compiler"] == "qsteed"


def test_run_warns_on_non_multiple_of_1024_shots():
    from tyxonq.devices.hardware.quafu import driver
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    with patch.object(Task, "session", _mock_session()):
        with pytest.warns(UserWarning, match="multiple of 1024"):
            driver.run(
                device="Dongling", token="t", source=_BELL_QASM, shots=500,
            )


def test_run_rejects_non_qasm_source():
    from tyxonq.devices.hardware.quafu import driver
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    with patch.object(Task, "session", _mock_session()):
        with pytest.raises(ValueError, match="OPENQASM 2.0"):
            driver.run(
                device="Dongling", token="t", source="not qasm at all", shots=1024,
            )


def test_run_rejects_empty_source():
    from tyxonq.devices.hardware.quafu import driver

    with pytest.raises(ValueError, match="source"):
        driver.run(device="Dongling", token="t", source=None, shots=1024)


def test_run_batch_returns_one_task_per_source():
    from tyxonq.devices.hardware.quafu import driver
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    sess = _mock_session()
    # Each .post() call returns the same mocked tid (12345); fine for shape test.
    with patch.object(Task, "session", sess):
        tasks = driver.run(
            device="Dongling", token="t",
            source=[_BELL_QASM, _BELL_QASM, _BELL_QASM],
            shots=1024,
        )
    assert len(tasks) == 3
    assert sess.post.call_count == 3


# ---------- Result retrieval ----------


def _mgr_with_result(result_payload):
    """Create a mocked Task manager whose .result(tid) returns the given dict."""
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    if hasattr(Task, "instance"):
        del Task.instance
    sess = MagicMock()
    sess.get.return_value = MagicMock(content=_json.dumps("ok").encode())
    sess.post.return_value = MagicMock(content=_json.dumps(12345).encode())
    with patch.object(Task, "session", sess):
        mgr = Task("test-token")
    # Now stub .result on the singleton instance
    mgr.result = MagicMock(return_value=result_payload)
    return mgr


@pytest.mark.parametrize(
    "upstream_status, expected_uni",
    [
        ("Finished", "completed"),
        ("Failed", "failed"),
        ("Running", "running"),
        ("Pending", "queued"),
        ("Queued", "queued"),
        ("Whatever", "unknown"),
    ],
)
def test_get_task_details_status_mapping(upstream_status, expected_uni):
    from tyxonq.devices.hardware.quafu import driver

    mgr = _mgr_with_result(
        {"count": {"00": 510, "11": 514}, "status": upstream_status}
    )
    task = driver.QuafuTask(id=12345, device="Dongling", _mgr=mgr)

    out = driver.get_task_details(task)

    assert out["uni_status"] == expected_uni
    assert out["result"] == {"00": 510, "11": 514}
    assert out["result_meta"]["device"] == "Dongling"
    assert out["result_meta"]["tid"] == 12345
    assert out["result_meta"]["raw"]["status"] == upstream_status


def test_get_task_details_handles_missing_count_field():
    from tyxonq.devices.hardware.quafu import driver

    mgr = _mgr_with_result({"status": "Pending"})  # no count yet
    task = driver.QuafuTask(id=12345, device="Dongling", _mgr=mgr)

    out = driver.get_task_details(task)
    assert out["result"] == {}
    assert out["uni_status"] == "queued"
    assert out["result_meta"]["shots"] is None


def test_get_task_details_surfaces_error_field():
    from tyxonq.devices.hardware.quafu import driver

    mgr = _mgr_with_result({"status": "Failed", "error": "calibration drift"})
    task = driver.QuafuTask(id=12345, device="Dongling", _mgr=mgr)

    out = driver.get_task_details(task)
    assert out["uni_status"] == "failed"
    assert out["error"] == "calibration drift"


# ---------- list_devices ----------

def test_list_devices_filters_offline_chips():
    from tyxonq.devices.hardware.quafu import driver
    from tyxonq.devices.hardware.quafu._vendor_quafu import Task

    sess = MagicMock()
    sess.get.side_effect = [
        # 1st call: verify (during Task.__init__)
        MagicMock(content=_json.dumps("ok").encode()),
        # 2nd call: status(0)
        MagicMock(content=_json.dumps(
            {"Dongling": 0, "Baihua": 10, "Miaofeng": "Offline"}
        ).encode()),
    ]
    with patch.object(Task, "session", sess):
        devs = driver.list_devices(token="t")

    assert sorted(devs) == ["quafu::Baihua", "quafu::Dongling"]
    assert "quafu::Miaofeng" not in devs


def test_list_devices_returns_empty_on_token_failure(monkeypatch):
    """If the user has no token configured, list_devices returns [] rather
    than crashing. Matches the qcos driver's defensive shape."""
    from tyxonq.devices.hardware.quafu import driver
    # No token set anywhere — _resolve_token would normally raise, but
    # list_devices wraps in a try/except and returns [].
    devs = driver.list_devices()
    assert devs == []


# ---------- cancel ----------

def test_cancel_calls_manager_cancel():
    from tyxonq.devices.hardware.quafu import driver

    mgr = MagicMock()
    mgr.cancel.return_value = {"cancelled": True, "tid": 12345}
    task = driver.QuafuTask(id=12345, device="Dongling", _mgr=mgr)

    out = driver.cancel(task)
    mgr.cancel.assert_called_once_with(12345)
    assert out == {"cancelled": True, "tid": 12345}


# ---------- Provider registration ----------

def test_resolve_driver_returns_quafu_module():
    from tyxonq.devices.base import resolve_driver
    from tyxonq.devices.hardware.quafu import driver as quafu_driver

    drv = resolve_driver("quafu", "Dongling")
    assert drv is quafu_driver
