"""国盾驱动测试；所有平台对象均为 mock，不产生网络请求。"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from tyxonq.devices.hardware import config as hwcfg
from tyxonq.devices.hardware.guodun import driver


_QCIS = "X2P Q0\nM Q0"


@pytest.fixture(autouse=True)
def _clear_tokens(monkeypatch):
    hwcfg._TOKENS.clear()
    monkeypatch.delenv("TYXONQ_GUODUN_TOKEN", raising=False)
    monkeypatch.delenv("TYXONQ_API_KEY", raising=False)
    yield
    hwcfg._TOKENS.clear()


def test_token_precedence(monkeypatch):
    hwcfg.set_token("from-set-token", provider="guodun")
    monkeypatch.setenv("TYXONQ_GUODUN_TOKEN", "from-env")
    assert driver._resolve_token("explicit") == "explicit"
    assert driver._resolve_token(None) == "from-set-token"


def test_token_uses_dedicated_env_and_ignores_global(monkeypatch):
    monkeypatch.setenv("TYXONQ_API_KEY", "wrong-global-token")
    with pytest.raises(RuntimeError, match="Guodun token required"):
        driver._resolve_token(None)
    monkeypatch.setenv("TYXONQ_GUODUN_TOKEN", "dedicated-token")
    assert driver._resolve_token(None) == "dedicated-token"


def test_device_layer_consumes_explicit_token(monkeypatch):
    from tyxonq.devices import base

    fake_driver = MagicMock()
    fake_driver.run.return_value = [MagicMock(async_result=True)]
    monkeypatch.setattr(base, "resolve_driver", lambda provider, device: fake_driver)

    base.run(
        provider="guodun",
        device="gd_test",
        source=_QCIS,
        shots=10,
        token="explicit-token",
    )
    fake_driver.run.assert_called_once_with(
        "gd_test",
        "explicit-token",
        source=_QCIS,
        shots=10,
    )


def test_disable_retry_requires_compatible_wrapper():
    platform = MagicMock()
    platform._send_request = lambda *args, **kwargs: None
    with pytest.raises(RuntimeError, match="无法安全关闭自动重试"):
        driver._disable_automatic_request_retry(platform)


def test_create_platform_disables_retry_before_login(monkeypatch):
    events = []

    class FakePlatform:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append("constructed")

        def original_request(self, *args, **kwargs):
            return "ok"

        def wrapped_request(self, *args, **kwargs):
            return self.original_request(*args, **kwargs)

        def login(self):
            assert self._send_request.__func__ is FakePlatform.original_request
            events.append("login")

    FakePlatform.wrapped_request.__wrapped__ = FakePlatform.original_request
    FakePlatform._send_request = FakePlatform.wrapped_request
    monkeypatch.setattr(driver, "_load_platform_class", lambda: FakePlatform)

    platform = driver._create_platform("secret", "gd_test")
    assert platform.kwargs == {
        "login_key": "secret",
        "machine_name": "gd_test",
        "auto_login": False,
    }
    assert events == ["constructed", "login"]


def test_run_calls_submit_exactly_once_with_verify(monkeypatch):
    platform = MagicMock()
    platform.submit_job.return_value = [101, 102]
    monkeypatch.setattr(driver, "_create_platform", lambda token, device: platform)

    tasks = driver.run(
        "guodun::gd_test",
        token="secret",
        source=[_QCIS, _QCIS],
        shots=[20, 20],
        exp_name="offline-test",
    )

    platform.submit_job.assert_called_once_with(
        circuit=[_QCIS, _QCIS],
        exp_name="offline-test",
        num_shots=20,
        is_verify=True,
    )
    assert [task.id for task in tasks] == [101, 102]
    assert all(task.platform is platform for task in tasks)


def test_beginner_chain_api_runs_end_to_end_with_mock_platform(monkeypatch):
    import tyxonq as tq

    platform = MagicMock()
    platform.submit_job.return_value = [150]
    platform.query_experiment.return_value = [
        {
            "runStatus": 2,
            "resultStatus": [["Q60", "Q55"], [0, 0], [1, 1]],
            "probability": [0.5, 0.0, 0.0, 0.5],
        }
    ]
    monkeypatch.setattr(driver, "_create_platform", lambda token, device: platform)
    tq.set_token("from-set-token", provider="guodun", device="gd_test")

    circuit = tq.Circuit(2)
    circuit.h(0).cx(0, 1).add_measure(0, 1)
    results = circuit.device(provider="guodun", device="gd_test").run(
        shots=2,
        physical_qubits=[60, 55],
        wait_async_result=True,
    )

    assert results[0]["result"] == {"00": 1, "11": 1}
    platform.submit_job.assert_called_once()
    submitted_qcis = platform.submit_job.call_args.kwargs["circuit"][0]
    assert "CZ Q60 Q55" in submitted_qcis
    assert submitted_qcis.endswith("M Q60\nM Q55")


def test_batch_requires_same_shots(monkeypatch):
    platform = MagicMock()
    monkeypatch.setattr(driver, "_create_platform", lambda token, device: platform)
    with pytest.raises(ValueError, match="相同 shots"):
        driver.run(
            "gd_test",
            token="secret",
            source=[_QCIS, _QCIS],
            shots=[10, 20],
        )
    platform.submit_job.assert_not_called()


def _config(disabled_qubits="", disabled_couplers=""):
    return {
        "disabledQubits": disabled_qubits,
        "disabledCouplers": disabled_couplers,
        "overview": {
            "coupler_map": {
                "C0": ["Q0", "Q1"],
                "C1": ["Q1", "Q2"],
            }
        },
    }


def test_topology_accepts_active_qubits_and_cz_edges():
    driver._validate_topology(["CZ Q0 Q1\nM Q0\nM Q1"], _config())


def test_topology_rejects_disabled_qubit():
    with pytest.raises(RuntimeError, match="禁用物理比特"):
        driver._validate_topology(["M Q1"], _config(disabled_qubits="Q1"))


def test_topology_rejects_disabled_coupler_edge():
    with pytest.raises(RuntimeError, match="不可用的 CZ 边"):
        driver._validate_topology(
            ["CZ Q0 Q1\nM Q0\nM Q1"],
            _config(disabled_couplers="C0"),
        )


def test_topology_rejects_missing_config_structure():
    with pytest.raises(RuntimeError, match="配置缺少"):
        driver._validate_topology(["M Q0"], {})


def test_run_downloads_current_config_before_simulator_submit(monkeypatch):
    platform = MagicMock()
    platform.download_config.return_value = _config()
    platform.submit_job.return_value = [201]
    monkeypatch.setattr(driver, "_create_platform", lambda token, device: platform)

    driver.run(
        "gd_sim1",
        token="secret",
        source="CZ Q0 Q1\nM Q0\nM Q1",
        shots=10,
    )
    platform.download_config.assert_called_once_with()
    assert platform.method_calls[0] == call.download_config()
    assert platform.method_calls[1] == call.submit_job(
        circuit=["CZ Q0 Q1\nM Q0\nM Q1"],
        exp_name="TyxonQJob",
        num_shots=10,
        is_verify=True,
    )


def _task_with_response(response):
    platform = MagicMock()
    platform.query_experiment.return_value = response
    return driver.GuodunTask(id=301, device="gd_test", platform=platform), platform


def test_completed_result_keeps_platform_bit_order_and_metadata():
    raw = {
        "runStatus": 2,
        "resultStatus": [["Q60", "Q55"], [0, 1], [0, 1], [1, 0]],
        "probability": [0.0, 2 / 3, 1 / 3, 0.0],
    }
    task, platform = _task_with_response([raw])
    result = driver.get_task_details(task)

    assert result["uni_status"] == "completed"
    assert result["result"] == {"01": 2, "10": 1}
    assert result["result_meta"]["measurement_order"] == ["Q60", "Q55"]
    assert result["result_meta"]["probability"] == raw["probability"]
    assert result["result_meta"]["raw"] is raw
    platform.query_experiment.assert_called_once_with(
        query_id=[301], max_wait_time=1, sleep_time=1
    )


def test_query_error_while_queued_is_nonterminal():
    class CqlibRequestError(Exception):
        pass

    task, platform = _task_with_response([])
    platform.query_experiment.side_effect = CqlibRequestError(
        "Failed to query the experimental result."
    )

    result = driver.get_task_details(task)

    assert result["uni_status"] == "unknown"
    assert result["result"] == {}
    assert result["error"] == ""
    assert "CqlibRequestError" in result["result_meta"]["query_error"]
    assert result["result_meta"]["raw"] is None
    platform.query_experiment.assert_called_once_with(
        query_id=[301], max_wait_time=1, sleep_time=1
    )


def test_unexpected_query_error_is_not_hidden():
    task, platform = _task_with_response([])
    platform.query_experiment.side_effect = RuntimeError("SDK contract changed")

    with pytest.raises(RuntimeError, match="SDK contract changed"):
        driver.get_task_details(task)


def test_failed_and_pending_status_mapping():
    failed_task, _ = _task_with_response([{"runStatus": 3, "msg": "bad"}])
    pending_task, _ = _task_with_response([{"runStatus": 1}])
    assert driver.get_task_details(failed_task)["uni_status"] == "failed"
    assert driver.get_task_details(pending_task)["uni_status"] == "1"


def test_empty_and_invalid_shot_width_results():
    empty_task, _ = _task_with_response([])
    invalid_task, _ = _task_with_response(
        [{"runStatus": 2, "resultStatus": [["Q0", "Q1"], [0]]}]
    )
    assert driver.get_task_details(empty_task)["result"] == {}
    invalid = driver.get_task_details(invalid_task)
    assert invalid["uni_status"] == "error"
    assert "宽度异常" in invalid["error"]


def test_remove_task_cancels_existing_query_id_only():
    task, platform = _task_with_response([])
    platform.stop_running_experiments.return_value = {"ok": True}
    assert driver.remove_task(task) == {"ok": True}
    platform.stop_running_experiments.assert_called_once_with(query_id=301)


def test_list_devices_without_token_is_empty():
    assert driver.list_devices() == []


def test_list_devices_extracts_only_guodun_codes(monkeypatch):
    platform = MagicMock()
    platform.query_quantum_computer_list.return_value = [
        ["gd_test", "online"],
        {"code": "gd_sim1"},
        ["guodun_sw", "other-machine"],
    ]
    monkeypatch.setattr(driver, "_create_platform", lambda token, device: platform)
    assert driver.list_devices("secret") == [
        "guodun::gd_sim1",
        "guodun::gd_test",
        "guodun::guodun_sw",
    ]


def test_unified_polling_stops_immediately_on_failure(monkeypatch):
    from tyxonq.devices import base

    handle = object()
    task = base.DeviceTask("guodun", "gd_test", handle, async_result=True)
    fake_driver = MagicMock()
    fake_driver.get_task_details.return_value = {
        "result": {},
        "uni_status": "failed",
        "error": "bad",
    }
    monkeypatch.setattr(base, "resolve_driver", lambda provider, device: fake_driver)
    result = base.get_task_details(task, wait=True, poll_interval=10, timeout=30)
    assert result["uni_status"] == "failed"
    fake_driver.get_task_details.assert_called_once_with(handle, None)
