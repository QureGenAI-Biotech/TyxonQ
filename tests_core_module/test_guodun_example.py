"""国盾终端示例必须在取得 query ID 后停止，不自动查询。"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cqlib") is None,
    reason="需要可选依赖 cqlib==1.3.11",
)


def test_x_example_uses_single_qubit_default_mapping(monkeypatch, capsys):
    from examples import run_circuit_on_guodun as example

    monkeypatch.delenv("TYXONQ_GUODUN_TOKEN", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        ["run_circuit_on_guodun.py", "--circuit", "x"],
    )

    example.main()

    output = capsys.readouterr().out
    assert "逻辑到物理映射: {0: 0}" in output
    assert "X Q0" in output
    assert "未登录、未下载配置、未提交任务" in output


def test_online_example_submits_once_prints_id_and_does_not_query(
    monkeypatch, capsys
):
    from examples import run_circuit_on_guodun as example

    handle = SimpleNamespace(id="example-query-id")
    fake_run = MagicMock(return_value=[SimpleNamespace(handle=handle)])
    monkeypatch.setattr(example.device_base, "run", fake_run)
    monkeypatch.setenv("TYXONQ_GUODUN_TOKEN", "mock-token")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_circuit_on_guodun.py",
            "--run-online",
            "--circuit",
            "x",
            "--device",
            "gd_test",
            "--shots",
            "10",
            "--physical-qubits",
            "0",
        ],
    )

    example.main()

    fake_run.assert_called_once()
    assert fake_run.call_args.kwargs["provider"] == "guodun"
    assert fake_run.call_args.kwargs["device"] == "gd_test"
    assert fake_run.call_args.kwargs["shots"] == 10
    assert fake_run.call_args.kwargs["source"].endswith("M Q0")
    assert fake_run.call_args.kwargs["exp_name"].startswith(
        "tyxonq_gd_test_x_"
    )
    output = capsys.readouterr().out
    assert "example-query-id" in output
    assert "exp_name=tyxonq_gd_test_x_" in output
    assert "未查询结果" in output


def test_variational_tutorial_example_runs_offline(monkeypatch, capsys):
    from examples import run_guodun_variational as example

    monkeypatch.delenv("TYXONQ_GUODUN_TOKEN", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_guodun_variational.py",
            "--physical-qubits",
            "0,6,12,7,1",
        ],
    )

    example.main()

    output = capsys.readouterr().out
    assert "逻辑到物理映射: {0: 0, 1: 6, 2: 12, 3: 7, 4: 1}" in output
    assert "'CZ': 16" in output
    assert "'M': 5" in output
    assert "未登录、未下载配置、未提交任务" in output
