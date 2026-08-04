from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tyxonq.applications.qml import RiverONEVQCSpec


def _load_example_module():
    example_path = Path(__file__).parents[1] / "examples" / "riverone_qml.py"
    spec = importlib.util.spec_from_file_location("riverone_qml_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # 用户可能把 example 配成真机设备；测试必须强制使用本地模拟器。
    module.DEVICE = "simulator"
    module.TYXONQ_API_KEY = ""
    return module


def _model_spec() -> RiverONEVQCSpec:
    return RiverONEVQCSpec(
        n_qubits=2,
        n_layers=1,
        angles=[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
    )


def _qasm() -> dict[str, str]:
    return {
        basis: f"OPENQASM 2.0; qreg q[2]; creg c[2]; // {basis}"
        for basis in ("X", "Y", "Z")
    }


def test_example_requires_checkpoint_when_not_configured(capsys):
    module = _load_example_module()

    with pytest.raises(SystemExit) as exc_info:
        module.main([])

    assert exc_info.value.code == 2
    assert "请在用户配置区设置 CHECKPOINT" in capsys.readouterr().err


def test_example_loads_npy_amplitudes(monkeypatch, tmp_path: Path):
    module = _load_example_module()
    amplitudes_path = tmp_path / "input.npy"
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    np.save(amplitudes_path, expected)
    received = []
    monkeypatch.setattr(module, "load_riverone_vqc", lambda *args, **kwargs: _model_spec())
    monkeypatch.setattr(
        module,
        "riverone_to_qasm2",
        lambda spec, amplitudes: received.append(amplitudes.copy()) or _qasm(),
    )
    monkeypatch.setattr(
        module.tq.api,
        "submit_task",
        lambda **kwargs: [SimpleNamespace(id="sim-task")],
    )
    monkeypatch.setattr(
        module.tq.api,
        "get_task_details",
        lambda task: {"result": {"00": 1}, "error": ""},
    )

    assert (
        module.main(
            [
                "--checkpoint",
                "/external/model.pt",
                "--amplitudes",
                str(amplitudes_path),
            ]
        )
        == 0
    )
    assert np.array_equal(received[0], expected)


def test_example_runs_xyz_qasm_on_local_simulator(monkeypatch, capsys):
    module = _load_example_module()
    qasm = _qasm()
    monkeypatch.setattr(module, "load_riverone_vqc", lambda *args, **kwargs: _model_spec())
    monkeypatch.setattr(module, "riverone_to_qasm2", lambda *args, **kwargs: qasm)
    monkeypatch.setattr(
        module.tq,
        "set_token",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("本地模拟器不应配置真机 token")
        ),
    )

    submitted = []

    def fake_submit_task(**kwargs):
        submitted.append(kwargs)
        return [SimpleNamespace(id=f"sim-{len(submitted)}")]

    monkeypatch.setattr(module.tq.api, "submit_task", fake_submit_task)
    monkeypatch.setattr(
        module.tq.api,
        "get_task_details",
        lambda task: {"result": {"00": 192, "10": 64}, "error": ""},
    )

    result = module.main(
        [
            "--checkpoint",
            "/external/checkpoint.pt",
            "--shots",
            "256",
        ]
    )

    assert result == 0
    assert submitted == [
        {
            "provider": "simulator",
            "device": "statevector",
            "source": qasm[basis],
            "shots": 256,
        }
        for basis in ("X", "Y", "Z")
    ]
    stdout = capsys.readouterr().out
    assert stdout.count("本地模拟完成") == 1
    assert "task_id=" not in stdout
    assert stdout.index("X/Y/Z 本地模拟完成") < stdout.index("X: X0=")
    assert "X: X0=0.50000000, X1=1.00000000" in stdout


def test_example_submits_xyz_in_one_api_call(monkeypatch, capsys):
    module = _load_example_module()
    qasm = _qasm()
    monkeypatch.setattr(module, "load_riverone_vqc", lambda *args, **kwargs: _model_spec())
    monkeypatch.setattr(module, "riverone_to_qasm2", lambda *args, **kwargs: qasm)
    monkeypatch.setenv("TYXONQ_API_KEY", "test-token")

    configured = []
    submitted = []
    monkeypatch.setattr(
        module.tq,
        "set_token",
        lambda token, provider, device: configured.append((token, provider, device)),
    )

    def fake_submit_task(**kwargs):
        submitted.append(kwargs)
        return [SimpleNamespace(id="batch-task")]

    monkeypatch.setattr(module.tq.api, "submit_task", fake_submit_task)

    result = module.main(
        [
            "--checkpoint",
            "/external/checkpoint.pt",
            "--shots",
            "256",
            "--device",
            "homebrew_s3",
        ]
    )

    assert result == 0
    assert configured == [("test-token", "tyxonq", "homebrew_s3")]
    assert submitted == [
        {
            "provider": "tyxonq",
            "device": "homebrew_s3",
            "source": [qasm["X"], qasm["Y"], qasm["Z"]],
            "shots": 256,
        }
    ]
    assert "batch-task" in capsys.readouterr().out
