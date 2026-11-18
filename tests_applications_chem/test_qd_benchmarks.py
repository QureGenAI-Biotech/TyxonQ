"""QD 基准数据、统计区间和只追加记录测试（无真机提交）。"""

from __future__ import annotations

import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from tyxonq.applications.chem.dynamics.benchmarks import (
    append_record,
    build_manifest,
    comparison_report,
    confidence_interval,
    convergence_report,
    load_records,
    pauli_standard_error,
    save_manifest,
)


def test_pauli_standard_error_and_confidence_interval():
    assert pauli_standard_error(0.0, 100) == pytest.approx(0.1)
    assert pauli_standard_error(1.0, 100) == 0.0
    assert pauli_standard_error(0.0, 400) == pytest.approx(
        pauli_standard_error(0.0, 100) / 2
    )
    lower, upper = confidence_interval(0.2, 0.1)
    assert lower == pytest.approx(0.2 - 1.959963984540054 * 0.1)
    assert upper == pytest.approx(0.2 + 1.959963984540054 * 0.1)
    for invalid in (0, -10, True):
        with pytest.raises((TypeError, ValueError)):
            pauli_standard_error(0.0, invalid)
    with pytest.raises(ValueError):
        pauli_standard_error(1.2, 100)


def test_convergence_report_preserves_nonmonotonic_statistical_data():
    report = convergence_report([0.2, 0.1, 0.05], [0.1, 0.04, 0.01])
    assert report["monotonically_decreasing"]
    np.testing.assert_allclose(report["error_ratios"], [2.5, 4.0])
    noisy = convergence_report([100, 400], [0.03, 0.04])
    assert not noisy["monotonically_decreasing"]
    np.testing.assert_allclose(noisy["errors"], [0.03, 0.04])


def test_manifest_hash_and_atomic_save(tmp_path):
    manifest = build_manifest(
        name="small-xx-chain",
        config={"n_qubits": 4, "dt": 0.05, "shots": 1024},
        reference={"kind": "ED", "source": "independent dense diagonalization"},
        preregistered_metrics={"peak_atol": 0.1},
    )
    path = save_manifest(tmp_path / "manifest.json", manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["manifest_hash"] == manifest["manifest_hash"]
    tampered = dict(manifest)
    tampered["config"] = dict(manifest["config"], shots=2048)
    with pytest.raises(ValueError, match="manifest_hash"):
        save_manifest(tmp_path / "tampered.json", tampered)


def test_raw_records_are_append_only_and_hashed(tmp_path):
    path = tmp_path / "counts.jsonl"
    first = append_record(path, {"record_id": "pilot-001", "counts": {"0": 10}})
    assert first["schema"] == "tyxonq.qd.record.v1"
    append_record(path, {"record_id": "pilot-002", "counts": {"1": 12}})
    assert len(load_records(path)) == 2
    with pytest.raises(ValueError, match="拒绝覆盖"):
        append_record(path, {"record_id": "pilot-001", "counts": {"0": 99}})

    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"0":10', '"0":99')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record_hash"):
        load_records(path)


def test_comparison_report_requires_common_grid(tmp_path):
    report = comparison_report(
        name="triple-check",
        curves={"classical": np.array([1.0, 0.5]), "hardware": [0.9, 0.45]},
        metadata={"unit": "meV"},
    )
    assert report["schema"] == "tyxonq.qd.benchmark.v1"
    with pytest.raises(ValueError, match="相同长度"):
        comparison_report(name="bad", curves={"a": [1], "b": [1, 2]})


def _q9_module():
    path = Path(__file__).parents[1] / "examples" / "qd" / "q9_lqcloud_rabi_benchmark.py"
    return runpy.run_path(str(path))


def test_q9_experiment_fit_manifest_and_offline_report(tmp_path):
    q9 = _q9_module()
    experiment = q9["load_experiment"]()
    assert len(experiment["duration_s"]) == 101
    assert set(experiment["repetitions"]) == {"0", "1"}
    assert np.all(experiment["repetitions"]["0"]["shots"] == 50)
    fit = q9["fit_classical_reference"](experiment)
    assert 250e3 < fit["frequency_hz"] < 320e3
    assert 0 <= fit["offset"] <= 0.5
    assert 0 <= fit["visibility"] <= 1
    manifest = q9["create_manifest"](experiment, fit, tmp_path, shots=100)
    assert manifest["config"]["cost_estimate"]["pilot_circuits"] == 5
    assert manifest["config"]["cost_estimate"]["batch_circuits"] == 101
    assert manifest["config"]["cost_estimate"]["batch_total_shots"] == 10100
    report = q9["offline_report"](experiment, fit, tmp_path)
    assert report["metadata"]["hardware_status"] == "not_submitted"
    assert report["metadata"]["validation_gate"]["passed"] == (
        fit["validation_rmse"] <= 0.075
    )


def test_q9_hardware_requires_confirmation_and_mock_pilot_is_rebuildable(tmp_path, monkeypatch):
    q9 = _q9_module()
    experiment = q9["load_experiment"]()
    fit = q9["fit_classical_reference"](experiment)
    q9["create_manifest"](experiment, fit, tmp_path, shots=100)
    with pytest.raises(PermissionError, match="明确"):
        q9["collect_hardware"](
            mode="pilot", device="mock", confirmed=False, fit=fit,
            experiment=experiment, data_dir=tmp_path,
        )

    from tyxonq.core.ir.circuit import Circuit

    submissions = []

    def fake_run(circuit, **kwargs):
        submissions.append((circuit, kwargs))
        return [{"result": {"0": 80, "1": 20}, "task_id": f"mock-{len(submissions)}", "uni_status": "completed"}]

    monkeypatch.setattr(Circuit, "run", fake_run)
    records = q9["collect_hardware"](
        mode="pilot", device="mock", confirmed=True, fit=fit,
        experiment=experiment, data_dir=tmp_path, shots=100,
    )
    assert len(records) == 5
    assert len(submissions) == 5
    for _, kwargs in submissions:
        assert kwargs["provider"] == "lqcloud"
        assert kwargs["device"] == "mock"
        assert kwargs["wait_async_result"] is True
    assert all(item["shots"] == 100 for item in records)
    assert all(item["counts"] == {"0": 80, "1": 20} for item in records)
    assert all(item["record_hash"] for item in records)

    with pytest.raises(PermissionError, match="人工审核"):
        q9["collect_hardware"](
            mode="batch", device="mock", confirmed=True,
            fit=fit,
            experiment=experiment, data_dir=tmp_path, pilot_reviewed=False,
        )
    assert len(submissions) == 5

    with pytest.raises(ValueError, match="拟合快照不一致"):
        q9["collect_hardware"](
            mode="batch", device="mock", confirmed=True,
            fit=dict(fit, validation_rmse=0.0),
            experiment=experiment, data_dir=tmp_path, pilot_reviewed=True,
        )
    assert len(submissions) == 5

    with pytest.raises(PermissionError, match="独立实验重复验证未通过"):
        q9["collect_hardware"](
            mode="batch", device="mock", confirmed=True, fit=fit,
            experiment=experiment, data_dir=tmp_path, pilot_reviewed=True,
        )
    assert len(submissions) == 5

    paired = q9["collect_hardware"](
        mode="batch", device="mock", confirmed=True, fit=fit,
        experiment=experiment, data_dir=tmp_path, pilot_reviewed=True,
        pairing_only=True, shots=100,
    )
    assert len(paired) == 101
    assert len(submissions) == 106
    assert all(
        item["collection_route"] == "hardware_experiment_pairing"
        and item["independent_validation_passed"] is False
        for item in paired
    )
    report = q9["offline_report"](experiment, fit, tmp_path)
    assert report["metadata"]["hardware_status"] == "collected"
    assert report["metadata"]["hardware_collection_routes"] == [
        "hardware_experiment_pairing"
    ]
    assert report["metadata"]["hardware_independent_validation_passed"] is False
    assert len(report["curves"]["hardware_shot_sem"]) == 101
    assert len(report["curves"]["hardware_ci95_low"]) == 101
    assert len(report["curves"]["hardware_ci95_high"]) == 101
