"""Floquet Ising 数值金标准与 QZ03 Q10 离线硬件门测试。"""

import runpy

import numpy as np
import pytest

from tyxonq.applications.chem.dynamics.circuits import measure_pauli_expectation
from tyxonq.applications.chem.dynamics.floquet import (
    autocorrelator_from_payload,
    autocorrelator_items,
    build_floquet_dtc_circuit,
    nearest_neighbor_edges,
    neel_bitstring,
    sample_dtc_disorder,
)
from tyxonq.core.ir.circuit import Circuit


def _exact_autocorrelator(circuit, initial):
    return measure_pauli_expectation(
        circuit, autocorrelator_items(initial), n_qubits=circuit.num_qubits, shots=0,
    )


def test_floquet_initial_neel_and_zero_cycle():
    initial = neel_bitstring(6)
    assert initial == "101010"
    circuit = build_floquet_dtc_circuit(6, 0, 0.94, np.zeros(6), np.zeros(5))
    assert _exact_autocorrelator(circuit, initial) == pytest.approx(1.0, abs=1e-12)
    items = autocorrelator_items(initial)
    assert [coeff for coeff, _ in items] == pytest.approx(np.array([1, -1, 1, -1, 1, -1]) / 6)


def test_single_qubit_floquet_matches_exact_kick_solution():
    for cycle in range(5):
        circuit = build_floquet_dtc_circuit(1, cycle, 1.0, [0.0], [])
        assert _exact_autocorrelator(circuit, "0") == pytest.approx((-1) ** cycle, abs=1e-12)
        circuit = build_floquet_dtc_circuit(1, cycle, 0.5, [0.0], [])
        expected = np.cos(cycle * 0.5 * np.pi)
        assert _exact_autocorrelator(circuit, "0") == pytest.approx(expected, abs=1e-12)


def test_diagonal_floquet_preserves_computational_neel_state():
    h = np.array([0.3, -0.8, 1.1])
    phi = np.array([-2.0, -1.3])
    initial = neel_bitstring(3)
    for cycle in (0, 1, 4):
        circuit = build_floquet_dtc_circuit(3, cycle, 0.0, h, phi)
        assert _exact_autocorrelator(circuit, initial) == pytest.approx(1.0, abs=1e-12)


def test_disorder_is_seed_reproducible_and_ranges_match_paper():
    h1, phi1, edges = sample_dtc_disorder(5, 123)
    h2, phi2, _ = sample_dtc_disorder(5, 123)
    h3, _, _ = sample_dtc_disorder(5, 124)
    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(phi1, phi2)
    assert not np.array_equal(h1, h3)
    assert edges == nearest_neighbor_edges(5)
    assert np.all((h1 >= -np.pi) & (h1 <= np.pi))
    assert np.all((phi1 >= -1.5 * np.pi) & (phi1 <= -0.5 * np.pi))


def test_counts_autocorrelator_uses_common_bit_order_postprocessing():
    initial = neel_bitstring(4)
    measured_initial = initial[::-1]
    assert autocorrelator_from_payload({"result": {measured_initial: 100}}, initial) == pytest.approx(1.0)
    flipped = "".join("1" if ch == "0" else "0" for ch in measured_initial)
    assert autocorrelator_from_payload({"result": {flipped: 100}}, initial) == pytest.approx(-1.0)


def test_q10_manifest_costs_and_mock_qz03_pilot(tmp_path, monkeypatch):
    q10 = runpy.run_path("examples/qd/q10_qz03_floquet_dtc.py", run_name="q10_test")
    manifest = q10["create_manifest"](tmp_path, shots=20)
    cost = manifest["config"]["cost_estimate"]
    assert cost["pilot_circuits"] == 8
    assert cost["pilot_shots"] == 160
    assert cost["batch_circuits"] == 264
    assert cost["batch_shots"] == 5280
    assert len(manifest["config"]["physical_qubits"]) == 20

    small = q10["small_scale_report"](tmp_path)
    assert small["initial_autocorrelator_max_error"] < 1e-12
    assert len(small["curves"]) == 10

    submissions = []
    initial = neel_bitstring(20)

    def fake_run(self, **kwargs):
        submissions.append((self, kwargs))
        return {
            "result": {initial[::-1]: 20},
            "result_meta": {"job_id": f"mock-{len(submissions)}"},
            "uni_status": "completed",
        }

    monkeypatch.setattr(Circuit, "run", fake_run)
    with pytest.raises(PermissionError, match="明确"):
        q10["collect_hardware"](
            mode="pilot", device="QZ03", confirmed=False,
            data_dir=tmp_path, shots=20,
        )
    assert not submissions

    records = q10["collect_hardware"](
        mode="pilot", device="QZ03", confirmed=True,
        data_dir=tmp_path, shots=20,
    )
    assert len(records) == 8
    assert len(submissions) == 8
    for _, kwargs in submissions:
        assert kwargs["provider"] == "lqcloud"
        assert kwargs["device"] == "QZ03"
        assert kwargs["physical_qubits"] == manifest["config"]["physical_qubits"]
        assert kwargs["wait_async_result"] is True
    assert all(item["error_mitigation"] == "none" for item in records)
    assert all(sum(item["counts"].values()) == 20 for item in records)

    resumed = q10["collect_hardware"](
        mode="pilot", device="QZ03", confirmed=True,
        data_dir=tmp_path, shots=20,
    )
    assert len(resumed) == 8
    assert len(submissions) == 8

    with pytest.raises(PermissionError, match="人工审核"):
        q10["collect_hardware"](
            mode="batch", device="QZ03", confirmed=True,
            data_dir=tmp_path, shots=20, pilot_reviewed=False,
        )
    assert len(submissions) == 8


def test_q10_failed_pilot_cannot_open_batch(tmp_path, monkeypatch):
    q10 = runpy.run_path("examples/qd/q10_qz03_floquet_dtc.py", run_name="q10_failed_test")
    q10["create_manifest"](tmp_path, shots=20)
    submissions = []

    def failed_run(self, **kwargs):
        submissions.append(kwargs)
        return {"result": {}, "result_meta": {"job_id": "failed"}, "uni_status": "failed"}

    monkeypatch.setattr(Circuit, "run", failed_run)
    with pytest.raises(RuntimeError, match="counts 无效"):
        q10["collect_hardware"](
            mode="pilot", device="QZ03", confirmed=True,
            data_dir=tmp_path, shots=20,
        )
    assert len(submissions) == 1
    with pytest.raises(PermissionError, match="完整有效 pilot"):
        q10["collect_hardware"](
            mode="batch", device="QZ03", confirmed=True,
            data_dir=tmp_path, shots=20, pilot_reviewed=True,
        )
    assert len(submissions) == 1
