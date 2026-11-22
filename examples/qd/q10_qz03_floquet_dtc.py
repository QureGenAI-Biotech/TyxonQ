"""Q10：QZ03 上的 20 比特 Floquet Ising 多体动力学与公开实验对照。

默认完全离线：运行 4 比特数值金标准、读取论文 Fig. 3a 原始未缓解曲线，
生成 pilot/batch manifest，并重建已有真机报告。

直接实验对应：Mi et al., Nature 600, 455–460 (2021)，DOI
10.1038/s41586-021-04257-w；CC BY 4.0 数据 DOI 10.5281/zenodo.5570676。
公开记录不含原始 disorder 种子和 counts，因此这是同模型、同观测量的独立
配对，不是 Google 原始线路逐任务复现，也不宣称量子优势。

真机必须显式 --device QZ03 --confirm-hardware；批量另需 --pilot-reviewed。
助手实际执行仍须先向用户报告线路数、深度和 shots 并分别获得确认。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from tyxonq.applications.chem.dynamics.benchmarks import (
    append_record,
    build_manifest,
    content_hash,
    load_records,
    save_manifest,
)
from tyxonq.applications.chem.dynamics.circuits import measure_pauli_expectation
from tyxonq.applications.chem.dynamics.floquet import (
    autocorrelator_from_payload,
    autocorrelator_items,
    build_floquet_dtc_circuit,
    nearest_neighbor_edges,
    neel_bitstring,
    sample_dtc_disorder,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "qd_benchmarks" / "floquet_dtc_zenodo_5570676"
N_QUBITS = 20
G_VALUE = 0.94
DEFAULT_PHYSICAL_QUBITS = [
    0, 1, 2, 8, 9, 3, 4, 10, 15, 14,
    20, 19, 18, 17, 23, 29, 30, 24, 25, 26,
]
PILOT_SEEDS = (811, 812)
PILOT_CYCLES = (0, 1, 2, 4)
BATCH_SEEDS = tuple(9100 + i for i in range(24))
BATCH_CYCLES = tuple(range(11))


def load_experiment(data_dir: Path = DATA_DIR) -> dict[str, np.ndarray]:
    path = Path(data_dir) / "raw" / "DTC_Data" / "fig_3a.csv"
    table = np.genfromtxt(path, delimiter=",", names=True)
    values = np.asarray(table["mbl_neel"], dtype=float)
    if values.shape != (61,) or not np.all(np.isfinite(values)):
        raise ValueError("Fig. 3a mbl_neel 应为 61 个有限值")
    return {"cycles": np.arange(61, dtype=int), "mbl_neel": values}


def circuit_for(seed: int, cycle: int, n_qubits: int = N_QUBITS):
    h, phi, edges = sample_dtc_disorder(n_qubits, seed)
    initial = neel_bitstring(n_qubits)
    circuit = build_floquet_dtc_circuit(
        n_qubits, cycle, G_VALUE, h, phi,
        edges=edges, init_bitstring=initial,
    )
    return circuit, initial, h, phi, edges


def task_specs(seeds: Sequence[int], cycles: Sequence[int]) -> list[dict[str, Any]]:
    specs = []
    for seed in seeds:
        for cycle in cycles:
            circuit, initial, _, _, _ = circuit_for(int(seed), int(cycle))
            gate_counts = dict(Counter(op[0] for op in circuit.ops))
            specs.append({
                "seed": int(seed),
                "cycle": int(cycle),
                "g": G_VALUE,
                "n_qubits": N_QUBITS,
                "init_bitstring": initial,
                "circuit_hash": content_hash({"system": "floquet_dtc_v1", "ops": circuit.ops}),
                "logical_gate_counts": gate_counts,
                "logical_ops": len(circuit.ops),
                "estimated_cx_from_rzz": 2 * (N_QUBITS - 1) * int(cycle),
            })
    return specs


def create_manifest(
    data_dir: Path = DATA_DIR,
    *,
    shots: int = 100,
    physical_qubits: Sequence[int] = DEFAULT_PHYSICAL_QUBITS,
) -> dict[str, Any]:
    if int(shots) <= 0:
        raise ValueError("shots 必须为正")
    layout = [int(q) for q in physical_qubits]
    if len(layout) != N_QUBITS or len(set(layout)) != N_QUBITS:
        raise ValueError("QZ03 物理布局必须包含 20 个不同比特")
    pilot = task_specs(PILOT_SEEDS, PILOT_CYCLES)
    batch = task_specs(BATCH_SEEDS, BATCH_CYCLES)
    config = {
        "provider": "lqcloud",
        "device": "QZ03",
        "n_qubits": N_QUBITS,
        "g": G_VALUE,
        "initial_state_logical": "01" * 10,
        "initial_state_ir": neel_bitstring(N_QUBITS),
        "logical_edges": nearest_neighbor_edges(N_QUBITS),
        "physical_qubits": layout,
        "disorder": {
            "rng": "numpy.random.default_rng/PCG64",
            "h_range": [-float(np.pi), float(np.pi)],
            "phi_range": [-1.5 * float(np.pi), -0.5 * float(np.pi)],
            "batch_instances": len(BATCH_SEEDS),
        },
        "cycle_gate_order": ["RX(pi*g)", "RZZ(phi/2)", "RZ(h)"],
        "shots_per_circuit": int(shots),
        "error_mitigation": "none",
        "pilot_seeds": list(PILOT_SEEDS),
        "pilot_cycles": list(PILOT_CYCLES),
        "batch_seeds": list(BATCH_SEEDS),
        "batch_cycles": list(BATCH_CYCLES),
        "pilot_specs": pilot,
        "batch_specs": batch,
        "cost_estimate": {
            "pilot_circuits": len(pilot),
            "pilot_shots": len(pilot) * int(shots),
            "batch_circuits": len(batch),
            "batch_shots": len(batch) * int(shots),
            "note": "LQCloud 异步任务提交延迟未计入 shots 成本",
        },
        "limits": [
            "Batch only covers cycles 0..10 of the published 0..60 curve.",
            "The 20-qubit benchmark is classically simulable and is not a quantum-advantage claim.",
            "The 35-qubit two-dimensional QZ03 extension is not part of this direct comparison.",
        ],
    }
    reference = {
        "article_doi": "10.1038/s41586-021-04257-w",
        "data_doi": "10.5281/zenodo.5570676",
        "license": "CC BY 4.0",
        "figure": "3a",
        "column": "mbl_neel",
        "published_disorder_instances": 24,
        "published_error_mitigation": "none",
        "original_seeds_available": False,
    }
    manifest = build_manifest(
        name="qz03_floquet_dtc_20q", config=config, reference=reference,
    )
    save_manifest(Path(data_dir) / "submission_manifest.json", manifest)
    return manifest


def small_scale_report(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """4 比特数值—线路金标准，不冒充 20 比特经典参照。"""
    curves = []
    for seed in (101, 102):
        for cycle in range(5):
            circuit, initial, _, _, _ = circuit_for(seed, cycle, n_qubits=4)
            value = measure_pauli_expectation(
                circuit, autocorrelator_items(initial), n_qubits=4, shots=0,
            )
            curves.append({"seed": seed, "cycle": cycle, "autocorrelator": float(value)})
    initial_error = max(abs(item["autocorrelator"] - 1.0) for item in curves if item["cycle"] == 0)
    report = {
        "schema": "qz03_floquet_small_scale_v1",
        "n_qubits": 4,
        "method": "Circuit + statevector Pauli expectation, shots=0",
        "initial_autocorrelator_max_error": float(initial_error),
        "curves": curves,
        "scope": "numerical/circuit regression only; not a 20-qubit classical comparison",
    }
    if initial_error > 1e-12:
        raise AssertionError("零周期自关联必须为 1")
    (Path(data_dir) / "small_scale_exact.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    return report


def _spec_key(spec: dict[str, Any]) -> tuple[int, int]:
    return int(spec["seed"]), int(spec["cycle"])


def _current_valid_records(
    records: Sequence[dict[str, Any]],
    specs: Sequence[dict[str, Any]],
    config: dict[str, Any],
    manifest_hash: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    """选择当前 manifest 的最新有效记录；原始 JSONL 中的失败/旧记录仍保留。"""
    spec_by_key = {_spec_key(spec): spec for spec in specs}
    selected = {}
    for item in records:
        try:
            key = int(item["seed"]), int(item["cycle"])
        except (KeyError, TypeError, ValueError):
            continue
        spec = spec_by_key.get(key)
        counts = item.get("counts")
        if spec is None or not isinstance(counts, dict) or not counts:
            continue
        valid = (
            item.get("status") == "completed"
            and item.get("manifest_hash") == manifest_hash
            and item.get("device") == config["device"]
            and int(item.get("shots", -1)) == int(config["shots_per_circuit"])
            and item.get("circuit_hash") == spec["circuit_hash"]
            and [int(q) for q in item.get("physical_qubits", [])] == list(config["physical_qubits"])
            and sum(counts.values()) == int(config["shots_per_circuit"])
            and all(
                len(str(k)) == N_QUBITS and set(str(k)) <= {"0", "1"}
                for k in counts
            )
        )
        if valid:
            selected[key] = item
    return selected


def pilot_report(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    root = Path(data_dir)
    manifest = json.loads((root / "submission_manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    raw_records = load_records(root / "pilot_counts.jsonl")
    specs = config["pilot_specs"]
    expected = {_spec_key(spec) for spec in specs}
    by_key = _current_valid_records(raw_records, specs, config, manifest["manifest_hash"])
    complete = expected == set(by_key)
    records = [by_key[_spec_key(spec)] for spec in specs if _spec_key(spec) in by_key]
    experiment = load_experiment(root)
    per_record = []
    for item in records:
        per_record.append({
            "task_id": item.get("task_id"),
            "seed": int(item["seed"]),
            "cycle": int(item["cycle"]),
            "unique_bitstrings": len(item["counts"]),
            "autocorrelator_raw": autocorrelator_from_payload(
                {"result": item["counts"]}, item["init_bitstring"],
            ),
        })
    cycle_summary = []
    for cycle in config["pilot_cycles"]:
        values = [item["autocorrelator_raw"] for item in per_record if item["cycle"] == cycle]
        cycle_summary.append({
            "cycle": int(cycle),
            "hardware_raw_mean": float(np.mean(values)) if values else None,
            "instances": len(values),
            "published_experiment_24_instances": float(experiment["mbl_neel"][cycle]),
        })
    report = {
        "schema": "qz03_floquet_pilot_v1",
        "complete": complete,
        "raw_records": len(raw_records),
        "selected_records": len(records),
        "error_mitigation": "none",
        "records": per_record,
        "cycle_summary": cycle_summary,
        "scope": "2 independently seeded instances; hardware validation only, not a scientific ensemble comparison",
    }
    (root / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def offline_report(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    root = Path(data_dir)
    manifest = json.loads((root / "submission_manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    experiment = load_experiment(root)
    cycles = np.asarray(config["batch_cycles"], dtype=int)
    specs = config["batch_specs"]
    expected = {_spec_key(spec) for spec in specs}
    records = load_records(root / "hardware_counts.jsonl")
    by_key = _current_valid_records(records, specs, config, manifest["manifest_hash"])
    hardware = []
    complete = expected == set(by_key)
    if complete:
        initial = neel_bitstring(N_QUBITS)
        for cycle in cycles:
            values = [
                autocorrelator_from_payload({"result": by_key[(seed, int(cycle))]["counts"]}, initial)
                for seed in config["batch_seeds"]
            ]
            hardware.append(float(np.mean(values)))
    report: dict[str, Any] = {
        "schema": "qz03_floquet_comparison_v1",
        "curves": {
            "cycles": cycles,
            "experiment_mbl_neel": experiment["mbl_neel"][cycles],
        },
        "metadata": {
            "hardware_status": "collected" if complete else "incomplete",
            "raw_hardware_records": len(records),
            "selected_hardware_records": len(by_key),
            "hardware_run_ids": sorted({item["run_id"] for item in by_key.values()}),
            "required_hardware_records": len(expected),
            "error_mitigation": "none",
            "classical_20q_status": "not_run",
            "experiment_type": "processed unmitigated Fig. 3a curve from an independent processor",
            "disorder_identity": "same protocol and ensemble size; independently fixed seeds, not original Google instances",
            "quantum_advantage_claim": False,
        },
    }
    if complete:
        hardware_arr = np.asarray(hardware)
        experiment_arr = experiment["mbl_neel"][cycles]
        report["curves"]["hardware_raw"] = hardware_arr
        report["metadata"]["hardware_experiment_rmse"] = float(
            np.sqrt(np.mean((hardware_arr - experiment_arr) ** 2))
        )
        report["metadata"]["rmse_scope"] = "cycles 0..10 only; no post-hoc window selection"
    (root / "offline_report.json").write_text(
        json.dumps(report, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x),
        encoding="utf-8",
    )
    return report


def collect_hardware(
    *,
    mode: str,
    device: str,
    confirmed: bool,
    data_dir: Path = DATA_DIR,
    shots: int = 100,
    pilot_reviewed: bool = False,
    physical_qubits: Sequence[int] = DEFAULT_PHYSICAL_QUBITS,
) -> list[dict[str, Any]]:
    if not confirmed:
        raise PermissionError("真机提交需要本次明确 confirmed=True")
    if mode not in {"pilot", "batch"}:
        raise ValueError("mode 必须为 pilot/batch")
    if not device:
        raise ValueError("真机提交必须显式提供 device")
    if int(shots) <= 0:
        raise ValueError("shots 必须为正")
    root = Path(data_dir)
    manifest = json.loads((root / "submission_manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if device != config["device"]:
        raise ValueError("device 必须与 manifest 的 QZ03 一致")
    if int(shots) != int(config["shots_per_circuit"]):
        raise ValueError("shots 与 manifest 不一致，请先离线重建并审核成本")
    layout = [int(q) for q in physical_qubits]
    if layout != [int(q) for q in config["physical_qubits"]]:
        raise ValueError("physical_qubits 与 manifest 不一致")
    if mode == "batch":
        if not pilot_reviewed:
            raise PermissionError("批量提交前必须人工审核 pilot")
        pilot_specs = config["pilot_specs"]
        valid_pilot = _current_valid_records(
            load_records(root / "pilot_counts.jsonl"), pilot_specs, config,
            manifest["manifest_hash"],
        )
        if {_spec_key(spec) for spec in pilot_specs} != set(valid_pilot):
            raise PermissionError("批量提交前必须存在当前 manifest 的完整有效 pilot")
    specs = config["pilot_specs" if mode == "pilot" else "batch_specs"]
    filename = "pilot_counts.jsonl" if mode == "pilot" else "hardware_counts.jsonl"
    records_path = root / filename
    existing = _current_valid_records(
        load_records(records_path), specs, config, manifest["manifest_hash"],
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for index, spec in enumerate(specs):
        if _spec_key(spec) in existing:
            continue
        circuit, initial, h, phi, edges = circuit_for(int(spec["seed"]), int(spec["cycle"]))
        measured = circuit.extended([("measure_z", q) for q in range(N_QUBITS)])
        result = measured.run(
            provider="lqcloud", device=device, shots=int(shots),
            physical_qubits=layout, wait_async_result=True,
        )
        payloads = result if isinstance(result, (list, tuple)) else [result]
        if len(payloads) != 1 or not isinstance(payloads[0], dict):
            raise RuntimeError(f"第 {index} 条线路的 LQCloud 返回数量或格式异常")
        payload = payloads[0]
        metadata = payload.get("result_meta") or {}
        task_id = payload.get("task_id") or payload.get("id") or metadata.get("job_id")
        counts = payload.get("result")
        status = payload.get("uni_status") or payload.get("status") or "unknown"
        error = payload.get("error") or ""
        append_record(records_path, {
            "record_id": f"floquet-{mode}-{run_id}-{index:04d}",
            "run_id": run_id,
            "provider": "lqcloud",
            "device": device,
            "task_id": task_id,
            "status": status,
            "error": error,
            "seed": int(spec["seed"]),
            "cycle": int(spec["cycle"]),
            "g": G_VALUE,
            "n_qubits": N_QUBITS,
            "init_bitstring": initial,
            "physical_qubits": layout,
            "shots": int(shots),
            "counts": counts if isinstance(counts, dict) else {},
            "circuit_hash": spec["circuit_hash"],
            "disorder_hash": content_hash({"h": h.tolist(), "phi": phi.tolist(), "edges": edges}),
            "manifest_hash": manifest["manifest_hash"],
            "error_mitigation": "none",
            "calibration_snapshot": payload.get("calibration"),
            "calibration_status": "provided" if payload.get("calibration") else "unavailable_from_lqcloud_driver",
            "submitted_at_utc": run_id,
        })
        valid = (
            isinstance(counts, dict) and bool(counts)
            and sum(counts.values()) > 0
            and all(len(str(k)) == N_QUBITS and set(str(k)) <= {"0", "1"} for k in counts)
        )
        if not valid:
            raise RuntimeError(
                f"第 {index} 条线路 counts 无效；task_id={task_id}，status={status}，"
                f"error={error}；已保存记录并停止后续提交"
            )
    return load_records(records_path)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--physical-qubits", type=int, nargs="+", default=DEFAULT_PHYSICAL_QUBITS)
    parser.add_argument("--submit-pilot", action="store_true")
    parser.add_argument("--submit-batch", action="store_true")
    parser.add_argument("--confirm-hardware", action="store_true")
    parser.add_argument("--pilot-reviewed", action="store_true")
    args = parser.parse_args(argv)
    if args.submit_pilot and args.submit_batch:
        raise ValueError("pilot 和 batch 必须分次执行")
    manifest = create_manifest(
        args.data_dir, shots=args.shots, physical_qubits=args.physical_qubits,
    )
    small = small_scale_report(args.data_dir)
    print("small-scale initial-state error:", small["initial_autocorrelator_max_error"])
    print("pilot cost:", manifest["config"]["cost_estimate"]["pilot_circuits"], "circuits /",
          manifest["config"]["cost_estimate"]["pilot_shots"], "shots")
    print("batch cost:", manifest["config"]["cost_estimate"]["batch_circuits"], "circuits /",
          manifest["config"]["cost_estimate"]["batch_shots"], "shots")
    experiment = load_experiment(args.data_dir)
    print("published Fig. 3a mbl_neel[0:5]:", experiment["mbl_neel"][:5].tolist())
    if args.submit_pilot:
        collect_hardware(
            mode="pilot", device=args.device, confirmed=args.confirm_hardware,
            data_dir=args.data_dir, shots=args.shots, physical_qubits=args.physical_qubits,
        )
    elif args.submit_batch:
        collect_hardware(
            mode="batch", device=args.device, confirmed=args.confirm_hardware,
            data_dir=args.data_dir, shots=args.shots, pilot_reviewed=args.pilot_reviewed,
            physical_qubits=args.physical_qubits,
        )
    pilot = pilot_report(args.data_dir)
    print("pilot complete:", pilot["complete"])
    report = offline_report(args.data_dir)
    print("hardware status:", report["metadata"]["hardware_status"])
    return report


if __name__ == "__main__":
    main()
