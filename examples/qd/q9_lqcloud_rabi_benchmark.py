"""Q9：单电子自旋 Rabi 的经典—LQCloud—实验闭环。

默认完全离线：读取 CC BY 4.0 Figshare 数据，使用 repetition 0 拟合、
repetition 1 验证，并生成单比特 RX 提交 manifest。不会自动提交任务。
真机需显式 ``--device`` 和 ``--confirm-hardware``；批量还需 pilot 已存在及
``--pilot-reviewed``。独立验证未通过时，只有额外指定 ``--paired-dataset``
才能按用户单独授权的配对数据路线采集，且原始记录仍保留验证失败状态。
助手实际运行时仍须分别获得用户确认。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from tyxonq.applications.chem.dynamics.benchmarks import (
    append_record,
    pauli_standard_error,
    build_manifest,
    comparison_report,
    confidence_interval,
    content_hash,
    load_records,
    save_manifest,
)
from tyxonq.core.ir.circuit import Circuit

DATA_DIR = Path(__file__).parents[2] / "data" / "qd_benchmarks" / "electron_rabi_figshare_14871087"
RAW_FILE = "repetition_0_nuclei_idx_0_0_ESR_ESR_pi_UU_duration_0_0_1.dat"


def rabi_probability(duration_s, frequency_hz, visibility=1.0, offset=0.0):
    """共振驱动的 p_up=offset+visibility*sin²(π f_R t)。"""
    t = np.asarray(duration_s, dtype=float)
    return offset + visibility * np.sin(np.pi * float(frequency_hz) * t) ** 2


def load_experiment(data_dir=DATA_DIR, *, nuclei_idx=3):
    """读取 QCoDeS GNUPlot 表；时间单位为秒，up_proportion 无量纲。"""
    path = Path(data_dir) / "raw" / RAW_FILE
    table = np.genfromtxt(path)
    if table.ndim != 2 or table.shape[1] != 7:
        raise ValueError("Rabi 数据应为七列 QCoDeS 表")
    group = table[table[:, 1] == float(nuclei_idx)]
    if not len(group):
        raise ValueError(f"缺少 nuclei_idx={nuclei_idx} 的实验数据")
    durations = np.unique(group[:, 2])
    repetitions = {}
    for repetition in np.unique(group[:, 0]):
        rows = group[group[:, 0] == repetition]
        if not np.array_equal(np.unique(rows[:, 2]), durations):
            raise ValueError("各实验重复必须使用相同时间网格")
        values = np.array([rows[rows[:, 2] == t, 5].mean() for t in durations])
        shots = np.array([rows[rows[:, 2] == t, 3].sum() for t in durations])
        if np.any(shots <= 0):
            raise ValueError("实验 shots 必须为正")
        repetitions[str(int(repetition))] = {"up": values, "shots": shots}
    if "0" not in repetitions or "1" not in repetitions:
        raise ValueError("需要 repetition 0 拟合和 repetition 1 独立验证")
    return {"duration_s": durations, "repetitions": repetitions, "source": str(path)}


def fit_classical_reference(experiment):
    """仅用 repetition 0 标定，repetition 1 不参与拟合。"""
    times = experiment["duration_s"]
    calibration = experiment["repetitions"]["0"]
    validation = experiment["repetitions"]["1"]

    def model(t, frequency, visibility, offset):
        return rabi_probability(t, frequency, visibility, offset)

    sigma = np.sqrt(np.maximum(calibration["up"] * (1 - calibration["up"]), 1e-4) / calibration["shots"])
    params, covariance = curve_fit(
        model, times, calibration["up"], p0=[3e5, 0.85, 0.03], sigma=sigma,
        absolute_sigma=True, bounds=([1e3, 0.0, 0.0], [2e6, 1.0, 0.5]),
        maxfev=50000,
    )
    predicted = model(times, *params)
    validation_error = validation["up"] - predicted
    validation_sigma = np.sqrt(np.maximum(validation["up"] * (1 - validation["up"]), 1e-4) / validation["shots"])
    return {
        "frequency_hz": float(params[0]),
        "visibility": float(params[1]),
        "offset": float(params[2]),
        "parameter_stderr": np.sqrt(np.diag(covariance)),
        "calibration_repetition": 0,
        "validation_repetition": 1,
        "calibration_rmse": float(np.sqrt(np.mean((calibration["up"] - predicted) ** 2))),
        "validation_rmse": float(np.sqrt(np.mean(validation_error ** 2))),
        "validation_reduced_chi_square": float(np.mean((validation_error / validation_sigma) ** 2)),
        "model": "offset + visibility * sin(pi * frequency_hz * duration_s)**2",
    }


def rabi_circuit(duration_s, fit):
    """RX(2π f_R t) 给出 sin²(π f_R t) 的理想单比特布居。"""
    circuit = Circuit(1)
    circuit.rx(0, theta=float(2 * np.pi * fit["frequency_hz"] * duration_s))
    circuit.ops.append(("measure_z", 0))
    return circuit


def circuit_payload(circuit):
    return [[str(item) if not isinstance(item, (float, int)) else item for item in op]
            for op in circuit.ops]


def create_manifest(experiment, fit, data_dir=DATA_DIR, *, shots=100):
    if int(shots) <= 0:
        raise ValueError("硬件 shots 必须为正")
    times = experiment["duration_s"]
    pilot_times = np.linspace(0.0, 1.0 / fit["frequency_hz"], 5)
    pilot_hashes = [content_hash(circuit_payload(rabi_circuit(t, fit))) for t in pilot_times]
    batch_hashes = [content_hash(circuit_payload(rabi_circuit(t, fit))) for t in times]
    manifest = build_manifest(
        name="electron-spin-rabi-lqcloud",
        config={
            "provider": "lqcloud",
            "device": "explicit CLI argument; no default hardware device",
            "observable": "p_up = counts['1']/sum(counts)",
            "initial_state": "|0>",
            "pilot_durations_s": pilot_times,
            "batch_durations_s": times,
            "shots_per_point": int(shots),
            "pilot_circuit_hashes": pilot_hashes,
            "batch_circuit_hashes": batch_hashes,
            "cost_estimate": {
                "pilot_circuits": len(pilot_times),
                "batch_circuits": len(times),
                "pilot_total_shots": len(pilot_times) * int(shots),
                "batch_total_shots": len(times) * int(shots),
                "credits": None,
                "note": "不在应用层假设 LQCloud 价格；提交前由用户核对实际设备成本",
            },
        },
        reference=fit,
        preregistered_metrics={
            "validation_rmse_max": 0.075,
            "comparison": "same duration grid; report shot SEM and model discrepancy separately",
            "no_post_hoc_filtering": True,
        },
    )
    save_manifest(Path(data_dir) / "submission_manifest.json", manifest)
    return manifest


def _current_valid_records(
    records, times, circuit_hashes, manifest, *, device=None,
) -> dict[float, dict[str, Any]]:
    config = manifest["config"]
    hash_by_time = {float(t): h for t, h in zip(times, circuit_hashes)}
    selected = {}
    for item in records:
        try:
            duration = float(item["duration_s"])
        except (KeyError, TypeError, ValueError):
            continue
        counts = item.get("counts")
        if duration not in hash_by_time or not isinstance(counts, dict) or not counts:
            continue
        if device is not None and item.get("device") != device:
            continue
        valid = (
            item.get("status") == "completed"
            and item.get("manifest_hash") == manifest["manifest_hash"]
            and item.get("circuit_hash") == hash_by_time[duration]
            and int(item.get("shots", -1)) == int(config["shots_per_point"])
            and sum(counts.values()) == int(config["shots_per_point"])
            and set(counts) <= {"0", "1"}
        )
        if valid:
            selected[duration] = item
    return selected


def offline_report(experiment, fit, data_dir=DATA_DIR):
    root = Path(data_dir)
    manifest = json.loads((root / "submission_manifest.json").read_text(encoding="utf-8"))
    if content_hash(fit) != content_hash(manifest["reference"]):
        raise ValueError("fit 与 manifest 的拟合快照不一致")
    times = experiment["duration_s"]
    classical = rabi_probability(
        times, fit["frequency_hz"], fit["visibility"], fit["offset"]
    )
    validation = experiment["repetitions"]["1"]["up"]
    report = comparison_report(
        name="electron-rabi-classical-experiment",
        curves={"classical_calibrated": classical, "experiment_validation": validation},
        metadata={
            "duration_s": times,
            "fit": fit,
            "validation_gate": {
                "metric": "validation_rmse",
                "threshold": 0.075,
                "passed": bool(fit["validation_rmse"] <= 0.075),
                "note": "验证未通过时禁止默认三方对标批量；配对数据路线须单独授权，pilot 也需审核",
            },
            "hardware_status": "not_submitted",
            "raw_hardware_records": 0,
        },
    )
    records_path = root / "hardware_counts.jsonl"
    records = load_records(records_path)
    by_time = _current_valid_records(
        records, times, manifest["config"]["batch_circuit_hashes"], manifest,
    )
    devices = {item.get("device") for item in by_time.values()}
    if set(by_time) == set(times) and len(devices) == 1:
        matched = [by_time[float(t)] for t in times]
        hardware = np.array([
            item["counts"].get("1", 0) / sum(item["counts"].values())
            for item in matched
        ])
        shots = np.array([sum(item["counts"].values()) for item in matched])
        shot_sem = np.array([
            0.5 * pauli_standard_error(2.0 * float(p) - 1.0, int(n))
            for p, n in zip(hardware, shots)
        ])
        intervals = np.array([
            confidence_interval(float(p), float(sem))
            for p, sem in zip(hardware, shot_sem)
        ])
        report["curves"]["hardware"] = hardware
        report["curves"]["hardware_shot_sem"] = shot_sem
        report["curves"]["hardware_ci95_low"] = intervals[:, 0]
        report["curves"]["hardware_ci95_high"] = intervals[:, 1]
        report["metadata"]["hardware_status"] = "collected"
        report["metadata"]["raw_hardware_records"] = len(records)
        report["metadata"]["selected_hardware_records"] = len(matched)
        report["metadata"]["hardware_device"] = next(iter(devices))
        report["metadata"]["hardware_collection_routes"] = sorted({
            item.get("collection_route", "unspecified") for item in matched
        })
        report["metadata"]["hardware_independent_validation_passed"] = all(
            item.get("independent_validation_passed", False) for item in matched
        )
        report["metadata"]["hardware_rmse_vs_classical"] = float(np.sqrt(np.mean((hardware - classical) ** 2)))
        report["metadata"]["hardware_rmse_vs_experiment"] = float(np.sqrt(np.mean((hardware - validation) ** 2)))
        report["metadata"]["classical_experiment_rmse"] = float(fit["validation_rmse"])
        report["metadata"]["mean_hardware_shot_sem"] = float(np.mean(shot_sem))
        report["metadata"]["error_decomposition_note"] = (
            "逐点 shot SEM/95% CI 与 RMSE 分列；hardware-classical 残差同时包含统计噪声和硬件偏差，"
            "单次运行不能将二者独立辨识，不作事后校正"
        )
    target = Path(data_dir) / "offline_report.json"
    target.write_text(json.dumps(report, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) + "\n", encoding="utf-8")
    return report


def collect_hardware(*, mode, device, confirmed, fit, experiment,
                     data_dir=DATA_DIR, shots=100, pilot_reviewed=False,
                     pairing_only=False):
    """唯一真机入口；未通过独立验证的批量仅限显式配对数据路线。"""
    if not confirmed:
        raise PermissionError("真机提交需要本次明确 confirmed=True")
    if mode not in {"pilot", "batch"}:
        raise ValueError("mode 必须为 pilot/batch")
    if not device:
        raise ValueError("真机提交必须显式提供 device")
    manifest_path = Path(data_dir) / "submission_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if content_hash(fit) != content_hash(manifest["reference"]):
        raise ValueError("fit 与 manifest 的拟合快照不一致")
    if int(shots) != int(manifest["config"]["shots_per_point"]):
        raise ValueError("shots 与 manifest 不一致")
    validation_passed = (
        fit["validation_rmse"]
        <= manifest["preregistered_metrics"]["validation_rmse_max"]
    )
    if mode == "batch":
        if not pilot_reviewed:
            raise PermissionError("批量提交前必须人工审核 pilot 并设置 pilot_reviewed=True")
        pilot_times = np.asarray(manifest["config"]["pilot_durations_s"])
        valid_pilot = _current_valid_records(
            load_records(Path(data_dir) / "pilot_counts.jsonl"),
            pilot_times, manifest["config"]["pilot_circuit_hashes"], manifest,
            device=device,
        )
        if set(valid_pilot) != set(float(t) for t in pilot_times):
            raise PermissionError("批量提交前必须存在当前 manifest 的完整有效 pilot")
        if not validation_passed and not pairing_only:
            raise PermissionError(
                "独立实验重复验证未通过；默认禁止批量，"
                "仅在用户明确授权配对数据路线时设置 pairing_only=True"
            )
        times = np.asarray(manifest["config"]["batch_durations_s"])
    else:
        if pairing_only:
            raise ValueError("pairing_only 仅适用于显式批量配对数据采集")
        times = np.asarray(manifest["config"]["pilot_durations_s"])
    circuits = [rabi_circuit(t, fit) for t in times]
    filename = "pilot_counts.jsonl" if mode == "pilot" else "hardware_counts.jsonl"
    records_path = Path(data_dir) / filename
    hashes = manifest["config"]["pilot_circuit_hashes" if mode == "pilot" else "batch_circuit_hashes"]
    existing = _current_valid_records(
        load_records(records_path), times, hashes, manifest, device=device,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for index, (time, circuit) in enumerate(zip(times, circuits)):
        if float(time) in existing:
            continue
        result = circuit.run(
            provider="lqcloud", device=device, shots=int(shots),
            wait_async_result=True,
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
            "record_id": f"{mode}-{run_id}-{index:04d}",
            "run_id": run_id,
            "provider": "lqcloud",
            "device": device,
            "task_id": task_id,
            "status": status,
            "error": error,
            "collection_route": (
                "pilot" if mode == "pilot"
                else "hardware_experiment_pairing" if pairing_only
                else "validated_comparison"
            ),
            "independent_validation_passed": bool(validation_passed),
            "duration_s": float(time),
            "shots": int(shots),
            "counts": counts if isinstance(counts, dict) else {},
            "circuit_hash": content_hash(circuit_payload(circuit)),
            "manifest_hash": manifest["manifest_hash"],
            "calibration_snapshot": payload.get("calibration"),
            "calibration_status": "provided" if payload.get("calibration") else "unavailable_from_lqcloud_driver",
            "submitted_at_utc": run_id,
        })
        if not isinstance(counts, dict) or not counts or sum(counts.values()) <= 0:
            raise RuntimeError(
                f"第 {index} 条线路缺少有效 counts；task_id={task_id}，"
                f"status={status}，error={error}；已保存原始记录并停止后续提交"
            )
    return load_records(records_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--submit-pilot", action="store_true")
    parser.add_argument("--submit-batch", action="store_true")
    parser.add_argument("--confirm-hardware", action="store_true")
    parser.add_argument("--pilot-reviewed", action="store_true")
    parser.add_argument(
        "--paired-dataset", action="store_true",
        help="独立验证未通过时，仅按明确授权采集真机—实验配对数据",
    )
    args = parser.parse_args(argv)
    if args.submit_pilot and args.submit_batch:
        raise ValueError("pilot 和 batch 必须分次执行")
    if args.paired_dataset and not args.submit_batch:
        raise ValueError("--paired-dataset 只能与 --submit-batch 一起使用")

    experiment = load_experiment(args.data_dir)
    fit = fit_classical_reference(experiment)
    manifest = create_manifest(experiment, fit, args.data_dir, shots=args.shots)
    report = offline_report(experiment, fit, args.data_dir)
    print(f"Rabi frequency: {float(fit['frequency_hz']) / 1e3:.6f} kHz")
    print(f"independent repetition-1 RMSE: {fit['validation_rmse']:.6f}")
    print(f"manifest: {args.data_dir / 'submission_manifest.json'}")
    print(f"offline report: {args.data_dir / 'offline_report.json'}")
    print(f"hardware status: {report['metadata']['hardware_status']}")
    if args.submit_pilot:
        collect_hardware(mode="pilot", device=args.device, confirmed=args.confirm_hardware,
                         fit=fit, experiment=experiment, data_dir=args.data_dir, shots=args.shots)
    elif args.submit_batch:
        collect_hardware(mode="batch", device=args.device, confirmed=args.confirm_hardware,
                         fit=fit, experiment=experiment, data_dir=args.data_dir, shots=args.shots,
                         pilot_reviewed=args.pilot_reviewed,
                         pairing_only=args.paired_dataset)
    if args.submit_pilot or args.submit_batch:
        report = offline_report(experiment, fit, args.data_dir)
        print(f"updated hardware status: {report['metadata']['hardware_status']}")
    return report


if __name__ == "__main__":
    main()
