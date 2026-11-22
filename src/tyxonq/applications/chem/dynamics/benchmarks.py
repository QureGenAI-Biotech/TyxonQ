"""量子动力学基准的统计、收敛和可复现数据工具。

本模块不实现动力学数值核心；演化、谱变换和测量仍由 ``exact.py``、
``circuits.py``、``spectroscopy.py`` 等既有模块负责。原始记录使用 JSONL
只追加格式，报告和 manifest 使用带内容哈希的 JSON。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

BENCHMARK_SCHEMA = "tyxonq.qd.benchmark.v1"
MANIFEST_SCHEMA = "tyxonq.qd.manifest.v1"
RECORD_SCHEMA = "tyxonq.qd.record.v1"


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """稳定 JSON 文本，用于内容哈希和跨进程复现。"""
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def pauli_standard_error(expectation: float, shots: int) -> float:
    """Pauli 期望在独立 shots 下的标准误，取值域为 [-1, 1]。"""
    if isinstance(shots, (bool, np.bool_)) or not isinstance(shots, (int, np.integer)):
        raise TypeError("shots 必须为整数")
    shots = int(shots)
    if shots <= 0:
        raise ValueError("有限采样标准误要求 shots>0")
    value = float(expectation)
    if not math.isfinite(value) or abs(value) > 1.0 + 1e-12:
        raise ValueError("Pauli 期望必须为 [-1,1] 内的有限值")
    variance = max(0.0, 1.0 - min(1.0, value * value))
    return math.sqrt(variance / shots)


def confidence_interval(estimate: float, standard_error: float,
                        *, confidence: float = 0.95) -> tuple[float, float]:
    """大样本正态置信区间；标准误必须来自独立采样估计。"""
    from scipy.stats import norm

    sigma = float(standard_error)
    if sigma < 0 or not math.isfinite(sigma):
        raise ValueError("standard_error 必须为有限非负值")
    level = float(confidence)
    if not 0 < level < 1:
        raise ValueError("confidence 必须在 (0,1) 内")
    half_width = float(norm.ppf(0.5 + level / 2.0)) * sigma
    center = float(estimate)
    return center - half_width, center + half_width


def convergence_report(parameters: Sequence[float], errors: Sequence[float]) -> dict:
    """记录细化网格误差及相邻误差比；不假装随机误差必然单调。"""
    params = np.asarray(parameters, dtype=float)
    err = np.asarray(errors, dtype=float)
    if params.ndim != 1 or err.ndim != 1 or len(params) != len(err) or len(err) < 2:
        raise ValueError("parameters/errors 必须为同长一维数组，且至少两个点")
    if not np.isfinite(params).all() or not np.isfinite(err).all() or np.any(err < 0):
        raise ValueError("parameters 和 errors 必须为有限值，errors 非负")
    ratios = err[:-1] / np.maximum(err[1:], 1e-300)
    return {
        "parameters": params,
        "errors": err,
        "error_ratios": ratios,
        "monotonically_decreasing": bool(np.all(np.diff(err) <= 1e-15)),
    }


def build_manifest(*, name: str, config: Mapping[str, Any],
                   reference: Mapping[str, Any] | None = None,
                   preregistered_metrics: Mapping[str, Any] | None = None) -> dict:
    """构造不含硬件任务结果的可复现提交/分析 manifest。"""
    if not str(name).strip():
        raise ValueError("manifest name 不能为空")
    payload = {
        "schema": MANIFEST_SCHEMA,
        "name": str(name),
        "config": _jsonable(dict(config)),
        "reference": _jsonable(dict(reference or {})),
        "preregistered_metrics": _jsonable(dict(preregistered_metrics or {})),
    }
    payload["manifest_hash"] = content_hash(payload)
    return payload


def save_manifest(path, manifest: Mapping[str, Any]) -> Path:
    """原子写入 manifest，并拒绝损坏或篡改的内容哈希。"""
    data = _jsonable(dict(manifest))
    expected = data.get("manifest_hash")
    unsigned = {key: value for key, value in data.items() if key != "manifest_hash"}
    if expected != content_hash(unsigned):
        raise ValueError("manifest_hash 与内容不一致")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(canonical_json(data) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def append_record(path, record: Mapping[str, Any]) -> dict:
    """追加一条带唯一 record_id 的记录；重复 ID 直接拒绝，不覆盖历史。"""
    target = Path(path)
    data = _jsonable(dict(record))
    data.setdefault("schema", RECORD_SCHEMA)
    record_id = data.get("record_id")
    if not record_id:
        raise ValueError("原始记录必须包含非空 record_id")
    existing = load_records(target)
    if any(item.get("record_id") == record_id for item in existing):
        raise ValueError(f"record_id 已存在，拒绝覆盖或重复追加: {record_id}")
    data["record_hash"] = content_hash(data)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(data) + "\n")
        handle.flush()
    return data


def load_records(path, *, verify_hashes: bool = True) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    records = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if verify_hashes:
            expected = item.get("record_hash")
            unsigned = {key: value for key, value in item.items() if key != "record_hash"}
            if expected != content_hash(unsigned):
                raise ValueError(f"第 {line_number} 行 record_hash 校验失败")
        records.append(item)
    return records


def comparison_report(*, name: str, curves: Mapping[str, Any],
                      metadata: Mapping[str, Any] | None = None) -> dict:
    """生成 JSON 可序列化报告；各曲线保留原值，不隐藏负谱或误差。"""
    if not curves:
        raise ValueError("报告至少需要一条曲线")
    normalized = {str(label): _jsonable(values) for label, values in curves.items()}
    lengths = {len(values) for values in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("报告曲线必须具有相同长度")
    return {
        "schema": BENCHMARK_SCHEMA,
        "name": str(name),
        "curves": normalized,
        "metadata": _jsonable(dict(metadata or {})),
    }


__all__ = [
    "BENCHMARK_SCHEMA",
    "MANIFEST_SCHEMA",
    "RECORD_SCHEMA",
    "append_record",
    "build_manifest",
    "comparison_report",
    "confidence_interval",
    "content_hash",
    "convergence_report",
    "load_records",
    "pauli_standard_error",
    "save_manifest",
]
