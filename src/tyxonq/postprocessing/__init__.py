"""Postprocessing router

职责：
- 提供统一的后处理入口，根据 `options["method"]` 路由到具体实现。
- 不直接依赖电路层，保持 `Circuit` 只负责收集参数并调用此处。
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import time

__all__ = ["apply_postprocessing"]


def apply_postprocessing(result: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Apply postprocessing to a single task result dict.

    Parameters:
        result: 单个任务的结果字典，期望包含 `result` 与可选的 `metadata`（含 shots）。
        options: 后处理选项，至少包含 `method`，其余参数根据方法不同而不同。

    Returns:
        一个形如 {"method": str|None, "result": Any|None} 的后处理产物。
    """
    opts = dict(options or {})
    method = opts.get("method")
    post = {"method": method, "result": None}

    if not method:
        return post

    if method == "readout_mitigation":
        try:
            from .readout import ReadoutMit  # 延迟导入，避免循环依赖

            cals = opts.get("cals")
            mit_method = opts.get("mitigation", "inverse")
            counts = result.get("result") or result.get("counts") or {}
            meta = result.get("metadata") or {}
            try:
                shots_meta = int(meta.get("shots", 0))
            except Exception:
                shots_meta = 0
            if shots_meta <= 0 and counts:
                # fallback: infer from counts sum when metadata missing or zero
                try:
                    shots_meta = int(sum(int(v) for v in counts.values()))
                except Exception:
                    shots_meta = 0

            if counts and cals:
                mit = ReadoutMit()
                mit.set_single_qubit_cals(dict(cals))
                _t0 = time.perf_counter()
                corrected = mit.apply_readout_mitigation(
                    counts, method=str(mit_method), qubits=None, shots=shots_meta
                )
                _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
                post["result"] = {
                    "raw_counts": counts,
                    "mitigated_counts": corrected,
                    "mitigation_time_ms": float(_elapsed_ms),
                }
            else:
                post["result"] = {
                    "raw_counts": counts,
                    "mitigated_counts": None,
                    "reason": "missing cals or empty result",
                }
        except Exception:
            # 失败时保持占位结构，避免在 Circuit 层抛出
            pass
        return post

    if method in ("expval_pauli_term", "expval_pauli_terms", "expval_pauli_sum"):
        try:
            from . import counts_expval as _m
            counts = result.get("result") or result.get("counts") or {}
            expectations = result.get("expectations") or {}
            probabilities = result.get("probabilities") if isinstance(result.get("probabilities"), (list, tuple)) or hasattr(result.get("probabilities"), "shape") else None
            if probabilities is None and result.get("statevector") is not None:
                try:
                    import numpy as _np
                    psi = _np.asarray(result.get("statevector"))
                    probabilities = _np.abs(psi) ** 2
                except Exception:
                    probabilities = None
            num_qubits = (result.get("metadata", {}) or {}).get("num_qubits")
            payload = None
            if method == "expval_pauli_term":
                idxs = options.get("idxs") or options.get("indices") or []
                payload = _m.expval_pauli_term(counts, idxs)
            elif method == "expval_pauli_terms":
                terms = options.get("terms") or (result.get("result_meta", {}) or {}).get("group_items") or []
                payload = _m.expval_pauli_terms(counts, terms)
            elif method == "expval_pauli_sum":
                meta = (result.get("result_meta", {}) or {})
                items = options.get("items") or options.get("group_items") or meta.get("group_items") or []
                identity_const = float(options.get("identity_const", meta.get("identity_const", 0.0)))
                # If expectations exist (shots==0, statevector), prefer analytic aggregation
                payload = _m.expval_pauli_sum(counts, items, identity_const=identity_const, expectations=expectations if expectations else None, probabilities=probabilities, num_qubits=num_qubits)
            post["result"] = payload
        except Exception:
            pass
        return post

    # 未识别的方法：返回占位信息，便于上层识别
    post["result"] = {"reason": f"unsupported method: {method}"}
    return post


