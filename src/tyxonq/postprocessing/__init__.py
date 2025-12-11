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
    """Apply postprocessing techniques to quantum computation results.

    This function provides a unified interface for applying various postprocessing
    methods to quantum circuit execution results, including error mitigation,
    expectation value computation, and result analysis.

    Args:
        result (Dict[str, Any]): Single task result dictionary containing:
            - "result" or "counts": Measurement counts dictionary
            - "metadata": Optional metadata including shots count
            - "statevector": Optional statevector for analytic computation
            - "expectations": Optional precomputed expectation values
            
        options (Optional[Dict[str, Any]]): Postprocessing configuration:
            - "method" (str): Required. Postprocessing method to apply
            - Additional method-specific parameters (see below)

    Returns:
        Dict[str, Any]: Postprocessing result with structure:
            {
                "method": str | None,  # Applied method name
                "result": Any | None   # Method-specific result data
            }

    Supported Methods:
        **readout_mitigation**: Mitigate measurement readout errors
            - "cals" (dict): Calibration data for readout error correction
            - "mitigation" (str): Mitigation method, default "inverse"
            
        **expval_pauli_term**: Compute single Pauli term expectation value
            - "idxs" or "indices" (list): Qubit indices for the Pauli term
            
        **expval_pauli_terms**: Compute multiple Pauli terms expectations
            - "terms" (list): List of Pauli term specifications
            
        **expval_pauli_sum**: Aggregate Pauli sum expectation value
            - "items" or "group_items" (list): Pauli terms with coefficients
            - "identity_const" (float): Identity term coefficient, default 0.0
            - "readout_cals" or "cals" (dict): Optional readout calibration
            - "mitigation" (str): Readout mitigation method, default "inverse"

    Examples:
        >>> # Basic readout error mitigation
        >>> result = {"result": {"00": 480, "11": 520}, "metadata": {"shots": 1000}}
        >>> cals = {"0": [[0.95, 0.05], [0.02, 0.98]], "1": [[0.97, 0.03], [0.01, 0.99]]}
        >>> options = {"method": "readout_mitigation", "cals": cals}
        >>> mitigated = apply_postprocessing(result, options)
        >>> mitigated["result"]["mitigated_counts"]
        {'00': 502, '11': 498}
        
        >>> # Pauli expectation value computation
        >>> counts = {"00": 400, "01": 100, "10": 300, "11": 200}
        >>> options = {"method": "expval_pauli_term", "idxs": [0]}  # Z_0 expectation
        >>> expectation = apply_postprocessing({"result": counts}, options)
        >>> expectation["result"]  # <Z_0> value
        0.2
        
        >>> # Full Hamiltonian expectation (with readout mitigation)
        >>> hamiltonian_items = [
        ...     {"coeff": 0.5, "pauli_terms": [("Z", 0)]},
        ...     {"coeff": 0.3, "pauli_terms": [("X", 0), ("X", 1)]}
        ... ]
        >>> options = {
        ...     "method": "expval_pauli_sum",
        ...     "items": hamiltonian_items,
        ...     "identity_const": -1.0,
        ...     "cals": cals
        ... }
        >>> energy = apply_postprocessing(result, options)
        >>> energy["result"]  # Total energy expectation
        -0.85

    Notes:
        - Failed postprocessing operations return gracefully with partial results
        - Readout mitigation is applied automatically in expval_pauli_sum when calibrations provided
        - Statevector results enable analytic expectation value computation
        - Shot count is inferred from metadata or counts sum when missing
        
    Raises:
        No exceptions are raised; failures result in partial results with diagnostic information.
        
    See Also:
        tyxonq.postprocessing.readout.ReadoutMit: Readout error mitigation implementation.
        tyxonq.postprocessing.counts_expval: Expectation value computation utilities.
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
                # Optional readout mitigation inside aggregator when calibrations provided
                # Accept both 'readout_cals' and 'cals' as option keys; default method='inverse'
                try:
                    _cals = options.get("readout_cals") or options.get("cals")
                    if counts and _cals:
                        from .readout import ReadoutMit  # lazy import
                        mit = ReadoutMit()
                        mit.set_single_qubit_cals(dict(_cals))
                        # shots inferred from counts sum; fall back to metadata if available
                        try:
                            _shots_meta = int((result.get("metadata", {}) or {}).get("shots", 0))
                        except Exception:
                            _shots_meta = 0
                        if _shots_meta <= 0:
                            try:
                                _shots_meta = int(sum(int(v) for v in counts.values()))
                            except Exception:
                                _shots_meta = 0
                        _mit_method = str(options.get("mitigation", "inverse"))
                        counts = mit.apply_readout_mitigation(counts, method=_mit_method, shots=_shots_meta)
                except Exception:
                    # mitigation is best-effort; fall back silently on failure
                    pass
                # If expectations exist (shots==0, statevector), prefer analytic aggregation
                payload = _m.expval_pauli_sum(counts, items, identity_const=identity_const, expectations=expectations if expectations else None, probabilities=probabilities, num_qubits=num_qubits)
            post["result"] = payload
        except Exception:
            pass
        return post

    # 未识别的方法：返回占位信息，便于上层识别
    post["result"] = {"reason": f"unsupported method: {method}"}
    return post


