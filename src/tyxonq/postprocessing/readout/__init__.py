from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize


class ReadoutMit:
    """Readout error mitigation.

    This refactored implementation fits the new architecture without relying on
    legacy modules. It supports local single-qubit calibration matrices and
    applies either matrix inversion or constrained least squares to correct
    measured counts.

    Notes:
    - Calibration is provided via `set_single_qubit_cals`.
    - For multi-qubit systems, the full calibration matrix is the Kronecker
      product of per-qubit matrices in ascending wire order.
    """

    def __init__(self, execute: Optional[Callable[..., List[Dict[str, int]]]] = None) -> None:
        self.execute_fun = execute
        self.single_qubit_cals: Dict[int, np.ndarray] = {}

    def set_single_qubit_cals(self, cals: Dict[int, np.ndarray]) -> None:
        """Set per-qubit 2x2 calibration matrices.

        The matrix maps true probabilities to measured probabilities.
        """

        for q, m in cals.items():
            arr = np.asarray(m, dtype=float)
            if arr.shape != (2, 2):
                raise ValueError(f"Calibration for qubit {q} must be 2x2")
            self.single_qubit_cals[q] = arr

    def _infer_qubits_from_counts(self, counts: Dict[str, int]) -> Sequence[int]:
        n = len(next(iter(counts.keys())))
        return list(range(n))

    def _kron_cal_matrix(self, qubits: Sequence[int]) -> np.ndarray:
        if not qubits:
            return np.eye(1)
        mats = []
        for q in qubits:
            if q not in self.single_qubit_cals:
                raise ValueError(f"Missing calibration for qubit {q}")
            mats.append(self.single_qubit_cals[q])
        full = mats[0]
        for m in mats[1:]:
            full = np.kron(full, m)
        return full

    @staticmethod
    def _count2vec(counts: Dict[str, int]) -> np.ndarray:
        n = len(next(iter(counts.keys())))
        size = 2**n
        vec = np.zeros(size, dtype=float)
        for bitstr, c in counts.items():
            idx = int(bitstr, 2)
            vec[idx] = float(c)
        shots = max(1.0, vec.sum())
        return vec / shots

    @staticmethod
    def _vec2count(prob: np.ndarray, shots: int) -> Dict[str, int]:
        prob = np.clip(prob, 0.0, 1.0)
        prob = prob / max(1e-12, prob.sum())
        vec = np.round(prob * shots).astype(int)
        n = int(np.log2(len(vec)))
        counts: Dict[str, int] = {}
        for idx, v in enumerate(vec):
            if v <= 0:
                continue
            bitstr = format(idx, f"0{n}b")
            counts[bitstr] = int(v)
        return counts

    def mitigate_probability(self, prob_measured: np.ndarray, qubits: Sequence[int], method: str = "inverse") -> np.ndarray:
        A = self._kron_cal_matrix(qubits)
        if method == "inverse":
            X = np.linalg.pinv(A)
            prob_true = X @ prob_measured
            prob_true = np.clip(prob_true, 0.0, 1.0)
            return prob_true / max(1e-12, prob_true.sum())

        # constrained least squares on simplex
        def fun(x: Any) -> Any:
            return float(np.sum((prob_measured - A @ x) ** 2))

        n = len(prob_measured)
        x0 = np.ones(n, dtype=float) / n
        cons = {"type": "eq", "fun": lambda x: 1.0 - float(np.sum(x))}
        bnds = tuple((0.0, 1.0) for _ in range(n))
        res = minimize(fun, x0, method="SLSQP", constraints=cons, bounds=bnds, tol=1e-6)
        x = np.clip(res.x, 0.0, 1.0)
        return x / max(1e-12, x.sum())

    def apply_readout_mitigation(self, raw_count: Dict[str, int], method: str = "inverse", qubits: Optional[Sequence[int]] = None, shots: Optional[int] = None) -> Dict[str, int]:
        if qubits is None:
            qubits = self._infer_qubits_from_counts(raw_count)
        prob_measured = self._count2vec(raw_count)
        shots0 = int(sum(raw_count.values())) if shots is None else int(shots)
        prob_true = self.mitigate_probability(prob_measured, qubits, method=method)
        return self._vec2count(prob_true, shots0)


__all__ = ["ReadoutMit"]


