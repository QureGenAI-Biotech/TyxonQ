from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

# Prefer numeric backend operations; use NumPy only when strictly necessary (e.g., SciPy interop)
from ...numerics.api import get_backend
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import pinv as scipy_pinv

numeric_backend = get_backend(None)
nb = numeric_backend


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
        self.single_qubit_cals: Dict[int, Any] = {}

    def set_single_qubit_cals(self, cals: Dict[int, Any]) -> None:
        """Set per-qubit 2x2 calibration matrices.

        The matrix maps true probabilities to measured probabilities.
        """

        for q, m in cals.items():
            arr = nb.asarray(m)
            if arr.shape != (2, 2):
                raise ValueError(f"Calibration for qubit {q} must be 2x2")
            self.single_qubit_cals[q] = arr

    def _infer_qubits_from_counts(self, counts: Dict[str, int]) -> Sequence[int]:
        n = len(next(iter(counts.keys())))
        return list(range(n))

    def _kron_cal_matrix(self, qubits: Sequence[int]) -> Any:
        if not qubits:
            return nb.eye(1)
        mats = []
        for q in qubits:
            if q not in self.single_qubit_cals:
                raise ValueError(f"Missing calibration for qubit {q}")
            mats.append(self.single_qubit_cals[q])
        full = mats[0]
        for m in mats[1:]:
            full = nb.kron(full, m)
        return full

    @staticmethod
    def _count2vec(counts: Dict[str, int]) -> Any:
        n = len(next(iter(counts.keys())))
        size = 2**n
        vec_np = np.zeros((size,), dtype=float)
        for bitstr, c in counts.items():
            idx = int(bitstr, 2)
            vec_np[idx] = float(c)
        shots = float(np.sum(vec_np))
        if shots <= 0:
            shots = 1.0
        vec_np = vec_np / shots
        return nb.asarray(vec_np)

    @staticmethod
    def _vec2count(prob: Any, shots: int) -> Dict[str, int]:
        arr_np = np.asarray(nb.to_numpy(prob), dtype=float)
        arr_np = np.clip(arr_np, 0.0, 1.0)
        s = float(np.sum(arr_np))
        if s <= 1e-12:
            s = 1.0
        arr_np = arr_np / s
        vec_np = np.rint(arr_np * float(shots)).astype(int)
        size = int(vec_np.size)
        n = int(np.log2(size)) if size > 0 else 0
        counts: Dict[str, int] = {}
        nz_idx = np.nonzero(vec_np)[0]
        for idx in nz_idx:
            bitstr = format(int(idx), f"0{n}b")
            counts[bitstr] = int(vec_np[int(idx)])
        return counts

    def mitigate_probability(self, prob_measured: Any, qubits: Sequence[int], method: str = "inverse") -> Any:
        A = self._kron_cal_matrix(qubits)
        if method == "inverse":
            # Use SciPy pinv for stability; convert via backend
            A_np = nb.to_numpy(A)
            pm_np = nb.to_numpy(prob_measured)
            X = scipy_pinv(A_np)
            pt_np = np.asarray(X @ pm_np, dtype=float)
            pt_np = np.clip(pt_np, 0.0, 1.0)
            s = float(np.sum(pt_np))
            if s <= 1e-12:
                s = 1.0
            pt_np = pt_np / s
            return [float(x) for x in pt_np]

        # constrained least squares on simplex
        A_np = nb.to_numpy(A)
        pm_np = nb.to_numpy(prob_measured)

        def fun(x: Any) -> Any:
            y = A_np @ x
            diff = y - pm_np
            return float(np.dot(diff, diff))

        n = len(pm_np)
        x0 = [1.0 / n] * n
        cons = {"type": "eq", "fun": lambda x: 1.0 - float(sum(x))}
        bnds = tuple((0.0, 1.0) for _ in range(n))
        res = minimize(fun, x0, method="SLSQP", constraints=cons, bounds=bnds, tol=1e-6)
        x_np = np.asarray(res.x, dtype=float)
        x_np = np.clip(x_np, 0.0, 1.0)
        s = float(np.sum(x_np))
        if s <= 1e-12:
            s = 1.0
        x_np = x_np / s
        return [float(v) for v in x_np]

    def apply_readout_mitigation(self, raw_count: Dict[str, int], method: str = "inverse", qubits: Optional[Sequence[int]] = None, shots: Optional[int] = None) -> Dict[str, int]:
        if qubits is None:
            qubits = self._infer_qubits_from_counts(raw_count)
        prob_measured = self._count2vec(raw_count)
        shots0 = int(sum(raw_count.values())) if shots is None else int(shots)
        prob_true = self.mitigate_probability(prob_measured, qubits, method=method)
        return self._vec2count(prob_true, shots0)


__all__ = ["ReadoutMit"]


