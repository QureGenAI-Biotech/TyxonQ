from __future__ import annotations

from typing import List, Tuple

import numpy as np


def generate_puccd_ex_ops(no: int, nv: int, t2_spatial: np.ndarray | None = None) -> Tuple[List[Tuple[int, ...]], List[int], List[float]]:
    """Generate paired excitations and initial guesses for pUCCD.

    This mirrors the legacy static implementation order to keep compatibility.
    """
    if t2_spatial is None:
        t2_spatial = np.zeros((no, no, nv, nv), dtype=float)

    ex_ops: List[Tuple[int, ...]] = []
    init_guess: List[float] = []
    for i in range(no):
        for a in range(nv - 1, -1, -1):
            ex_ops.append((no + a, i))
            init_guess.append(float(t2_spatial[i, i, a, a]))
    param_ids = list(range(len(ex_ops)))
    return ex_ops, param_ids, init_guess


__all__ = ["generate_puccd_ex_ops"]


