from __future__ import annotations

from typing import List, Tuple

import numpy as np


def generate_kupccgsd_ex1_ops(no: int, nv: int) -> Tuple[List[Tuple[int, ...]], List[int], np.ndarray]:
    """Generalized singles for KUpCCGSD; parameter sharing follows static implementation."""
    ex1_ops: List[Tuple[int, ...]] = []
    ex1_param_id: List[int] = [-1]
    for a in range(no + nv):
        for i in range(a):
            # alpha->alpha (upper block)
            ex_op_a = (no + nv + a, no + nv + i)
            # beta->beta (lower block)
            ex_op_b = (a, i)
            ex1_ops.extend([ex_op_a, ex_op_b])
            ex1_param_id.extend([ex1_param_id[-1] + 1] * 2)
    # Number of parameters equals max index + 1 (indices start from 0)
    ex1_init_guess = np.zeros(max(ex1_param_id) + 1, dtype=float) if len(ex1_param_id) > 1 else np.zeros(0)
    return ex1_ops, ex1_param_id[1:], ex1_init_guess


def generate_kupccgsd_ex2_ops(no: int, nv: int) -> Tuple[List[Tuple[int, ...]], List[int], np.ndarray]:
    """Generalized paired doubles for KUpCCGSD."""
    ex2_ops: List[Tuple[int, ...]] = []
    ex2_param_id: List[int] = [-1]
    for a in range(no + nv):
        for i in range(a):
            # paired (i,a)
            ex_op_ab = (a, no + nv + a, no + nv + i, i)
            ex2_ops.append(ex_op_ab)
            ex2_param_id.append(ex2_param_id[-1] + 1)
    ex2_init_guess = np.zeros(max(ex2_param_id) + 1, dtype=float) if len(ex2_param_id) > 1 else np.zeros(0)
    return ex2_ops, ex2_param_id[1:], ex2_init_guess


def generate_kupccgsd_ex_ops(no: int, nv: int, k: int) -> Tuple[List[Tuple[int, ...]], List[int], np.ndarray]:
    ex1_ops, ex1_param_id, _ = generate_kupccgsd_ex1_ops(no, nv)
    ex2_ops, ex2_param_id, _ = generate_kupccgsd_ex2_ops(no, nv)

    ex_ops: List[Tuple[int, ...]] = []
    param_ids: List[int] = [-1]
    for _ in range(k):
        ex_ops.extend(ex2_ops + ex1_ops)
        param_ids.extend([i + param_ids[-1] + 1 for i in ex2_param_id])
        param_ids.extend([i + param_ids[-1] + 1 for i in ex1_param_id])
    init_guess = np.random.rand(max(param_ids) + 1) - 0.5 if len(param_ids) > 1 else np.zeros(0)
    return ex_ops, param_ids[1:], init_guess


__all__ = [
    "generate_kupccgsd_ex1_ops",
    "generate_kupccgsd_ex2_ops",
    "generate_kupccgsd_ex_ops",
]


