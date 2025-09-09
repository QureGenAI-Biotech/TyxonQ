from __future__ import annotations

from typing import List, Tuple

import numpy as np
from pyscf.cc.addons import spatial2spin  # type: ignore


def generate_uccsd_ex1_ops(
    no: int,
    nv: int,
    t1: np.ndarray | None = None,
    *,
    mode: str = "fermion",
) -> Tuple[List[Tuple[int, ...]], List[int], List[float]]:
    """
    Generate UCCSD single-excitation operators, parameter ids and initial guesses.

    The mapping matches the legacy static implementation so that downstream behavior
    (e.g., parameter screening and ordering) remains consistent.
    """
    if t1 is None:
        t1 = np.zeros((no, nv), dtype=float)
    # If spin-orbital shape is passed, reduce to spatial for init guess
    if t1.ndim == 2 and t1.shape == (2 * no, 2 * nv):
        # Use alpha block for initial guess
        t1 = t1[0::2, 0::2]
    else:
        assert t1.shape == (no, nv)

    ex1_ops: List[Tuple[int, ...]] = []
    ex1_param_ids: List[int] = [-1]
    ex1_init_guess: List[float] = []

    for i in range(no):
        for a in range(nv):
            # alpha -> alpha
            ex_op_a = (2 * no + nv + a, no + nv + i)
            # beta  -> beta
            ex_op_b = (no + a, i)
            ex1_ops.extend([ex_op_a, ex_op_b])
            ex1_param_ids.extend([ex1_param_ids[-1] + 1] * 2)
            ex1_init_guess.append(float(t1[i, a]))

    return ex1_ops, ex1_param_ids[1:], ex1_init_guess


def generate_uccsd_ex2_ops(
    no: int,
    nv: int,
    t2: np.ndarray | None = None,
    *,
    mode: str = "fermion",
) -> Tuple[List[Tuple[int, ...]], List[int], List[float]]:
    """
    Generate UCCSD double-excitation operators (aa, bb, ab) with parameter ids and initial guesses.

    The returned operator index convention matches the legacy static implementation.
    Initial guesses are derived from the provided spatial t2 amplitudes when available.
    """
    # Prepare t2 in spin-orbital shape (2no,2no,2nv,2nv) to match TCC indexing
    if t2 is None:
        t2_spin = np.zeros((2 * no, 2 * no, 2 * nv, 2 * nv), dtype=float)
    else:
        if t2.shape == (no, no, nv, nv):
            t2_spin = spatial2spin(t2)
        else:
            assert t2.shape == (2 * no, 2 * no, 2 * nv, 2 * nv)
            t2_spin = np.asarray(t2, dtype=float)

    def alpha_o(_i: int) -> int:
        return no + nv + _i

    def alpha_v(_i: int) -> int:
        return 2 * no + nv + _i

    def beta_o(_i: int) -> int:
        return _i

    def beta_v(_i: int) -> int:
        return no + _i

    ex2_ops: List[Tuple[int, ...]] = []
    ex2_param_ids: List[int] = [-1]
    ex2_init_guess: List[float] = []

    # 2 alphas / 2 betas (AA/BB)
    for i in range(no):
        for j in range(i):
            for a in range(nv):
                for b in range(a):
                    ex_op_aa = (alpha_v(b), alpha_v(a), alpha_o(i), alpha_o(j))
                    ex_op_bb = (beta_v(b), beta_v(a), beta_o(i), beta_o(j))
                    ex2_ops.extend([ex_op_aa, ex_op_bb])
                    ex2_param_ids.extend([ex2_param_ids[-1] + 1] * 2)
                    ex2_init_guess.append(float(t2_spin[2 * i, 2 * j, 2 * a, 2 * b]))

    # alpha-beta (AB)
    for i in range(no):
        for j in range(i + 1):
            for a in range(nv):
                for b in range(a + 1):
                    if i == j and a == b:
                        # paired
                        ex_op_ab = (beta_v(a), alpha_v(a), alpha_o(i), beta_o(i))
                        ex2_ops.append(ex_op_ab)
                        ex2_param_ids.append(ex2_param_ids[-1] + 1)
                        ex2_init_guess.append(float(t2_spin[2 * i, 2 * i + 1, 2 * a, 2 * a + 1]))
                        continue
                    # simple reflection first
                    ex_op_ab1 = (beta_v(b), alpha_v(a), alpha_o(i), beta_o(j))
                    ex_op_ab2 = (alpha_v(b), beta_v(a), beta_o(i), alpha_o(j))
                    ex2_ops.extend([ex_op_ab1, ex_op_ab2])
                    ex2_param_ids.extend([ex2_param_ids[-1] + 1] * 2)
                    ex2_init_guess.append(float(t2_spin[2 * i, 2 * j + 1, 2 * a, 2 * b + 1]))
                    if (i != j) and (a != b):
                        # exchange alpha and beta after
                        ex_op_ab3 = (beta_v(a), alpha_v(b), alpha_o(i), beta_o(j))
                        ex_op_ab4 = (alpha_v(a), beta_v(b), beta_o(i), alpha_o(j))
                        ex2_ops.extend([ex_op_ab3, ex_op_ab4])
                        ex2_param_ids.extend([ex2_param_ids[-1] + 1] * 2)
                        ex2_init_guess.append(float(t2_spin[2 * i, 2 * j + 1, 2 * b, 2 * a + 1]))

    return ex2_ops, ex2_param_ids[1:], ex2_init_guess


__all__ = [
    "generate_uccsd_ex1_ops",
    "generate_uccsd_ex2_ops",
]


