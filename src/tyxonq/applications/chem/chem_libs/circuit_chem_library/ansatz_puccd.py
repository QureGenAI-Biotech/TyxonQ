from __future__ import annotations

from typing import List, Tuple

import numpy as np


def generate_puccd_ex_ops(no: int, nv: int, t2_spatial: np.ndarray | None = None) -> Tuple[List[Tuple[int, ...]], List[int], List[float]]:
    """Generate paired excitations and initial guesses for pUCCD.

    Notes
    -----
    - Current design intentionally keeps the legacy "paired-occupancy two-body"
      excitation form `(no + a, i)` for stability and compatibility with
      existing tests (H4 passes across numeric engines). This matches TCC's
      practical usage where pUCCD parameters are organized per occupied index i
      with virtual index a in descending order.
    - We evaluated switching to a four-body paired-double representation to
      mirror UCC's ex-ops shape. That change introduced ordering/mapping drift
      and broke the previously passing H4 case until all mappings and numeric
      paths were realigned. Since there's no immediate functional gain, we keep
      the two-body paired form here and align tests (random-integral branch)
      using PUCCD's own civector as the RDM reference.

    Test history
    ------------
    - Before: H4 passed with the two-body paired form. Random-integral energy
      mismatched when directly compared to UCC's four-body list due to different
      excitation manifolds.
    - After: We adopted "Scheme A" in tests: for random-integral, UCC either
      consumes the same paired two-body ex-ops or uses PUCCD's parameter space
      directly; RDM checks use PUCCD's civector as reference via PySCF
      direct_spin1, achieving consistent alignment without changing this file.

    TODO
    ----
    - If we later need full four-body paired-double parity with UCC, introduce
      an optional switch here (default stays two-body). That refactor must also
      update civector/statevector evolution order and tests together to avoid
      regressions.
    """
    ex_ops: List[Tuple[int, ...]] = []
    init_guess: List[float] = []
    for i in range(no):
        for a in range(nv - 1, -1, -1):
            ex_ops.append((no + a, i))
            init_guess.append(t2_spatial[2 * i, 2 * i + 1, 2 * a, 2 * a + 1])
    param_ids = list(range(len(ex_ops)))
    return ex_ops, param_ids, init_guess


__all__ = ["generate_puccd_ex_ops"]


