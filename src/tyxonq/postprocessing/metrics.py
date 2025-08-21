from __future__ import annotations

"""Postprocessing metrics: normalization, divergences, expectations.

This module provides lightweight, dependency-free utilities for common
postprocessing metrics used across devices and simulators.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np

ct = Dict[str, int]


def normalized_count(count: ct) -> Dict[str, float]:
    shots = max(1, sum(count.values()))
    return {k: v / shots for k, v in count.items()}


def kl_divergence(c1: ct, c2: ct, *, eps: float = 1e-12) -> float:
    p = normalized_count(c1)
    q = normalized_count(c2)
    kl = 0.0
    for k, v in p.items():
        qk = q.get(k, eps)
        kl += float(v) * (float(np.log(max(v, eps))) - float(np.log(max(qk, eps))))
    return float(kl)


def expectation(
    count: ct, z: Optional[Sequence[int]] = None, diagonal_op: Optional[Sequence[Sequence[float]]] = None
) -> float:
    """Compute diagonal observable expectation from bitstring counts.

    If `z` is provided, it denotes wires with Z measurement; otherwise supply a
    diagonal operator per qubit (length-2 arrays) for I/Z mix.
    """

    if z is None and diagonal_op is None:
        raise ValueError("One of `z` and `diagonal_op` must be set")
    n = len(next(iter(count.keys())))
    if z is not None:
        diagonal_op = [[1.0, -1.0] if i in z else [1.0, 1.0] for i in range(n)]
    assert diagonal_op is not None
    total = 0.0
    shots = 0
    for bitstr, v in count.items():
        val = 1.0
        for i in range(n):
            val *= float(diagonal_op[i][int(bitstr[i])])
        total += val * float(v)
        shots += int(v)
    return float(total / max(1, shots))


__all__ = ["normalized_count", "kl_divergence", "expectation"]


