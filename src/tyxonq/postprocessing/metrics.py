from __future__ import annotations

"""Postprocessing metrics for measurement data.

This module provides utilities for processing measurement count data:
- Normalization and divergence metrics
- Observable expectation values from bitstring counts

Quantum information theory functions have been moved to:
    tyxonq.libs.quantum_library.kernels.quantum_info
"""

from typing import Dict, Optional, Sequence

import numpy as np

ct = Dict[str, int]


def normalized_count(count: ct) -> Dict[str, float]:
    """Normalize count dictionary to probability distribution.
    
    Args:
        count: Dictionary mapping bitstrings to counts
        
    Returns:
        Dictionary mapping bitstrings to probabilities
    """
    shots = max(1, sum(count.values()))
    return {k: v / shots for k, v in count.items()}


def kl_divergence(c1: ct, c2: ct, *, eps: float = 1e-12) -> float:
    """Calculate Kullback-Leibler divergence D_KL(c1 || c2).
    
    Args:
        c1: First count distribution
        c2: Second count distribution
        eps: Regularization to avoid log(0)
        
    Returns:
        KL divergence in nats
    """
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

    Args:
        count: Dictionary mapping bitstrings to counts
        z: Qubit indices with Z measurement (for Pauli-Z expectation)
        diagonal_op: Diagonal operator per qubit (length-2 arrays for I/Z mix)
        
    Returns:
        Expectation value
        
    Note:
        Either `z` or `diagonal_op` must be provided.
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


__all__ = [
    "normalized_count",
    "kl_divergence",
    "expectation",
]


