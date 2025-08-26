from __future__ import annotations

from typing import Any

import numpy as np

"""Matrix Product State (MPS) minimal utilities for compressed-state simulation.

This module provides a small, readable subset of MPS operations used by the
compressed-state simulator path:

- MPSState: container of site tensors with shapes (Dl, 2, Dr)
- init_product_state: build |0..0> (or |1..1>) as an MPS with unit bonds
- apply_1q: in-place application of 1-qubit unitary on a site
- apply_2q_nn: in-place application of a nearest-neighbor 2-qubit unitary with
  SVD recompression (optional max bond dimension)
- to_statevector: reconstruct the full statevector (for small systems/testing)

Design notes
------------
- This implementation focuses on clarity and a minimal feature set that our
  tests exercise. It uses NumPy and does not depend on the numerics backends.
  Device engines can wrap and call these functions while selecting backends for
  heavy kernels as needed.
- All tensors use conventional qubit ordering; multi-qubit unitaries are
  expected in shape (2, 2, 2, 2) with index order (s0', s1', s0, s1).
- Truncation policy is basic: hard cut to ``max_bond`` if provided. Future
  extensions may include error-based truncation and backend-accelerated SVDs.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class MPSState:
    """Container of site tensors for an MPS chain.

    Each tensor has shape (Dl, 2, Dr) where Dl and Dr are the left and right
    bond dimensions respectively.
    """

    tensors: List[np.ndarray]


def init_product_state(num_qubits: int, bit: int = 0) -> MPSState:
    """Initialize a product state as an MPS.

    Parameters
    ----------
    num_qubits:
        Number of qubits in the chain.
    bit:
        If 0, initialize |0..0>; if 1, initialize |1..1>.
    """

    v = np.array([1.0, 0.0], dtype=np.complex128) if bit == 0 else np.array([0.0, 1.0], dtype=np.complex128)
    tensors = [v.reshape(1, 2, 1).copy() for _ in range(num_qubits)]
    return MPSState(tensors)


def apply_1q(mps: MPSState, U: np.ndarray, site: int) -> None:
    """Apply a single-qubit unitary on the specified site in-place."""

    A = mps.tensors[site]
    # U_{a s} * A_{i s j} -> A'_{i a j}
    A2 = np.einsum("as, isj -> iaj", U, A)
    mps.tensors[site] = A2


def _truncate_svd(
    Um: np.ndarray,
    Sm: np.ndarray,
    Vh: np.ndarray,
    max_bond: Optional[int],
    svd_cutoff: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Start with all singular values
    r = Sm.shape[0]
    # Apply cutoff by absolute threshold on singular values
    if svd_cutoff is not None:
        keep = (Sm >= float(svd_cutoff))
        # Ensure at least rank-1 to avoid collapsing entirely
        if not np.any(keep):
            keep = np.zeros_like(Sm, dtype=bool)
            keep[0] = True
        Um = Um[:, keep]
        Sm = Sm[keep]
        Vh = Vh[keep, :]
        r = Sm.shape[0]
    # Apply hard cap
    if max_bond is not None and max_bond < r:
        r = max_bond
        Um = Um[:, :r]
        Sm = Sm[:r]
        Vh = Vh[:r, :]
    return Um, Sm, Vh


def apply_2q_nn(
    mps: MPSState,
    U4: np.ndarray,
    left_site: int,
    max_bond: Optional[int] = None,
    *,
    svd_cutoff: Optional[float] = None,
) -> None:
    """Apply a nearest-neighbor 2-qubit unitary in-place with SVD recompression.

    Parameters
    ----------
    mps:
        The target MPS state.
    U4:
        Unitary with shape (2, 2, 2, 2) in order (s0', s1', s0, s1).
    left_site:
        Apply between sites ``left_site`` and ``left_site + 1``.
    max_bond:
        Optional hard cap on the new bond dimension after applying the gate.
    """

    i = left_site
    A = mps.tensors[i]
    B = mps.tensors[i + 1]
    # Merge two site tensors into theta_{Dl, s0, s1, Dr}
    theta = np.einsum("isj, jtk -> istk", A, B)
    # Apply gate: U_{a b s t} * theta_{i s t k} -> theta'_{i a b k}
    theta2 = np.einsum("abst, istk -> iabk", U4.reshape(2, 2, 2, 2), theta)
    Dl, _, _, Dr = theta2.shape
    # Reshape to matrix for SVD: (Dl*2) x (2*Dr)
    mat = theta2.reshape(Dl * 2, 2 * Dr)
    Um, Sm, Vh = np.linalg.svd(mat, full_matrices=False)
    Um, Sm, Vh = _truncate_svd(Um, Sm, Vh, max_bond, svd_cutoff)
    r = Sm.shape[0]
    # Left tensor: reshape U to (Dl, 2, r)
    A_new = Um.reshape(Dl, 2, r)
    # Right tensor: S @ Vh -> (r, 2*Dr) -> (r, 2, Dr)
    SV = (Sm[:, None] * Vh)
    B_new = SV.reshape(r, 2, Dr)
    mps.tensors[i] = A_new
    mps.tensors[i + 1] = B_new


def _gate_swap_4x4() -> np.ndarray:
    # SWAP two qubits
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )


def apply_swap_nn(
    mps: MPSState,
    left_site: int,
    *,
    max_bond: Optional[int] = None,
    svd_cutoff: Optional[float] = None,
) -> None:
    U = _gate_swap_4x4().reshape(2, 2, 2, 2)
    apply_2q_nn(mps, U, left_site, max_bond, svd_cutoff=svd_cutoff)


def apply_2q(
    mps: MPSState,
    U4: np.ndarray,
    q0: int,
    q1: int,
    *,
    max_bond: Optional[int] = None,
    svd_cutoff: Optional[float] = None,
) -> None:
    """Apply a general 2-qubit unitary by routing with SWAPs if needed."""
    if q0 == q1:
        return
    if abs(q0 - q1) == 1:
        left = min(q0, q1)
        apply_2q_nn(mps, U4, left, max_bond, svd_cutoff=svd_cutoff)
        return
    # Route: move q0 next to q1 by successive swaps, apply, then swap back
    a, b = (q0, q1) if q0 < q1 else (q1, q0)
    # Move a rightwards until at b-1
    for s in range(a, b - 1):
        apply_swap_nn(mps, s, max_bond=max_bond, svd_cutoff=svd_cutoff)
    # Apply gate at (b-1, b)
    apply_2q_nn(mps, U4, b - 1, max_bond, svd_cutoff=svd_cutoff)
    # Move a back leftwards to original position
    for s in range(b - 2, a - 1, -1):
        apply_swap_nn(mps, s, max_bond=max_bond, svd_cutoff=svd_cutoff)


def to_statevector(mps: MPSState) -> np.ndarray:
    """Reconstruct the full statevector from an MPS.

    Intended for small systems and testing; complexity is exponential in the
    number of qubits.
    """

    T = mps.tensors[0]
    # T has shape (1,2,D1)
    cur = T
    for j in range(1, len(mps.tensors)):
        B = mps.tensors[j]
        # cur_{L, s, M} * B_{M, t, R} -> cur_{L, s, t, R}
        cur = np.einsum("lsm, mtr -> lstr", cur, B)
        # merge the two physical indices into one leading physical axis (2^k) while keeping bonds on ends
        L = cur.shape[0]
        R = cur.shape[-1]
        cur = cur.reshape(L, -1, R)
    # Now cur shape (1, 2^n, 1)
    psi = cur.reshape(-1)
    return psi


def bond_dims(mps: MPSState) -> List[Tuple[int, int]]:
    """Return list of (Dl, Dr) bond dimensions for each site tensor."""
    dims: List[Tuple[int, int]] = []
    for T in mps.tensors:
        Dl, _, Dr = T.shape
        dims.append((Dl, Dr))
    return dims

