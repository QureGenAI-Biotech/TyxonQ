from __future__ import annotations

"""Converters for external MPO formats into TyxonQ-native representations.

This module intentionally avoids importing heavy optional dependencies
(`tensornetwork`, `quimb`). It operates on duck-typed inputs exposing only the
minimal attributes we need, and contracts MPOs using NumPy.

Supported inputs (duck-typed):
- Tensornetwork-like: object with `.tensors` being a list of 4D arrays with
  shape (Dl, d_out, d_in, Dr). If elements are wrapper objects, they should
  provide `.tensor` or be array-like.
- Quimb-like: object with `.tensors`, each having `.data` as a 4D array
  (Dl, d_out, d_in, Dr). The `.inds` attribute is not required if tensors are
  already ordered as (left-bond, out, in, right-bond).

Outputs:
- `contract_mpo_to_matrix` returns a dense NumPy matrix of shape (2^n, 2^n).
"""

from typing import Any, Iterable, List
import numpy as np


def _to_array4(t: Any) -> np.ndarray:
    # Accept raw ndarray, object with .tensor, or object with .data
    if isinstance(t, np.ndarray):
        arr = t
    elif hasattr(t, "tensor"):
        arr = getattr(t, "tensor")
    elif hasattr(t, "data"):
        arr = getattr(t, "data")
    else:
        arr = np.asarray(t)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D MPO tensor, got shape {arr.shape}")
    return arr


def contract_mpo_to_matrix(mpo_tensors: Iterable[np.ndarray]) -> np.ndarray:
    """Contract a chain of 4D MPO tensors into a dense matrix.

    Each site tensor must be shaped (Dl, d_out, d_in, Dr). Physical dimensions
    d_out and d_in are assumed to be 2 for qubit systems.
    """
    tensors: List[np.ndarray] = [
        _to_array4(t).astype(np.complex128, copy=False) for t in mpo_tensors
    ]
    n = len(tensors)
    if n == 0:
        return np.array([[1.0 + 0.0j]])

    # Begin with first tensor: (1, d_out, d_in, D1)
    Dl0, d_out, d_in, Dr0 = tensors[0].shape
    if Dl0 != 1:
        raise ValueError("Leftmost MPO tensor must have Dl=1")
    core = tensors[0]  # (1, do, di, D)

    # Iteratively contract bonds: core_{..., r} with next_{l, ...}
    for k in range(1, n):
        Tk = tensors[k]  # (Dk, do, di, Dk+1)
        # Contract right bond of core with left bond of Tk
        core = np.einsum("...r, rabc -> ...abc", core, Tk)

    # After stacking all sites, core has shape (1, do1, di1, ..., doN, diN, 1)
    # Extract inner physical legs, reorder to (do1, ..., doN, di1, ..., diN)
    if core.ndim < 3:
        inner = core
    else:
        inner = core[(0,)*1 + (slice(None),)*(core.ndim-2) + (0,)]
    odims = list(range(0, inner.ndim, 2))
    idims = list(range(1, inner.ndim, 2))
    inner = np.transpose(inner, axes=odims + idims)
    # reshape to matrix
    do_total = 2 ** n
    di_total = 2 ** n
    mat = inner.reshape(do_total, di_total)
    return mat


def from_tensornetwork_mpo(mpo_obj: Any) -> np.ndarray:
    """Contract a Tensornetwork-like MPO object to a dense matrix.

    The object must expose `.tensors`, each 4D array-like with (Dl, d_out, d_in, Dr).
    """
    return contract_mpo_to_matrix(getattr(mpo_obj, "tensors"))


def from_quimb_mpo(qb_mpo: Any) -> np.ndarray:
    """Contract a Quimb-like MPO object to a dense matrix.

    The object must expose `.tensors`, each item with attribute `.data` being 4D array
    with (Dl, d_out, d_in, Dr). The index labels are not used here.
    """
    tensors = [getattr(t, "data") for t in getattr(qb_mpo, "tensors")]
    return contract_mpo_to_matrix(tensors)


