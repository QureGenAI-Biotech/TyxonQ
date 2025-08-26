from __future__ import annotations

"""Pauli helpers and dense Hamiltonian builders for small systems.

This module provides:
- ps2xyz / xyz2ps conversions between list-encoded Pauli strings and dict form
- pauli_string_to_matrix for a single Pauli term
- pauli_string_sum_dense to build a dense Hamiltonian from many Pauli terms

All implementations use NumPy only to avoid optional heavy deps.
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


def ps2xyz(ps: List[int]) -> Dict[str, List[int]]:
    """Convert a Pauli string list to xyz dict.

    ps[i] in {0,1,2,3} encodes I,X,Y,Z respectively.
    Returns dict with keys "x","y","z" mapping to index lists.
    """
    xyz: Dict[str, List[int]] = {"x": [], "y": [], "z": []}
    for i, v in enumerate(ps):
        if v == 1:
            xyz["x"].append(i)
        elif v == 2:
            xyz["y"].append(i)
        elif v == 3:
            xyz["z"].append(i)
    return xyz


def xyz2ps(xyz: Dict[str, List[int]], n: Optional[int] = None) -> List[int]:
    """Convert xyz dict back to a Pauli string list of length n.

    Missing qubits default to identity (0).
    """
    if n is None:
        all_idx = (xyz.get("x", []) + xyz.get("y", []) + xyz.get("z", [])) or [0]
        n = max(all_idx) + 1
    ps = [0] * n
    for i in xyz.get("x", []):
        ps[i] = 1
    for i in xyz.get("y", []):
        ps[i] = 2
    for i in xyz.get("z", []):
        ps[i] = 3
    return ps


def _pauli_matrix(code: int) -> np.ndarray:
    if code == 0:
        return np.eye(2, dtype=np.complex128)
    if code == 1:
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if code == 2:
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if code == 3:
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    raise ValueError(f"Invalid Pauli code: {code}")


def pauli_string_to_matrix(ps: Sequence[int]) -> np.ndarray:
    """Return the dense matrix for a single Pauli string on n qubits.

    ps[i] in {0,1,2,3} encodes I,X,Y,Z on qubit i (little-endian not required
    for matrix building; conventional left-to-right kron is used).
    """
    mats = [_pauli_matrix(code) for code in ps]
    out = np.array([[1.0 + 0.0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def pauli_string_sum_dense(
    ls: Sequence[Sequence[int]], weights: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Build a dense Hamiltonian matrix from a list of Pauli strings.

    Parameters
    ----------
    ls: list of Pauli strings, each entry ps is length-n with entries in {0,1,2,3}
    weights: real weights per term (defaults to 1.0)
    """
    if not ls:
        return np.array([[0.0 + 0.0j]])
    n = len(ls[0])
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)
    if weights is None:
        weights = [1.0] * len(ls)
    for ps, w in zip(ls, weights):
        H += float(w) * pauli_string_to_matrix(ps)
    return H


__all__ = [
    "ps2xyz",
    "xyz2ps",
    "pauli_string_to_matrix",
    "pauli_string_sum_dense",
]


def pauli_string_sum_coo(
    ls: Sequence[Sequence[int]], weights: Optional[Sequence[float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Build a sparse COO (rows, cols, values, shape) for a Pauli string sum.

    This dense-backed implementation is suitable for small n; for large n, a
    specialized bitwise implementation should be used.
    """
    if not ls:
        return (
            np.array([0], dtype=int),
            np.array([0], dtype=int),
            np.array([0.0 + 0.0j], dtype=np.complex128),
            (1, 1),
        )
    H = pauli_string_sum_dense(ls, weights)
    rows, cols = np.nonzero(H)
    vals = H[rows, cols]
    shape = H.shape
    return rows.astype(int), cols.astype(int), vals.astype(np.complex128), shape


def heisenberg_hamiltonian(
    num_qubits: int,
    edges: Sequence[Tuple[int, int]],
    *,
    hzz: float = 1.0,
    hxx: float = 1.0,
    hyy: float = 1.0,
    hz: float = 0.0,
    hx: float = 0.0,
    hy: float = 0.0,
) -> np.ndarray:
    """Build Heisenberg Hamiltonian (dense) from edge list and fields.

    Parameters
    ----------
    num_qubits: int
        Number of qubits.
    edges: list of (i,j)
        Undirected edges indicating coupled pairs.
    hzz,hxx,hyy: float
        Pair couplings for Z.Z, X.X, Y.Y.
    hz,hx,hy: float
        On-site fields for Z,X,Y per qubit (uniform).
    """
    terms: List[List[int]] = []
    weights: List[float] = []
    # pair terms
    for (a, b) in edges:
        if hzz != 0.0:
            ps = [0] * num_qubits
            ps[a] = 3; ps[b] = 3
            terms.append(ps); weights.append(hzz)
        if hxx != 0.0:
            ps = [0] * num_qubits
            ps[a] = 1; ps[b] = 1
            terms.append(ps); weights.append(hxx)
        if hyy != 0.0:
            ps = [0] * num_qubits
            ps[a] = 2; ps[b] = 2
            terms.append(ps); weights.append(hyy)
    # local fields
    for q in range(num_qubits):
        if hz != 0.0:
            ps = [0] * num_qubits; ps[q] = 3
            terms.append(ps); weights.append(hz)
        if hx != 0.0:
            ps = [0] * num_qubits; ps[q] = 1
            terms.append(ps); weights.append(hx)
        if hy != 0.0:
            ps = [0] * num_qubits; ps[q] = 2
            terms.append(ps); weights.append(hy)
    return pauli_string_sum_dense(terms, weights)


