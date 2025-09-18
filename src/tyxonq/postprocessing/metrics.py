from __future__ import annotations

"""Postprocessing metrics: normalization, divergences, expectations.

This module provides lightweight, dependency-free utilities for common
postprocessing metrics used across devices and simulators.
"""

from typing import Any, Dict, Optional, Sequence, List, Union

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


def entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    rho = np.asarray(rho, dtype=np.complex128)
    # Ensure density matrix
    vals = np.linalg.eigvalsh(rho)
    vals = np.clip(vals.real, 0.0, None)
    vals = vals / max(vals.sum(), eps)
    vals = np.clip(vals, eps, 1.0)
    return float(-(vals * np.log(vals)).sum())


def renyi_entropy(rho: np.ndarray, k: int = 2) -> float:
    vals = np.linalg.eigvalsh(np.asarray(rho, dtype=np.complex128)).real
    vals = np.clip(vals, 0.0, None)
    s = vals.sum()
    if s <= 0:
        return 0.0
    vals = vals / s
    if k == 1:
        return entropy(vals)
    return float((1.0 / (1 - k)) * np.log(np.power(vals, k).sum()))


def free_energy(rho: np.ndarray, h: np.ndarray, beta: float = 1.0, eps: float = 1e-12) -> float:
    rho = np.asarray(rho, dtype=np.complex128)
    h = np.asarray(h, dtype=np.complex128)
    energy = float(np.real(np.trace(rho @ h)))
    s = entropy(rho, eps)
    return float(energy - s / beta)


def renyi_free_energy(rho: np.ndarray, h: np.ndarray, beta: float = 1.0, k: int = 2) -> float:
    energy = float(np.real(np.trace(np.asarray(rho, dtype=np.complex128) @ np.asarray(h, dtype=np.complex128))))
    s = renyi_entropy(rho, k)
    return float(energy - s / beta)


def _sqrtm_psd(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(a)
    w = np.clip(w.real, 0.0, None)
    return (v * np.sqrt(w)) @ v.conj().T


def trace_distance(rho: np.ndarray, rho0: np.ndarray, eps: float = 1e-12) -> float:
    d = np.asarray(rho, dtype=np.complex128) - np.asarray(rho0, dtype=np.complex128)
    x = d.conj().T @ d
    vals = np.linalg.eigvalsh(x).real
    vals = np.clip(vals, 0.0, None)
    return float(0.5 * np.sum(np.sqrt(vals + eps)))


def fidelity(rho: np.ndarray, rho0: np.ndarray) -> float:
    rho = np.asarray(rho, dtype=np.complex128)
    rho0 = np.asarray(rho0, dtype=np.complex128)
    rs = _sqrtm_psd(rho)
    inner = rs @ rho0 @ rs
    val = _sqrtm_psd(inner)
    f = float(np.real(np.trace(val)))
    return float(f**2)


def gibbs_state(h: np.ndarray, beta: float = 1.0) -> np.ndarray:
    w, v = np.linalg.eigh(np.asarray(h, dtype=np.complex128))
    ew = np.exp(-beta * w.real)
    rho = (v * ew) @ v.conj().T
    rho = rho / np.trace(rho)
    return rho


def double_state(h: np.ndarray, beta: float = 1.0) -> np.ndarray:
    w, v = np.linalg.eigh(np.asarray(h, dtype=np.complex128))
    ew = np.exp(-0.5 * beta * w.real)
    rho_half = (v * ew) @ v.conj().T
    psi = rho_half.reshape(-1)
    psi = psi / np.linalg.norm(psi)
    return psi


def partial_transpose(rho: np.ndarray, transposed_sites: List[int]) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.complex128)
    dim = rho.shape[0]
    n = int(round(np.log2(dim)))
    assert rho.shape == (dim, dim)
    t = rho.reshape([2] * (2 * n))
    axes = list(range(2 * n))
    for q in transposed_sites:
        axes[q], axes[q + n] = axes[q + n], axes[q]
    t = np.transpose(t, axes)
    return t.reshape(dim, dim)


def entanglement_negativity(rho: np.ndarray, transposed_sites: List[int]) -> float:
    rhot = partial_transpose(rho, transposed_sites)
    es = np.linalg.eigvalsh(rhot).real
    rhot_m = float(np.sum(np.abs(es)))
    return float((np.log(rhot_m) - 1.0) / 2.0)


def log_negativity(rho: np.ndarray, transposed_sites: List[int], base: Union[str, int] = "e") -> float:
    rhot = partial_transpose(rho, transposed_sites)
    es = np.linalg.eigvalsh(rhot).real
    rhot_m = float(np.sum(np.abs(es)))
    val = float(np.log(rhot_m))
    if base in ("2", 2):
        return val / np.log(2.0)
    return val


def reduced_density_matrix(state: np.ndarray, cut: Union[int, List[int]], p: Optional[np.ndarray] = None) -> np.ndarray:
    s = np.asarray(state, dtype=np.complex128)
    # Determine n qubits
    if s.ndim == 2 and s.shape[0] == s.shape[1]:
        # density operator case
        dim = s.shape[0]
        n = int(round(np.log2(dim)))
        traced = list(cut) if isinstance(cut, (list, tuple, set)) else list(range(int(cut)))
        kept = [i for i in range(n) if i not in traced]
        t = s.reshape([2] * (2 * n))
        perm = kept + traced + [i + n for i in kept + traced]
        t = np.transpose(t, perm)
        k = len(kept)
        t = t.reshape(2**k, 2 ** (n - k), 2**k, 2 ** (n - k))
        rho = np.tensordot(t, np.eye(2 ** (n - k)), axes=([1, 3], [0, 1]))
        rho = rho.reshape(2**k, 2**k)
        rho = rho / max(1e-12, np.trace(rho))
        return rho
    else:
        # pure state vector
        dim = s.shape[0]
        n = int(round(np.log2(dim)))
        traced = list(cut) if isinstance(cut, (list, tuple, set)) else list(range(int(cut)))
        kept = [i for i in range(n) if i not in traced]
        t = s.reshape([2] * n)
        perm = kept + traced
        t = np.transpose(t, perm)
        t = t.reshape(2 ** len(kept), 2 ** len(traced))
        if p is None:
            rho = t @ t.conj().T
        else:
            p = np.asarray(p).reshape(-1)
            rho = t @ np.diag(p) @ t.conj().T
        rho = rho / max(1e-12, np.trace(rho))
        return rho


def mutual_information(s: np.ndarray, cut: Union[int, List[int]]) -> float:
    arr = np.asarray(s, dtype=np.complex128)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        n = int(round(np.log2(arr.shape[0])))
        hab = entropy(arr)
        rhoa = reduced_density_matrix(arr, cut)
        ha = entropy(rhoa)
        other = [i for i in range(n) if i not in (list(cut) if isinstance(cut, (list, tuple, set)) else list(range(int(cut))))]
        rhob = reduced_density_matrix(arr, other)
        hb = entropy(rhob)
    else:
        hab = 0.0
        rhoa = reduced_density_matrix(arr, cut)
        ha = hb = entropy(rhoa)
    return float(ha + hb - hab)


# ---- Extras: truncated free energy and reduced wavefunction ----

def taylorlnm(x: np.ndarray, k: int) -> np.ndarray:
    """Truncated Taylor series of ln(1 + x) up to order k.

    ln(1+x) ≈ Σ_{i=1..k} (-1)^{i+1} x^i / i
    """
    x = np.asarray(x, dtype=np.complex128)
    y = np.zeros_like(x)
    pow_x = np.eye(x.shape[0], dtype=np.complex128)
    for i in range(1, k + 1):
        pow_x = pow_x @ x
        coef = ((-1) ** (i + 1)) / i
        y = y + coef * pow_x
    return y


def truncated_free_energy(rho: np.ndarray, h: np.ndarray, beta: float = 1.0, k: int = 2) -> float:
    """Truncated free energy using Taylor approximation of log.

    F ≈ Tr(rho h) - (1/β) * Tr[rho ln(rho)]_truncated
    with ln(ρ) ≈ ln(I + (ρ - I)) truncated to order (k-1).
    """
    rho = np.asarray(rho, dtype=np.complex128)
    h = np.asarray(h, dtype=np.complex128)
    energy = float(np.real(np.trace(rho @ h)))
    # ln(rho) ≈ taylorlnm(rho - I, k-1)
    I = np.eye(rho.shape[0], dtype=np.complex128)
    if k <= 1:
        approx_ln = np.zeros_like(rho)
    else:
        approx_ln = taylorlnm(rho - I, k - 1)
    renyi = float(np.real(np.trace(rho @ approx_ln)))
    return float(energy - renyi / beta)


def reduced_wavefunction(state: np.ndarray, cut: List[int], measure: Optional[List[int]] = None) -> np.ndarray:
    """Project `state` on computational outcomes of `cut` and return remaining ket.

    - state: ket vector of length 2^n
    - cut: list of qubit indices to project out
    - measure: same length as cut, 0/1 outcomes; defaults to zeros
    """
    s = np.asarray(state, dtype=np.complex128).reshape(-1)
    n = int(round(np.log2(s.shape[0])))
    if measure is None:
        measure = [0 for _ in cut]
    # Reshape to [2]*n and index measured axes
    t = s.reshape([2] * n)
    # Build slices for all axes
    idx = [slice(None)] * n
    for q, m in zip(cut, measure):
        idx[q] = int(m)
    t = t[tuple(idx)]
    # Flatten remaining axes in ascending order of axes not in cut
    remaining = [i for i in range(n) if i not in cut]
    if remaining:
        t = np.transpose(t, axes=remaining) if t.ndim > 1 else t
        out = t.reshape(-1)
    else:
        out = np.array([t], dtype=np.complex128).reshape(-1)
    return out


__all__ = [
    "normalized_count",
    "kl_divergence",
    "expectation",
    "entropy",
    "renyi_entropy",
    "free_energy",
    "renyi_free_energy",
    "trace_distance",
    "fidelity",
    "gibbs_state",
    "double_state",
    "partial_transpose",
    "entanglement_negativity",
    "log_negativity",
    "reduced_density_matrix",
    "mutual_information",
    "taylorlnm",
    "truncated_free_energy",
    "reduced_wavefunction",
]


