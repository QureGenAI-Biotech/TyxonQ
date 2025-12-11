from __future__ import annotations

"""Quantum information theory utilities.

This module provides backend-agnostic quantum information metrics and calculations:
- Entropy measures (von Neumann, Rényi)
- State fidelity and trace distance
- Partial transpose and entanglement negativity
- Reduced density matrices and mutual information
- Thermal states (Gibbs, purified)
"""

from typing import List, Optional, Union

import numpy as np


def entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    """Calculate von Neumann entropy S(ρ) = -Tr(ρ log ρ).

    Args:
        rho: Density matrix (can be pure state vector or density operator)
        eps: Small regularization to avoid log(0)

    Returns:
        Von Neumann entropy in nats
    """
    rho = np.asarray(rho, dtype=np.complex128)
    # Ensure density matrix
    vals = np.linalg.eigvalsh(rho)
    vals = np.clip(vals.real, 0.0, None)
    vals = vals / max(vals.sum(), eps)
    vals = np.clip(vals, eps, 1.0)
    return float(-(vals * np.log(vals)).sum())


def renyi_entropy(rho: np.ndarray, k: int = 2) -> float:
    """Calculate Rényi entropy of order k.

    S_k(ρ) = (1/(1-k)) log(Tr(ρ^k))

    Args:
        rho: Density matrix
        k: Order of Rényi entropy (k=1 gives von Neumann entropy)

    Returns:
        Rényi entropy
    """
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
    """Calculate free energy F = Tr(ρH) - S(ρ)/β.

    Args:
        rho: Density matrix
        h: Hamiltonian operator
        beta: Inverse temperature
        eps: Regularization for entropy

    Returns:
        Free energy
    """
    rho = np.asarray(rho, dtype=np.complex128)
    h = np.asarray(h, dtype=np.complex128)
    energy = float(np.real(np.trace(rho @ h)))
    s = entropy(rho, eps)
    return float(energy - s / beta)


def renyi_free_energy(rho: np.ndarray, h: np.ndarray, beta: float = 1.0, k: int = 2) -> float:
    """Calculate Rényi free energy F_k = Tr(ρH) - S_k(ρ)/β.

    Args:
        rho: Density matrix
        h: Hamiltonian operator
        beta: Inverse temperature
        k: Rényi order

    Returns:
        Rényi free energy
    """
    energy = float(np.real(np.trace(np.asarray(rho, dtype=np.complex128) @ np.asarray(h, dtype=np.complex128))))
    s = renyi_entropy(rho, k)
    return float(energy - s / beta)


def _sqrtm_psd(a: np.ndarray) -> np.ndarray:
    """Matrix square root for positive semi-definite matrix."""
    w, v = np.linalg.eigh(a)
    w = np.clip(w.real, 0.0, None)
    return (v * np.sqrt(w)) @ v.conj().T


def trace_distance(rho: np.ndarray, rho0: np.ndarray, eps: float = 1e-12) -> float:
    """Calculate trace distance D(ρ, σ) = (1/2)||ρ - σ||_1.

    Args:
        rho: First density matrix
        rho0: Second density matrix
        eps: Regularization

    Returns:
        Trace distance
    """
    d = np.asarray(rho, dtype=np.complex128) - np.asarray(rho0, dtype=np.complex128)
    x = d.conj().T @ d
    vals = np.linalg.eigvalsh(x).real
    vals = np.clip(vals, 0.0, None)
    return float(0.5 * np.sum(np.sqrt(vals + eps)))


def fidelity(rho: np.ndarray, rho0: np.ndarray) -> float:
    """Calculate state fidelity F(ρ, σ) = [Tr(√(√ρ σ √ρ))]².

    Args:
        rho: First density matrix
        rho0: Second density matrix

    Returns:
        Fidelity in [0, 1]
    """
    rho = np.asarray(rho, dtype=np.complex128)
    rho0 = np.asarray(rho0, dtype=np.complex128)
    rs = _sqrtm_psd(rho)
    inner = rs @ rho0 @ rs
    val = _sqrtm_psd(inner)
    f = float(np.real(np.trace(val)))
    return float(f**2)


def gibbs_state(h: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Generate Gibbs thermal state ρ = exp(-βH) / Z.

    Args:
        h: Hamiltonian operator
        beta: Inverse temperature

    Returns:
        Gibbs density matrix
    """
    w, v = np.linalg.eigh(np.asarray(h, dtype=np.complex128))
    ew = np.exp(-beta * w.real)
    rho = (v * ew) @ v.conj().T
    rho = rho / np.trace(rho)
    return rho


def double_state(h: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Generate purified thermal state |ψ⟩ where Tr_B(|ψ⟩⟨ψ|) = Gibbs(H, β).

    Args:
        h: Hamiltonian operator
        beta: Inverse temperature

    Returns:
        Purified state vector
    """
    w, v = np.linalg.eigh(np.asarray(h, dtype=np.complex128))
    ew = np.exp(-0.5 * beta * w.real)
    rho_half = (v * ew) @ v.conj().T
    psi = rho_half.reshape(-1)
    psi = psi / np.linalg.norm(psi)
    return psi


def partial_transpose(rho: np.ndarray, transposed_sites: List[int]) -> np.ndarray:
    """Apply partial transpose on specified qubits.

    Args:
        rho: Density matrix (2^n × 2^n)
        transposed_sites: List of qubit indices to transpose

    Returns:
        Partially transposed density matrix
    """
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
    """Calculate entanglement negativity E_N = (log||ρ^T_A|| - 1) / 2.

    Args:
        rho: Density matrix
        transposed_sites: Qubits to partially transpose

    Returns:
        Entanglement negativity
    """
    rhot = partial_transpose(rho, transposed_sites)
    es = np.linalg.eigvalsh(rhot).real
    rhot_m = float(np.sum(np.abs(es)))
    return float((np.log(rhot_m) - 1.0) / 2.0)


def log_negativity(rho: np.ndarray, transposed_sites: List[int], base: Union[str, int] = "e") -> float:
    """Calculate logarithmic negativity E_L = log||ρ^T_A||.

    Args:
        rho: Density matrix
        transposed_sites: Qubits to partially transpose
        base: Logarithm base ('e' or 2)

    Returns:
        Log negativity
    """
    rhot = partial_transpose(rho, transposed_sites)
    es = np.linalg.eigvalsh(rhot).real
    rhot_m = float(np.sum(np.abs(es)))
    val = float(np.log(rhot_m))
    if base in ("2", 2):
        return val / np.log(2.0)
    return val


def reduced_density_matrix(state: np.ndarray, cut: Union[int, List[int]], p: Optional[np.ndarray] = None) -> np.ndarray:
    """Calculate reduced density matrix by tracing out specified qubits.

    Args:
        state: State vector (2^n) or density matrix (2^n × 2^n)
        cut: Qubit index or list of indices to trace out
               OR number of qubits to trace from beginning (if int)
        p: Optional probability weights for mixed state from ensemble

    Returns:
        Reduced density matrix
    """
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
    """Calculate mutual information I(A:B) = S(A) + S(B) - S(AB).

    Args:
        s: State vector or density matrix
        cut: Partition specifying subsystem A

    Returns:
        Mutual information
    """
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


def taylorlnm(x: np.ndarray, k: int) -> np.ndarray:
    """Truncated Taylor series of ln(1 + x) up to order k.

    ln(1+x) ≈ Σ_{i=1..k} (-1)^{i+1} x^i / i

    Args:
        x: Input matrix
        k: Order of Taylor expansion

    Returns:
        Approximated ln(1 + x)
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

    Args:
        rho: Density matrix
        h: Hamiltonian operator
        beta: Inverse temperature
        k: Order of truncation

    Returns:
        Truncated free energy
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
    """Project state on computational outcomes of specified qubits.

    Args:
        state: State vector (2^n)
        cut: Qubit indices to measure and project
        measure: Measurement outcomes (0/1) for each qubit in cut; defaults to all zeros

    Returns:
        Reduced wavefunction with measured qubits projected out
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
