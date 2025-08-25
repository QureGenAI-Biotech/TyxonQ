from __future__ import annotations

from typing import Any
import numpy as np
from .common import _einsum_backend 


def init_density(num_qubits: int) -> np.ndarray:
    dim = 1 << num_qubits
    rho = np.zeros((dim, dim), dtype=np.complex128)
    rho[0, 0] = 1.0 + 0.0j
    return rho


def apply_1q_density(backend: Any, rho: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    psi = rho.reshape([2] * (2 * n))
    letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    reserved = set(['a', 'b', 'x', 'y'])
    # choose axis symbols disjoint from reserved
    axis_symbols = [ch for ch in letters if ch not in reserved]
    r_axes = axis_symbols[:n]
    c_axes = axis_symbols[n:2 * n]
    r_in = r_axes.copy(); c_in = c_axes.copy()
    r_in[q] = 'a'; c_in[q] = 'b'
    r_out = r_axes.copy(); c_out = c_axes.copy()
    r_out[q] = 'x'; c_out[q] = 'y'
    spec = f"xa,{''.join(r_in + c_in)},by->{''.join(r_out + c_out)}"
    Udag = np.conj(U.T)
    out = _einsum_backend(backend, spec, U, psi, Udag)
    return out.reshape(rho.shape)


def apply_2q_density(backend: Any, rho: np.ndarray, U4: np.ndarray, q0: int, q1: int, n: int) -> np.ndarray:
    if q0 == q1:
        return rho
    psi = rho.reshape([2] * (2 * n))
    letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    reserved = set(['a', 'b', 'c', 'd', 'w', 'x', 'y', 'z'])
    axis_symbols = [ch for ch in letters if ch not in reserved]
    r_axes = axis_symbols[:n]
    c_axes = axis_symbols[n:2 * n]
    r_in = r_axes.copy(); c_in = c_axes.copy()
    r_in[q0] = 'a'; r_in[q1] = 'b'
    c_in[q0] = 'c'; c_in[q1] = 'd'
    r_out = r_axes.copy(); c_out = c_axes.copy()
    r_out[q0] = 'w'; r_out[q1] = 'x'
    c_out[q0] = 'y'; c_out[q1] = 'z'
    spec = f"wxab,{''.join(r_in + c_in)},yzcd->{''.join(r_out + c_out)}"
    U4 = U4.reshape(2, 2, 2, 2)
    U4d = np.conj(U4.transpose(2, 3, 0, 1))
    out = _einsum_backend(backend, spec, U4, psi, U4d)
    return out.reshape(rho.shape)


def exp_z_density(backend: Any, rho: np.ndarray, q: int, n: int) -> float:
    # Fast path via diagonal populations; correct for Z expectation
    dim = 1 << n
    diag = np.real(np.diag(rho))
    bits = (np.arange(dim) >> (n - 1 - q)) & 1
    signs = 1.0 - 2.0 * bits
    return float(np.sum(diag * signs))


__all__ = [
    "init_density",
    "apply_1q_density",
    "apply_2q_density",
    "exp_z_density",
]


