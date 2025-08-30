from __future__ import annotations

from typing import Any
import numpy as np
from ....numerics import NumericBackend as nb
from ....numerics.api import ArrayBackend


def init_density(num_qubits: int, backend: ArrayBackend | None = None) -> Any:
    K = backend or nb
    dim = 1 << num_qubits
    rho = K.zeros((dim, dim), dtype=K.complex128)
    # set |0...0><0...0|
    one = K.array(1.0 + 0.0j, dtype=K.complex128)
    rho_np = K.to_numpy(rho)
    rho_np[0, 0] = 1.0 + 0.0j
    return K.asarray(rho_np)


def apply_1q_density(backend: Any, rho: Any, U: Any, q: int, n: int) -> Any:
    K = backend or nb
    psi = K.reshape(K.asarray(rho), (2,) * (2 * n))
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
    U_bk = K.asarray(U)
    U_conj = K.conj(U_bk)
    # use indices 'yb' to represent conjugate-transpose without materializing a transpose
    spec = f"xa,{''.join(r_in + c_in)},yb->{''.join(r_out + c_out)}"
    out = K.einsum(spec, U_bk, psi, U_conj)
    return K.reshape(K.asarray(out), (1 << n, 1 << n))


def apply_2q_density(backend: Any, rho: Any, U4: Any, q0: int, q1: int, n: int) -> Any:
    if q0 == q1:
        return rho
    K = backend or nb
    psi = K.reshape(K.asarray(rho), (2,) * (2 * n))
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
    U4n = K.asarray(np.reshape(np.asarray(U4), (2, 2, 2, 2)))
    U4c = K.conj(U4n)
    # specify indices 'yzcd' for conj(U4) to represent conjugate-transpose on the appropriate axes
    out = K.einsum(spec, U4n, psi, U4c)
    return K.reshape(K.asarray(out), (1 << n, 1 << n))


def exp_z_density(backend: Any, rho: Any, q: int, n: int) -> Any:
    # Fast path via diagonal populations; correct for Z expectation
    K = backend or nb
    dim = 1 << n
    diag = K.real(K.diag(rho))
    bits = (np.arange(dim) >> (n - 1 - q)) & 1
    signs = 1.0 - 2.0 * bits
    return K.sum(K.asarray(K.to_numpy(diag) * signs))


__all__ = [
    "init_density",
    "apply_1q_density",
    "apply_2q_density",
    "exp_z_density",
]


