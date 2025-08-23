from __future__ import annotations

from typing import List
import numpy as np


def depolarizing(p: float) -> List[np.ndarray]:
    p = float(p)
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    k0 = np.sqrt(1 - p) * I
    k1 = np.sqrt(p / 3.0) * X
    k2 = np.sqrt(p / 3.0) * Y
    k3 = np.sqrt(p / 3.0) * Z
    return [k0, k1, k2, k3]


def amplitude_damping(gamma: float) -> List[np.ndarray]:
    g = float(gamma)
    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - g)]], dtype=np.complex128)
    K1 = np.array([[0.0, np.sqrt(g)], [0.0, 0.0]], dtype=np.complex128)
    return [K0, K1]


def phase_damping(lmbda: float) -> List[np.ndarray]:
    l = float(lmbda)
    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - l)]], dtype=np.complex128)
    K1 = np.array([[0.0, 0.0], [0.0, np.sqrt(l)]], dtype=np.complex128)
    return [K0, K1]


def pauli_channel(px: float, py: float, pz: float) -> List[np.ndarray]:
    px = float(px); py = float(py); pz = float(pz)
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    p0 = max(0.0, 1.0 - (px + py + pz))
    return [np.sqrt(p0) * I, np.sqrt(px) * X, np.sqrt(py) * Y, np.sqrt(pz) * Z]


def apply_to_density_matrix(rho: np.ndarray, kraus: List[np.ndarray], wire: int, num_qubits: int | None = None) -> np.ndarray:
    if num_qubits is None:
        dim = int(np.round(np.log2(rho.shape[0])))
    else:
        dim = int(num_qubits)
    assert rho.shape == (1 << dim, 1 << dim)
    t = rho.reshape([2] * (2 * dim))
    letters = list("abcdefghijklmnopqrstuvwxyz")
    r = letters[:dim]; c = letters[dim:2*dim]
    r_in = r.copy(); c_in = c.copy()
    r_in[wire] = 'a'; c_in[wire] = 'b'
    r_out = r.copy(); c_out = c.copy()
    r_out[wire] = 'x'; c_out[wire] = 'y'
    out = np.zeros_like(t)
    for K in kraus:
        spec = f"xa,{''.join(r_in + c_in)},by->{''.join(r_out + c_out)}"
        Kd = np.conj(K.T)
        out = out + np.einsum(spec, K, t, Kd)
    return out.reshape(rho.shape)


