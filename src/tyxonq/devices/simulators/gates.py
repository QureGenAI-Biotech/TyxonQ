from __future__ import annotations

from typing import Any
import numpy as np


def _einsum_backend(backend: Any, spec: str, *ops: Any) -> np.ndarray:
    einsum = getattr(backend, "einsum", None)
    if callable(einsum):
        try:
            asarray = getattr(backend, "asarray", None)
            to_numpy = getattr(backend, "to_numpy", None)
            tops = [asarray(x) if callable(asarray) else x for x in ops]
            res = einsum(spec, *tops)
            return to_numpy(res) if callable(to_numpy) else np.asarray(res)
        except Exception:
            pass
    return np.einsum(spec, *ops)


# ---- Gate matrices (NumPy) ----
def gate_h() -> np.ndarray:
    return (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)


def gate_rz(theta: float) -> np.ndarray:
    e = np.exp(-0.5j * theta)
    return np.array([[e, 0.0], [0.0, np.conj(e)]], dtype=np.complex128)


def gate_rx(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    return np.array([[c, s], [s, c]], dtype=np.complex128)


def gate_cx_4x4() -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.complex128)


def gate_cx_rank4() -> np.ndarray:
    U = gate_cx_4x4().reshape(2, 2, 2, 2)
    return U


# ---- Statevector helpers ----
def init_statevector(num_qubits: int) -> np.ndarray:
    if num_qubits <= 0:
        return np.array([1.0 + 0.0j])
    state = np.zeros(1 << num_qubits, dtype=np.complex128)
    state[0] = 1.0 + 0.0j
    return state


def apply_1q_statevector(backend: Any, state: np.ndarray, gate2: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    psi = state.reshape([2] * num_qubits)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    in_letters = letters[:num_qubits]
    in_letters_q = in_letters.copy(); in_letters_q[qubit] = 'b'
    out_letters = in_letters.copy(); out_letters[qubit] = 'a'
    spec = f"ab,{''.join(in_letters_q)}->{''.join(out_letters)}"
    arr = _einsum_backend(backend, spec, gate2, psi)
    return arr.reshape(-1)


def apply_2q_statevector(backend: Any, state: np.ndarray, gate4: np.ndarray, q0: int, q1: int, num_qubits: int) -> np.ndarray:
    if q0 == q1:
        return state
    psi = state.reshape([2] * num_qubits)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    in_letters = letters[:num_qubits]
    in_letters_q = in_letters.copy(); in_letters_q[q0] = 'c'; in_letters_q[q1] = 'd'
    out_letters = in_letters.copy(); out_letters[q0] = 'a'; out_letters[q1] = 'b'
    spec = f"abcd,{''.join(in_letters_q)}->{''.join(out_letters)}"
    arr = _einsum_backend(backend, spec, gate4.reshape(2, 2, 2, 2), psi)
    return arr.reshape(-1)


def expect_z_statevector(state: np.ndarray, qubit: int, num_qubits: int) -> float:
    s_perm = np.moveaxis(state.reshape([2] * num_qubits), qubit, 0)
    s2 = np.abs(s_perm.reshape(2, -1)) ** 2
    probs = np.sum(s2, axis=1)
    return float(probs[0] - probs[1])


# ---- Density matrix helpers ----
def init_density(num_qubits: int) -> np.ndarray:
    dim = 1 << num_qubits
    rho = np.zeros((dim, dim), dtype=np.complex128)
    rho[0, 0] = 1.0 + 0.0j
    return rho


def apply_1q_density(backend: Any, rho: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    psi = rho.reshape([2] * (2 * n))
    letters = list("abcdefghijklmnopqrstuvwxyz")
    r = letters[:n]; c = letters[n:2*n]
    r_in = r.copy(); c_in = c.copy()
    r_in[q] = 'a'; c_in[q] = 'b'
    r_out = r.copy(); c_out = c.copy()
    r_out[q] = 'x'; c_out[q] = 'y'
    spec = f"xa,{''.join(r_in + c_in)},by->{''.join(r_out + c_out)}"
    Udag = np.conj(U.T)
    out = _einsum_backend(backend, spec, U, psi, Udag)
    return out.reshape(rho.shape)


def apply_2q_density(backend: Any, rho: np.ndarray, U4: np.ndarray, q0: int, q1: int, n: int) -> np.ndarray:
    if q0 == q1:
        return rho
    psi = rho.reshape([2] * (2 * n))
    letters = list("abcdefghijklmnopqrstuvwxyz")
    r = letters[:n]; c = letters[n:2*n]
    r_in = r.copy(); c_in = c.copy()
    r_in[q0] = 'a'; r_in[q1] = 'b'
    c_in[q0] = 'c'; c_in[q1] = 'd'
    r_out = r.copy(); c_out = c.copy()
    r_out[q0] = 'w'; r_out[q1] = 'x'
    c_out[q0] = 'y'; c_out[q1] = 'z'
    spec = f"wxab,{''.join(r_in + c_in)},yzcd->{''.join(r_out + c_out)}"
    U4 = U4.reshape(2, 2, 2, 2)
    U4d = np.conj(U4.transpose(2, 3, 0, 1))
    out = _einsum_backend(backend, spec, U4, psi, U4d)
    return out.reshape(rho.shape)


def exp_z_density(backend: Any, rho: np.ndarray, q: int, n: int) -> float:
    rho_t = rho.reshape([2] * (2 * n))
    letters = list("abcdefghijklmnopqrstuvwxyz")
    r = letters[:n]; c = letters[n:2*n]
    r_in = r.copy(); c_in = c.copy()
    r_in[q] = 'a'; c_in[q] = 'b'
    spec = f"ab,{''.join(r_in + c_in)}->"
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    val = _einsum_backend(backend, spec, Z, rho_t)
    return float(np.real_if_close(val))


