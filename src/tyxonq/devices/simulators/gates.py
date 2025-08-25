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


def gate_ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def gate_phase(theta: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, np.exp(1j * theta)]], dtype=np.complex128)


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


def gate_cz_4x4() -> np.ndarray:
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ], dtype=np.complex128)


def gate_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def gate_s() -> np.ndarray:
    return gate_phase(np.pi / 2.0)


def gate_sd() -> np.ndarray:
    return gate_phase(-np.pi / 2.0)


def gate_t() -> np.ndarray:
    return gate_phase(np.pi / 4.0)


def gate_td() -> np.ndarray:
    return gate_phase(-np.pi / 4.0)


def gate_rxx(theta: float) -> np.ndarray:
    # exp(-i theta/2 X⊗X) = cos(theta/2) I - i sin(theta/2) X⊗X
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    X = gate_x()
    XX = np.kron(X, X)
    return (c * np.eye(4) + s * XX).astype(np.complex128)


def gate_ryy(theta: float) -> np.ndarray:
    Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    YY = np.kron(Y, Y)
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    return (c * np.eye(4) + s * YY).astype(np.complex128)


def gate_rzz(theta: float) -> np.ndarray:
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    ZZ = np.kron(Z, Z)
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    return (c * np.eye(4) + s * ZZ).astype(np.complex128)


def build_controlled_unitary(U: np.ndarray, num_controls: int, ctrl_state: list[int] | None = None) -> np.ndarray:
    """Build a dense multi-controlled unitary.

    Layout: [controls..., targets...]. If controls match ctrl_state, apply U on targets, else identity.
    U must be shape (2^k, 2^k) for some k>=1.
    """
    if num_controls < 1:
        return U.astype(np.complex128)
    dim_t = U.shape[0]
    k = int(np.log2(dim_t))
    assert dim_t == (1 << k) and U.shape == (dim_t, dim_t)
    m = num_controls
    if ctrl_state is None:
        ctrl_state = [1] * m
    assert len(ctrl_state) == m
    dim_c = 1 << m
    dim = dim_c * dim_t
    out = np.zeros((dim, dim), dtype=np.complex128)
    for mask in range(dim_c):
        # block row/col slice for this control pattern
        row = mask * dim_t
        if all(((mask >> i) & 1) == ctrl_state[m - 1 - i] for i in range(m)):
            out[row:row + dim_t, row:row + dim_t] = U
        else:
            out[row:row + dim_t, row:row + dim_t] = np.eye(dim_t, dtype=np.complex128)
    return out


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
    # Reserve 'a','b' for gate indices; use distinct axis symbols starting from 'c'
    axes = letters[2:2 + num_qubits]
    in_axes = axes.copy(); in_axes[qubit] = 'b'
    out_axes = axes.copy(); out_axes[qubit] = 'a'
    spec = f"ab,{''.join(in_axes)}->{''.join(out_axes)}"
    arr = _einsum_backend(backend, spec, gate2, psi)
    return arr.reshape(-1)


def apply_2q_statevector(backend: Any, state: np.ndarray, gate4: np.ndarray, q0: int, q1: int, num_qubits: int) -> np.ndarray:
    if q0 == q1:
        return state
    psi = state.reshape([2] * num_qubits)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    # Reserve 'a','b','c','d' for gate indices; use distinct axis symbols starting from 'e'
    axes = letters[4:4 + num_qubits]
    in_axes = axes.copy(); in_axes[q0] = 'c'; in_axes[q1] = 'd'
    out_axes = axes.copy(); out_axes[q0] = 'a'; out_axes[q1] = 'b'
    spec = f"abcd,{''.join(in_axes)}->{''.join(out_axes)}"
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


