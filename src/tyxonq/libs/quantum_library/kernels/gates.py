from __future__ import annotations

from typing import Any
import numpy as np



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

