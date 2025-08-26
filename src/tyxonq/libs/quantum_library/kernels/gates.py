from __future__ import annotations

from typing import Any
import numpy as np
from ....numerics import NumericBackend as nb



# ---- Gate matrices (backend-native) ----
def gate_h() -> Any:
    one = nb.array(1.0, dtype=nb.complex128)
    minus_one = nb.array(-1.0, dtype=nb.complex128)
    mat = nb.array([[one, one], [one, minus_one]], dtype=nb.complex128)
    factor = nb.array(1.0, dtype=nb.complex128) / nb.sqrt(nb.array(2.0, dtype=nb.float64))
    return factor * mat


def gate_rz(theta: Any) -> Any:
    th = nb.asarray(theta)
    half = nb.array(0.5, dtype=nb.float64)
    j = nb.array(1j, dtype=nb.complex128)
    e = nb.exp(-j * half * th)
    zero = nb.array(0.0, dtype=nb.complex128)
    return nb.array([[e, zero], [zero, nb.conj(e)]], dtype=nb.complex128)


def gate_rx(theta: Any) -> Any:
    # Rx = cos(th/2) I - i sin(th/2) X
    th = nb.asarray(theta)
    half = nb.array(0.5, dtype=nb.float64)
    c = nb.cos(th * half)
    s = nb.sin(th * half)
    I = nb.eye(2, dtype=nb.complex128)
    X = gate_x()
    return c * I + (-nb.array(1j, dtype=nb.complex128) * s) * X


def gate_ry(theta: Any) -> Any:
    # Ry = cos(th/2) I - i sin(th/2) Y, but conventional definition yields real matrix
    th = nb.asarray(theta)
    half = nb.array(0.5, dtype=nb.float64)
    c = nb.cos(th * half)
    s = nb.sin(th * half)
    # Build matrix directly to keep real form
    return nb.array([[c, -s], [s, c]], dtype=nb.complex128)


def gate_phase(theta: Any) -> Any:
    th = nb.asarray(theta)
    j = nb.array(1j, dtype=nb.complex128)
    e = nb.exp(j * th)
    one = nb.array(1.0, dtype=nb.complex128)
    zero = nb.array(0.0, dtype=nb.complex128)
    return nb.array([[one, zero], [zero, e]], dtype=nb.complex128)


def gate_cx_4x4() -> Any:
    one = nb.array(1.0, dtype=nb.complex128)
    zero = nb.array(0.0, dtype=nb.complex128)
    return nb.array([
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, zero, one],
        [zero, zero, one, zero],
    ], dtype=nb.complex128)


def gate_cx_rank4() -> Any:
    U = gate_cx_4x4()
    return nb.reshape(U, (2, 2, 2, 2))


def gate_cz_4x4() -> Any:
    one = nb.array(1.0, dtype=nb.complex128)
    zero = nb.array(0.0, dtype=nb.complex128)
    minus_one = nb.array(-1.0, dtype=nb.complex128)
    return nb.array([
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, one, zero],
        [zero, zero, zero, minus_one],
    ], dtype=nb.complex128)


def gate_x() -> Any:
    zero = nb.array(0.0, dtype=nb.complex128)
    one = nb.array(1.0, dtype=nb.complex128)
    return nb.array([[zero, one], [one, zero]], dtype=nb.complex128)


def gate_s() -> Any:
    return gate_phase(nb.array(np.pi / 2.0, dtype=nb.float64))


def gate_sd() -> Any:
    return gate_phase(nb.array(-np.pi / 2.0, dtype=nb.float64))


def gate_t() -> Any:
    return gate_phase(nb.array(np.pi / 4.0, dtype=nb.float64))


def gate_td() -> Any:
    return gate_phase(nb.array(-np.pi / 4.0, dtype=nb.float64))


def gate_rxx(theta: Any) -> Any:
    # exp(-i theta/2 X⊗X) = cos(theta/2) I - i sin(theta/2) X⊗X
    th = nb.asarray(theta)
    half = nb.array(0.5, dtype=nb.float64)
    c = nb.cos(th * half)
    s = nb.sin(th * half)
    X = gate_x()
    XX = nb.kron(X, X)
    I4 = nb.eye(4, dtype=nb.complex128)
    return c * I4 + (-nb.array(1j, dtype=nb.complex128) * s) * XX


def gate_ryy(theta: Any) -> Any:
    Y = nb.array([[0.0 + 0.0j, -1j], [1j, 0.0 + 0.0j]], dtype=nb.complex128)
    YY = nb.kron(Y, Y)
    th = nb.asarray(theta)
    half = nb.array(0.5, dtype=nb.float64)
    c = nb.cos(th * half)
    s = nb.sin(th * half)
    I4 = nb.eye(4, dtype=nb.complex128)
    return c * I4 + (-nb.array(1j, dtype=nb.complex128) * s) * YY


def gate_rzz(theta: Any) -> Any:
    Z = nb.array([[1.0, 0.0], [0.0, -1.0]], dtype=nb.complex128)
    ZZ = nb.kron(Z, Z)
    th = nb.asarray(theta)
    half = nb.array(0.5, dtype=nb.float64)
    c = nb.cos(th * half)
    s = nb.sin(th * half)
    I4 = nb.eye(4, dtype=nb.complex128)
    return c * I4 + (-nb.array(1j, dtype=nb.complex128) * s) * ZZ


def build_controlled_unitary(U: np.ndarray, num_controls: int, ctrl_state: list[int] | None = None) -> Any:
    """Build a dense multi-controlled unitary (backend-native array).

    Layout: [controls..., targets...]. If controls match ctrl_state, apply U on targets, else identity.
    U must be shape (2^k, 2^k) for some k>=1.
    """
    if num_controls < 1:
        return nb.asarray(U)
    dim_t = U.shape[0]
    k = int(np.log2(dim_t))
    assert dim_t == (1 << k) and U.shape == (dim_t, dim_t)
    m = num_controls
    if ctrl_state is None:
        ctrl_state = [1] * m
    assert len(ctrl_state) == m
    dim_c = 1 << m
    dim = dim_c * dim_t
    # Build in Python lists to avoid requiring slicing on backends
    zero = 0.0 + 0.0j
    out_rows: list[list[complex]] = [[zero for _ in range(dim)] for _ in range(dim)]
    for mask in range(dim_c):
        row = mask * dim_t
        if all(((mask >> i) & 1) == ctrl_state[m - 1 - i] for i in range(m)):
            # place U block
            for r in range(dim_t):
                for c in range(dim_t):
                    out_rows[row + r][row + c] = complex(U[r, c])
        else:
            # place identity block
            for r in range(dim_t):
                out_rows[row + r][row + r] = 1.0 + 0.0j
    return nb.array(out_rows, dtype=nb.complex128)

