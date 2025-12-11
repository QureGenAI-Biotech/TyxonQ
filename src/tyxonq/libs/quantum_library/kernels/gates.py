from __future__ import annotations

from typing import Any
import numpy as np
from ....numerics import NumericBackend as nb
from ....numerics.api import ArrayBackend, get_backend



# ---- Gate matrices (backend-native) ----
def gate_h() -> Any:
    one = nb.array(1.0, dtype=nb.complex128)
    minus_one = nb.array(-1.0, dtype=nb.complex128)
    mat = nb.array([[one, one], [one, minus_one]], dtype=nb.complex128)
    factor = nb.array(1.0, dtype=nb.complex128) / nb.sqrt(nb.array(2.0, dtype=nb.float64))
    return factor * mat


def gate_rz(theta: Any, backend: ArrayBackend | None = None) -> Any:
    K = backend if backend is not None else get_backend(None)
    # Convert theta to backend tensor if it's a Python scalar, preserving autograd
    if isinstance(theta, (int, float)):
        theta = K.array(theta, dtype=K.float64)
    # Rz = cos(th/2) I - i sin(th/2) Z
    c = K.cos(theta * 0.5)
    s = K.sin(theta * 0.5)
    I = K.eye(2, dtype=K.complex128)
    # CRITICAL: Use stack to preserve gradient chain
    one = K.array(1.0, dtype=K.complex128)
    zero = K.array(0.0, dtype=K.complex128)
    minus_one = K.array(-1.0, dtype=K.complex128)
    Z_row0 = K.stack([one, zero])
    Z_row1 = K.stack([zero, minus_one])
    Z = K.stack([Z_row0, Z_row1])
    return c * I + (-1j * s) * Z


def gate_rx(theta: Any, backend: ArrayBackend | None = None) -> Any:
    # Rx = cos(th/2) I - i sin(th/2) X
    K = backend if backend is not None else get_backend(None)
    # Convert theta to backend tensor if it's a Python scalar
    if isinstance(theta, (int, float)):
        theta = K.array(theta, dtype=K.float64)
    c = K.cos(theta * 0.5)
    s = K.sin(theta * 0.5)
    I = K.eye(2, dtype=K.complex128)
    X = gate_x(backend=backend)
    return c * I + (-1j * s) * X


def gate_ry(theta: Any, backend: ArrayBackend | None = None) -> Any:
    # Ry = cos(th/2) I - i sin(th/2) Y, but conventional definition yields real matrix
    K = backend if backend is not None else get_backend(None)
    # Convert theta to backend tensor if it's a Python scalar
    if isinstance(theta, (int, float)):
        theta = K.array(theta, dtype=K.float64)
    c = K.cos(theta * 0.5)
    s = K.sin(theta * 0.5)
    # CRITICAL: Use stack to preserve gradient chain, not K.array([[c, -s], [s, c]])
    # Build rows as tensors, then stack them
    row0 = K.stack([c, -s])
    row1 = K.stack([s, c])
    mat = K.stack([row0, row1])
    # Cast to complex128 (this preserves gradients in PyTorch)
    return K.cast(mat, K.complex128)


def gate_phase(theta: Any, backend: ArrayBackend | None = None) -> Any:
    K = backend if backend is not None else get_backend(None)
    # Convert theta to backend tensor if it's a Python scalar
    if isinstance(theta, (int, float)):
        theta = K.array(theta, dtype=K.float64)
    e = K.exp(1j * theta)
    one = K.array(1.0, dtype=K.complex128)
    zero = K.array(0.0, dtype=K.complex128)
    # CRITICAL: Use stack to preserve gradient chain
    row0 = K.stack([one, zero])
    row1 = K.stack([zero, e])
    return K.stack([row0, row1])


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


def gate_iswap_4x4() -> Any:
    """iSWAP gate: exchanges qubits and applies relative phase.
    
    Matrix representation:
    [[1,  0,  0,  0],
     [0,  0, 1j,  0],
     [0, 1j,  0,  0],
     [0,  0,  0,  1]]
    
    Physical model: iSWAP = exp(-i π/4 · σ_x ⊗ σ_x)
    Swaps |01⟩ ↔ |10⟩ with relative phase i
    
    Reference:
        Shende & Markov, PRA 72, 062305 (2005)
    """
    one = nb.array(1.0, dtype=nb.complex128)
    zero = nb.array(0.0, dtype=nb.complex128)
    j = nb.array(1j, dtype=nb.complex128)
    return nb.array([
        [one, zero, zero, zero],
        [zero, zero, j, zero],
        [zero, j, zero, zero],
        [zero, zero, zero, one],
    ], dtype=nb.complex128)


def gate_swap_4x4() -> Any:
    """SWAP gate: exchanges qubits without phase.
    
    Matrix representation:
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]]
    
    Pure state exchange: |01⟩ ↔ |10⟩
    No relative phase factor
    
    Properties:
        - SWAP² = I (applying twice gives identity)
        - SWAP is Hermitian
        - Useful for qubit routing and layout optimization
    """
    one = nb.array(1.0, dtype=nb.complex128)
    zero = nb.array(0.0, dtype=nb.complex128)
    return nb.array([
        [one, zero, zero, zero],
        [zero, zero, one, zero],
        [zero, one, zero, zero],
        [zero, zero, zero, one],
    ], dtype=nb.complex128)


def gate_x(backend: ArrayBackend | None = None) -> Any:
    K = backend if backend is not None else get_backend(None)
    zero = K.array(0.0, dtype=K.complex128)
    one = K.array(1.0, dtype=K.complex128)
    # CRITICAL: Use stack to preserve gradient chain
    row0 = K.stack([zero, one])
    row1 = K.stack([one, zero])
    return K.stack([row0, row1])


def gate_y() -> Any:
    zero = nb.array(0.0, dtype=nb.complex128)
    j = nb.array(1j, dtype=nb.complex128)
    minus_j = nb.array(-1j, dtype=nb.complex128)
    return nb.array([[zero, minus_j], [j, zero]], dtype=nb.complex128)


def gate_z() -> Any:
    one = nb.array(1.0, dtype=nb.complex128)
    zero = nb.array(0.0, dtype=nb.complex128)
    minus_one = nb.array(-1.0, dtype=nb.complex128)
    return nb.array([[one, zero], [zero, minus_one]], dtype=nb.complex128)


def gate_s() -> Any:
    return gate_phase(nb.array(np.pi / 2.0, dtype=nb.float64))


def gate_sd() -> Any:
    return gate_phase(nb.array(-np.pi / 2.0, dtype=nb.float64))


def gate_t() -> Any:
    return gate_phase(nb.array(np.pi / 4.0, dtype=nb.float64))


def gate_td() -> Any:
    return gate_phase(nb.array(-np.pi / 4.0, dtype=nb.float64))


def gate_rxx(theta: Any, backend: ArrayBackend | None = None) -> Any:
    # exp(-i theta/2 X⊗X) = cos(theta/2) I - i sin(theta/2) X⊗X
    K = backend if backend is not None else get_backend(None)
    # Convert theta to backend tensor if it's a Python scalar
    if isinstance(theta, (int, float)):
        theta = K.array(theta, dtype=K.float64)
    c = K.cos(theta * 0.5)
    s = K.sin(theta * 0.5)
    X = gate_x(backend=backend)
    XX = K.kron(X, X)
    I4 = K.eye(4, dtype=K.complex128)
    return c * I4 + (-1j * s) * XX


def gate_ryy(theta: Any, backend: ArrayBackend | None = None) -> Any:
    K = backend if backend is not None else get_backend(None)
    # CRITICAL: Use stack to preserve gradient chain for Y matrix
    zero = K.array(0.0 + 0.0j, dtype=K.complex128)
    j = K.array(1j, dtype=K.complex128)
    minus_j = K.array(-1j, dtype=K.complex128)
    Y_row0 = K.stack([zero, minus_j])
    Y_row1 = K.stack([j, zero])
    Y = K.stack([Y_row0, Y_row1])
    YY = K.kron(Y, Y)
    # Convert theta to backend tensor if it's a Python scalar
    if isinstance(theta, (int, float)):
        theta = K.array(theta, dtype=K.float64)
    c = K.cos(theta * 0.5)
    s = K.sin(theta * 0.5)
    I4 = K.eye(4, dtype=K.complex128)
    return c * I4 + (-1j * s) * YY


def gate_rzz(theta: Any, backend: ArrayBackend | None = None) -> Any:
    K = backend if backend is not None else get_backend(None)
    # CRITICAL: Use stack to preserve gradient chain for Z matrix
    one = K.array(1.0, dtype=K.complex128)
    zero = K.array(0.0, dtype=K.complex128)
    minus_one = K.array(-1.0, dtype=K.complex128)
    Z_row0 = K.stack([one, zero])
    Z_row1 = K.stack([zero, minus_one])
    Z = K.stack([Z_row0, Z_row1])
    ZZ = K.kron(Z, Z)
    # Convert theta to backend tensor if it's a Python scalar
    if isinstance(theta, (int, float)):
        theta = K.array(theta, dtype=K.float64)
    c = K.cos(theta * 0.5)
    s = K.sin(theta * 0.5)
    I4 = K.eye(4, dtype=K.complex128)
    return c * I4 + (-1j * s) * ZZ


# --- ZZ Hamiltonian matrix (not exponential) ---

def zz_matrix() -> Any:
    """Return Z⊗Z (4x4 Hermitian) as backend-native array.
    Useful for exp(i theta Z⊗Z) style APIs that take a Hamiltonian matrix.
    """
    Z = nb.array([[1.0, 0.0], [0.0, -1.0]], dtype=nb.complex128)
    return nb.kron(Z, Z)


def gate_cry_4x4(theta: Any, backend: ArrayBackend | None = None) -> Any:
    """Controlled-RY on target with control as the first qubit.

    Basis order is |00>, |01>, |10>, |11> with control as the most-significant qubit,
    consistent with gate_cx_4x4.
    """
    K = backend if backend is not None else get_backend(None)
    # Convert theta to backend tensor if it's a Python scalar
    if isinstance(theta, (int, float)):
        theta = K.array(theta, dtype=K.float64)
    c = K.cos(theta * 0.5)
    s = K.sin(theta * 0.5)
    one = K.array(1.0, dtype=K.complex128)
    zero = K.array(0.0, dtype=K.complex128)
    # CRITICAL: Use stack to preserve gradient chain for 4x4 matrix
    row0 = K.stack([one, zero, zero, zero])
    row1 = K.stack([zero, one, zero, zero])
    row2 = K.stack([zero, zero, c, -s])
    row3 = K.stack([zero, zero, s, c])
    return K.stack([row0, row1, row2, row3])


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

