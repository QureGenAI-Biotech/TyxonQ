from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
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


def gate_cy_4x4() -> Any:
    """Controlled-Y gate.

    Basis order is |00>, |01>, |10>, |11> with the control as the
    most-significant qubit, consistent with gate_cx_4x4 / gate_cz_4x4 /
    gate_cry_4x4. When control=1 the Pauli-Y acts on the target.
    """
    one = nb.array(1.0, dtype=nb.complex128)
    zero = nb.array(0.0, dtype=nb.complex128)
    j = nb.array(1j, dtype=nb.complex128)
    minus_j = nb.array(-1j, dtype=nb.complex128)
    return nb.array([
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, zero, minus_j],
        [zero, zero, j, zero],
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


# ---------------------------------------------------------------------------
# 权威 op 词汇表与门矩阵解析（模拟器引擎分发的单一真相源）
#
# 三个模拟器引擎（statevector / density_matrix / matrix_product_state）曾各自维护
# run() 与 state() 两套 op 分发循环，门集互不一致且对未知 op 静默 ``continue``，导致
# ``cry`` / ``y`` / ``z`` / ``t`` / ``tdg`` / ``cy`` 等门被静默丢弃而算错。下面把"哪些
# op 是幺正门、各自对应哪个门矩阵、参数在 op 元组的哪个位置"集中定义**一次**：各引擎
# 的单一 ``_evolve`` 循环通过 :func:`resolve_unitary` 取矩阵，用各自表示的 apply 内核施加；
# 非幺正的特殊/控制 op 由引擎显式处理；两者都不认识的 op 必须 loudly ``raise``，绝不静默跳过。
#
# 约定：
#   - 比特序 qubit 0 = 最高有效位（MSB，左端），与 apply_*_statevector / 采样 bitstr 一致。
#   - 2q 门的 4x4 矩阵基序为 |q0 q1>，其中第一个参数 q0 为该对的高位（控制位）。
# ---------------------------------------------------------------------------

# 1q 幺正门：name -> (has_param, builder(theta, backend) -> 2x2)
# 无参门的 builder 忽略 theta；有参门（rx/ry/rz）从 op[2] 取 theta。
UNITARY_1Q: Dict[str, Tuple[bool, Callable[[Any, Any], Any]]] = {
    "h": (False, lambda th, bk: gate_h()),
    "x": (False, lambda th, bk: gate_x(backend=bk)),
    "y": (False, lambda th, bk: gate_y()),
    "z": (False, lambda th, bk: gate_z()),
    "s": (False, lambda th, bk: gate_s()),
    "sdg": (False, lambda th, bk: gate_sd()),
    "t": (False, lambda th, bk: gate_t()),
    "tdg": (False, lambda th, bk: gate_td()),
    "rx": (True, lambda th, bk: gate_rx(th, backend=bk)),
    "ry": (True, lambda th, bk: gate_ry(th, backend=bk)),
    "rz": (True, lambda th, bk: gate_rz(th, backend=bk)),
}

# 2q 幺正门：name -> (has_param, builder(theta, backend) -> 4x4)
# 无参门从 op[1],op[2] 取 (q0,q1)；有参门（cry/rxx/ryy/rzz）另从 op[3] 取 theta。
UNITARY_2Q: Dict[str, Tuple[bool, Callable[[Any, Any], Any]]] = {
    "cx": (False, lambda th, bk: gate_cx_4x4()),
    "cy": (False, lambda th, bk: gate_cy_4x4()),
    "cz": (False, lambda th, bk: gate_cz_4x4()),
    "iswap": (False, lambda th, bk: gate_iswap_4x4()),
    "swap": (False, lambda th, bk: gate_swap_4x4()),
    "cry": (True, lambda th, bk: gate_cry_4x4(th, backend=bk)),
    "rxx": (True, lambda th, bk: gate_rxx(th, backend=bk)),
    "ryy": (True, lambda th, bk: gate_ryy(th, backend=bk)),
    "rzz": (True, lambda th, bk: gate_rzz(th, backend=bk)),
}

# 控制 op：非幺正、不改变量子态，全引擎接受（measure_z 收集测量比特，barrier 空操作）。
CONTROL_OPS = frozenset({"measure_z", "barrier"})

# 特殊 op：非纯幺正门矩阵，需引擎按各自表示专门处理（不支持则 loudly raise）。
SPECIAL_OPS = frozenset(
    {"unitary", "kraus", "project_z", "reset", "pulse", "pulse_inline"}
)

UNITARY_OPS = frozenset(UNITARY_1Q) | frozenset(UNITARY_2Q)

# 全引擎公认"应当支持或明确处置"的 op 全集（护栏测试据此断言分发完备性）。
KNOWN_OPS = UNITARY_OPS | CONTROL_OPS | SPECIAL_OPS


def resolve_unitary(
    name: str, op: Tuple, backend: Any = None
) -> Optional[Tuple[str, Tuple[int, ...], Any]]:
    """把一个幺正 op 解析为 (arity, qubits, matrix)。

    Returns:
        ("1q", (q,), mat2)  或 ("2q", (q0, q1), mat4)；若 name 不是纯幺正门则返回
        None（交由引擎处理控制/特殊 op 或 raise）。
    """
    if name in UNITARY_1Q:
        has_param, build = UNITARY_1Q[name]
        q = int(op[1])
        th = op[2] if has_param else None
        return ("1q", (q,), build(th, backend))
    if name in UNITARY_2Q:
        has_param, build = UNITARY_2Q[name]
        q0, q1 = int(op[1]), int(op[2])
        th = op[3] if has_param else None
        return ("2q", (q0, q1), build(th, backend))
    return None

