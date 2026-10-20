"""LUCJ 和 SKQD 共用的费米子 Gaussian 数学与原生门。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TwoModeRotation:
    """一条相邻二模 orbital rotation。"""

    modes: tuple[int, int]
    matrix: np.ndarray


def decompose_unitary_to_adjacent_rotations(unitary: np.ndarray) -> tuple[np.ndarray, list[TwoModeRotation]]:
    """把 full unitary 分解成 diagonal phase 和相邻二模 rotation。"""
    matrix = np.asarray(unitary, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("unitary must have shape (N, N)")
    n = matrix.shape[0]
    work = matrix.copy()
    left_rotations: list[TwoModeRotation] = []

    # 逐列用相邻 Givens 消元：G_m ... G_1 U = D。
    for col in range(n - 1):
        for row in range(n - 1, col, -1):
            i = row - 1
            j = row
            g_matrix = _zeroing_givens(work[i, col], work[j, col])
            work[[i, j], :] = g_matrix @ work[[i, j], :]
            left_rotations.append(TwoModeRotation((i, j), g_matrix))

    phases = np.diag(work).copy()
    phase_norms = np.abs(phases)
    phases = np.where(phase_norms > 0, phases / phase_norms, 1.0 + 0.0j)

    # U = G_1† ... G_m† D；线路先放 D，再按反序放 G_m† ... G_1†。
    rotations = [
        TwoModeRotation(rotation.modes, rotation.matrix.conj().T)
        for rotation in reversed(left_rotations)
    ]
    return phases, rotations


def decompose_slater_determinant_to_adjacent_rotations(
    orbital_coeffs: np.ndarray,
    *,
    tol: float = 1e-12,
) -> list[TwoModeRotation]:
    """把 Slater determinant 系数矩阵分解成相邻二模 rotation。

    输入 shape 为 ``(n_occupied, n_orbitals)``。与分解完整 unitary 相比，
    这里只保留真正改变已占据子空间的 rotation，最多需要
    ``n_occupied * (n_orbitals - n_occupied)`` 个门。
    """
    work = np.asarray(orbital_coeffs, dtype=complex).copy()
    if work.ndim != 2:
        raise ValueError("orbital_coeffs must have shape (n_occupied, n_orbitals)")
    n_occupied, n_orbitals = work.shape
    if n_occupied > n_orbitals:
        raise ValueError("n_occupied must be <= n_orbitals")

    # 先用行旋转消去右上角；行旋转只改变已占据轨道的基，不改变 Slater 态。
    if n_orbitals > n_occupied:
        n_virtual = n_orbitals - n_occupied
        for col in range(n_orbitals - 1, n_virtual, -1):
            for row in range(col - n_virtual):
                if abs(work[row, col]) <= tol:
                    continue
                givens = _zeroing_givens(work[row + 1, col], work[row, col])
                work[[row + 1, row], :] = givens @ work[[row + 1, row], :]

    rotations: list[TwoModeRotation] = []
    for row in range(n_occupied):
        last_col = n_orbitals - n_occupied + row
        for col in range(last_col, row, -1):
            if abs(work[row, col]) <= tol:
                continue
            givens = _zeroing_givens(work[row, col - 1], work[row, col])
            work[:, [col - 1, col]] = work[:, [col - 1, col]] @ givens.T
            c = float(givens[0, 0].real)
            s = complex(givens[0, 1])
            rotations.append(
                TwoModeRotation(
                    (col - 1, col),
                    np.array([[c, -s], [np.conjugate(s), c]], dtype=complex),
                )
            )

    rotations.reverse()
    return rotations


def two_mode_fock_matrix(single_particle_matrix: np.ndarray) -> np.ndarray:
    """把 2x2 orbital rotation 提升成两个 qubit 上的 4x4 Fock unitary。

    qubit 顺序为 `(mode_i, mode_j)`，局部 basis 为 `|00>, |01>, |10>, |11>`。
    """
    v = np.asarray(single_particle_matrix, dtype=complex)
    if v.shape != (2, 2):
        raise ValueError("single_particle_matrix must have shape (2, 2)")
    matrix = np.eye(4, dtype=complex)
    matrix[1, 1] = v[1, 1]
    matrix[1, 2] = v[1, 0]
    matrix[2, 1] = v[0, 1]
    matrix[2, 2] = v[0, 0]
    matrix[3, 3] = v[0, 0] * v[1, 1] - v[0, 1] * v[1, 0]
    return matrix


def _zeroing_givens(a: complex, b: complex) -> np.ndarray:
    """构造左乘后把 `[a, b]` 第二个分量消为 0 的 Givens 矩阵。"""
    radius = float(np.hypot(abs(a), abs(b)))
    if radius == 0:
        return np.eye(2, dtype=complex)
    if abs(a) == 0:
        c = 0.0
        s = 1.0 + 0.0j
    else:
        c = abs(a) / radius
        s = c * np.conjugate(b / a)
    return np.array([[c, s], [-np.conjugate(s), c]], dtype=complex)


def append_orbital_rotation(circuit, rotation, modes):
    """在一条连续自旋链上追加 Gaussian 酉，整体相位可忽略。"""
    modes = tuple(modes)
    rotation = np.asarray(rotation, complex)
    if rotation.shape != (len(modes), len(modes)) or not np.allclose(
        rotation.conj().T @ rotation, np.eye(len(modes)), atol=1e-10, rtol=0
    ):
        raise ValueError("rotation must be unitary and match the mode count")
    if any(b != a + 1 for a, b in zip(modes, modes[1:])):
        raise ValueError("orbital rotation requires consecutive JW modes")
    phases, rotations = decompose_unitary_to_adjacent_rotations(rotation)
    for mode, phase in zip(modes, phases):
        if abs(np.angle(phase)) > 1e-14:
            circuit.rz(mode, float(np.angle(phase)))
    for item in rotations:
        if not np.allclose(item.matrix, np.eye(2), atol=1e-14, rtol=0):
            circuit.unitary(*(modes[i] for i in item.modes),
                            matrix=two_mode_fock_matrix(item.matrix))


def append_cphase(circuit, p, q, angle):
    """CP(angle) 的原生 RZ/RZZ 分解，等价到整体相位。"""
    circuit.rz(p, angle / 2)
    circuit.rz(q, angle / 2)
    circuit.rzz(p, q, -angle / 2)


def append_pauli_evolution(circuit, pauli, angle):
    """追加 exp(-i*angle*P)，保留 JW Z 链及原始 qubit 索引。"""
    if not pauli or abs(angle) < 1e-15:
        return
    for mode, axis in pauli:
        if axis == "X":
            circuit.h(mode)
        elif axis == "Y":
            circuit.sdg(mode)
            circuit.h(mode)
    modes = [p for p, _ in pauli]
    for p, q in zip(modes, modes[1:]):
        circuit.cx(p, q)
    circuit.rz(modes[-1], 2 * float(angle))
    for p, q in reversed(list(zip(modes, modes[1:]))):
        circuit.cx(p, q)
    for mode, axis in reversed(pauli):
        if axis == "X":
            circuit.h(mode)
        elif axis == "Y":
            circuit.h(mode)
            circuit.s(mode)
