"""matrix LUCJ 使用的本地线性代数工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class TwoModeRotation:
    """一条相邻二模 orbital rotation。"""

    modes: tuple[int, int]
    matrix: np.ndarray


def antihermitians_to_parameters(mats: np.ndarray, *, real: bool = False) -> np.ndarray:
    """把一批 anti-Hermitian 矩阵打包成实参数向量。"""
    matrices = np.asarray(mats, dtype=float if real else complex)
    n_mats, dim, _ = matrices.shape
    n_triu = dim * (dim - 1) // 2
    n_params_per_mat = n_triu if real else dim**2
    params = np.zeros((n_mats, n_params_per_mat), dtype=float)

    rows, cols = np.triu_indices(dim, k=1)
    params[:, :n_triu] = matrices[:, rows, cols].real
    if not real:
        rows, cols = np.triu_indices(dim)
        params[:, n_triu:] = matrices[:, rows, cols].imag
    return params.reshape(-1)


def antihermitians_from_parameters(
    params: np.ndarray,
    *,
    dim: int,
    n_mats: int,
    real: bool = False,
) -> np.ndarray:
    """从实参数向量还原一批 anti-Hermitian 矩阵。"""
    n_triu = dim * (dim - 1) // 2
    n_params_per_mat = n_triu if real else dim**2
    values = np.asarray(params, dtype=float).reshape(n_mats, n_params_per_mat)
    mats = np.zeros((n_mats, dim, dim), dtype=float if real else complex)

    if not real:
        rows, cols = np.triu_indices(dim)
        imag_values = 1j * values[:, n_triu:]
        mats[:, rows, cols] = imag_values
        mats[:, cols, rows] = imag_values

    rows, cols = np.triu_indices(dim, k=1)
    real_values = values[:, :n_triu]
    mats[:, rows, cols] += real_values
    mats[:, cols, rows] -= real_values
    return mats


def unitaries_to_parameters(mats: np.ndarray, *, real: bool = False) -> np.ndarray:
    """把一批 unitary 矩阵通过矩阵对数打包成实参数。"""
    matrices = np.asarray(mats, dtype=complex)
    logs = np.stack([scipy.linalg.logm(matrix) for matrix in matrices])
    return antihermitians_to_parameters(logs, real=real)


def unitaries_from_parameters(
    params: np.ndarray,
    *,
    dim: int,
    n_mats: int,
    real: bool = False,
) -> np.ndarray:
    """从实参数向量还原一批 unitary 矩阵。"""
    generators = antihermitians_from_parameters(params, dim=dim, n_mats=n_mats, real=real)
    return np.stack([scipy.linalg.expm(generator) for generator in generators])


def real_symmetrics_to_parameters(
    mats: np.ndarray,
    indices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """把一批实对称矩阵按上三角或指定 pair 打包成实参数。"""
    matrices = np.asarray(mats, dtype=float)
    _, dim, _ = matrices.shape
    if indices is None:
        rows, cols = np.triu_indices(dim)
    else:
        rows, cols = zip(*indices, strict=True)
    return matrices[:, rows, cols].reshape(-1)


def real_symmetrics_from_parameters(
    params: np.ndarray,
    *,
    dim: int,
    n_mats: int,
    indices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """从实参数向量还原一批实对称矩阵。"""
    if indices is None:
        rows, cols = np.triu_indices(dim)
        n_params_per_mat = dim * (dim + 1) // 2
    else:
        rows, cols = zip(*indices, strict=True)
        n_params_per_mat = len(indices)
    values = np.asarray(params, dtype=float).reshape(n_mats, n_params_per_mat)
    mats = np.zeros((n_mats, dim, dim), dtype=float)
    mats[:, rows, cols] = values
    mats[:, cols, rows] = values
    return mats


def df_tensors_to_params(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """把 compressed DF 的 `Z_mu/U_mu` 打包成优化向量。"""
    return np.concatenate(
        [
            unitaries_to_parameters(np.asarray(orbital_rotations, dtype=complex)),
            real_symmetrics_to_parameters(
                np.asarray(diag_coulomb_mats, dtype=float),
                diag_coulomb_indices,
            ),
        ]
    )


def df_tensors_from_params(
    params: np.ndarray,
    *,
    n_tensors: int,
    norb: int,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """从优化向量还原 compressed DF 的 `Z_mu/U_mu`。"""
    n_orbital_params = n_tensors * norb**2
    orbital_rotations = unitaries_from_parameters(
        np.asarray(params[:n_orbital_params], dtype=float),
        dim=norb,
        n_mats=n_tensors,
    )
    diag_coulomb_mats = real_symmetrics_from_parameters(
        np.asarray(params[n_orbital_params:], dtype=float),
        dim=norb,
        n_mats=n_tensors,
        indices=diag_coulomb_indices,
    )
    return diag_coulomb_mats, orbital_rotations


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
