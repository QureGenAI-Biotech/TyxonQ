"""matrix LUCJ 使用的本地线性代数工具。"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from ...chem_libs.fermionic_circuits import (
    TwoModeRotation as TwoModeRotation,
    decompose_unitary_to_adjacent_rotations as decompose_unitary_to_adjacent_rotations,
    decompose_slater_determinant_to_adjacent_rotations as decompose_slater_determinant_to_adjacent_rotations,
    two_mode_fock_matrix as two_mode_fock_matrix,
    _zeroing_givens as _zeroing_givens,
)


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
        rows, cols = [p for p, _ in indices], [q for _, q in indices]
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
        rows, cols = [p for p, _ in indices], [q for _, q in indices]
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
