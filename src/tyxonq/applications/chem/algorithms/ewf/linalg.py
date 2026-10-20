"""提供嵌入计算所需的轨道正交化、重叠度量和投影工具。"""

from __future__ import annotations

import numpy as np
import scipy.linalg


def eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对实对称矩阵对角化，并在 NumPy 驱动失败时使用 SciPy。"""
    try:
        return np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        return scipy.linalg.eigh(matrix)


def fix_orbital_sign(coeff: np.ndarray) -> np.ndarray:
    """令每个轨道绝对值最大的 AO 系数为正，消除任意符号差异。"""
    coeff = np.asarray(coeff, dtype=float).copy()
    if coeff.shape[1] == 0:
        return coeff
    rows = np.argmax(np.abs(coeff), axis=0)
    signs = np.sign(coeff[rows, np.arange(coeff.shape[1])])
    signs[signs == 0] = 1.0
    return coeff * signs


def hstack(*blocks: np.ndarray | None, nrow: int | None = None) -> np.ndarray:
    """忽略 ``None`` 并横向拼接轨道系数。"""
    arrays = [np.asarray(x, dtype=float) for x in blocks if x is not None]
    arrays = [x for x in arrays if x.shape[1] > 0]
    if arrays:
        return np.hstack(arrays)
    if nrow is None:
        for block in blocks:
            if block is not None:
                nrow = int(block.shape[0])
                break
    if nrow is None:
        raise ValueError("nrow is required when all orbital blocks are None.")
    return np.zeros((nrow, 0), dtype=float)


def metric_overlap(left: np.ndarray, overlap: np.ndarray, right: np.ndarray) -> np.ndarray:
    """返回 ``left.T @ overlap @ right``。"""
    return np.linalg.multi_dot((left.T, overlap, right))


def orthonormality_error(coeff: np.ndarray, overlap: np.ndarray) -> float:
    """返回轨道在 AO overlap metric 下的最大正交误差。"""
    if coeff.shape[1] == 0:
        return 0.0
    metric = metric_overlap(coeff, overlap, coeff)
    return float(np.max(np.abs(metric - np.eye(coeff.shape[1]))))


def assert_orthonormal(
    coeff: np.ndarray,
    overlap: np.ndarray,
    *,
    name: str,
    tolerance: float = 1e-8,
) -> None:
    """验证一组轨道正交归一。"""
    error = orthonormality_error(coeff, overlap)
    if error > tolerance:
        raise RuntimeError(f"{name} are not S-orthonormal: max error={error:.3e}.")


def symmetric_orthogonalizer(
    coeff: np.ndarray,
    overlap: np.ndarray,
    *,
    tolerance: float = 1e-15,
) -> tuple[np.ndarray, float]:
    """返回对称 Lowdin 正交化矩阵以及最小 metric 本征值。"""
    metric = metric_overlap(coeff, overlap, coeff)
    values, vectors = eigh(metric)
    minimum = float(np.min(values))
    keep = values >= tolerance
    if np.count_nonzero(keep) != coeff.shape[1]:
        raise RuntimeError(
            "IAO Lowdin metric is rank deficient: "
            f"kept {np.count_nonzero(keep)} of {coeff.shape[1]} orbitals."
        )
    transform = (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T
    return fix_orbital_sign(transform), minimum


def canonicalize_subspace(
    coeff: np.ndarray,
    fock: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在给定轨道子空间内对角化 AO Fock 矩阵。"""
    if coeff.shape[1] == 0:
        empty = np.zeros(0, dtype=float)
        return np.asarray(coeff, dtype=float).copy(), np.zeros((0, 0)), empty
    subspace_fock = coeff.T @ fock @ coeff
    energies, rotation = eigh(subspace_fock)
    canonical = fix_orbital_sign(coeff @ rotation)
    # fix_orbital_sign 改变列符号；重新由 overlap 得到与 canonical 一致的旋转。
    rotation = np.linalg.lstsq(coeff, canonical, rcond=None)[0]
    return canonical, rotation, energies
