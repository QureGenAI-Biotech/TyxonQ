"""定义矩阵 LUCJ 参数的形状、计数、规范化及有效性检查。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from operator import index
from typing import Any

import numpy as np

from .topology import (
    InteractionPairs,
    interaction_pairs_spin_balanced,
    normalize_topology,
    validate_layers,
    validate_n_orbitals,
)


ParameterShapes = dict[str, tuple[int, ...]]
NormalizedParameters = dict[str, np.ndarray | None]


def lucj_parameter_shapes(
    n_orbitals: int,
    layers: int,
    topology: str = "linear",
    *,
    interaction_pairs: InteractionPairs | None = None,
    spin_unbalanced: bool = False,
    with_final_orbital_rotation: bool = False,
) -> ParameterShapes:
    """返回 matrix LUCJ 参数数组的 shape。"""
    n = validate_n_orbitals(n_orbitals)
    layer_count = validate_layers(layers)
    interaction_pairs_spin_balanced(n, topology, interaction_pairs, ordered_ab=spin_unbalanced)
    if spin_unbalanced:
        shapes: ParameterShapes = {
            "orbital_rotations": (layer_count, 2, n, n),
            "diag_coulomb_mats": (layer_count, 3, n, n),
        }
    else:
        shapes = {
            "orbital_rotations": (layer_count, n, n),
            "diag_coulomb_mats": (layer_count, 2, n, n),
        }
    if with_final_orbital_rotation:
        shapes["final_orbital_rotation"] = (2, n, n) if spin_unbalanced else (n, n)
    return shapes


def lucj_parameter_count(
    n_orbitals: int,
    layers: int,
    topology: str = "linear",
    *,
    interaction_pairs: InteractionPairs | None = None,
    spin_unbalanced: bool = False,
    with_final_orbital_rotation: bool = False,
) -> int:
    """返回 ffsim matrix UCJ 的独立实参数数量，不是数组元素总数。"""
    n = validate_n_orbitals(n_orbitals)
    layer_count = validate_layers(layers)
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        n, topology, interaction_pairs, ordered_ab=spin_unbalanced
    )
    if spin_unbalanced:
        total = layer_count * (2 * n**2 + 2 * len(pairs_aa) + len(pairs_ab))
    else:
        total = layer_count * (n**2 + len(pairs_aa) + len(pairs_ab))
    if with_final_orbital_rotation:
        total += (2 if spin_unbalanced else 1) * n**2
    return total


def normalize_lucj_params(
    params: Mapping[str, Any],
    n_orbitals: int,
    layers: int,
    topology: str = "linear",
    *,
    interaction_pairs: InteractionPairs | None = None,
    spin_unbalanced: bool | None = None,
    excluded_orbitals: Iterable[int] | tuple[Iterable[int], Iterable[int]] | None = None,
) -> NormalizedParameters:
    """校验并规范化 matrix LUCJ 参数字典。"""
    if not isinstance(params, Mapping):
        raise ValueError("LUCJ params must be a mapping")

    n = validate_n_orbitals(n_orbitals)
    layer_count = validate_layers(layers)
    name = normalize_topology(topology)
    if spin_unbalanced is None:
        # 直接调用该 helper 时，根据明确的 spin 轴判断；builder 会传入显式模式。
        raw_rotations = np.asarray(params.get("orbital_rotations"))
        spin_unbalanced = raw_rotations.ndim == 4 and raw_rotations.shape[1] == 2
    shapes = lucj_parameter_shapes(
        n,
        layer_count,
        name,
        spin_unbalanced=spin_unbalanced,
        interaction_pairs=interaction_pairs,
    )
    excluded = normalize_excluded_orbitals(excluded_orbitals, n)

    normalized: NormalizedParameters = {}
    orbital_rotations = _as_complex_array(
        params,
        "orbital_rotations",
        shapes["orbital_rotations"],
    )
    orbital_rotations = _project_excluded_rotations(
        orbital_rotations,
        excluded,
        spin_unbalanced=spin_unbalanced,
    )
    if spin_unbalanced:
        for layer in range(layer_count):
            for spin_index in range(2):
                _validate_unitary(
                    orbital_rotations[layer, spin_index],
                    f"orbital_rotations[{layer},{spin_index}]",
                )
    else:
        for layer, matrix in enumerate(orbital_rotations):
            _validate_unitary(matrix, f"orbital_rotations[{layer}]")
    normalized["orbital_rotations"] = orbital_rotations

    diag_coulomb_mats = _as_real_array(
        params,
        "diag_coulomb_mats",
        shapes["diag_coulomb_mats"],
    )
    diag_coulomb_mats = _zero_excluded_coulomb_entries(
        diag_coulomb_mats,
        excluded,
        spin_unbalanced=spin_unbalanced,
    )
    for layer in range(layer_count):
        _validate_real_symmetric(diag_coulomb_mats[layer, 0], f"diag_coulomb_mats[{layer},0]")
        if spin_unbalanced:
            _validate_real_symmetric(
                diag_coulomb_mats[layer, 2],
                f"diag_coulomb_mats[{layer},2]",
            )
        else:
            _validate_real_symmetric(
                diag_coulomb_mats[layer, 1],
                f"diag_coulomb_mats[{layer},1]",
            )
    if interaction_pairs is not None:
        pairs_same, pairs_ab = interaction_pairs_spin_balanced(
            n, name, interaction_pairs, ordered_ab=spin_unbalanced
        )
        # 只有显式 pair 才规范化耦合掩码，默认参数数组行为保持兼容。
        for channel in range(3 if spin_unbalanced else 2):
            pairs = pairs_ab if channel == 1 else pairs_same
            mask = np.zeros((n, n), dtype=bool)
            for p, q in pairs:
                mask[p, q] = True
                if channel != 1 or not spin_unbalanced:
                    mask[q, p] = True
            diag_coulomb_mats[:, channel] *= mask
    normalized["diag_coulomb_mats"] = diag_coulomb_mats

    final = params.get("final_orbital_rotation")
    if final is None:
        normalized["final_orbital_rotation"] = None
    else:
        final_matrix = np.asarray(final, dtype=complex)
        expected_shape = (2, n, n) if spin_unbalanced else (n, n)
        if final_matrix.shape != expected_shape:
            raise ValueError(
                "Invalid shape for final_orbital_rotation: "
                f"expected {expected_shape}, got {final_matrix.shape}"
            )
        final_matrix = _project_excluded_rotations(
            final_matrix,
            excluded,
            spin_unbalanced=spin_unbalanced,
        )
        if spin_unbalanced:
            for spin_index in range(2):
                _validate_unitary(
                    final_matrix[spin_index],
                    f"final_orbital_rotation[{spin_index}]",
                )
        else:
            _validate_unitary(final_matrix, "final_orbital_rotation")
        normalized["final_orbital_rotation"] = final_matrix

    return normalized


def normalize_excluded_orbitals(
    excluded_orbitals: Iterable[int] | tuple[Iterable[int], Iterable[int]] | None,
    n_orbitals: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """把共享或分自旋的 dummy 轨道编号规范成 `(alpha, beta)`。"""
    n = validate_n_orbitals(n_orbitals)
    if excluded_orbitals is None:
        return (), ()

    if (
        isinstance(excluded_orbitals, tuple)
        and len(excluded_orbitals) == 2
        and all(not isinstance(values, (str, bytes, int, np.integer)) for values in excluded_orbitals)
    ):
        groups = excluded_orbitals
    else:
        groups = (excluded_orbitals, excluded_orbitals)

    normalized: list[tuple[int, ...]] = []
    for values in groups:
        try:
            raw_values = list(values)
        except TypeError as exc:
            raise ValueError("excluded_orbitals must contain orbital indices") from exc
        indices: list[int] = []
        for value in raw_values:
            try:
                orbital = index(value)
            except TypeError as exc:
                raise ValueError("excluded_orbitals must contain integer indices") from exc
            if orbital < 0 or orbital >= n:
                raise ValueError(f"excluded orbital index out of range: {orbital}")
            indices.append(int(orbital))
        normalized.append(tuple(sorted(set(indices))))
    return normalized[0], normalized[1]


def _project_excluded_rotations(
    rotations: np.ndarray,
    excluded: tuple[tuple[int, ...], tuple[int, ...]],
    *,
    spin_unbalanced: bool,
) -> np.ndarray:
    """把 dummy 轨道从旋转中解耦，并保持物理子块为 unitary。"""
    result = np.asarray(rotations, dtype=complex).copy()
    if not any(excluded):
        return result

    if spin_unbalanced:
        matrices = result.reshape(-1, 2, result.shape[-2], result.shape[-1])
        for pair in matrices:
            for spin_index, indices in enumerate(excluded):
                pair[spin_index] = _project_one_unitary(pair[spin_index], indices)
    else:
        shared_excluded = tuple(sorted(set(excluded[0]) | set(excluded[1])))
        matrices = result.reshape(-1, result.shape[-2], result.shape[-1])
        for matrix_index, matrix in enumerate(matrices):
            matrices[matrix_index] = _project_one_unitary(matrix, shared_excluded)
    return result


def _project_one_unitary(matrix: np.ndarray, excluded: tuple[int, ...]) -> np.ndarray:
    """以最近的 unitary 物理子块替换含 dummy mixing 的矩阵。"""
    if not excluded:
        return np.asarray(matrix, dtype=complex)
    n = matrix.shape[0]
    active = [orbital for orbital in range(n) if orbital not in excluded]
    projected = np.eye(n, dtype=complex)
    if active:
        block = np.asarray(matrix, dtype=complex)[np.ix_(active, active)]
        left, _, right = np.linalg.svd(block)
        projected[np.ix_(active, active)] = left @ right
    return projected


def _zero_excluded_coulomb_entries(
    mats: np.ndarray,
    excluded: tuple[tuple[int, ...], tuple[int, ...]],
    *,
    spin_unbalanced: bool,
) -> np.ndarray:
    """清零所有包含 dummy 轨道的 `aa/ab/bb` 相互作用。"""
    result = np.asarray(mats, dtype=float).copy()
    alpha, beta = excluded
    if spin_unbalanced:
        if alpha:
            result[:, 0, alpha, :] = 0.0
            result[:, 0, :, alpha] = 0.0
            result[:, 1, alpha, :] = 0.0
        if beta:
            result[:, 1, :, beta] = 0.0
            result[:, 2, beta, :] = 0.0
            result[:, 2, :, beta] = 0.0
    else:
        shared = tuple(sorted(set(alpha) | set(beta)))
        if shared:
            result[:, :, shared, :] = 0.0
            result[:, :, :, shared] = 0.0
    return result


def _as_complex_array(
    params: Mapping[str, Any],
    key: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """从参数字典取出 complex 数组并检查 shape。"""
    if key not in params:
        raise ValueError(f"Missing LUCJ parameter {key!r}")
    value = np.asarray(params[key], dtype=complex)
    if value.shape != expected_shape:
        raise ValueError(f"Invalid shape for {key}: expected {expected_shape}, got {value.shape}")
    return value


def _as_real_array(
    params: Mapping[str, Any],
    key: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """从参数字典取出实数组并检查 shape 和虚部。"""
    if key not in params:
        raise ValueError(f"Missing LUCJ parameter {key!r}")
    value = np.asarray(params[key])
    if np.iscomplexobj(value):
        if not np.allclose(value.imag, 0.0):
            raise ValueError(f"{key} must be real-valued")
        value = value.real
    value = np.asarray(value, dtype=float)
    if value.shape != expected_shape:
        raise ValueError(f"Invalid shape for {key}: expected {expected_shape}, got {value.shape}")
    return value


def _validate_unitary(matrix: np.ndarray, label: str) -> None:
    """检查 orbital rotation 是否为 unitary。"""
    eye = np.eye(matrix.shape[0], dtype=complex)
    if not np.allclose(matrix.conj().T @ matrix, eye, rtol=1e-5, atol=1e-8):
        raise ValueError(f"{label} must be unitary")


def _validate_real_symmetric(matrix: np.ndarray, label: str) -> None:
    """检查 diagonal Coulomb 矩阵是否为实对称矩阵。"""
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{label} must be real symmetric")
