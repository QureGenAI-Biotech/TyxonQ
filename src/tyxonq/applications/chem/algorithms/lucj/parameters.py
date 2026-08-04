"""ffsim 风格 matrix LUCJ 参数 shape、计数和校验工具。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .topology import (
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
    topology: str = "square",
    *,
    with_final_orbital_rotation: bool = False,
) -> ParameterShapes:
    """返回 matrix LUCJ 参数数组的 shape。"""
    n = validate_n_orbitals(n_orbitals)
    layer_count = validate_layers(layers)
    normalize_topology(topology)
    shapes: ParameterShapes = {
        "orbital_rotations": (layer_count, n, n),
        "diag_coulomb_mats": (layer_count, 2, n, n),
    }
    if with_final_orbital_rotation:
        shapes["final_orbital_rotation"] = (n, n)
    return shapes


def lucj_parameter_count(
    n_orbitals: int,
    layers: int,
    topology: str = "square",
    *,
    with_final_orbital_rotation: bool = False,
) -> int:
    """返回 ffsim matrix UCJ 的独立实参数数量，不是数组元素总数。"""
    n = validate_n_orbitals(n_orbitals)
    layer_count = validate_layers(layers)
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(n, topology)
    total = layer_count * (n**2 + len(pairs_aa) + len(pairs_ab))
    if with_final_orbital_rotation:
        total += n**2
    return total


def normalize_lucj_params(
    params: Mapping[str, Any],
    n_orbitals: int,
    layers: int,
    topology: str = "square",
) -> NormalizedParameters:
    """校验并规范化 matrix LUCJ 参数字典。"""
    if not isinstance(params, Mapping):
        raise ValueError("LUCJ params must be a mapping")

    n = validate_n_orbitals(n_orbitals)
    layer_count = validate_layers(layers)
    name = normalize_topology(topology)
    shapes = lucj_parameter_shapes(n, layer_count, name)

    normalized: NormalizedParameters = {}
    orbital_rotations = _as_complex_array(
        params,
        "orbital_rotations",
        shapes["orbital_rotations"],
    )
    for layer, matrix in enumerate(orbital_rotations):
        _validate_unitary(matrix, f"orbital_rotations[{layer}]")
    normalized["orbital_rotations"] = orbital_rotations

    diag_coulomb_mats = _as_real_array(
        params,
        "diag_coulomb_mats",
        shapes["diag_coulomb_mats"],
    )
    for layer in range(layer_count):
        _validate_real_symmetric(diag_coulomb_mats[layer, 0], f"diag_coulomb_mats[{layer},0]")
        _validate_real_symmetric(diag_coulomb_mats[layer, 1], f"diag_coulomb_mats[{layer},1]")
    normalized["diag_coulomb_mats"] = diag_coulomb_mats

    final = params.get("final_orbital_rotation")
    if final is None:
        normalized["final_orbital_rotation"] = None
    else:
        final_matrix = np.asarray(final, dtype=complex)
        expected_shape = (n, n)
        if final_matrix.shape != expected_shape:
            raise ValueError(
                "Invalid shape for final_orbital_rotation: "
                f"expected {expected_shape}, got {final_matrix.shape}"
            )
        _validate_unitary(final_matrix, "final_orbital_rotation")
        normalized["final_orbital_rotation"] = final_matrix

    return normalized


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
