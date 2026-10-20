"""把 compressed DF 结果打包成 matrix LUCJ 参数字典。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .double_factorization import DoubleFactorizationResult
from .parameters import NormalizedParameters, normalize_lucj_params
from .topology import InteractionPairs, normalize_topology, validate_layers, validate_n_orbitals


def factors_to_lucj_parameters(
    factors: DoubleFactorizationResult,
    n_spatial_orbitals: int,
    n_layers: int,
    topology: str = "square",
    *,
    interaction_pairs: InteractionPairs | None = None,
    excluded_orbitals: Iterable[int] | tuple[Iterable[int], Iterable[int]] | None = None,
) -> NormalizedParameters:
    """把 `U_mu/J_mu` 因子打包成 builder 可消费的 matrix UCJ 参数。"""
    n = validate_n_orbitals(n_spatial_orbitals)
    layers = validate_layers(n_layers)
    name = normalize_topology(topology)
    if factors.n_spatial_orbitals != n:
        raise ValueError(
            "Factor orbital dimension does not match n_spatial_orbitals; "
            f"got factors.N={factors.n_spatial_orbitals}, N={n}"
        )
    params = {
        "orbital_rotations": np.asarray(factors.orbital_rotations, dtype=complex)[:layers],
        "diag_coulomb_mats": np.asarray(factors.diag_coulomb_mats, dtype=float)[:layers],
        "final_orbital_rotation": factors.final_orbital_rotation,
    }
    spin_unbalanced = np.asarray(factors.orbital_rotations).ndim == 4
    return normalize_lucj_params(
        params,
        n,
        layers,
        name,
        spin_unbalanced=spin_unbalanced,
        interaction_pairs=(factors.interaction_pairs if interaction_pairs is None else interaction_pairs),
        excluded_orbitals=excluded_orbitals,
    )
