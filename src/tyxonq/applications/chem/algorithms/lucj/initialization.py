"""matrix LUCJ 初始参数生成入口。"""

from __future__ import annotations

from .conversion import factors_to_lucj_parameters
from .double_factorization import double_factorize_t2
from .parameters import NormalizedParameters


def initialize_lucj_parameters_from_ccsd(
    t2_amplitudes,
    *,
    t1=None,
    n_spatial_orbitals: int,
    n_layers: int,
    topology: str = "square",
    optimize: bool = True,
    regularization: float = 0.0,
    maxiter: int = 100,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
) -> NormalizedParameters:
    """从 CCSD `t1/t2` 振幅生成 matrix LUCJ 初始参数。"""
    factors = double_factorize_t2(
        t2_amplitudes,
        n_spatial_orbitals,
        n_layers,
        topology,
        t1_amplitudes=t1,
        optimize=optimize,
        regularization=regularization,
        maxiter=maxiter,
        multi_stage_start=multi_stage_start,
        multi_stage_step=multi_stage_step,
    )
    return factors_to_lucj_parameters(
        factors,
        n_spatial_orbitals,
        n_layers,
        topology,
    )
