"""matrix LUCJ 初始参数生成入口。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .conversion import factors_to_lucj_parameters
from .double_factorization import DoubleFactorizationResult, double_factorize_t2
from .parameters import NormalizedParameters
from .topology import InteractionPairs


def initialize_lucj_parameters_from_ccsd(
    t2_amplitudes,
    *,
    t1=None,
    n_spatial_orbitals: int,
    n_layers: int,
    topology: str = "linear",
    interaction_pairs: InteractionPairs | None = None,
    optimize: bool = True,
    regularization: float = 0.0,
    maxiter: int = 500,
    optimizer_options: Mapping[str, Any] | None = None,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    callback=None,
    excluded_orbitals: Iterable[int] | tuple[Iterable[int], Iterable[int]] | None = None,
    return_factorization_result: bool = False,
) -> NormalizedParameters | tuple[NormalizedParameters, DoubleFactorizationResult]:
    """从 RCCSD 数组或 UCCSD α/β tuple 生成 matrix LUCJ 初始参数。

    ``interaction_pairs=(pairs_same, pairs_ab)`` 显式覆盖默认 topology。
    拟合与构造线路必须使用同一组 pair；论文闭壳层设置为同自旋相邻轨道，
    异自旋 ``[(p, p) for p in (0, 4, 8, 12) if p < n_spatial_orbitals]``。
    """
    factors = double_factorize_t2(
        t2_amplitudes,
        n_spatial_orbitals,
        n_layers,
        topology,
        t1_amplitudes=t1,
        interaction_pairs=interaction_pairs,
        optimize=optimize,
        regularization=regularization,
        maxiter=maxiter,
        optimizer_options=optimizer_options,
        multi_stage_start=multi_stage_start,
        multi_stage_step=multi_stage_step,
        callback=callback,
    )
    params = factors_to_lucj_parameters(
        factors,
        n_spatial_orbitals,
        n_layers,
        topology,
        excluded_orbitals=excluded_orbitals,
        interaction_pairs=interaction_pairs,
    )
    if return_factorization_result:
        return params, factors
    return params
