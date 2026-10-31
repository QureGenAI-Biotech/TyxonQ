"""LUCJ 的 compressed double factorization 初始化数学工具。"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product, zip_longest
from typing import Any

import numpy as np
import scipy.linalg

from .linalg import (
    df_tensors_from_params,
    df_tensors_to_params,
    real_symmetrics_from_parameters,
    real_symmetrics_to_parameters,
    unitaries_from_parameters,
    unitaries_to_parameters,
)
from .topology import (
    InteractionPairs,
    allowed_diag_coulomb_indices,
    interaction_pairs_spin_balanced,
    normalize_topology,
    validate_interaction_pairs,
    validate_layers,
    validate_n_orbitals,
)


def _callback_accepts_intermediate_result(callback) -> bool:
    """复现 SciPy 对 ``callback(intermediate_result=...)`` 的签名约定。"""
    if callback is None:
        return False
    try:
        return set(inspect.signature(callback).parameters) == {"intermediate_result"}
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class PreparedT2:
    """整理后的 restricted closed-shell CCSD `t2` 数据。"""

    amplitudes: np.ndarray
    nocc: int
    nvir: int
    n_spatial_orbitals: int


@dataclass(frozen=True)
class PreparedT2Unrestricted:
    """整理后的 UCCSD `(t2aa, t2ab, t2bb)` 数据。"""

    amplitudes: tuple[np.ndarray, np.ndarray, np.ndarray]
    nocc: tuple[int, int]
    nvir: tuple[int, int]
    n_spatial_orbitals: int


@dataclass(frozen=True)
class DoubleFactorizationResult:
    """compressed DF 初始化的结果。"""

    orbital_rotations: np.ndarray
    diagonal_coulomb_matrices: np.ndarray
    diag_coulomb_mats: np.ndarray
    final_orbital_rotation: np.ndarray | None
    nocc: int | tuple[int, int]
    nvir: int | tuple[int, int]
    n_spatial_orbitals: int
    topology: str
    loss: float
    optimize_result: Any | None = None
    interaction_pairs: InteractionPairs | None = None


def prepare_t2_amplitudes(t2_amplitudes, n_spatial_orbitals: int) -> PreparedT2:
    """校验并整理 CCSD `t2` 振幅。"""
    n = validate_n_orbitals(n_spatial_orbitals)
    array = np.asarray(t2_amplitudes)
    if np.iscomplexobj(array):
        if not np.allclose(array.imag, 0.0):
            raise ValueError("t2_amplitudes must be real-valued")
        array = array.real
    amplitudes = np.asarray(array, dtype=float)
    if amplitudes.ndim != 4:
        raise ValueError("t2_amplitudes must have shape (nocc, nocc, nvir, nvir)")
    nocc0, nocc1, nvir0, nvir1 = amplitudes.shape
    if nocc0 != nocc1 or nvir0 != nvir1:
        raise ValueError(
            "t2_amplitudes must have shape (nocc, nocc, nvir, nvir); "
            f"got {amplitudes.shape}"
        )
    if nocc0 + nvir0 != n:
        raise ValueError(
            "n_spatial_orbitals must equal nocc + nvir for t2_amplitudes; "
            f"got N={n}, nocc={nocc0}, nvir={nvir0}"
        )
    return PreparedT2(amplitudes, nocc0, nvir0, n)


def prepare_t2_amplitudes_unrestricted(
    t2_amplitudes,
    n_spatial_orbitals: int,
) -> PreparedT2Unrestricted:
    """校验 UCCSD 的 `aa/ab/bb` 振幅与公共空间轨道数。"""
    n = validate_n_orbitals(n_spatial_orbitals)
    if not isinstance(t2_amplitudes, tuple) or len(t2_amplitudes) != 3:
        raise ValueError("unrestricted t2_amplitudes must be a (t2aa, t2ab, t2bb) tuple")
    t2aa, t2ab, t2bb = (_as_real_t2(value) for value in t2_amplitudes)
    nocc_a, nocc_b, nvir_a, nvir_b = t2ab.shape
    if t2aa.shape != (nocc_a, nocc_a, nvir_a, nvir_a):
        raise ValueError(
            "t2aa shape must agree with the alpha dimensions of t2ab; "
            f"got {t2aa.shape} and {t2ab.shape}"
        )
    if t2bb.shape != (nocc_b, nocc_b, nvir_b, nvir_b):
        raise ValueError(
            "t2bb shape must agree with the beta dimensions of t2ab; "
            f"got {t2bb.shape} and {t2ab.shape}"
        )
    if nocc_a + nvir_a != n or nocc_b + nvir_b != n:
        raise ValueError(
            "each spin sector must satisfy nocc + nvir == n_spatial_orbitals; "
            f"got alpha=({nocc_a},{nvir_a}), beta=({nocc_b},{nvir_b}), N={n}"
        )
    return PreparedT2Unrestricted(
        (t2aa, t2ab, t2bb),
        (nocc_a, nocc_b),
        (nvir_a, nvir_b),
        n,
    )


def _as_real_t2(value) -> np.ndarray:
    """把单个 t2 block 转成实 rank-4 数组。"""
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if not np.allclose(array.imag, 0.0):
            raise ValueError("t2_amplitudes must be real-valued")
        array = array.real
    result = np.asarray(array, dtype=float)
    if result.ndim != 4:
        raise ValueError("each unrestricted t2 block must be a rank-4 array")
    return result


def orbital_rotation_from_t1_amplitudes(t1_amplitudes) -> np.ndarray:
    """按 ffsim 约定由 CCSD `t1` 构造 final orbital rotation。"""
    if t1_amplitudes is None:
        raise ValueError("t1_amplitudes must not be None")
    t1 = np.asarray(t1_amplitudes)
    if np.iscomplexobj(t1):
        if not np.allclose(t1.imag, 0.0):
            raise ValueError("t1_amplitudes must be real-valued")
        t1 = t1.real
    t1 = np.asarray(t1, dtype=float)
    if t1.ndim != 2:
        raise ValueError("t1_amplitudes must have shape (nocc, nvir)")
    nocc, nvir = t1.shape
    norb = nocc + nvir
    generator = np.zeros((norb, norb), dtype=float)
    generator[:nocc, nocc:] = -t1
    generator[nocc:, :nocc] = t1.T
    return scipy.linalg.expm(generator)


def double_factorized_t2(
    t2_amplitudes,
    *,
    tol: float = 1e-8,
    max_terms: int | None = None,
    optimize: bool = False,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    diag_coulomb_indices: list[tuple[int, int]] | None = None,
    regularization: float = 0.0,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    return_optimize_result: bool = False,
    ):
    """ffsim 风格的 restricted `t2` double factorization。"""
    if max_terms is not None and max_terms < 1:
        raise ValueError(f"max_terms must be at least 1. Got {max_terms}.")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if diag_coulomb_indices is not None:
        validate_interaction_pairs(diag_coulomb_indices, ordered=False)

    if not optimize:
        return _double_factorized_t2_explicit(t2_amplitudes, tol=tol, max_terms=max_terms)
    return _double_factorized_t2_compressed(
        t2_amplitudes,
        tol=tol,
        max_terms=max_terms,
        method=method,
        callback=callback,
        options=options,
        diag_coulomb_indices=diag_coulomb_indices,
        regularization=regularization,
        multi_stage_start=multi_stage_start,
        multi_stage_step=multi_stage_step,
        return_optimize_result=return_optimize_result,
    )


def double_factorize_t2(
    t2_amplitudes,
    n_spatial_orbitals: int,
    n_layers: int,
    topology: str = "square",
    *,
    interaction_pairs: InteractionPairs | None = None,
    t1_amplitudes=None,
    optimize: bool = True,
    regularization: float = 0.0,
    maxiter: int = 500,
    optimizer_options: Mapping[str, Any] | None = None,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    callback=None,
) -> DoubleFactorizationResult:
    """从 CCSD `t1/t2` 生成 matrix UCJ 初始化参数。"""
    if isinstance(t2_amplitudes, tuple):
        return double_factorize_t2_unrestricted(
            t2_amplitudes,
            n_spatial_orbitals,
            n_layers,
            topology,
            t1_amplitudes=t1_amplitudes,
            interaction_pairs=interaction_pairs,
            optimize=optimize,
            regularization=regularization,
            maxiter=maxiter,
            optimizer_options=optimizer_options,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
            callback=callback,
        )
    prepared = prepare_t2_amplitudes(t2_amplitudes, n_spatial_orbitals)
    layers = validate_layers(n_layers)
    name = normalize_topology(topology)
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(prepared.n_spatial_orbitals, name, interaction_pairs)
    diag_indices = allowed_diag_coulomb_indices(prepared.n_spatial_orbitals, name, interaction_pairs)

    options = dict(optimizer_options or {})
    options["maxiter"] = int(maxiter)
    factorization = double_factorized_t2(
        prepared.amplitudes,
        max_terms=layers,
        optimize=optimize,
        callback=callback,
        options=options,
        diag_coulomb_indices=diag_indices,
        regularization=regularization,
        multi_stage_start=multi_stage_start,
        multi_stage_step=multi_stage_step,
        return_optimize_result=optimize,
    )
    optimize_result = None
    if optimize:
        diagonal_coulomb_matrices, orbital_rotations, optimize_result = factorization
    else:
        diagonal_coulomb_matrices, orbital_rotations = factorization

    diagonal_coulomb_matrices, orbital_rotations = _pad_factorization(
        diagonal_coulomb_matrices,
        orbital_rotations,
        layers,
        prepared.n_spatial_orbitals,
    )
    diag_coulomb_mats = _spin_balanced_mats_from_shared_z(
        diagonal_coulomb_matrices,
        pairs_aa=pairs_aa,
        pairs_ab=pairs_ab,
    )
    final_orbital_rotation = (
        None if t1_amplitudes is None else orbital_rotation_from_t1_amplitudes(t1_amplitudes)
    )
    loss = t2_reconstruction_loss(
        prepared.amplitudes,
        orbital_rotations,
        diagonal_coulomb_matrices,
    )
    return DoubleFactorizationResult(
        orbital_rotations=orbital_rotations,
        diagonal_coulomb_matrices=diagonal_coulomb_matrices,
        diag_coulomb_mats=diag_coulomb_mats,
        final_orbital_rotation=final_orbital_rotation,
        nocc=prepared.nocc,
        nvir=prepared.nvir,
        n_spatial_orbitals=prepared.n_spatial_orbitals,
        topology=name,
        loss=loss,
        optimize_result=optimize_result,
        interaction_pairs=interaction_pairs,
    )


def double_factorize_t2_unrestricted(
    t2_amplitudes,
    n_spatial_orbitals: int,
    n_layers: int,
    topology: str = "square",
    *,
    interaction_pairs: InteractionPairs | None = None,
    t1_amplitudes=None,
    optimize: bool = True,
    regularization: float = 0.0,
    maxiter: int = 500,
    optimizer_options: Mapping[str, Any] | None = None,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    callback=None,
) -> DoubleFactorizationResult:
    """按 ffsim 顺序先取 αβ 因子，再补充 αα/ββ 因子。"""
    prepared = prepare_t2_amplitudes_unrestricted(t2_amplitudes, n_spatial_orbitals)
    layers = validate_layers(n_layers)
    name = normalize_topology(topology)
    pairs_same, pairs_ab = interaction_pairs_spin_balanced(
        prepared.n_spatial_orbitals, name, interaction_pairs, ordered_ab=True
    )
    factor_pairs = (pairs_same, pairs_ab, pairs_same)
    t2aa, t2ab, t2bb = prepared.amplitudes
    options = dict(optimizer_options or {})
    options["maxiter"] = int(maxiter)

    optimize_results: list[Any | None] = []
    # αβ 因子优先占用有限层数，这是 ffsim spin-unbalanced UCJ 的约定。
    alpha_beta_result = double_factorized_t2_alpha_beta(
        t2ab,
        max_terms=layers,
        optimize=optimize,
        callback=callback,
        options=options,
        diag_coulomb_indices=factor_pairs,
        regularization=regularization,
        multi_stage_start=multi_stage_start,
        multi_stage_step=multi_stage_step,
        return_optimize_result=optimize,
    )
    if optimize:
        diag_ab, rotations_ab, optimize_result = alpha_beta_result
        optimize_results.append(optimize_result)
    else:
        diag_ab, rotations_ab = alpha_beta_result
    remaining = layers - len(diag_ab)
    diag_same = np.empty((0, 3, prepared.n_spatial_orbitals, prepared.n_spatial_orbitals))
    rotations_same = np.empty(
        (0, 2, prepared.n_spatial_orbitals, prepared.n_spatial_orbitals),
        dtype=complex,
    )
    if remaining > 0:
        same_spin_results = []
        for block in (t2aa, t2bb):
            result = double_factorized_t2(
                block,
                max_terms=remaining,
                optimize=optimize,
                options=options,
                callback=callback,
                diag_coulomb_indices=pairs_same,
                regularization=regularization,
                multi_stage_start=multi_stage_start,
                multi_stage_step=multi_stage_step,
                return_optimize_result=optimize,
            )
            if optimize:
                diag, rotations, optimize_result = result
                optimize_results.append(optimize_result)
            else:
                diag, rotations = result
            same_spin_results.append((diag, rotations))

        (diag_aa, rotations_aa), (diag_bb, rotations_bb) = same_spin_results
        zero = np.zeros((prepared.n_spatial_orbitals, prepared.n_spatial_orbitals))
        eye = np.eye(prepared.n_spatial_orbitals, dtype=complex)
        diag_rows = [
            np.stack((mat_aa, zero, mat_bb))
            for mat_aa, mat_bb in zip_longest(diag_aa, diag_bb, fillvalue=zero)
        ]
        rotation_rows = [
            np.stack((rotation_aa, rotation_bb))
            for rotation_aa, rotation_bb in zip_longest(
                rotations_aa,
                rotations_bb,
                fillvalue=eye,
            )
        ]
        if diag_rows:
            diag_same = np.stack(diag_rows)
            rotations_same = np.stack(rotation_rows)

    diag = np.concatenate((diag_ab, diag_same), axis=0)[:layers]
    rotations = np.concatenate((rotations_ab, rotations_same), axis=0)[:layers]
    diag, rotations = _pad_unrestricted_factorization(
        diag,
        rotations,
        layers,
        prepared.n_spatial_orbitals,
    )

    final_orbital_rotation = None
    if t1_amplitudes is not None:
        if not isinstance(t1_amplitudes, tuple) or len(t1_amplitudes) != 2:
            raise ValueError("unrestricted t1_amplitudes must be a (t1a, t1b) tuple")
        final_orbital_rotation = np.stack(
            [orbital_rotation_from_t1_amplitudes(block) for block in t1_amplitudes]
        )
        if final_orbital_rotation.shape != (
            2,
            prepared.n_spatial_orbitals,
            prepared.n_spatial_orbitals,
        ):
            raise ValueError("unrestricted t1 dimensions must match n_spatial_orbitals")

    reconstructed_aa = reconstruct_t2_from_factors(
        rotations[:, 0],
        diag[:, 0],
        prepared.nocc[0],
        prepared.nvir[0],
    )
    reconstructed_ab = reconstruct_t2_alpha_beta_from_factors(
        rotations,
        diag,
        prepared.nocc[0],
        prepared.nocc[1],
    )
    reconstructed_bb = reconstruct_t2_from_factors(
        rotations[:, 1],
        diag[:, 2],
        prepared.nocc[1],
        prepared.nvir[1],
    )
    loss = 0.5 * sum(
        float(np.vdot(current - target, current - target).real)
        for current, target in (
            (reconstructed_aa, t2aa),
            (reconstructed_ab, t2ab),
            (reconstructed_bb, t2bb),
        )
    )
    return DoubleFactorizationResult(
        orbital_rotations=rotations,
        diagonal_coulomb_matrices=diag,
        diag_coulomb_mats=diag,
        final_orbital_rotation=final_orbital_rotation,
        nocc=prepared.nocc,
        nvir=prepared.nvir,
        n_spatial_orbitals=prepared.n_spatial_orbitals,
        topology=name,
        loss=loss,
        optimize_result=tuple(optimize_results) if optimize_results else None,
        interaction_pairs=interaction_pairs,
    )


def double_factorized_t2_alpha_beta(
    t2_amplitudes,
    *,
    tol: float = 1e-8,
    max_terms: int | None = None,
    optimize: bool = False,
    method: str = "L-BFGS-B",
    callback=None,
    options: dict | None = None,
    diag_coulomb_indices: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ]
    | None = None,
    regularization: float = 0.0,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
    return_optimize_result: bool = False,
):
    """对实数 αβ 双激发振幅执行显式双重分解及可选压缩拟合。"""
    if max_terms is not None and max_terms < 1:
        raise ValueError(f"max_terms must be at least 1. Got {max_terms}.")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if diag_coulomb_indices is None:
        pairs_aa = pairs_ab = pairs_bb = None
    else:
        pairs_aa, pairs_ab, pairs_bb = diag_coulomb_indices
        validate_interaction_pairs(pairs_aa, ordered=False)
        # αβ block 是有向矩阵，因此允许下三角 pair；这里只检查重复项。
        validate_interaction_pairs(pairs_ab, ordered=True)
        validate_interaction_pairs(pairs_bb, ordered=False)

    t2 = _as_real_t2(t2_amplitudes)
    nocc_a, nocc_b, nvir_a, nvir_b = t2.shape
    if nocc_a + nvir_a != nocc_b + nvir_b:
        raise ValueError("alpha and beta t2 dimensions must define the same n_orbitals")
    norb = nocc_a + nvir_a
    if not t2.size or not np.any(t2):
        full_diag = np.empty((0, 3, norb, norb), dtype=float)
        full_rotations = np.empty((0, 2, norb, norb), dtype=complex)
    else:
        t2_matrix = t2.transpose(0, 2, 1, 3).reshape(
            nocc_a * nvir_a,
            nocc_b * nvir_b,
        )
        left_vecs, singular_vals, right_vecs = _truncated_svd(t2_matrix, tol=tol)
        n_vecs = len(singular_vals)
        one_body_tensors = np.zeros((n_vecs, 2, 2, 2, norb, norb), dtype=complex)
        for left_vec, right_vec, tensors in zip(
            left_vecs.T,
            right_vecs,
            one_body_tensors,
            strict=True,
        ):
            left_mat = np.zeros((norb, norb))
            col, row = zip(*product(range(nocc_a), range(nocc_a, norb)), strict=True)
            left_mat[row, col] = left_vec
            right_mat = np.zeros((norb, norb))
            col, row = zip(*product(range(nocc_b), range(nocc_b, norb)), strict=True)
            right_mat[row, col] = right_vec
            tensors[0, :, 0] = _quadrature(left_mat, 1)
            tensors[0, 0, 1] = _quadrature(right_mat, 1)
            tensors[0, 1, 1] = _quadrature(-right_mat, 1)
            tensors[1, :, 0] = _quadrature(left_mat, -1)
            tensors[1, 0, 1] = _quadrature(right_mat, -1)
            tensors[1, 1, 1] = _quadrature(-right_mat, -1)
        one_body_tensors = one_body_tensors.reshape((n_vecs, 4, 2, norb, norb))
        eigs, rotations = np.linalg.eigh(one_body_tensors)
        eigs = np.concatenate((eigs[:, :, 0], eigs[:, :, 1]), axis=-1)
        coeffs = 0.5 * np.array([1, -1, -1, 1]) * singular_vals[:, None]
        big_diag = coeffs[:, :, None, None] * eigs[:, :, :, None] * eigs[:, :, None, :]
        full_diag = np.stack(
            (
                big_diag[:, :, :norb, :norb],
                big_diag[:, :, :norb, norb:],
                big_diag[:, :, norb:, norb:],
            ),
            axis=2,
        ).reshape(-1, 3, norb, norb)
        full_rotations = rotations.reshape(-1, 2, norb, norb)

    interaction_pairs = (pairs_aa, pairs_ab, pairs_bb)
    if optimize:
        return _double_factorized_t2_alpha_beta_compressed(
            t2,
            full_diag,
            full_rotations,
            max_terms=max_terms,
            method=method,
            callback=callback,
            options=options,
            interaction_pairs=interaction_pairs,
            regularization=regularization,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
            return_optimize_result=return_optimize_result,
        )

    diag = full_diag[:max_terms]
    rotations = full_rotations[:max_terms]
    diag = _mask_unrestricted_diag(diag, interaction_pairs)
    return diag, rotations


def _double_factorized_t2_alpha_beta_compressed(
    t2_amplitudes: np.ndarray,
    full_diag: np.ndarray,
    full_rotations: np.ndarray,
    *,
    max_terms: int | None,
    method: str,
    callback,
    options: dict | None,
    interaction_pairs: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ],
    regularization: float,
    multi_stage_start: int | None,
    multi_stage_step: int | None,
    return_optimize_result: bool,
):
    """在本地用 PyTorch 梯度执行 spin-resolved compressed DF 优化。"""
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("optimize=True for LUCJ initialization requires scipy") from exc
    try:
        import torch
    except ImportError as exc:
        raise ImportError("optimize=True for LUCJ initialization requires torch") from exc

    t2 = np.asarray(t2_amplitudes, dtype=float)
    nocc_a, nocc_b, nvir_a, _ = t2.shape
    norb = nocc_a + nvir_a
    n_terms_full = len(full_diag)
    init_diag_norm = float(np.sum(np.abs(full_diag) ** 2))

    # 没有发生压缩时保留精确显式分解，不伪造一次优化结果。
    if max_terms is None or n_terms_full < max_terms:
        diag = _mask_unrestricted_diag(full_diag, interaction_pairs)
        if return_optimize_result:
            return diag, full_rotations, None
        return diag, full_rotations

    if multi_stage_start is None and multi_stage_step is None:
        stages = [max_terms]
    else:
        start = min(n_terms_full, multi_stage_start or n_terms_full)
        step = multi_stage_step or 1
        stages = list(range(start, max_terms, -step))
        stages.append(max_terms)

    diag = full_diag
    rotations = full_rotations
    target = torch.as_tensor(t2, dtype=torch.complex128)
    result = None
    for n_tensors in stages:
        diag = _mask_unrestricted_diag(diag[:n_tensors], interaction_pairs)
        rotations = rotations[:n_tensors]
        x0 = _alpha_beta_tensors_to_params(diag, rotations, interaction_pairs)

        last_evaluation: dict[str, Any] = {}

        def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
            variable = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            reconstructed, current_diag = _torch_reconstruct_alpha_beta_from_params(
                variable,
                n_tensors=n_tensors,
                norb=norb,
                nocc_a=nocc_a,
                nocc_b=nocc_b,
                interaction_pairs=interaction_pairs,
            )
            diff = reconstructed - target
            loss = 0.5 * torch.sum((torch.conj(diff) * diff).real)
            if regularization:
                diag_norm = torch.sum((torch.conj(current_diag) * current_diag).real)
                loss = loss + float(regularization) * torch.abs(diag_norm - init_diag_norm)
            loss.backward()
            value = float(loss.detach().cpu().item())
            last_evaluation["x"] = np.asarray(values, dtype=float).copy()
            last_evaluation["loss"] = value
            return value, variable.grad.detach().cpu().numpy()

        # 保存初值和每个被接受迭代点的目标函数，供大线路优化审计与作图。
        # 优化器回调仍收到原始 xk，避免改变既有公共接口。
        initial_loss, _ = objective(x0)
        loss_history = [initial_loss]

        def traced_callback(intermediate_result):
            values = np.asarray(intermediate_result.x, dtype=float)
            cached_x = last_evaluation.get("x")
            if cached_x is None or not np.array_equal(values, cached_x):
                current_loss, _ = objective(values)
            else:
                current_loss = float(last_evaluation["loss"])
            loss_history.append(current_loss)
            if callback is not None:
                if _callback_accepts_intermediate_result(callback):
                    return callback(intermediate_result)
                return callback(values)
            return None

        result = minimize(
            objective,
            x0,
            method=method,
            jac=True,
            callback=traced_callback,
            options=options,
        )
        result.loss_history = np.asarray(loss_history, dtype=float)
        diag, rotations = _alpha_beta_tensors_from_params(
            result.x,
            n_tensors=n_tensors,
            norb=norb,
            interaction_pairs=interaction_pairs,
        )

    if return_optimize_result:
        return diag, rotations, result
    return diag, rotations


def _alpha_beta_tensors_to_params(
    diag: np.ndarray,
    rotations: np.ndarray,
    interaction_pairs: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ],
) -> np.ndarray:
    """把 αα/αβ/ββ 矩阵和两套 orbital rotations 打包成实参数。"""
    n_tensors, _, norb, _ = diag.shape
    pairs_aa, pairs_ab, pairs_bb = interaction_pairs
    rotation_params = unitaries_to_parameters(rotations.reshape(-1, norb, norb))
    aa_params = real_symmetrics_to_parameters(diag[:, 0], pairs_aa)
    if pairs_ab is None:
        ab_params = np.asarray(diag[:, 1], dtype=float).reshape(n_tensors, -1).reshape(-1)
    else:
        rows, cols = [p for p, _ in pairs_ab], [q for _, q in pairs_ab]
        ab_params = np.asarray(diag[:, 1], dtype=float)[:, rows, cols].reshape(-1)
    bb_params = real_symmetrics_to_parameters(diag[:, 2], pairs_bb)
    return np.concatenate((rotation_params, aa_params, ab_params, bb_params))


def _alpha_beta_tensors_from_params(
    params: np.ndarray,
    *,
    n_tensors: int,
    norb: int,
    interaction_pairs: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ],
) -> tuple[np.ndarray, np.ndarray]:
    """从实参数还原 spin-resolved compressed DF 张量。"""
    pairs_aa, pairs_ab, pairs_bb = interaction_pairs
    n_rotation = n_tensors * 2 * norb**2
    n_aa = n_tensors * (norb * (norb + 1) // 2 if pairs_aa is None else len(pairs_aa))
    n_ab = n_tensors * (norb**2 if pairs_ab is None else len(pairs_ab))
    rotation_params, aa_params, ab_params, bb_params = np.split(
        np.asarray(params, dtype=float),
        (n_rotation, n_rotation + n_aa, n_rotation + n_aa + n_ab),
    )
    rotations = unitaries_from_parameters(
        rotation_params,
        dim=norb,
        n_mats=2 * n_tensors,
    ).reshape(n_tensors, 2, norb, norb)
    aa = real_symmetrics_from_parameters(
        aa_params,
        dim=norb,
        n_mats=n_tensors,
        indices=pairs_aa,
    )
    if pairs_ab is None:
        ab = ab_params.reshape(n_tensors, norb, norb)
    else:
        ab = np.zeros((n_tensors, norb, norb), dtype=float)
        rows, cols = [p for p, _ in pairs_ab], [q for _, q in pairs_ab]
        ab[:, rows, cols] = ab_params.reshape(n_tensors, len(pairs_ab))
    bb = real_symmetrics_from_parameters(
        bb_params,
        dim=norb,
        n_mats=n_tensors,
        indices=pairs_bb,
    )
    return np.stack((aa, ab, bb), axis=1), rotations


def _torch_reconstruct_alpha_beta_from_params(
    values,
    *,
    n_tensors: int,
    norb: int,
    nocc_a: int,
    nocc_b: int,
    interaction_pairs: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ],
):
    """PyTorch 版参数还原和 αβ `t2` 重构。"""
    import torch

    pairs_aa, pairs_ab, pairs_bb = interaction_pairs
    n_rotation = n_tensors * 2 * norb**2
    n_aa = n_tensors * (norb * (norb + 1) // 2 if pairs_aa is None else len(pairs_aa))
    n_ab = n_tensors * (norb**2 if pairs_ab is None else len(pairs_ab))
    rotation_params, aa_params, ab_params, bb_params = torch.tensor_split(
        values,
        (n_rotation, n_rotation + n_aa, n_rotation + n_aa + n_ab),
    )
    generators = _torch_antihermitians_from_parameters(
        rotation_params,
        norb,
        2 * n_tensors,
    )
    rotations = torch.matrix_exp(generators).reshape(n_tensors, 2, norb, norb)
    aa = _torch_real_symmetrics_from_parameters(
        aa_params,
        dim=norb,
        n_mats=n_tensors,
        indices=pairs_aa,
    )
    ab = _torch_real_matrices_from_parameters(
        ab_params,
        dim=norb,
        n_mats=n_tensors,
        indices=pairs_ab,
    )
    bb = _torch_real_symmetrics_from_parameters(
        bb_params,
        dim=norb,
        n_mats=n_tensors,
        indices=pairs_bb,
    )
    diag = torch.stack((aa, ab, bb), dim=1).to(torch.complex128)

    zero = torch.zeros_like(rotations[:, 0])
    expanded_diag = torch.cat(
        (
            torch.cat((diag[:, 0], diag[:, 1]), dim=2),
            torch.cat((diag[:, 1].transpose(1, 2), diag[:, 2]), dim=2),
        ),
        dim=1,
    )
    expanded_rotations = torch.cat(
        (
            torch.cat((rotations[:, 0], zero), dim=2),
            torch.cat((zero, rotations[:, 1]), dim=2),
        ),
        dim=1,
    )
    reconstructed = (
        1j
        * torch.einsum(
            "kpq,kap,kip,kbq,kjq->ijab",
            expanded_diag,
            expanded_rotations,
            torch.conj(expanded_rotations),
            expanded_rotations,
            torch.conj(expanded_rotations),
        )[:nocc_a, norb : norb + nocc_b, nocc_a:norb, norb + nocc_b :]
    )
    return reconstructed, diag


def _torch_real_matrices_from_parameters(
    values,
    *,
    dim: int,
    n_mats: int,
    indices: list[tuple[int, int]] | None,
):
    """从参数还原一批不要求对称的实矩阵。"""
    import torch

    if indices is None:
        return values.reshape(n_mats, dim, dim)
    rows = torch.tensor([pair[0] for pair in indices], dtype=torch.long, device=values.device)
    cols = torch.tensor([pair[1] for pair in indices], dtype=torch.long, device=values.device)
    matrices = torch.zeros((n_mats, dim, dim), dtype=torch.float64, device=values.device)
    matrices[:, rows, cols] = values.reshape(n_mats, len(indices))
    return matrices


def reconstruct_t2_alpha_beta_from_factors(
    orbital_rotations,
    diagonal_coulomb_matrices,
    nocc_a: int,
    nocc_b: int,
) -> np.ndarray:
    """由 spin-resolved 因子重构 `t2ab`，用于数值回归与 loss。"""
    rotations = np.asarray(orbital_rotations, dtype=complex)
    diag = np.asarray(diagonal_coulomb_matrices, dtype=complex)
    if rotations.ndim != 4 or rotations.shape[1] != 2:
        raise ValueError("orbital_rotations must have shape (n_terms, 2, norb, norb)")
    if diag.ndim != 4 or diag.shape[1] != 3:
        raise ValueError("diagonal_coulomb_matrices must have shape (n_terms, 3, norb, norb)")
    if rotations.shape[0] != diag.shape[0] or rotations.shape[-1] != diag.shape[-1]:
        raise ValueError("spin-resolved factors must have matching term and orbital dimensions")
    norb = rotations.shape[-1]
    expanded_diag = np.empty((len(diag), 2 * norb, 2 * norb), dtype=complex)
    expanded_rotations = np.empty_like(expanded_diag)
    for term, ((mat_aa, mat_ab, mat_bb), (rotation_a, rotation_b)) in enumerate(
        zip(diag, rotations, strict=True)
    ):
        expanded_diag[term] = np.block([[mat_aa, mat_ab], [mat_ab.T, mat_bb]])
        expanded_rotations[term] = scipy.linalg.block_diag(rotation_a, rotation_b)
    return (
        1j
        * np.einsum(
            "kpq,kap,kip,kbq,kjq->ijab",
            expanded_diag,
            expanded_rotations,
            expanded_rotations.conj(),
            expanded_rotations,
            expanded_rotations.conj(),
            optimize=True,
        )[:nocc_a, norb : norb + nocc_b, nocc_a:norb, norb + nocc_b :]
    )


def reconstruct_t2_from_factors(
    orbital_rotations,
    diagonal_coulomb_matrices,
    nocc: int,
    nvir: int | None = None,
) -> np.ndarray:
    """用 `U_mu/Z_mu` 重构 CCSD `t2`。"""
    rotations = np.asarray(orbital_rotations, dtype=complex)
    matrices = np.asarray(diagonal_coulomb_matrices)
    if matrices.ndim == 4:
        matrices = matrices[:, 0]
    matrices = np.asarray(matrices, dtype=complex)
    if rotations.ndim != 3 or matrices.ndim != 3:
        raise ValueError("orbital_rotations and diagonal_coulomb_matrices must be rank-3 arrays")
    if rotations.shape != matrices.shape:
        raise ValueError(
            "orbital_rotations and diagonal_coulomb_matrices must have the same shape; "
            f"got {rotations.shape} and {matrices.shape}"
        )
    norb = rotations.shape[1]
    if nvir is None:
        nvir = norb - int(nocc)
    if nocc < 1 or nvir < 1 or nocc + nvir != norb:
        raise ValueError("nocc and nvir must be positive and sum to N")
    return (
        1j
        * np.einsum(
            "kpq,kap,kip,kbq,kjq->ijab",
            matrices,
            rotations,
            rotations.conj(),
            rotations,
            rotations.conj(),
            optimize=True,
        )[:nocc, :nocc, nocc:, nocc:]
    )


def t2_reconstruction_loss(
    t2_amplitudes,
    orbital_rotations,
    diagonal_coulomb_matrices,
    *,
    regularization: float = 0.0,
    reference_j_norm: float | None = None,
) -> float:
    """计算 `0.5 * ||reconstructed_t2 - t2||^2`。"""
    target = np.asarray(t2_amplitudes)
    if np.iscomplexobj(target):
        if not np.allclose(target.imag, 0.0):
            raise ValueError("t2_amplitudes must be real-valued")
        target = target.real
    target = np.asarray(target, dtype=float)
    nocc, _, nvir, _ = target.shape
    reconstructed = reconstruct_t2_from_factors(
        orbital_rotations,
        diagonal_coulomb_matrices,
        nocc,
        nvir,
    )
    diff = reconstructed - target
    loss = 0.5 * float(np.vdot(diff, diff).real)
    if regularization:
        matrices = np.asarray(diagonal_coulomb_matrices, dtype=float)
        current_norm = float(np.vdot(matrices, matrices).real)
        if reference_j_norm is None:
            loss += 0.5 * float(regularization) * current_norm
        else:
            loss += float(regularization) * abs(current_norm - float(reference_j_norm))
    return loss


def _double_factorized_t2_explicit(
    t2_amplitudes: np.ndarray,
    *,
    tol: float,
    max_terms: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """执行 ffsim 的 nested eigenvalue explicit double factorization。"""
    t2 = np.asarray(t2_amplitudes, dtype=float)
    nocc, _, nvir, _ = t2.shape
    norb = nocc + nvir
    if not np.any(t2):
        return np.zeros((0, norb, norb), dtype=float), np.zeros((0, norb, norb), dtype=complex)
    t2_matrix = t2.transpose(0, 2, 1, 3).reshape(nocc * nvir, nocc * nvir)
    outer_eigs, outer_vecs = _truncated_eigh(t2_matrix, tol=tol)
    if len(outer_eigs) == 0:
        return np.zeros((0, norb, norb), dtype=float), np.zeros((0, norb, norb), dtype=complex)

    n_vecs = len(outer_eigs)
    one_body_tensors = np.zeros((n_vecs, 2, norb, norb), dtype=complex)
    row_col = list((row, col) for col in range(nocc) for row in range(nocc, norb))
    for outer_vec, one_body_tensor in zip(outer_vecs.T, one_body_tensors, strict=True):
        matrix = np.zeros((norb, norb), dtype=complex)
        rows, cols = zip(*row_col, strict=True)
        matrix[rows, cols] = outer_vec
        one_body_tensor[0] = _quadrature(matrix, sign=1)
        one_body_tensor[1] = _quadrature(matrix, sign=-1)

    eigs, orbital_rotations = np.linalg.eigh(one_body_tensors)
    coeffs = np.array([1, -1], dtype=float) * outer_eigs[:, None]
    diag_coulomb_mats = coeffs[:, :, None, None] * eigs[:, :, :, None] * eigs[:, :, None, :]

    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)
    if max_terms is not None:
        orbital_rotations = orbital_rotations[:max_terms]
        diag_coulomb_mats = diag_coulomb_mats[:max_terms]
    return diag_coulomb_mats.real, orbital_rotations


def _double_factorized_t2_compressed(
    t2_amplitudes: np.ndarray,
    *,
    tol: float,
    max_terms: int | None,
    method: str,
    callback,
    options: dict | None,
    diag_coulomb_indices: list[tuple[int, int]] | None,
    regularization: float,
    multi_stage_start: int | None,
    multi_stage_step: int | None,
    return_optimize_result: bool,
    ):
    """用 L-BFGS-B 压缩优化 `Z_mu/U_mu`。"""
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("optimize=True for LUCJ initialization requires scipy") from exc
    try:
        import torch
    except ImportError as exc:
        raise ImportError("optimize=True for LUCJ initialization requires torch") from exc

    t2 = np.asarray(t2_amplitudes, dtype=float)
    nocc, _, nvir, _ = t2.shape
    norb = nocc + nvir
    full_diag, full_rotations = _double_factorized_t2_explicit(t2, tol=tol, max_terms=None)
    init_diag_norm = float(np.sum(np.abs(full_diag) ** 2))
    n_terms_full = full_rotations.shape[0]
    if max_terms is None or n_terms_full < max_terms:
        if return_optimize_result:
            return full_diag, full_rotations, None
        return full_diag, full_rotations

    if multi_stage_start is None and multi_stage_step is None:
        list_reps = [max_terms]
    else:
        multi_stage_start = min(n_terms_full, multi_stage_start or n_terms_full)
        multi_stage_step = multi_stage_step or 1
        list_reps = list(range(multi_stage_start, max_terms, -multi_stage_step))
        list_reps.append(max_terms)

    diag_coulomb_mats = full_diag
    orbital_rotations = full_rotations
    result = None
    for n_tensors in list_reps:
        diag_coulomb_mats = diag_coulomb_mats[:n_tensors]
        orbital_rotations = orbital_rotations[:n_tensors]
        x0 = df_tensors_to_params(
            diag_coulomb_mats,
            orbital_rotations,
            diag_coulomb_indices,
        )
        target = torch.as_tensor(t2, dtype=torch.complex128)

        last_evaluation: dict[str, Any] = {}

        def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
            variable = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            reconstructed, current_diag = _torch_reconstruct_from_params(
                variable,
                target,
                n_tensors=n_tensors,
                norb=norb,
                nocc=nocc,
                diag_coulomb_indices=diag_coulomb_indices,
            )
            diff = reconstructed - target
            loss = 0.5 * torch.sum((torch.conj(diff) * diff).real)
            if regularization:
                diag_norm = torch.sum((torch.conj(current_diag) * current_diag).real)
                loss = loss + float(regularization) * torch.abs(diag_norm - init_diag_norm)
            loss.backward()
            value = float(loss.detach().cpu().item())
            last_evaluation["x"] = np.asarray(values, dtype=float).copy()
            last_evaluation["loss"] = value
            return value, variable.grad.detach().cpu().numpy()

        # 记录初值与每个被接受的 L-BFGS-B 迭代点，同时保持用户 callback 语义。
        initial_loss, _ = objective(x0)
        loss_history = [initial_loss]

        def traced_callback(intermediate_result):
            values = np.asarray(intermediate_result.x, dtype=float)
            cached_x = last_evaluation.get("x")
            if cached_x is None or not np.array_equal(values, cached_x):
                current_loss, _ = objective(values)
            else:
                current_loss = float(last_evaluation["loss"])
            loss_history.append(current_loss)
            if callback is not None:
                if _callback_accepts_intermediate_result(callback):
                    return callback(intermediate_result)
                return callback(values)
            return None

        result = minimize(
            objective,
            x0,
            method=method,
            jac=True,
            callback=traced_callback,
            options=options,
        )
        result.loss_history = np.asarray(loss_history, dtype=float)
        diag_coulomb_mats, orbital_rotations = df_tensors_from_params(
            result.x,
            n_tensors=n_tensors,
            norb=norb,
            diag_coulomb_indices=diag_coulomb_indices,
        )

    if return_optimize_result:
        return diag_coulomb_mats, orbital_rotations, result
    return diag_coulomb_mats, orbital_rotations


def _torch_reconstruct_from_params(
    values,
    target,
    *,
    n_tensors: int,
    norb: int,
    nocc: int,
    diag_coulomb_indices: list[tuple[int, int]] | None,
):
    """PyTorch 版 `df_tensors_from_params + reconstruct_t2`。"""
    import torch

    n_orbital_params = n_tensors * norb**2
    generator_params = values[:n_orbital_params]
    diag_params = values[n_orbital_params:]
    generators = _torch_antihermitians_from_parameters(generator_params, norb, n_tensors)
    orbital_rotations = torch.matrix_exp(generators)
    diag_coulomb_mats = _torch_real_symmetrics_from_parameters(
        diag_params,
        dim=norb,
        n_mats=n_tensors,
        indices=diag_coulomb_indices,
    ).to(torch.complex128)
    reconstructed = (
        1j
        * torch.einsum(
            "kpq,kap,kip,kbq,kjq->ijab",
            diag_coulomb_mats,
            orbital_rotations,
            torch.conj(orbital_rotations),
            orbital_rotations,
            torch.conj(orbital_rotations),
        )[:nocc, :nocc, nocc:, nocc:]
    )
    return reconstructed, diag_coulomb_mats


def _torch_antihermitians_from_parameters(values, dim: int, n_mats: int):
    """PyTorch 版 complex anti-Hermitian 参数还原。"""
    import torch

    n_triu = dim * (dim - 1) // 2
    matrix = values.reshape(n_mats, dim**2)
    mats = torch.zeros((n_mats, dim, dim), dtype=torch.complex128, device=values.device)
    rows, cols = torch.triu_indices(dim, dim, offset=0, device=values.device)
    imag_values = 1j * matrix[:, n_triu:].to(torch.complex128)
    mats[:, rows, cols] = imag_values
    mats[:, cols, rows] = imag_values

    rows, cols = torch.triu_indices(dim, dim, offset=1, device=values.device)
    real_values = matrix[:, :n_triu].to(torch.complex128)
    mats[:, rows, cols] = mats[:, rows, cols] + real_values
    mats[:, cols, rows] = mats[:, cols, rows] - real_values
    return mats


def _torch_real_symmetrics_from_parameters(
    values,
    *,
    dim: int,
    n_mats: int,
    indices: list[tuple[int, int]] | None,
):
    """PyTorch 版实对称矩阵参数还原。"""
    import torch

    if indices is None:
        rows, cols = torch.triu_indices(dim, dim, offset=0, device=values.device)
        n_params_per_mat = dim * (dim + 1) // 2
    else:
        rows = torch.tensor([pair[0] for pair in indices], dtype=torch.long, device=values.device)
        cols = torch.tensor([pair[1] for pair in indices], dtype=torch.long, device=values.device)
        n_params_per_mat = len(indices)
    matrix = values.reshape(n_mats, n_params_per_mat)
    mats = torch.zeros((n_mats, dim, dim), dtype=torch.float64, device=values.device)
    mats[:, rows, cols] = matrix
    mats[:, cols, rows] = matrix
    return mats


def _truncated_eigh(
    matrix: np.ndarray,
    *,
    tol: float,
    max_vecs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """按 ffsim 规则对本征向量按绝对本征值截断。"""
    eigs, vecs = scipy.linalg.eigh(matrix)
    if max_vecs is None:
        max_vecs = len(eigs)
    indices = np.argsort(np.abs(eigs))[::-1]
    eigs = eigs[indices]
    vecs = vecs[:, indices]
    n_discard = int(np.searchsorted(np.cumsum(np.abs(eigs[::-1])), tol))
    n_vecs = min(max_vecs, len(eigs) - n_discard)
    return eigs[:n_vecs], vecs[:, :n_vecs]


def _truncated_svd(
    matrix: np.ndarray,
    *,
    tol: float,
    max_vecs: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 ffsim 规则截断 αβ 分解使用的奇异向量。"""
    left, singular_values, right = scipy.linalg.svd(matrix, full_matrices=False)
    if max_vecs is None:
        max_vecs = len(singular_values)
    n_discard = int(np.searchsorted(np.cumsum(singular_values[::-1]), tol))
    n_vecs = min(max_vecs, len(singular_values) - n_discard)
    return left[:, :n_vecs], singular_values[:n_vecs], right[:n_vecs]


def _quadrature(matrix: np.ndarray, sign: int) -> np.ndarray:
    """ffsim explicit DF 使用的 Hermitian quadrature。"""
    return 0.5 * (1 - sign * 1j) * (matrix + sign * 1j * matrix.T.conj())


def _pad_factorization(
    diagonal_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    n_layers: int,
    norb: int,
) -> tuple[np.ndarray, np.ndarray]:
    """当 DF 项少于请求层数时，用 no-op 补齐。"""
    diag = np.asarray(diagonal_coulomb_mats, dtype=float)
    rotations = np.asarray(orbital_rotations, dtype=complex)
    if diag.shape[0] >= n_layers:
        return diag[:n_layers], rotations[:n_layers]
    missing = n_layers - diag.shape[0]
    diag = np.concatenate([diag, np.zeros((missing, norb, norb), dtype=float)], axis=0)
    rotations = np.concatenate(
        [rotations, np.tile(np.eye(norb, dtype=complex), (missing, 1, 1))],
        axis=0,
    )
    return diag, rotations


def _pad_unrestricted_factorization(
    diagonal_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    n_layers: int,
    norb: int,
) -> tuple[np.ndarray, np.ndarray]:
    """用三通道零矩阵和两自旋单位阵补齐 spin-resolved 层数。"""
    diag = np.asarray(diagonal_coulomb_mats, dtype=float)
    rotations = np.asarray(orbital_rotations, dtype=complex)
    if len(diag) >= n_layers:
        return diag[:n_layers], rotations[:n_layers]
    missing = n_layers - len(diag)
    diag = np.concatenate(
        (diag, np.zeros((missing, 3, norb, norb), dtype=float)),
        axis=0,
    )
    identity_pair = np.tile(np.eye(norb, dtype=complex), (missing, 2, 1, 1))
    rotations = np.concatenate((rotations, identity_pair), axis=0)
    return diag, rotations


def _mask_unrestricted_diag(
    diagonal_coulomb_mats: np.ndarray,
    interaction_pairs: tuple[
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
        list[tuple[int, int]] | None,
    ],
) -> np.ndarray:
    """分别应用 `aa/ab/bb` topology mask；αβ 不强制对称。"""
    result = np.asarray(diagonal_coulomb_mats, dtype=float).copy()
    norb = result.shape[-1]
    for channel, pairs in enumerate(interaction_pairs):
        if pairs is None:
            continue
        mask = np.zeros((norb, norb), dtype=bool)
        if pairs:
            rows, cols = zip(*pairs, strict=True)
            mask[rows, cols] = True
            if channel != 1:
                mask[cols, rows] = True
        result[:, channel] *= mask
    return result


def _spin_balanced_mats_from_shared_z(
    diagonal_coulomb_mats: np.ndarray,
    *,
    pairs_aa: list[tuple[int, int]],
    pairs_ab: list[tuple[int, int]],
) -> np.ndarray:
    """把共享 `Z_mu` 分别 mask 成 builder 使用的 `Jaa/Jab`。"""
    shared = np.asarray(diagonal_coulomb_mats, dtype=float)
    n_layers, norb, _ = shared.shape
    mats = np.zeros((n_layers, 2, norb, norb), dtype=float)
    for channel, pairs in enumerate((pairs_aa, pairs_ab)):
        for p, q in pairs:
            mats[:, channel, p, q] = shared[:, p, q]
            mats[:, channel, q, p] = shared[:, q, p]
    return mats
