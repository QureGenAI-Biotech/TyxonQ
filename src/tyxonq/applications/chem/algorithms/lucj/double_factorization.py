"""LUCJ 的 compressed double factorization 初始化数学工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg

from .linalg import df_tensors_from_params, df_tensors_to_params
from .topology import (
    allowed_diag_coulomb_indices,
    interaction_pairs_spin_balanced,
    normalize_topology,
    validate_interaction_pairs,
    validate_layers,
    validate_n_orbitals,
)


@dataclass(frozen=True)
class PreparedT2:
    """整理后的 restricted closed-shell CCSD `t2` 数据。"""

    amplitudes: np.ndarray
    nocc: int
    nvir: int
    n_spatial_orbitals: int


@dataclass(frozen=True)
class DoubleFactorizationResult:
    """compressed DF 初始化的结果。"""

    orbital_rotations: np.ndarray
    diagonal_coulomb_matrices: np.ndarray
    diag_coulomb_mats: np.ndarray
    final_orbital_rotation: np.ndarray | None
    nocc: int
    nvir: int
    n_spatial_orbitals: int
    topology: str
    loss: float
    optimize_result: Any | None = None


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
    t1_amplitudes=None,
    optimize: bool = True,
    regularization: float = 0.0,
    maxiter: int = 100,
    multi_stage_start: int | None = None,
    multi_stage_step: int | None = None,
) -> DoubleFactorizationResult:
    """从 CCSD `t1/t2` 生成 matrix UCJ 初始化参数。"""
    prepared = prepare_t2_amplitudes(t2_amplitudes, n_spatial_orbitals)
    layers = validate_layers(n_layers)
    name = normalize_topology(topology)
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(prepared.n_spatial_orbitals, name)
    diag_indices = allowed_diag_coulomb_indices(prepared.n_spatial_orbitals, name)

    factorization = double_factorized_t2(
        prepared.amplitudes,
        max_terms=layers,
        optimize=optimize,
        options={"maxiter": int(maxiter)},
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
            return float(loss.detach().cpu().item()), variable.grad.detach().cpu().numpy()

        result = minimize(
            objective,
            x0,
            method=method,
            jac=True,
            callback=callback,
            options=options,
        )
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
