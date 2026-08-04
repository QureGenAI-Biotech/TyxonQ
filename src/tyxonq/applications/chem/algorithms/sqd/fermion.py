"""TyxonQ 的费米子 SQD 主流程。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Callable, cast

import numpy as np
from pyscf import fci
from pyscf.fci.selected_ci import (
    _as_SCIvector,
    make_rdm1,
    make_rdm1s,
    make_rdm2,
    make_rdm2s,
    spin_square,
)

from .recovery import recover_configurations
from .samples import bitstring_matrix_to_integers, samples_to_arrays
from .subsampling import postselect_by_hamming_right_and_left, subsample


@dataclass(frozen=True)
class SCIState:
    """保存一次 selected-CI 求解得到的量子态。"""

    amplitudes: np.ndarray
    ci_strs_a: np.ndarray
    ci_strs_b: np.ndarray
    norb: int
    nelec: tuple[int, int]

    def __post_init__(self) -> None:
        """检查 CI 系数矩阵形状是否和 alpha/beta determinant 数量匹配。"""
        object.__setattr__(self, "amplitudes", np.asarray(self.amplitudes))
        if self.amplitudes.shape != (len(self.ci_strs_a), len(self.ci_strs_b)):
            raise ValueError(
                f"'amplitudes' shape must be ({len(self.ci_strs_a)}, {len(self.ci_strs_b)}) "
                f"but got {self.amplitudes.shape}."
            )

    def save(self, filename: str) -> None:
        """把 SCIState 保存成 npz 文件，方便后续复查。"""
        np.savez(
            filename,
            amplitudes=self.amplitudes,
            ci_strs_a=self.ci_strs_a,
            ci_strs_b=self.ci_strs_b,
            norb=self.norb,
            nelec=self.nelec,
        )

    @classmethod
    def load(cls, filename: str) -> "SCIState":
        """从 ``save`` 写出的 npz 文件恢复 SCIState。"""
        with np.load(filename) as data:
            return cls(
                data["amplitudes"],
                data["ci_strs_a"],
                data["ci_strs_b"],
                norb=int(data["norb"]),
                nelec=tuple(int(x) for x in data["nelec"]),
            )

    def rdm(self, rank: int = 1, spin_summed: bool = False) -> np.ndarray:
        """用 PySCF selected-CI 工具计算 1-RDM 或 2-RDM。"""
        sci_vector = _as_SCIvector(self.amplitudes, (self.ci_strs_a, self.ci_strs_b))
        if rank == 1:
            if spin_summed:
                return make_rdm1(sci_vector, self.norb, self.nelec)  # type: ignore[no-any-return]
            return make_rdm1s(sci_vector, self.norb, self.nelec)  # type: ignore[no-any-return]
        if rank == 2:
            if spin_summed:
                return make_rdm2(sci_vector, self.norb, self.nelec)  # type: ignore[no-any-return]
            return make_rdm2s(sci_vector, self.norb, self.nelec)  # type: ignore[no-any-return]
        raise NotImplementedError(
            f"Computing the rank {rank} reduced density matrix is currently not supported."
        )

    def spin_square(self) -> float:
        """返回总自旋平方期望值。"""
        sci_vector = _as_SCIvector(self.amplitudes, (self.ci_strs_a, self.ci_strs_b))
        spin_squared, _ = spin_square(sci_vector, norb=self.norb, nelec=self.nelec)
        return cast(float, spin_squared)

    def orbital_occupancies(self) -> tuple[np.ndarray, np.ndarray]:
        """从 1-RDM 对角线返回 alpha 和 beta 轨道平均占据数。"""
        dm_a, dm_b = self.rdm(rank=1, spin_summed=False)
        return np.diagonal(dm_a), np.diagonal(dm_b)


@dataclass(frozen=True)
class SCIResult:
    """保存一次 selected-CI 对角化的结果。"""

    energy: float
    sci_state: SCIState
    orbital_occupancies: tuple[np.ndarray, np.ndarray]
    rdm1: np.ndarray | None = None
    rdm2: np.ndarray | None = None
    nuclear_repulsion_energy: float = 0.0

    @property
    def total_energy(self) -> float:
        """电子能加核排斥能，也就是常用的分子总能量。"""
        return float(self.energy) + float(self.nuclear_repulsion_energy)


def run_sqd_fermion(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    samples: Mapping[str, float | int] | np.ndarray | Sequence[Sequence[bool | int]],
    samples_per_batch: int,
    norb: int,
    nelec: tuple[int, int],
    *,
    probabilities: Sequence[float] | np.ndarray | None = None,
    sample_order: str = "alpha_beta",
    nuclear_repulsion_energy: float = 0.0,
    num_batches: int = 1,
    energy_tol: float = 1e-8,
    occupancies_tol: float = 1e-5,
    max_iterations: int = 100,
    sci_solver: Callable[
        [list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, int, tuple[int, int]],
        list[SCIResult],
    ]
    | None = None,
    symmetrize_spin: bool = False,
    max_dim: int | tuple[int, int] | None = None,
    include_configurations: list[int] | tuple[list[int], list[int]] | np.ndarray | None = None,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    carryover_threshold: float = 1e-4,
    callback: Callable[[list[SCIResult]], None] | None = None,
    seed: int | np.random.Generator | None = None,
) -> SCIResult:
    """从 counts 或 bitstring 矩阵运行费米子 SQD。"""
    bitstring_matrix, probability_array = samples_to_arrays(
        samples,
        probabilities,
        sample_order=sample_order,
    )
    return diagonalize_fermionic_hamiltonian(
        one_body_tensor,
        two_body_tensor,
        bitstring_matrix,
        probability_array,
        samples_per_batch,
        norb,
        nelec,
        num_batches=num_batches,
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        max_iterations=max_iterations,
        sci_solver=sci_solver,
        symmetrize_spin=symmetrize_spin,
        max_dim=max_dim,
        include_configurations=include_configurations,
        initial_occupancies=initial_occupancies,
        carryover_threshold=carryover_threshold,
        callback=callback,
        seed=seed,
        nuclear_repulsion_energy=nuclear_repulsion_energy,
    )


def diagonalize_fermionic_hamiltonian(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    samples_per_batch: int,
    norb: int,
    nelec: tuple[int, int],
    *,
    num_batches: int = 1,
    energy_tol: float = 1e-8,
    occupancies_tol: float = 1e-5,
    max_iterations: int = 100,
    sci_solver: Callable[
        [list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, int, tuple[int, int]],
        list[SCIResult],
    ]
    | None = None,
    symmetrize_spin: bool = False,
    max_dim: int | tuple[int, int] | None = None,
    include_configurations: list[int] | tuple[list[int], list[int]] | np.ndarray | None = None,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    carryover_threshold: float = 1e-4,
    callback: Callable[[list[SCIResult]], None] | None = None,
    seed: int | np.random.Generator | None = None,
    nuclear_repulsion_energy: float = 0.0,
) -> SCIResult:
    """在内部 ``[beta | alpha]`` 样本顺序上运行 SQD 主循环。"""
    bitstring_matrix = np.asarray(bitstring_matrix, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    if max_iterations < 1:
        raise ValueError("Maximum number of iterations must be at least 1.")
    if bitstring_matrix.ndim != 2:
        raise ValueError("bitstring_matrix must be a 2D array.")
    if bitstring_matrix.shape[1] != 2 * norb:
        raise ValueError("bitstring_matrix must have exactly 2 * norb columns.")
    if len(probabilities) != bitstring_matrix.shape[0]:
        raise ValueError("The number of probabilities must match the number of bitstrings.")
    if np.sum(probabilities) <= 0:
        raise ValueError("The probability sum must be positive.")
    probabilities = probabilities / np.sum(probabilities)

    n_alpha, n_beta = nelec
    if symmetrize_spin and n_alpha != n_beta:
        raise ValueError(
            "Spin symmetrization is only possible if alpha and beta electron counts match. "
            f"Instead, got {n_alpha} and {n_beta}."
        )

    if max_dim is None:
        max_dim_a = max_dim_b = None
    elif isinstance(max_dim, tuple):
        max_dim_a, max_dim_b = max_dim
    else:
        max_dim_a = max_dim_b = max_dim
    if symmetrize_spin and max_dim_a != max_dim_b:
        raise ValueError("max_dim must be the same for alpha and beta when symmetrizing spin.")

    if include_configurations is None:
        include_a: list[int] | np.ndarray = np.array([], dtype=int)
        include_b: list[int] | np.ndarray = np.array([], dtype=int)
    elif isinstance(include_configurations, tuple):
        include_a, include_b = include_configurations
    else:
        include_a = include_configurations
        include_b = include_configurations

    rng = np.random.default_rng(seed)
    current_occupancies = initial_occupancies
    best_result: SCIResult | None = None
    current_result: SCIResult | None = None
    if sci_solver is None:
        sci_solver = solve_sci_batch

    include_a = np.unique(include_a)
    include_b = np.unique(include_b)
    carryover_strings_a = np.array([], dtype=np.int64)
    carryover_strings_b = np.array([], dtype=np.int64)

    for _ in range(max_iterations):
        if current_occupancies is None:
            bitstrings, probs = postselect_by_hamming_right_and_left(
                bitstring_matrix,
                probabilities,
                hamming_right=n_alpha,
                hamming_left=n_beta,
            )
            if not bitstrings.size:
                raise ValueError(
                    "The input samples did not contain any bitstrings with the required "
                    "left and right Hamming weights. Pass valid samples or provide "
                    "initial_occupancies to enable configuration recovery."
                )
        else:
            # recovery 使用上一轮 selected-CI 得到的 occupancy 作为修复依据。
            bitstrings, probs = recover_configurations(
                bitstring_matrix,
                probabilities,
                current_occupancies,
                n_alpha,
                n_beta,
                rand_seed=rng,
            )

        subsamples = subsample(
            bitstrings,
            probs,
            samples_per_batch=samples_per_batch,
            num_batches=num_batches,
            rand_seed=rng,
        )

        ci_strings = []
        for sample_batch in subsamples:
            # sample_batch 内部是 [beta | alpha]；右半边转 alpha，左半边转 beta。
            samples_a, counts_a = np.unique(
                bitstring_matrix_to_integers(sample_batch[:, norb:]), return_counts=True
            )
            samples_b, counts_b = np.unique(
                bitstring_matrix_to_integers(sample_batch[:, :norb]), return_counts=True
            )
            if symmetrize_spin:
                samples_ab = np.concatenate((samples_a, samples_b))
                counts_ab = np.concatenate((counts_a, counts_b))
                samples_ab = samples_ab[np.argsort(counts_ab)[::-1]]
                strings = np.concatenate((include_a, include_b, carryover_strings_a, samples_ab))
                strs_a = strs_b = _unique_with_order_preserved(strings)[:max_dim_a]
            else:
                samples_a = samples_a[np.argsort(counts_a)[::-1]]
                samples_b = samples_b[np.argsort(counts_b)[::-1]]
                strs_a = np.concatenate((include_a, carryover_strings_a, samples_a))
                strs_b = np.concatenate((include_b, carryover_strings_b, samples_b))
                strs_a = _unique_with_order_preserved(strs_a)[:max_dim_a]
                strs_b = _unique_with_order_preserved(strs_b)[:max_dim_b]
            strs_a.sort()
            strs_b.sort()
            ci_strings.append((strs_a, strs_b))

        # PySCF 会把 Hamiltonian 限制到这些 CI strings 张成的 selected-CI 子空间。
        results = sci_solver(ci_strings, one_body_tensor, two_body_tensor, norb, nelec)
        if callback is not None:
            callback(results)

        best_result_in_batch = min(results, key=lambda result: result.energy)
        if best_result is None or best_result_in_batch.energy < best_result.energy:
            best_result = best_result_in_batch

        if (
            current_result is not None
            and abs(current_result.energy - best_result_in_batch.energy) < energy_tol
            and np.linalg.norm(
                np.ravel(cast(tuple[np.ndarray, np.ndarray], current_occupancies))
                - np.ravel(best_result_in_batch.orbital_occupancies),
                ord=np.inf,
            )
            < occupancies_tol
        ):
            break

        current_result = best_result_in_batch
        current_occupancies = current_result.orbital_occupancies
        carryover_strings_a, carryover_strings_b = _carryover_strings(
            current_result,
            carryover_threshold,
            symmetrize_spin,
        )

    return replace(
        cast(SCIResult, best_result),
        nuclear_repulsion_energy=float(nuclear_repulsion_energy),
    )


def solve_sci_batch(
    ci_strings: list[tuple[np.ndarray, np.ndarray]],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    **kwargs,
) -> list[SCIResult]:
    """批量对角化多个 selected-CI 子空间。"""
    return [
        solve_sci(
            strings,
            one_body_tensor,
            two_body_tensor,
            norb=norb,
            nelec=nelec,
            spin_sq=spin_sq,
            **kwargs,
        )
        for strings in ci_strings
    ]


def solve_sci(
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    **kwargs,
) -> SCIResult:
    """在一个 selected-CI 子空间里投影并对角化 Hamiltonian。"""
    norb, _ = one_body_tensor.shape
    myci = fci.selected_ci.SelectedCI()
    if spin_sq is not None:
        myci = fci.addons.fix_spin_(myci, ss=spin_sq)

    _, sci_vec = fci.selected_ci.kernel_fixed_space(
        myci,
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
        ci_strs=ci_strings,
        **kwargs,
    )
    dm1s = myci.make_rdm1s(sci_vec, norb, nelec)
    occupancies = (np.diagonal(dm1s[0]), np.diagonal(dm1s[1]))
    dm1 = myci.make_rdm1(sci_vec, norb, nelec)
    dm2 = myci.make_rdm2(sci_vec, norb, nelec)
    energy = np.einsum("pr,pr->", dm1, one_body_tensor) + 0.5 * np.einsum(
        "prqs,prqs->",
        dm2,
        two_body_tensor,
    )
    sci_state = SCIState(
        amplitudes=np.array(sci_vec),
        ci_strs_a=sci_vec._strs[0],
        ci_strs_b=sci_vec._strs[1],
        norb=norb,
        nelec=nelec,
    )
    return SCIResult(energy, sci_state, orbital_occupancies=occupancies, rdm1=dm1, rdm2=dm2)


def _unique_with_order_preserved(vals: np.ndarray) -> np.ndarray:
    """去重但保留原来的优先级顺序。"""
    _, indices = np.unique(vals, return_index=True)
    indices.sort()
    return vals[indices]


def _carryover_strings(
    result: SCIResult,
    carryover_threshold: float,
    symmetrize_spin: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """挑出下一轮必须继续保留的重要 CI strings。"""
    sci_state = result.sci_state
    flattened = sci_state.amplitudes.reshape(-1)
    absolute_vals = np.abs(flattened)
    indices = np.argsort(absolute_vals)
    carryover_index = np.searchsorted(absolute_vals, carryover_threshold, sorter=indices)
    carryover_indices = indices[carryover_index:]
    _, n_strings_b = sci_state.amplitudes.shape
    alpha_indices, beta_indices = np.divmod(carryover_indices, n_strings_b)
    alpha_indices = np.unique(alpha_indices)
    beta_indices = np.unique(beta_indices)
    carryover_strings_a = sci_state.ci_strs_a[alpha_indices]
    carryover_strings_b = sci_state.ci_strs_b[beta_indices]
    weights_a = np.sum(np.abs(sci_state.amplitudes[alpha_indices]) ** 2, axis=1)
    weights_b = np.sum(np.abs(sci_state.amplitudes[:, beta_indices]) ** 2, axis=0)

    if symmetrize_spin:
        carryover_strings = np.concatenate((carryover_strings_a, carryover_strings_b))
        weights = np.concatenate((weights_a, weights_b))
        carryover_strings = carryover_strings[np.argsort(weights)[::-1]]
        carryover_strings = _unique_with_order_preserved(carryover_strings)
        return carryover_strings, carryover_strings

    return (
        carryover_strings_a[np.argsort(weights_a)[::-1]],
        carryover_strings_b[np.argsort(weights_b)[::-1]],
    )
