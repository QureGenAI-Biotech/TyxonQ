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
)

from .excitations import enlarge_batch_from_single_excitations
from .recovery import recover_configurations
from .samples import bitstring_matrix_to_integers, samples_to_arrays
from .selected_ci_uhf import kernel_fixed_space_uhf
from .subsampling import postselect_by_hamming_right_and_left, subsample


OneBodyTensor = np.ndarray | tuple[np.ndarray, np.ndarray]
TwoBodyTensor = np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]
ExcludedOrbitals = (
    Sequence[int]
    | np.ndarray
    | tuple[Sequence[int] | np.ndarray, Sequence[int] | np.ndarray]
)


def spin_square_from_rdms(
    rdm1s: tuple[np.ndarray, np.ndarray],
    rdm2s: tuple[np.ndarray, np.ndarray, np.ndarray],
    overlap: np.ndarray | None = None,
) -> float:
    """直接从已有自旋分辨 RDM 计算总自旋平方，避免重复构造 RDM。"""

    dm1a, dm1b = rdm1s
    dm2aa, dm2ab, dm2bb = rdm2s
    norb = int(dm1a.shape[0])
    identity = np.eye(norb)
    overlap_ab = identity if overlap is None else np.asarray(overlap)
    if overlap_ab.shape != (norb, norb):
        raise ValueError(f"overlap must have shape ({norb}, {norb}).")
    overlap_ba = overlap_ab.T.conj()

    ssz = (
        np.einsum("ijkl,ij,kl->", dm2aa, identity, identity)
        - 2.0 * np.einsum("ijkl,ij,kl->", dm2ab, identity, identity)
        + np.einsum("ijkl,ij,kl->", dm2bb, identity, identity)
    ) * 0.25
    ssz += (
        np.einsum("ji,ij->", dm1a, identity)
        + np.einsum("ji,ij->", dm1b, identity)
    ) * 0.25

    dm2abba = -dm2ab.transpose(0, 3, 2, 1)
    dm2baab = -dm2ab.transpose(2, 1, 0, 3)
    ssxy = (
        np.einsum("ijkl,ij,kl->", dm2baab, overlap_ba, overlap_ab)
        + np.einsum("ijkl,ij,kl->", dm2abba, overlap_ab, overlap_ba)
        + np.einsum("ji,ij->", dm1a, identity)
        + np.einsum("ji,ij->", dm1b, identity)
    ) * 0.5
    return float(np.real_if_close(ssz + ssxy))


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

    def rdm(
        self, rank: int = 1, spin_summed: bool = False
    ) -> (
        np.ndarray
        | tuple[np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray]
    ):
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

    def spin_square(self, overlap: np.ndarray | None = None) -> float:
        """返回总自旋平方期望值。

        ``overlap`` 是 alpha 与 beta 活性轨道间的交叉重叠
        ``C_alpha.T @ S @ C_beta``；省略时使用公共正交轨道基。
        """
        return spin_square_from_rdms(
            cast(tuple[np.ndarray, np.ndarray], self.rdm(1, False)),
            cast(tuple[np.ndarray, np.ndarray, np.ndarray], self.rdm(2, False)),
            overlap,
        )

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
    rdm1s: tuple[np.ndarray, np.ndarray] | None = None
    rdm2s: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    # 仅表示本次 CI 对角化收敛，不表示 SQD 恢复迭代已经收敛。
    converged: bool = True

    @property
    def total_energy(self) -> float:
        """电子能加核排斥能，也就是常用的分子总能量。"""
        return float(self.energy) + float(self.nuclear_repulsion_energy)

    def spin_square(self, overlap: np.ndarray | None = None) -> float:
        """优先复用求解时已有的 RDM 返回 ``<S^2>`` 诊断量。"""

        if self.rdm1s is not None and self.rdm2s is not None:
            return spin_square_from_rdms(self.rdm1s, self.rdm2s, overlap)
        return self.sci_state.spin_square(overlap)


def run_sqd_fermion(
    one_body_tensor: OneBodyTensor,
    two_body_tensor: TwoBodyTensor,
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
        [
            list[tuple[np.ndarray, np.ndarray]],
            OneBodyTensor,
            TwoBodyTensor,
            int,
            tuple[int, int],
        ],
        list[SCIResult],
    ]
    | None = None,
    spin_sq: float | None = None,
    symmetrize_spin: bool = False,
    max_dim: int | tuple[int, int] | None = None,
    include_configurations: list[int]
    | tuple[list[int], list[int]]
    | np.ndarray
    | None = None,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    carryover_threshold: float = 1e-4,
    average_batch_occupancies: bool = False,
    callback: Callable[[list[SCIResult]], None] | None = None,
    seed: int | np.random.Generator | None = None,
    ext_sqd: bool = False,
    ext_sqd_threshold: float | None = None,
    ext_sqd_threshold_mode: str = "amplitude",
    excluded_orbitals: ExcludedOrbitals | None = None,
) -> SCIResult:
    """从 counts 或 bitstring 矩阵运行费米子 SQD。

    ``sample_order`` 只描述输入左右半区是 ``[alpha | beta]`` 还是
    ``[beta | alpha]``，不反转半区内部。默认 ``sample_order="alpha_beta"``
    期望输入使用 SQD/PySCF order：
    ``[alphaN-1..alpha0 | betaN-1..beta0]``。
    TyxonQ/LUCJ raw qubit order ``[alpha0..alphaN-1 | beta0..betaN-1]``
    进入本函数前应先用 ``reverse_bitstring_halves`` 转换。

    一体/二体积分同时传数组时沿用 restricted 求解；同时传
    ``(h_alpha, h_beta)`` 与 ``(g_aa, g_ab, g_bb)`` 时启用 unrestricted
    selected-CI。``excluded_orbitals`` 可传公共轨道列表，或传
    ``(excluded_alpha, excluded_beta)``，这些轨道在恢复和 CI 子空间中始终为零占据。

    ``spin_sq`` 是 restricted 默认 selected-CI 求解器的目标
    :math:`S(S+1)`；自定义 ``sci_solver`` 应自行实现自旋约束，因此不能同时传入。

    ``ext_sqd_threshold`` 独立控制末轮扩展的种子筛选；None 沿用
    ``carryover_threshold``。``amplitude`` 使用 ``abs(c) >= threshold``，
    ``probability`` 使用 ``abs(c)**2 > threshold``，后者对应论文程序设置。
    扩展始终从历轮最低能结果出发，并保留原有行列式空间。
    """
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
        spin_sq=spin_sq,
        symmetrize_spin=symmetrize_spin,
        max_dim=max_dim,
        include_configurations=include_configurations,
        initial_occupancies=initial_occupancies,
        carryover_threshold=carryover_threshold,
        average_batch_occupancies=average_batch_occupancies,
        callback=callback,
        seed=seed,
        nuclear_repulsion_energy=nuclear_repulsion_energy,
        ext_sqd=ext_sqd,
        ext_sqd_threshold=ext_sqd_threshold,
        ext_sqd_threshold_mode=ext_sqd_threshold_mode,
        excluded_orbitals=excluded_orbitals,
    )


def diagonalize_fermionic_hamiltonian(
    one_body_tensor: OneBodyTensor,
    two_body_tensor: TwoBodyTensor,
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
        [
            list[tuple[np.ndarray, np.ndarray]],
            OneBodyTensor,
            TwoBodyTensor,
            int,
            tuple[int, int],
        ],
        list[SCIResult],
    ]
    | None = None,
    spin_sq: float | None = None,
    symmetrize_spin: bool = False,
    max_dim: int | tuple[int, int] | None = None,
    include_configurations: list[int]
    | tuple[list[int], list[int]]
    | np.ndarray
    | None = None,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    carryover_threshold: float = 1e-4,
    average_batch_occupancies: bool = False,
    callback: Callable[[list[SCIResult]], None] | None = None,
    seed: int | np.random.Generator | None = None,
    nuclear_repulsion_energy: float = 0.0,
    ext_sqd: bool = False,
    ext_sqd_threshold: float | None = None,
    ext_sqd_threshold_mode: str = "amplitude",
    excluded_orbitals: ExcludedOrbitals | None = None,
) -> SCIResult:
    """在内部 ``[beta | alpha]`` 样本顺序上运行 SQD 主循环。"""
    one_body_tensor, two_body_tensor, is_unrestricted = _normalize_hamiltonian(
        one_body_tensor, two_body_tensor, norb
    )
    bitstring_matrix = np.asarray(bitstring_matrix, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    if spin_sq is not None:
        if not np.isfinite(spin_sq) or spin_sq < 0.0:
            raise ValueError("spin_sq must be a finite non-negative number or None.")
        if sci_solver is not None:
            raise ValueError("spin_sq cannot be combined with a custom sci_solver.")
        if is_unrestricted:
            raise ValueError(
                "spin_sq constraints are not supported for unrestricted Hamiltonians."
            )
    if max_iterations < 1:
        raise ValueError("Maximum number of iterations must be at least 1.")
    extension_threshold = carryover_threshold if ext_sqd_threshold is None else ext_sqd_threshold
    if ext_sqd_threshold_mode not in {"amplitude", "probability"}:
        raise ValueError("ext_sqd_threshold_mode must be 'amplitude' or 'probability'.")
    if not np.isfinite(extension_threshold) or extension_threshold < 0:
        raise ValueError("ext_sqd_threshold must be finite and non-negative.")
    if bitstring_matrix.ndim != 2:
        raise ValueError("bitstring_matrix must be a 2D array.")
    if bitstring_matrix.shape[1] != 2 * norb:
        raise ValueError("bitstring_matrix must have exactly 2 * norb columns.")
    if len(probabilities) != bitstring_matrix.shape[0]:
        raise ValueError(
            "The number of probabilities must match the number of bitstrings."
        )
    if np.sum(probabilities) <= 0:
        raise ValueError("The probability sum must be positive.")
    probabilities = probabilities / np.sum(probabilities)

    n_alpha, n_beta = nelec
    excluded_a, excluded_b = _normalize_excluded_orbitals(excluded_orbitals, norb)
    if n_alpha > norb - len(excluded_a) or n_beta > norb - len(excluded_b):
        raise ValueError(
            "The requested electron count exceeds the non-excluded orbitals."
        )
    if is_unrestricted and symmetrize_spin:
        raise ValueError(
            "Spin symmetrization is not supported for unrestricted Hamiltonians."
        )
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
        raise ValueError(
            "max_dim must be the same for alpha and beta when symmetrizing spin."
        )
    if symmetrize_spin and not np.array_equal(excluded_a, excluded_b):
        raise ValueError(
            "Alpha and beta excluded_orbitals must match when symmetrizing spin."
        )

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
        # 默认求解器继续接收相同五个位置参数，dummy 约束由闭包补充。
        def default_sci_solver(strings, h1e, eri, n_orbitals, electron_counts):
            return solve_sci_batch(
                strings,
                h1e,
                eri,
                n_orbitals,
                electron_counts,
                spin_sq=spin_sq,
                excluded_orbitals=(excluded_a, excluded_b),
            )

        sci_solver = default_sci_solver

    include_a = _filter_excluded_strings(np.unique(include_a), excluded_a)
    include_b = _filter_excluded_strings(np.unique(include_b), excluded_b)
    carryover_strings_a = np.array([], dtype=np.int64)
    carryover_strings_b = np.array([], dtype=np.int64)

    for _ in range(max_iterations):
        if current_occupancies is None:
            bitstrings, probs = postselect_by_hamming_right_and_left(
                bitstring_matrix,
                probabilities,
                hamming_right=n_alpha,
                hamming_left=n_beta,
                excluded_right=excluded_a,
                excluded_left=excluded_b,
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
                excluded_orbitals_a=excluded_a,
                excluded_orbitals_b=excluded_b,
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
                strings = np.concatenate(
                    (include_a, include_b, carryover_strings_a, samples_ab)
                )
                strings = _filter_excluded_strings(strings, excluded_a)
                strs_a = strs_b = _unique_with_order_preserved(strings)[:max_dim_a]
            else:
                samples_a = samples_a[np.argsort(counts_a)[::-1]]
                samples_b = samples_b[np.argsort(counts_b)[::-1]]
                strs_a = np.concatenate((include_a, carryover_strings_a, samples_a))
                strs_b = np.concatenate((include_b, carryover_strings_b, samples_b))
                strs_a = _filter_excluded_strings(strs_a, excluded_a)
                strs_b = _filter_excluded_strings(strs_b, excluded_b)
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
        if average_batch_occupancies:
            # David 等的 PCM-SQD 流程用所有 batch 的平均 1-RDM 对角占据数
            # 驱动下一轮 S-CORE；保留最低能 batch 作为该轮报告能量。
            next_occupancies = tuple(
                np.mean(
                    [
                        np.asarray(result.orbital_occupancies[spin], dtype=float)
                        for result in results
                    ],
                    axis=0,
                )
                for spin in (0, 1)
            )
        else:
            next_occupancies = best_result_in_batch.orbital_occupancies
        if best_result is None or best_result_in_batch.energy < best_result.energy:
            best_result = best_result_in_batch

        if (
            current_result is not None
            and abs(current_result.energy - best_result_in_batch.energy) < energy_tol
            and np.linalg.norm(
                np.ravel(cast(tuple[np.ndarray, np.ndarray], current_occupancies))
                - np.ravel(next_occupancies),
                ord=np.inf,
            )
            < occupancies_tol
        ):
            break

        current_result = best_result_in_batch
        current_occupancies = next_occupancies
        carryover_strings_a, carryover_strings_b = _carryover_strings(
            current_result,
            carryover_threshold,
            symmetrize_spin,
        )

    result = cast(SCIResult, best_result)
    if ext_sqd:
        result = _run_ext_sqd_final_diagonalization(
            result,
            one_body_tensor,
            two_body_tensor,
            norb,
            nelec,
            extension_threshold,
            symmetrize_spin,
            sci_solver,
            excluded_a,
            excluded_b,
            threshold_mode=ext_sqd_threshold_mode,
        )

    return replace(result, nuclear_repulsion_energy=float(nuclear_repulsion_energy))


def solve_sci_batch(
    ci_strings: list[tuple[np.ndarray, np.ndarray]],
    one_body_tensor: OneBodyTensor,
    two_body_tensor: TwoBodyTensor,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    spin_shift: float = 0.1,
    excluded_orbitals: ExcludedOrbitals | None = None,
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
            spin_shift=spin_shift,
            excluded_orbitals=excluded_orbitals,
            **kwargs,
        )
        for strings in ci_strings
    ]


def solve_sci(
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: OneBodyTensor,
    two_body_tensor: TwoBodyTensor,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    spin_shift: float = 0.1,
    excluded_orbitals: ExcludedOrbitals | None = None,
    **kwargs,
) -> SCIResult:
    """在一个 selected-CI 子空间里投影并对角化 restricted/UHF Hamiltonian。"""
    one_body_tensor, two_body_tensor, is_unrestricted = _normalize_hamiltonian(
        one_body_tensor, two_body_tensor, norb
    )
    excluded_a, excluded_b = _normalize_excluded_orbitals(excluded_orbitals, norb)
    _validate_selected_strings(ci_strings[0], excluded_a, "alpha")
    _validate_selected_strings(ci_strings[1], excluded_b, "beta")

    myci = fci.selected_ci.SelectedCI()
    if is_unrestricted and spin_sq is not None:
        raise ValueError(
            "spin_sq constraints are not supported for unrestricted Hamiltonians."
        )
    if spin_sq is not None:
        if not np.isfinite(spin_shift) or spin_shift <= 0.0:
            raise ValueError("spin_shift must be a finite positive number.")
        myci = fci.addons.fix_spin_(myci, shift=spin_shift, ss=spin_sq)

    if is_unrestricted:
        _, sci_vec = kernel_fixed_space_uhf(
            myci,
            cast(tuple[np.ndarray, np.ndarray], one_body_tensor),
            cast(tuple[np.ndarray, np.ndarray, np.ndarray], two_body_tensor),
            norb,
            nelec,
            ci_strs=ci_strings,
            **kwargs,
        )
    else:
        _, sci_vec = fci.selected_ci.kernel_fixed_space(
            myci,
            cast(np.ndarray, one_body_tensor),
            cast(np.ndarray, two_body_tensor),
            norb,
            nelec,
            ci_strs=ci_strings,
            **kwargs,
        )
    dm1s = myci.make_rdm1s(sci_vec, norb, nelec)
    occupancies = (np.diagonal(dm1s[0]), np.diagonal(dm1s[1]))
    dm2s = myci.make_rdm2s(sci_vec, norb, nelec)
    dm1 = myci.make_rdm1(sci_vec, norb, nelec)
    dm2 = myci.make_rdm2(sci_vec, norb, nelec)
    if is_unrestricted:
        h1a, h1b = cast(tuple[np.ndarray, np.ndarray], one_body_tensor)
        gaa, gab, gbb = cast(tuple[np.ndarray, np.ndarray, np.ndarray], two_body_tensor)
        energy = (
            np.einsum("pr,pr->", dm1s[0], h1a)
            + np.einsum("pr,pr->", dm1s[1], h1b)
            + 0.5 * np.einsum("prqs,prqs->", dm2s[0], gaa)
            + np.einsum("prqs,prqs->", dm2s[1], gab)
            + 0.5 * np.einsum("prqs,prqs->", dm2s[2], gbb)
        )
    else:
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
    return SCIResult(
        float(np.real_if_close(energy)),
        sci_state,
        orbital_occupancies=occupancies,
        rdm1=dm1,
        rdm2=dm2,
        rdm1s=dm1s,
        rdm2s=dm2s,
        converged=bool(myci.converged),
    )


def _normalize_hamiltonian(
    one_body_tensor: OneBodyTensor,
    two_body_tensor: TwoBodyTensor,
    norb: int,
) -> tuple[OneBodyTensor, TwoBodyTensor, bool]:
    """检查 restricted/unrestricted 积分表示，并统一为 NumPy 数组。"""
    one_is_tuple = isinstance(one_body_tensor, tuple)
    two_is_tuple = isinstance(two_body_tensor, tuple)
    if one_is_tuple != two_is_tuple:
        raise ValueError(
            "one_body_tensor and two_body_tensor must both use restricted arrays or "
            "unrestricted tuples."
        )

    if one_is_tuple:
        if len(one_body_tensor) != 2 or len(cast(tuple, two_body_tensor)) != 3:
            raise ValueError(
                "Unrestricted tensors must contain (h_alpha, h_beta) and "
                "(g_aa, g_ab, g_bb)."
            )
        h1_blocks = tuple(np.asarray(block) for block in one_body_tensor)
        eri_blocks = tuple(np.asarray(block) for block in cast(tuple, two_body_tensor))
        if any(block.shape != (norb, norb) for block in h1_blocks):
            raise ValueError(
                f"Each unrestricted one-body tensor must have shape ({norb}, {norb})."
            )
        if any(block.shape != (norb, norb, norb, norb) for block in eri_blocks):
            raise ValueError(
                "Each unrestricted two-body tensor must have shape "
                f"({norb}, {norb}, {norb}, {norb})."
            )
        if any(np.iscomplexobj(block) for block in (*h1_blocks, *eri_blocks)):
            raise NotImplementedError(
                "Unrestricted SQD currently supports real tensors only."
            )
        return (
            cast(tuple[np.ndarray, np.ndarray], h1_blocks),
            cast(tuple[np.ndarray, np.ndarray, np.ndarray], eri_blocks),
            True,
        )

    h1 = np.asarray(one_body_tensor)
    eri = np.asarray(two_body_tensor)
    if h1.shape != (norb, norb):
        raise ValueError(f"one_body_tensor must have shape ({norb}, {norb}).")
    if eri.shape != (norb, norb, norb, norb):
        raise ValueError(
            f"two_body_tensor must have shape ({norb}, {norb}, {norb}, {norb})."
        )
    return h1, eri, False


def _normalize_excluded_orbitals(
    excluded_orbitals: ExcludedOrbitals | None,
    norb: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把公共或 spin-resolved dummy 轨道输入统一成 ``(alpha, beta)``。"""
    if excluded_orbitals is None:
        empty = np.array([], dtype=int)
        return empty, empty.copy()

    is_spin_pair = (
        isinstance(excluded_orbitals, tuple)
        and len(excluded_orbitals) == 2
        and not np.isscalar(excluded_orbitals[0])
        and not np.isscalar(excluded_orbitals[1])
    )
    if is_spin_pair:
        excluded_a = _as_excluded_array(excluded_orbitals[0], norb)
        excluded_b = _as_excluded_array(excluded_orbitals[1], norb)
        return excluded_a, excluded_b

    shared = _as_excluded_array(
        cast(Sequence[int] | np.ndarray, excluded_orbitals), norb
    )
    return shared, shared.copy()


def _as_excluded_array(values: Sequence[int] | np.ndarray, norb: int) -> np.ndarray:
    """检查一组空间轨道编号。"""
    array = np.asarray(values)
    try:
        integer_array = array.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("excluded_orbitals must contain integers.") from exc
    if array.ndim != 1 or np.any(array != integer_array):
        raise ValueError(
            "excluded_orbitals must be a one-dimensional sequence of integers."
        )
    integer_array = np.unique(integer_array)
    if np.any(integer_array < 0) or np.any(integer_array >= norb):
        raise ValueError(f"excluded_orbitals must be between 0 and {norb - 1}.")
    return integer_array


def _filter_excluded_strings(
    strings: np.ndarray, excluded_orbitals: np.ndarray
) -> np.ndarray:
    """删除占据任一 dummy 轨道的 determinant string。"""
    strings = np.asarray(strings)
    if not excluded_orbitals.size or not strings.size:
        return strings
    mask = sum(1 << int(orbital) for orbital in excluded_orbitals)
    keep = np.fromiter(
        ((int(string) & mask) == 0 for string in strings),
        dtype=bool,
        count=len(strings),
    )
    return strings[keep]


def _validate_selected_strings(
    strings: np.ndarray,
    excluded_orbitals: np.ndarray,
    spin: str,
) -> None:
    """阻止直接 ``solve_sci`` 调用把 dummy determinant 带入子空间。"""
    filtered = _filter_excluded_strings(np.asarray(strings), excluded_orbitals)
    if len(filtered) != len(strings):
        raise ValueError(
            f"Selected {spin} CI strings must not occupy excluded_orbitals."
        )


def _filter_excluded_bitstrings(
    bitstrings: np.ndarray,
    norb: int,
    excluded_a: np.ndarray,
    excluded_b: np.ndarray,
) -> np.ndarray:
    """过滤 ext-SQD 单激发中落入 dummy 轨道的样本。"""
    if not bitstrings.size:
        return bitstrings
    keep = np.ones(len(bitstrings), dtype=bool)
    if excluded_a.size:
        columns_a = 2 * norb - 1 - excluded_a
        keep &= ~np.any(bitstrings[:, columns_a], axis=1)
    if excluded_b.size:
        columns_b = norb - 1 - excluded_b
        keep &= ~np.any(bitstrings[:, columns_b], axis=1)
    return bitstrings[keep]


def _unique_with_order_preserved(vals: np.ndarray) -> np.ndarray:
    """去重但保留原来的优先级顺序。"""
    _, indices = np.unique(vals, return_index=True)
    indices.sort()
    return vals[indices]


def _run_ext_sqd_final_diagonalization(
    result: SCIResult,
    one_body_tensor: OneBodyTensor,
    two_body_tensor: TwoBodyTensor,
    norb: int,
    nelec: tuple[int, int],
    carryover_threshold: float,
    symmetrize_spin: bool,
    sci_solver: Callable[
        [
            list[tuple[np.ndarray, np.ndarray]],
            OneBodyTensor,
            TwoBodyTensor,
            int,
            tuple[int, int],
        ],
        list[SCIResult],
    ],
    excluded_a: np.ndarray,
    excluded_b: np.ndarray,
    *,
    threshold_mode: str = "amplitude",
) -> SCIResult:
    """用 dominant product configurations 的单激发子空间再对角化一次。"""
    dominant_bitstrings = _dominant_product_bitstrings(
        result, carryover_threshold, threshold_mode=threshold_mode
    )
    expanded_bitstrings = enlarge_batch_from_single_excitations(dominant_bitstrings)
    expanded_bitstrings = _filter_excluded_bitstrings(
        expanded_bitstrings, norb, excluded_a, excluded_b
    )
    ci_strings = _merge_ext_sqd_ci_strings(
        result, expanded_bitstrings, norb, symmetrize_spin
    )
    return min(
        sci_solver([ci_strings], one_body_tensor, two_body_tensor, norb, nelec),
        key=lambda ext_result: ext_result.energy,
    )


def _dominant_product_bitstrings(
    result: SCIResult, carryover_threshold: float, *, threshold_mode: str = "amplitude"
) -> np.ndarray:
    """从 final SCI coefficient matrix 里取 dominant alpha/beta product pairs。"""
    sci_state = result.sci_state
    coefficients = np.abs(sci_state.amplitudes)
    if threshold_mode == "amplitude":
        dominant_indices = np.argwhere(coefficients >= carryover_threshold)
    elif threshold_mode == "probability":
        dominant_indices = np.argwhere(coefficients**2 > carryover_threshold)
    else:
        raise ValueError("threshold_mode must be 'amplitude' or 'probability'.")
    if dominant_indices.size == 0:
        dominant_indices = np.asarray(
            [np.unravel_index(np.argmax(coefficients), coefficients.shape)]
        )

    weights = coefficients[dominant_indices[:, 0], dominant_indices[:, 1]]
    dominant_indices = dominant_indices[np.argsort(weights)[::-1]]
    alpha_strings = sci_state.ci_strs_a[dominant_indices[:, 0]]
    beta_strings = sci_state.ci_strs_b[dominant_indices[:, 1]]

    alpha_bits = _integers_to_bitstring_matrix(alpha_strings, sci_state.norb)
    beta_bits = _integers_to_bitstring_matrix(beta_strings, sci_state.norb)
    return np.concatenate((beta_bits, alpha_bits), axis=1)


def _merge_ext_sqd_ci_strings(
    result: SCIResult,
    expanded_bitstrings: np.ndarray,
    norb: int,
    symmetrize_spin: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """把单激发扩展得到的 determinant strings 与 final SCI 子空间合并。"""
    sci_state = result.sci_state
    expanded_a = bitstring_matrix_to_integers(expanded_bitstrings[:, norb:])
    expanded_b = bitstring_matrix_to_integers(expanded_bitstrings[:, :norb])
    if symmetrize_spin:
        strings = np.concatenate(
            (sci_state.ci_strs_a, sci_state.ci_strs_b, expanded_a, expanded_b)
        )
        strings = np.unique(strings)
        return strings, strings

    return (
        np.unique(np.concatenate((sci_state.ci_strs_a, expanded_a))),
        np.unique(np.concatenate((sci_state.ci_strs_b, expanded_b))),
    )


def _integers_to_bitstring_matrix(strings: np.ndarray, n_bits: int) -> np.ndarray:
    """把 PySCF selected-CI string 整数转回固定宽度 bitstring 矩阵。"""
    strings = np.asarray(strings)
    bitstring_matrix = np.zeros((len(strings), n_bits), dtype=bool)
    for row, value in enumerate(strings):
        integer = int(value)
        for col in range(n_bits):
            bitstring_matrix[row, col] = bool((integer >> (n_bits - 1 - col)) & 1)
    return bitstring_matrix


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
    carryover_index = np.searchsorted(
        absolute_vals, carryover_threshold, sorter=indices
    )
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
