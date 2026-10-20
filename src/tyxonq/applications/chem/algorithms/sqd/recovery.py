"""SQD 的 configuration recovery 模块。"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def recover_configurations(
    bitstring_matrix: np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    avg_occupancies: tuple[np.ndarray, np.ndarray],
    num_elec_a: int,
    num_elec_b: int,
    rand_seed: np.random.Generator | int | None = None,
    *,
    excluded_orbitals_a: Sequence[int] | np.ndarray | None = None,
    excluded_orbitals_b: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """用上一轮轨道占据数修复 alpha/beta 电子数不对的 bitstring。"""
    bitstring_matrix = np.asarray(bitstring_matrix, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(rand_seed)

    occ_dims = len(np.array(avg_occupancies).shape)
    if occ_dims == 1:
        warnings.warn(
            "Passing avg_occupancies as a 1D array is deprecated. Pass a length-2 tuple "
            "containing the spin-up and spin-down occupancies respectively.",
            DeprecationWarning,
            stacklevel=2,
        )
        norb = bitstring_matrix.shape[1] // 2
        avg_occupancies = (np.flip(avg_occupancies[norb:]), np.flip(avg_occupancies[:norb]))

    if num_elec_a < 0 or num_elec_b < 0:
        raise ValueError("The numbers of electrons must be specified as non-negative integers.")
    if probabilities.shape[0] != bitstring_matrix.shape[0]:
        raise ValueError(
            "The number of probabilities must match the number of rows in bitstring_matrix."
        )

    norb = bitstring_matrix.shape[1] // 2
    excluded_a = _validate_excluded_orbitals(excluded_orbitals_a, norb)
    excluded_b = _validate_excluded_orbitals(excluded_orbitals_b, norb)
    if num_elec_a > norb - len(excluded_a) or num_elec_b > norb - len(excluded_b):
        raise ValueError("The requested electron count exceeds the non-excluded orbitals.")

    allowed_internal = np.ones(2 * norb, dtype=bool)
    allowed_internal[norb - 1 - excluded_b] = False
    allowed_internal[2 * norb - 1 - excluded_a] = False

    corrected_dict: defaultdict[str, float] = defaultdict(float)
    # 内部 bitstring 顺序是 [beta | alpha]，occupancies 也要翻成同样顺序。
    occs_array = np.flip(avg_occupancies).flatten()
    for bitstring, freq in zip(bitstring_matrix, probabilities):
        corrected = _bipartite_bitstring_correcting(
            bitstring,
            occs_array,
            num_elec_a,
            num_elec_b,
            rng=rng,
            allowed_orbitals=allowed_internal,
        )
        corrected_key = "".join("1" if bit else "0" for bit in corrected)
        corrected_dict[corrected_key] += float(freq)

    bitstrings_out = np.array([[bit == "1" for bit in key] for key in corrected_dict], dtype=bool)
    probabilities_out = np.array(list(corrected_dict.values()), dtype=float)
    probabilities_out = np.abs(probabilities_out) / np.sum(np.abs(probabilities_out))
    return bitstrings_out, probabilities_out


def _p_flip_0_to_1(ratio_exp: float, occ: float, eps: float = 0.01) -> float:
    """估计把某个 bit 从 0 翻成 1 的概率。"""
    if occ < ratio_exp:
        return occ * eps / ratio_exp
    if ratio_exp == 1.0:
        return eps
    slope = (1 - eps) / (1 - ratio_exp)
    intercept = 1 - slope
    return occ * slope + intercept


def _p_flip_1_to_0(ratio_exp: float, occ: float, eps: float = 0.01) -> float:
    """估计把某个 bit 从 1 翻成 0 的概率。"""
    return _p_flip_0_to_1(1 - ratio_exp, 1 - occ, eps)


def _bipartite_bitstring_correcting(
    bit_array: np.ndarray,
    avg_occupancies: np.ndarray,
    hamming_right: int,
    hamming_left: int,
    rng: np.random.Generator,
    allowed_orbitals: np.ndarray | None = None,
) -> np.ndarray:
    """修正单个 bitstring，同时保证左右两半的目标电子数。"""
    bit_array = bit_array.copy()
    num_bits = bit_array.shape[0]
    partition_size = num_bits // 2
    if allowed_orbitals is None:
        allowed_orbitals = np.ones(num_bits, dtype=bool)
    else:
        allowed_orbitals = np.asarray(allowed_orbitals, dtype=bool)
        if allowed_orbitals.shape != bit_array.shape:
            raise ValueError("allowed_orbitals must have the same shape as one bitstring.")
    # dummy 轨道先强制清零，后续修复也只允许在物理轨道间翻转。
    bit_array[~allowed_orbitals] = False

    probs_left = np.zeros(partition_size)
    probs_right = np.zeros(partition_size)
    for i in range(partition_size):
        if bit_array[i]:
            probs_left[i] = _p_flip_1_to_0(hamming_left / partition_size, avg_occupancies[i])
        else:
            probs_left[i] = _p_flip_0_to_1(hamming_left / partition_size, avg_occupancies[i])

        if bit_array[i + partition_size]:
            probs_right[i] = _p_flip_1_to_0(
                hamming_right / partition_size,
                avg_occupancies[i + partition_size],
            )
        else:
            probs_right[i] = _p_flip_0_to_1(
                hamming_right / partition_size,
                avg_occupancies[i + partition_size],
            )

    probs_left[~allowed_orbitals[:partition_size]] = 0.0
    probs_right[~allowed_orbitals[partition_size:]] = 0.0

    probs_left = np.minimum(1, np.maximum(0, probs_left))
    probs_right = np.minimum(1, np.maximum(0, probs_right))

    # 左半边是 beta，右半边是 alpha，两边必须分开修。
    _correct_partition(
        bit_array[:partition_size],
        probs_left,
        hamming_left,
        rng,
        allowed_orbitals[:partition_size],
    )
    _correct_partition(
        bit_array[partition_size:],
        probs_right,
        hamming_right,
        rng,
        allowed_orbitals[partition_size:],
    )
    return bit_array


def _correct_partition(
    partition: np.ndarray,
    probabilities: np.ndarray,
    target_hamming: int,
    rng: np.random.Generator,
    allowed_orbitals: np.ndarray | None = None,
) -> None:
    """把某一半 bitstring 的 1 的数量修到目标值。"""
    if allowed_orbitals is None:
        allowed_orbitals = np.ones_like(partition, dtype=bool)
    if target_hamming > int(np.sum(allowed_orbitals)):
        raise ValueError("Target Hamming weight exceeds the allowed orbitals.")

    n_diff = int(np.sum(partition) - target_hamming)
    # 已满足目标电子数时不需要概率，也不能让全零权重产生 NaN。
    if n_diff == 0:
        return

    if n_diff > 0:
        # 删除电子时同样只在允许的物理轨道中选择。
        occupied = np.where(np.logical_and(partition, allowed_orbitals))[0]
        p_choice = probabilities[occupied]
        # 严格 HF 占据可使所有翻转权重为零；此时均匀随机选择。
        if not np.any(p_choice):
            p_choice = np.ones(len(occupied), dtype=float)
        p_choice = p_choice / np.sum(p_choice)
        to_flip = rng.choice(occupied, size=n_diff, replace=False, p=p_choice)
        partition[to_flip] = False
    elif n_diff < 0:
        candidates = np.logical_and(np.logical_not(partition), allowed_orbitals)
        empty = np.where(candidates)[0]
        p_choice = probabilities[candidates]
        # 严格 HF 占据可使所有翻转权重为零；此时均匀随机选择。
        if not np.any(p_choice):
            p_choice = np.ones(len(empty), dtype=float)
        p_choice = p_choice / np.sum(p_choice)
        to_flip = rng.choice(empty, size=abs(n_diff), replace=False, p=p_choice)
        partition[to_flip] = True


def _validate_excluded_orbitals(
    excluded_orbitals: Sequence[int] | np.ndarray | None,
    norb: int,
) -> np.ndarray:
    """检查 recovery 使用的 dummy 轨道编号。"""
    if excluded_orbitals is None:
        return np.array([], dtype=int)
    orbitals = np.asarray(excluded_orbitals)
    if orbitals.ndim != 1 or np.any(orbitals != orbitals.astype(int)):
        raise ValueError("excluded orbitals must be a one-dimensional sequence of integers.")
    orbitals = np.unique(orbitals.astype(int))
    if np.any(orbitals < 0) or np.any(orbitals >= norb):
        raise ValueError(f"excluded orbitals must be between 0 and {norb - 1}.")
    return orbitals
