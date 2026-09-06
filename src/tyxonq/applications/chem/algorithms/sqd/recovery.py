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
) -> np.ndarray:
    """修正单个 bitstring，同时保证左右两半的目标电子数。"""
    bit_array = bit_array.copy()
    num_bits = bit_array.shape[0]
    partition_size = num_bits // 2

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

    probs_left = np.minimum(1, np.maximum(0, probs_left))
    probs_right = np.minimum(1, np.maximum(0, probs_right))

    # 左半边是 beta，右半边是 alpha，两边必须分开修。
    _correct_partition(bit_array[:partition_size], probs_left, hamming_left, rng)
    _correct_partition(bit_array[partition_size:], probs_right, hamming_right, rng)
    return bit_array


def _correct_partition(
    partition: np.ndarray,
    probabilities: np.ndarray,
    target_hamming: int,
    rng: np.random.Generator,
) -> None:
    """把某一半 bitstring 的 1 的数量修到目标值。"""
    n_diff = int(np.sum(partition) - target_hamming)
    if n_diff == 0:
        return

    if n_diff > 0:
        occupied = np.where(partition)[0]
        p_choice = probabilities[partition]
        # 严格 HF 占据可能使所有候选权重为零，此时均匀选择。
        if not np.any(p_choice):
            p_choice = np.ones(len(occupied), dtype=float)
        p_choice = p_choice / np.sum(p_choice)
        to_flip = rng.choice(occupied, size=n_diff, replace=False, p=p_choice)
        partition[to_flip] = False
    elif n_diff < 0:
        empty = np.where(np.logical_not(partition))[0]
        p_choice = probabilities[np.logical_not(partition)]
        # 严格空占据同样可能给出全零权重，仍需满足目标电子数。
        if not np.any(p_choice):
            p_choice = np.ones(len(empty), dtype=float)
        p_choice = p_choice / np.sum(p_choice)
        to_flip = rng.choice(empty, size=abs(n_diff), replace=False, p=p_choice)
        partition[to_flip] = True
