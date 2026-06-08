"""SQD 的电子数筛选和批量抽样工具。"""

from __future__ import annotations

import numpy as np


def postselect_by_hamming_right_and_left(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    *,
    hamming_right: int,
    hamming_left: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按左右两半的 Hamming weight 筛选合法 bitstring。"""
    bitstring_matrix = np.asarray(bitstring_matrix, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    if hamming_left < 0 or hamming_right < 0:
        raise ValueError("Hamming weight must be specified with a non-negative integer.")
    if bitstring_matrix.ndim != 2:
        raise ValueError("bitstring_matrix must be a 2D array.")

    n_bitstrings, n_bits = bitstring_matrix.shape
    if n_bits % 2:
        raise ValueError(f"The length of the bitstrings must be even. Instead, got {n_bits}.")
    if len(probabilities) != n_bitstrings:
        raise ValueError(
            "The number of probabilities must match the number of rows in bitstring_matrix."
        )

    # SQD 内部约定 left=beta、right=alpha；nelec 输入顺序是 (n_alpha, n_beta)。
    norb = n_bits // 2
    valid_right = np.sum(bitstring_matrix[:, norb:], axis=1) == hamming_right
    valid_left = np.sum(bitstring_matrix[:, :norb], axis=1) == hamming_left
    valid_indices = np.logical_and(valid_right, valid_left)

    bitstrings_post = bitstring_matrix[valid_indices]
    probabilities_post = probabilities[valid_indices]
    if not len(probabilities_post):
        return bitstrings_post, probabilities_post
    probabilities_post = probabilities_post / np.sum(probabilities_post)
    return bitstrings_post, probabilities_post


def subsample(
    bitstring_matrix: np.ndarray,
    probabilities: np.ndarray,
    samples_per_batch: int,
    num_batches: int,
    rand_seed: np.random.Generator | int | None = None,
) -> list[np.ndarray]:
    """从合法 bitstring 中抽取若干 batch。"""
    bitstring_matrix = np.asarray(bitstring_matrix, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    if samples_per_batch < 1:
        raise ValueError("Samples per batch must be specified with a positive integer.")
    if num_batches < 1:
        raise ValueError("The number of batches must be specified with a positive integer.")
    if bitstring_matrix.shape[0] < 1:
        return [np.empty((0, bitstring_matrix.shape[1]), dtype=bool) for _ in range(num_batches)]
    if len(probabilities) != bitstring_matrix.shape[0]:
        raise ValueError(
            "The number of probabilities must match the number of rows in bitstring_matrix."
        )

    rng = np.random.default_rng(rand_seed)
    num_bitstrings = bitstring_matrix.shape[0]
    randomly_sample = samples_per_batch < num_bitstrings
    indices = np.arange(num_bitstrings, dtype=int)

    batches = []
    for _ in range(num_batches):
        if randomly_sample:
            indices = rng.choice(
                np.arange(num_bitstrings, dtype=int),
                samples_per_batch,
                replace=False,
                p=probabilities,
            )
        batches.append(bitstring_matrix[indices])
    return batches
