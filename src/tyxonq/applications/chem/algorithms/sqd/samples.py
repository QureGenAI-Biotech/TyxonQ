"""TyxonQ SQD 的样本转换工具。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

SampleOrder = str


def samples_to_arrays(
    samples: Mapping[str, float | int] | np.ndarray | Sequence[Sequence[bool | int]],
    probabilities: Sequence[float] | np.ndarray | None = None,
    *,
    sample_order: SampleOrder = "alpha_beta",
) -> tuple[np.ndarray, np.ndarray]:
    """把 counts 或 bitstring 矩阵转换成内部 ``[beta | alpha]`` 顺序。

    ``sample_order="alpha_beta"`` 只交换 alpha/beta 两个半区，不反转
    半区内部。TyxonQ/LUCJ raw qubit order 进入本函数前，需要先调用
    ``reverse_bitstring_halves`` 转成 SQD/PySCF order。
    """
    if isinstance(samples, Mapping):
        if probabilities is not None:
            raise ValueError("probabilities must be None when samples is a counts dictionary.")
        bitstring_matrix, probabilities_array = counts_to_arrays(samples)
    else:
        bitstring_matrix = _as_bitstring_matrix(samples)
        if probabilities is None:
            bitstring_matrix, probabilities_array = _unique_rows_with_probabilities(bitstring_matrix)
        else:
            probabilities_array = _normalize_probabilities(bitstring_matrix, probabilities)

    bitstring_matrix = _convert_sample_order(bitstring_matrix, sample_order)
    return bitstring_matrix, probabilities_array


def counts_to_arrays(counts: Mapping[str, float | int]) -> tuple[np.ndarray, np.ndarray]:
    """把 counts 字典转换成 bitstring 矩阵和概率数组。"""
    if not counts:
        return np.empty((0, 0), dtype=bool), np.array([], dtype=float)

    _validate_counts_bitstrings(counts)
    prob_dict = normalize_counts_dict(counts)
    bitstring_matrix = np.array([[bit == "1" for bit in bitstring] for bitstring in prob_dict])
    probabilities = np.array(list(prob_dict.values()), dtype=float)
    return bitstring_matrix, probabilities


def generate_counts_uniform(
    num_samples: int, num_bits: int, rand_seed: np.random.Generator | int | None = None
) -> dict[str, int]:
    """生成均匀随机 counts，主要用于简单测试和示例。"""
    if num_samples < 1:
        raise ValueError("The number of samples must be specified with a positive integer.")
    if num_bits < 1:
        raise ValueError("The number of bits must be specified with a positive integer.")

    rng = np.random.default_rng(rand_seed)
    bitstring_matrix = rng.choice([0, 1], size=(num_samples, num_bits))
    counts: dict[str, int] = {}
    for row in bitstring_matrix:
        bitstring = "".join("1" if bit else "0" for bit in row.astype(int))
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def generate_counts_bipartite_hamming(
    num_samples: int,
    num_bits: int,
    *,
    hamming_right: int,
    hamming_left: int,
    rand_seed: np.random.Generator | int | None = None,
) -> dict[str, int]:
    """生成左右两半 Hamming weight 固定的 counts。"""
    if num_bits % 2:
        raise ValueError("The number of bits must be specified with an even integer.")
    if num_samples < 1:
        raise ValueError("The number of samples must be specified with a positive integer.")
    if num_bits < 1:
        raise ValueError("The number of bits must be specified with a positive integer.")
    if hamming_left < 0 or hamming_right < 0:
        raise ValueError("Hamming weights must be specified as non-negative integers.")

    rng = np.random.default_rng(rand_seed)
    half = num_bits // 2
    counts: dict[str, int] = {}
    for _ in range(num_samples):
        right_flips = rng.choice(np.arange(half), hamming_right, replace=False).astype(int)
        left_flips = rng.choice(np.arange(half), hamming_left, replace=False).astype(int)
        row = np.zeros(num_bits, dtype=int)
        row[left_flips] = 1
        row[right_flips + half] = 1
        bitstring = "".join("1" if bit else "0" for bit in row)
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


def normalize_counts_dict(counts: Mapping[str, float | int]) -> dict[str, float]:
    """把 counts 出现次数归一化成概率。"""
    if not counts:
        return {}
    total_counts = float(sum(counts.values()))
    if total_counts <= 0:
        raise ValueError("The total count must be positive.")
    return {bitstring: float(count) / total_counts for bitstring, count in counts.items()}


def bitstring_matrix_to_integers(bitstring_matrix: np.ndarray) -> np.ndarray:
    """把 bitstring 矩阵的每一行转成 PySCF selected-CI string 整数。"""
    bitstring_matrix = np.asarray(bitstring_matrix, dtype=bool)
    if bitstring_matrix.ndim != 2:
        raise ValueError("bitstring_matrix must be a 2D array.")

    n_bitstrings, n_bits = bitstring_matrix.shape
    if n_bits < 64:
        dtype: type = int
    else:
        dtype = object
        bitstring_matrix = bitstring_matrix.astype(object)

    result = np.zeros(n_bitstrings, dtype=dtype)
    for i in range(n_bits):
        result += bitstring_matrix[:, i] * (1 << (n_bits - 1 - i))
    return result


def reverse_bitstring_halves(bitstring: str) -> str:
    """在 TyxonQ/LUCJ raw order 和 SQD/PySCF order 之间转换 bitstring。

    TyxonQ/LUCJ raw order 为
    ``[alpha0..alphaN-1 | beta0..betaN-1]``；SQD/PySCF order 为
    ``[alphaN-1..alpha0 | betaN-1..beta0]``。
    """
    if not isinstance(bitstring, str):
        raise TypeError("bitstring must be a string.")
    if len(bitstring) % 2:
        raise ValueError("The length of the bitstring must be even.")
    if any(bit not in {"0", "1"} for bit in bitstring):
        raise ValueError("bitstring must contain only '0' and '1'.")
    half = len(bitstring) // 2
    return bitstring[:half][::-1] + bitstring[half:][::-1]


def _as_bitstring_matrix(
    samples: np.ndarray | Sequence[Sequence[bool | int]],
) -> np.ndarray:
    """把用户传入的矩阵样本整理成二维布尔矩阵。"""
    bitstring_matrix = np.asarray(samples, dtype=bool)
    if bitstring_matrix.ndim != 2:
        raise ValueError("samples must be a 2D bitstring matrix.")
    if bitstring_matrix.shape[1] % 2:
        raise ValueError("The length of the bitstrings must be even.")
    return bitstring_matrix


def _unique_rows_with_probabilities(bitstring_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """矩阵没有显式概率时，按重复行的次数估计概率。"""
    if bitstring_matrix.shape[0] == 0:
        return bitstring_matrix, np.array([], dtype=float)
    unique_rows, counts = np.unique(bitstring_matrix, axis=0, return_counts=True)
    probabilities = counts.astype(float) / float(bitstring_matrix.shape[0])
    return unique_rows, probabilities


def _normalize_probabilities(
    bitstring_matrix: np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """检查并归一化用户显式传入的概率。"""
    probabilities_array = np.asarray(probabilities, dtype=float)
    if probabilities_array.ndim != 1:
        raise ValueError("probabilities must be a 1D array.")
    if len(probabilities_array) != bitstring_matrix.shape[0]:
        raise ValueError(
            "The number of probabilities must match the number of rows in the sample matrix."
        )
    total = float(np.sum(probabilities_array))
    if total <= 0:
        raise ValueError("The probability sum must be positive.")
    return probabilities_array / total


def _convert_sample_order(bitstring_matrix: np.ndarray, sample_order: SampleOrder) -> np.ndarray:
    """把用户顺序转换成 SQD 内部顺序。"""
    if sample_order not in {"alpha_beta", "beta_alpha"}:
        raise ValueError("sample_order must be either 'alpha_beta' or 'beta_alpha'.")
    if bitstring_matrix.size == 0:
        return bitstring_matrix
    if bitstring_matrix.shape[1] % 2:
        raise ValueError("The length of the bitstrings must be even.")
    if sample_order == "beta_alpha":
        return bitstring_matrix

    half = bitstring_matrix.shape[1] // 2
    return np.concatenate((bitstring_matrix[:, half:], bitstring_matrix[:, :half]), axis=1)


def _validate_counts_bitstrings(counts: Mapping[str, float | int]) -> None:
    """检查 counts 的键和值是否能作为 bitstring 样本使用。"""
    lengths = {len(bitstring) for bitstring in counts}
    if len(lengths) != 1:
        raise ValueError("All bitstrings in counts must have the same length.")
    num_bits = next(iter(lengths))
    if num_bits % 2:
        raise ValueError("The length of the bitstrings must be even.")
    for bitstring, count in counts.items():
        if any(bit not in {"0", "1"} for bit in bitstring):
            raise ValueError("Counts keys must be bitstrings containing only '0' and '1'.")
        if count < 0:
            raise ValueError("Counts values must be non-negative.")
