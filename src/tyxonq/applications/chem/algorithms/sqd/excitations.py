"""SQD 子空间的费米子激发扩展工具。"""

from __future__ import annotations

import numpy as np


def enlarge_batch_from_transitions(
    bitstring_matrix: np.ndarray,
    transition_operators: np.ndarray,
) -> np.ndarray:
    """把 transition operators 作用到一批 bitstring 上。

    ``bitstring_matrix`` 使用 TyxonQ SQD 内部顺序 ``[beta | alpha]``。
    transition 的 ``+`` 只能作用在空轨道，``-`` 只能作用在已占轨道，
    ``n`` 要求该轨道已占但不改变 bit。
    """
    bitstring_matrix = np.asarray(bitstring_matrix, dtype=bool)
    transitions = np.asarray(transition_operators)
    if bitstring_matrix.ndim != 2:
        raise ValueError("bitstring_matrix must be a 2D array.")
    if transitions.ndim == 1:
        transitions = transitions.reshape(1, -1)
    if transitions.ndim != 2:
        raise ValueError("transition_operators must be a 1D or 2D array.")
    if transitions.shape[1] != bitstring_matrix.shape[1]:
        raise ValueError("transition_operators must have the same width as bitstring_matrix.")
    if not np.isin(transitions, ["I", "+", "-", "n"]).all():
        raise ValueError("transition_operators may only contain 'I', '+', '-', and 'n'.")

    rows = []
    for transition in transitions:
        create = transition == "+"
        annihilate = transition == "-"
        number = transition == "n"

        valid = np.ones(bitstring_matrix.shape[0], dtype=bool)
        if np.any(create):
            valid &= ~np.any(bitstring_matrix[:, create], axis=1)
        if np.any(annihilate):
            valid &= np.all(bitstring_matrix[:, annihilate], axis=1)
        if np.any(number):
            valid &= np.all(bitstring_matrix[:, number], axis=1)
        if not np.any(valid):
            continue

        updated = bitstring_matrix[valid].copy()
        updated[:, create] = True
        updated[:, annihilate] = False
        rows.append(updated)

    if not rows:
        return np.empty((0, bitstring_matrix.shape[1]), dtype=bool)
    return np.concatenate(rows, axis=0)


def enlarge_batch_from_single_excitations(bitstring_matrix: np.ndarray) -> np.ndarray:
    """用同自旋 single particle-hole excitations 扩展 bitstring batch。"""
    bitstring_matrix = np.asarray(bitstring_matrix, dtype=bool)
    if bitstring_matrix.ndim != 2:
        raise ValueError("bitstring_matrix must be a 2D array.")
    if bitstring_matrix.shape[1] % 2:
        raise ValueError("The length of the bitstrings must be even.")

    norb = bitstring_matrix.shape[1] // 2
    transitions = [np.full(2 * norb, "I", dtype="<U1")]
    for offset in (0, norb):
        for i in range(norb):
            for j in range(i + 1, norb):
                forward = np.full(2 * norb, "I", dtype="<U1")
                forward[offset + i] = "+"
                forward[offset + j] = "-"
                transitions.append(forward)

                backward = np.full(2 * norb, "I", dtype="<U1")
                backward[offset + i] = "-"
                backward[offset + j] = "+"
                transitions.append(backward)

    expanded = enlarge_batch_from_transitions(bitstring_matrix, np.asarray(transitions))
    return _unique_rows_preserving_order(expanded)


def _unique_rows_preserving_order(bitstring_matrix: np.ndarray) -> np.ndarray:
    """按首次出现顺序去掉重复 bitstring。"""
    seen: set[bytes] = set()
    rows = []
    for row in bitstring_matrix:
        key = row.tobytes()
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    if not rows:
        return np.empty((0, bitstring_matrix.shape[1]), dtype=bool)
    return np.asarray(rows, dtype=bool)
