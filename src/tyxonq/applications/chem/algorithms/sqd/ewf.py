"""将 selected-CI 的能量、波函数及 RDM 转换为 EWF 片段求解结果。"""

from __future__ import annotations

import warnings

import numpy as np
from pyscf import cc

from ..ewf.types import Cluster, SolverResult, SpinArray, SpinERI
from .fermion import SCIResult, SCIState


def _validated_string_positions(
    strings: np.ndarray,
    *,
    norb: int,
    nelec: int,
    dummy_orbitals: tuple[int, ...],
    spin: str,
) -> dict[int, int]:
    """验证 determinant pool，并返回 bitstring 到 selected-CI 下标的映射。"""

    values = np.asarray(strings)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{spin} CI strings must be a one-dimensional integer array.")
    determinants = [int(value) for value in values]
    if len(set(determinants)) != len(determinants):
        raise ValueError(f"{spin} CI strings contain duplicate determinants.")

    upper_bound = 1 << norb
    dummy_mask = sum(1 << orbital for orbital in dummy_orbitals)
    for determinant in determinants:
        if determinant < 0 or determinant >= upper_bound:
            raise ValueError(f"{spin} determinant {determinant} is outside {norb} orbitals.")
        if determinant.bit_count() != nelec:
            raise ValueError(
                f"{spin} determinant {determinant} has {determinant.bit_count()} electrons; "
                f"expected {nelec}."
            )
        if determinant & dummy_mask:
            raise ValueError(f"{spin} determinant {determinant} occupies a dummy orbital.")
    return {determinant: index for index, determinant in enumerate(determinants)}


def _single_excitation_strings(norb: int, nocc: int) -> tuple[list[int], np.ndarray]:
    """按 PySCF CISD 的 ``(occupied, virtual)`` 顺序生成单激发及费米符号。"""

    reference = (1 << nocc) - 1
    strings: list[int] = []
    signs: list[int] = []
    for occupied in range(nocc):
        sign = -1 if (nocc - occupied - 1) % 2 else 1
        hole = reference ^ (1 << occupied)
        for virtual in range(nocc, norb):
            strings.append(hole | (1 << virtual))
            signs.append(sign)
    return strings, np.asarray(signs, dtype=int)


def _double_excitation_strings(norb: int, nocc: int) -> tuple[list[int], np.ndarray]:
    """按 PySCF 四重反对称展开所需的 compact 顺序生成同自旋双激发。"""

    reference = (1 << nocc) - 1
    nvir = norb - nocc
    occupied_pairs = [
        (low, high) for high in range(1, nocc) for low in range(high)
    ]
    virtual_pairs = [
        (low, high) for high in range(1, nvir) for low in range(high)
    ]
    strings: list[int] = []
    signs: list[int] = []
    for first, second in occupied_pairs:
        sign = -1 if (first + second + 1) % 2 else 1
        holes = reference ^ (1 << first) ^ (1 << second)
        for first_virtual, second_virtual in virtual_pairs:
            strings.append(
                holes
                | (1 << (nocc + first_virtual))
                | (1 << (nocc + second_virtual))
            )
            signs.append(sign)
    return strings, np.asarray(signs, dtype=int)


def _selected_values(
    strings: list[int],
    signs: np.ndarray,
    positions: dict[int, int],
    amplitudes: np.ndarray,
) -> np.ndarray:
    """从一个 selected determinant 轴提取系数；缺失激发严格按零处理。"""

    values = np.zeros(len(strings), dtype=np.result_type(amplitudes.dtype, np.float64))
    for output, (determinant, sign) in enumerate(zip(strings, signs, strict=True)):
        position = positions.get(determinant)
        if position is not None:
            values[output] = amplitudes[position] * sign
    return values


def _sci_state_to_cisd(
    state: SCIState,
    cluster: Cluster,
    *,
    zero_threshold: float,
    warning_threshold: float,
) -> tuple[float | complex, SpinArray, np.ndarray | SpinERI]:
    """直接从 selected determinant pools 提取参考、C1 和 C2。"""

    if state.norb != cluster.norb or tuple(state.nelec) != tuple(cluster.nelec):
        raise ValueError(
            "SCIState metadata does not match the EWF cluster: "
            f"norb={state.norb}/{cluster.norb}, nelec={state.nelec}/{cluster.nelec}."
        )
    amplitudes = np.asarray(state.amplitudes)
    expected_shape = (len(state.ci_strs_a), len(state.ci_strs_b))
    if amplitudes.shape != expected_shape:
        raise ValueError(f"SCI amplitudes have shape {amplitudes.shape}; expected {expected_shape}.")

    nocca, noccb = cluster.nelec
    dummy_a, dummy_b = cluster.dummy_orbitals
    positions_a = _validated_string_positions(
        state.ci_strs_a,
        norb=cluster.norb,
        nelec=nocca,
        dummy_orbitals=dummy_a,
        spin="alpha",
    )
    positions_b = _validated_string_positions(
        state.ci_strs_b,
        norb=cluster.norb,
        nelec=noccb,
        dummy_orbitals=dummy_b,
        spin="beta",
    )
    reference_a = (1 << nocca) - 1
    reference_b = (1 << noccb) - 1
    if reference_a not in positions_a or reference_b not in positions_b:
        raise ValueError("The SCIState does not contain the EWF reference determinant.")
    reference = amplitudes[positions_a[reference_a], positions_b[reference_b]]
    if abs(reference) <= zero_threshold:
        raise ValueError(
            "The SCI reference amplitude is too small for intermediate normalization: "
            f"{reference!r}."
        )
    if abs(reference) < warning_threshold:
        warnings.warn(
            "The SCI reference amplitude is small; intermediate-normalized C1/C2 "
            "may be numerically unstable.",
            RuntimeWarning,
            stacklevel=3,
        )

    singles_a, signs_a = _single_excitation_strings(cluster.norb, nocca)
    singles_b, signs_b = _single_excitation_strings(cluster.norb, noccb)
    ref_index_a = positions_a[reference_a]
    ref_index_b = positions_b[reference_b]
    c1a = _selected_values(
        singles_a,
        signs_a,
        positions_a,
        amplitudes[:, ref_index_b],
    ).reshape(nocca, cluster.norb - nocca)
    c1b = _selected_values(
        singles_b,
        signs_b,
        positions_b,
        amplitudes[ref_index_a, :],
    ).reshape(noccb, cluster.norb - noccb)

    if not cluster.is_unrestricted:
        cross = np.zeros(
            (len(singles_a), len(singles_b)),
            dtype=np.result_type(amplitudes.dtype, np.float64),
        )
        for row, (det_a, sign_a) in enumerate(zip(singles_a, signs_a, strict=True)):
            selected_a = positions_a.get(det_a)
            if selected_a is None:
                continue
            for column, (det_b, sign_b) in enumerate(
                zip(singles_b, signs_b, strict=True)
            ):
                selected_b = positions_b.get(det_b)
                if selected_b is not None:
                    cross[row, column] = (
                        amplitudes[selected_a, selected_b] * sign_a * sign_b
                    )
        c2 = cross.reshape(
            nocca,
            cluster.norb - nocca,
            noccb,
            cluster.norb - noccb,
        ).transpose(0, 2, 1, 3)
        return reference.item(), c1b / reference, c2 / reference

    doubles_a, double_signs_a = _double_excitation_strings(cluster.norb, nocca)
    doubles_b, double_signs_b = _double_excitation_strings(cluster.norb, noccb)
    compact_aa = _selected_values(
        doubles_a,
        double_signs_a,
        positions_a,
        amplitudes[:, ref_index_b],
    ).reshape(
        nocca * (nocca - 1) // 2,
        (cluster.norb - nocca) * (cluster.norb - nocca - 1) // 2,
    )
    compact_bb = _selected_values(
        doubles_b,
        double_signs_b,
        positions_b,
        amplitudes[ref_index_a, :],
    ).reshape(
        noccb * (noccb - 1) // 2,
        (cluster.norb - noccb) * (cluster.norb - noccb - 1) // 2,
    )
    c2aa = cc.ccsd._unpack_4fold(compact_aa, nocca, cluster.norb - nocca)
    c2bb = cc.ccsd._unpack_4fold(compact_bb, noccb, cluster.norb - noccb)

    cross = np.zeros(
        (len(singles_a), len(singles_b)),
        dtype=np.result_type(amplitudes.dtype, np.float64),
    )
    for row, (det_a, sign_a) in enumerate(zip(singles_a, signs_a, strict=True)):
        selected_a = positions_a.get(det_a)
        if selected_a is None:
            continue
        for column, (det_b, sign_b) in enumerate(zip(singles_b, signs_b, strict=True)):
            selected_b = positions_b.get(det_b)
            if selected_b is not None:
                cross[row, column] = amplitudes[selected_a, selected_b] * sign_a * sign_b
    c2ab = cross.reshape(
        nocca,
        cluster.norb - nocca,
        noccb,
        cluster.norb - noccb,
    ).transpose(0, 2, 1, 3)
    return (
        reference.item(),
        (c1a / reference, c1b / reference),
        (c2aa / reference, c2ab / reference, c2bb / reference),
    )


def sci_state_to_cisd(
    state: SCIState,
    cluster: Cluster,
    *,
    zero_threshold: float = 1e-12,
    warning_threshold: float = 0.1,
) -> tuple[float, SpinArray, np.ndarray | SpinERI]:
    """不构造完整 FCI 矩阵，提取中间归一化 ``C0/C1/C2``。"""

    _, c1, c2 = _sci_state_to_cisd(
        state,
        cluster,
        zero_threshold=zero_threshold,
        warning_threshold=warning_threshold,
    )
    return 1.0, c1, c2


def sci_result_to_solver_result(
    result: SCIResult,
    cluster: Cluster,
    *,
    zero_threshold: float = 1e-12,
    warning_threshold: float = 0.1,
    include_cisd: bool = True,
) -> SolverResult:
    """将 selected-CI 结果接入 EWF；密度路径可跳过依赖非零 C0 的转换。

    ``include_cisd=False`` 只传 RDM；仍验证轨道、电子数与组态合法性。
    返回的收敛标记来自 CI 求解器，不代表 SQD 恢复迭代收敛。
    """
    state = result.sci_state
    if state.norb != cluster.norb or tuple(state.nelec) != tuple(cluster.nelec):
        raise ValueError("SCIState metadata does not match the EWF cluster.")
    reference = c1 = c2 = None
    if include_cisd:
        reference, c1, c2 = _sci_state_to_cisd(
            state, cluster, zero_threshold=zero_threshold,
            warning_threshold=warning_threshold,
        )
    else:
        for spin, strings, nelec, dummy in zip(
            ("alpha", "beta"), (state.ci_strs_a, state.ci_strs_b),
            cluster.nelec, cluster.dummy_orbitals, strict=True,
        ):
            _validated_string_positions(
                strings, norb=cluster.norb, nelec=nelec,
                dummy_orbitals=dummy, spin=spin,
            )
    if cluster.is_unrestricted:
        if result.rdm1s is None or result.rdm2s is None:
            raise ValueError("An unrestricted SCIResult must contain spin-resolved RDMs.")
        rdm1: SpinArray = tuple(np.asarray(value) for value in result.rdm1s)
        rdm2: np.ndarray | SpinERI = tuple(np.asarray(value) for value in result.rdm2s)
    else:
        if result.rdm1 is None or result.rdm2 is None:
            raise ValueError("A restricted SCIResult must contain spin-summed RDMs.")
        rdm1 = np.asarray(result.rdm1)
        rdm2 = np.asarray(result.rdm2)
    matrices = rdm1 if cluster.is_unrestricted else (rdm1,)
    tensors = rdm2 if cluster.is_unrestricted else (rdm2,)
    if any(x.shape != (cluster.norb,) * 2 for x in matrices):
        raise ValueError("SCI 1-RDM shape does not match the EWF cluster.")
    if any(x.shape != (cluster.norb,) * 4 for x in tensors):
        raise ValueError("SCI 2-RDM shape does not match the EWF cluster.")
    if not all(np.all(np.isfinite(x)) for x in (*matrices, *tensors)):
        raise ValueError("SCI RDMs must be finite.")
    counts = cluster.nelec if cluster.is_unrestricted else (sum(cluster.nelec),)
    if any(not np.isclose(np.trace(x), n, atol=1e-7, rtol=0)
           for x, n in zip(matrices, counts, strict=True)):
        raise ValueError("SCI 1-RDM trace does not match the electron count.")
    return SolverResult(
        method="SCI",
        converged=bool(result.converged),
        energy=float(result.energy),
        rdm1=rdm1,
        rdm2=rdm2,
        reference_amplitude=reference,
        cisd_c1=c1,
        cisd_c2=c2,
    )


__all__ = ["sci_result_to_solver_result", "sci_state_to_cisd"]
