"""构造分子 RHF/UHF 嵌入计算的 DMET 浴与 MP2 浴自然轨道。"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Literal

import numpy as np
from pyscf import ao2mo

from .linalg import (
    assert_orthonormal,
    canonicalize_subspace,
    eigh,
    fix_orbital_sign,
    hstack,
    metric_overlap,
    orthonormality_error,
)
from .types import BathData


OrbitalType = Literal["occupied", "virtual"]


@dataclass(frozen=True)
class DMETBath:
    """DMET bath 及其 occupied/virtual 环境划分。"""

    c_dmet: np.ndarray | tuple[np.ndarray, np.ndarray]
    occupations: np.ndarray | tuple[np.ndarray, np.ndarray]
    c_cluster_occ: np.ndarray | tuple[np.ndarray, np.ndarray]
    c_cluster_vir: np.ndarray | tuple[np.ndarray, np.ndarray]
    c_env_occ: np.ndarray | tuple[np.ndarray, np.ndarray]
    c_env_vir: np.ndarray | tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class MP2BNO:
    """一个 occupied 或 virtual MP2-BNO bath 的完整截断结果。"""

    c_bath: np.ndarray | tuple[np.ndarray, np.ndarray]
    c_frozen: np.ndarray | tuple[np.ndarray, np.ndarray]
    c_all: np.ndarray | tuple[np.ndarray, np.ndarray]
    occupations: np.ndarray | tuple[np.ndarray, np.ndarray]
    correlation_energy: float | None


def _get_mean_field(fragment: object) -> object:
    """从分片或其 embedding 容器取得 PySCF mean-field。"""
    mf = getattr(fragment, "mf", None)
    if mf is None:
        mf = getattr(getattr(fragment, "base", None), "mf", None)
    if mf is None:
        raise TypeError("fragment must expose 'mf' or 'base.mf'.")
    return mf


def _get_overlap(fragment: object, mf: object) -> np.ndarray:
    base = getattr(fragment, "base", None)
    getter = getattr(base, "get_ovlp", None)
    overlap = getter() if callable(getter) else mf.get_ovlp()
    return _as_real(overlap, name="AO overlap")


def _get_fock(fragment: object, mf: object) -> np.ndarray:
    base = getattr(fragment, "base", None)
    getter = getattr(base, "get_fock_for_bath", None)
    if not callable(getter):
        getter = getattr(base, "get_fock", None)
    fock = getter() if callable(getter) else mf.get_fock()
    return _as_real(fock, name="AO Fock")


def _as_real(array: np.ndarray, *, name: str) -> np.ndarray:
    """当前分子 RHF/UHF 实现只接受实数轨道和积分。"""
    value = np.real_if_close(np.asarray(array), tol=1000)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real for the EWF implementation.")
    return np.asarray(value, dtype=float)


def _empty_coeff(nrow: int) -> np.ndarray:
    return np.zeros((nrow, 0), dtype=float)


def _canonicalize(
    coeff: np.ndarray,
    fock: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回轨道、从原轨道到新轨道的旋转和轨道能量。"""
    norb = coeff.shape[1]
    if enabled:
        return canonicalize_subspace(coeff, fock)
    rotation = np.eye(norb)
    energies = np.einsum("ai,ab,bi->i", coeff, fock, coeff, optimize=True)
    return coeff.copy(), rotation, energies


def build_dmet_bath(
    fragment: object,
    *,
    dmet_threshold: float = 1e-8,
    occupation_tolerance: float = 1e-6,
) -> DMETBath:
    """由分片环境的一阶 RHF 密度矩阵构造 DMET bath。

    ``fragment`` 只需提供 ``c_frag``、``c_env``，以及 ``mf`` 或
    ``base.mf``。所有轨道系数均按 AO overlap metric 正交归一。
    """
    if not 0 <= dmet_threshold < 0.5:
        raise ValueError("dmet_threshold must be in [0, 0.5).")
    if occupation_tolerance <= 0:
        raise ValueError("occupation_tolerance must be positive.")

    c_frag_value = getattr(fragment, "c_frag")
    if isinstance(c_frag_value, tuple):
        return _build_dmet_bath_uhf(
            fragment,
            dmet_threshold=dmet_threshold,
            occupation_tolerance=occupation_tolerance,
        )

    mf = _get_mean_field(fragment)
    overlap = _get_overlap(fragment, mf)
    c_frag = _as_real(getattr(fragment, "c_frag"), name="fragment orbitals")
    c_env = _as_real(getattr(fragment, "c_env"), name="environment orbitals")
    nao = c_frag.shape[0]
    if c_env.shape[0] != nao:
        raise ValueError("fragment and environment orbitals have different AO dimensions.")
    assert_orthonormal(
        hstack(c_frag, c_env, nrow=nao),
        overlap,
        name="fragment plus environment orbitals",
    )

    if c_env.shape[1] == 0:
        c_dmet = _empty_coeff(nao)
        occupations = np.zeros(0)
        c_env_occ = _empty_coeff(nao)
        c_env_vir = _empty_coeff(nao)
    else:
        dm1 = _as_real(mf.make_rdm1(), name="RHF one-particle density matrix")
        if dm1.ndim != 2:
            raise TypeError("build_dmet_bath only supports a restricted RHF density matrix.")

        # RHF occupation为 0/2；除以 2 后，环境谱位于 [0, 1]。
        sc_env = overlap @ c_env
        dm_env = np.linalg.multi_dot((sc_env.T, dm1, sc_env)) / 2
        occupations, rotation = eigh(dm_env)
        order = np.argsort(occupations)[::-1]
        occupations = occupations[order]
        c_env_rot = fix_orbital_sign(c_env @ rotation[:, order])

        if occupations[0] > 1 + 1e-8 or occupations[-1] < -1e-8:
            warnings.warn(
                "DMET environment occupations fall outside [0, 1] by more than 1e-8.",
                RuntimeWarning,
                stacklevel=2,
            )

        is_dmet = (occupations >= dmet_threshold) & (
            occupations <= 1 - dmet_threshold
        )
        is_occ_env = occupations > 1 - dmet_threshold
        is_vir_env = occupations < dmet_threshold
        if np.count_nonzero(is_dmet | is_occ_env | is_vir_env) != c_env.shape[1]:
            raise RuntimeError("DMET bath masks do not partition the environment.")

        c_dmet = c_env_rot[:, is_dmet].copy()
        c_env_occ = c_env_rot[:, is_occ_env].copy()
        c_env_vir = c_env_rot[:, is_vir_env].copy()
        occupations = occupations[is_dmet].copy()

    # fragment+DMET 空间的 RHF 密度谱应重新成为 2/0，随后分成占据和虚轨道。
    c_raw = hstack(c_frag, c_dmet, nrow=nao)
    assert_orthonormal(c_raw, overlap, name="fragment plus DMET orbitals")
    dm1 = _as_real(mf.make_rdm1(), name="RHF one-particle density matrix")
    sc_raw = overlap @ c_raw
    cluster_dm = np.linalg.multi_dot((sc_raw.T, dm1, sc_raw))
    cluster_occ, rotation = eigh(cluster_dm)
    order = np.argsort(cluster_occ)[::-1]
    cluster_occ = cluster_occ[order]
    c_raw = fix_orbital_sign(c_raw @ rotation[:, order])

    residual = np.minimum(np.abs(cluster_occ), np.abs(cluster_occ - 2))
    if residual.size and np.max(residual) > 2 * occupation_tolerance:
        raise RuntimeError(
            "DMET cluster density eigenvalues are not all close to 0 or 2: "
            f"max residual={np.max(residual):.3e}."
        )
    nocc = int(np.count_nonzero(cluster_occ >= 1))
    fock = _get_fock(fragment, mf)
    c_cluster_occ = canonicalize_subspace(c_raw[:, :nocc], fock)[0]
    c_cluster_vir = canonicalize_subspace(c_raw[:, nocc:], fock)[0]

    all_orbitals = hstack(
        c_cluster_occ,
        c_cluster_vir,
        c_env_occ,
        c_env_vir,
        nrow=nao,
    )
    assert_orthonormal(all_orbitals, overlap, name="DMET orbital partition")
    return DMETBath(
        c_dmet=c_dmet,
        occupations=occupations,
        c_cluster_occ=c_cluster_occ,
        c_cluster_vir=c_cluster_vir,
        c_env_occ=c_env_occ,
        c_env_vir=c_env_vir,
    )


def _build_dmet_bath_uhf(
    fragment: object,
    *,
    dmet_threshold: float,
    occupation_tolerance: float,
) -> DMETBath:
    """按 Vayesta 约定逐 spin 调用 restricted DMET 构造。"""

    mf = _get_mean_field(fragment)
    overlap = _get_overlap(fragment, mf)
    fock = tuple(_as_real(value, name="UHF AO Fock") for value in mf.get_fock())
    dm1 = tuple(
        _as_real(value, name="UHF one-particle density matrix")
        for value in mf.make_rdm1()
    )

    class _SpinMF:
        def __init__(self, density, fock_matrix):
            self._density = density
            self._fock = fock_matrix

        def make_rdm1(self):
            # restricted helper 以 0/2 归一化密度工作。
            return 2.0 * self._density

        def get_ovlp(self):
            return overlap

        def get_fock(self):
            return self._fock

    class _SpinFragment:
        pass

    results = []
    for spin in range(2):
        proxy = _SpinFragment()
        proxy.mf = _SpinMF(dm1[spin], fock[spin])
        proxy.c_frag = fragment.c_frag[spin]
        proxy.c_env = fragment.c_env[spin]
        results.append(
            build_dmet_bath(
                proxy,
                dmet_threshold=dmet_threshold,
                occupation_tolerance=occupation_tolerance,
            )
        )
    return DMETBath(
        **{
            field: tuple(getattr(result, field) for result in results)
            for field in (
                "c_dmet",
                "occupations",
                "c_cluster_occ",
                "c_cluster_vir",
                "c_env_occ",
                "c_env_vir",
            )
        }
    )


def dmet_projector_weights(
    occupations: np.ndarray,
    mode: str,
) -> np.ndarray:
    """返回 Vayesta MP2-BNO 使用的 DMET 纠缠权重。"""
    occupations = np.asarray(occupations, dtype=float)
    if np.any(occupations < -1e-10) or np.any(occupations > 1 + 1e-10):
        raise ValueError("DMET occupations must lie in [0, 1].")
    values = np.clip(occupations, 0, 1)
    if mode == "full":
        weights = np.zeros_like(values)
    elif mode == "half":
        weights = np.full_like(values, 0.5)
    elif mode == "linear":
        weights = 2 * np.minimum(values, 1 - values)
    elif mode == "cosine":
        weights = (1 - np.cos(2 * np.pi * values)) / 2
    elif mode == "cosine-half":
        weights = (1 - np.cos(2 * np.pi * values)) / 4
    elif mode == "entropy":
        weights = 4 * values * (1 - values)
    elif mode == "sqrt-entropy":
        weights = 2 * np.sqrt(values * (1 - values))
    elif mode == "squared-entropy":
        weights = (4 * values * (1 - values)) ** 2
    else:
        raise ValueError(f"Invalid project_dmet_mode: {mode!r}.")
    return np.clip(weights, 0, 1)


def _project_t2(
    t2: np.ndarray,
    *,
    occtype: OrbitalType,
    c_occ: np.ndarray,
    c_vir: np.ndarray,
    c_frag: np.ndarray,
    dmet: DMETBath,
    overlap: np.ndarray,
    order: int,
    mode: str,
) -> np.ndarray:
    """在与目标 BNO 相反的 T2 指标上施加 fragment+DMET projector。"""
    if order == 0:
        return t2
    weights = np.concatenate(
        (np.ones(c_frag.shape[1]), dmet_projector_weights(dmet.occupations, mode))
    )
    c_fragdmet = hstack(c_frag, dmet.c_dmet, nrow=c_frag.shape[0])

    if occtype == "occupied":
        rotation = metric_overlap(c_vir, overlap, c_fragdmet)
        projector = (rotation * weights) @ rotation.T
        if order == 1:
            t2 = np.einsum("xa,ijab->ijxb", projector, t2, optimize=True)
        else:
            t2 = np.einsum(
                "xa,yb,ijab->ijxy", projector, projector, t2, optimize=True
            )
    else:
        rotation = metric_overlap(c_occ, overlap, c_fragdmet)
        projector = (rotation * weights) @ rotation.T
        if order == 1:
            t2 = np.einsum("xi,ijab->xjab", projector, t2, optimize=True)
        else:
            t2 = np.einsum(
                "xi,yj,ijab->xyab", projector, projector, t2, optimize=True
            )
    return (t2 + t2.transpose(1, 0, 3, 2)) / 2


def _transform_ovov(
    mf: object,
    c_occ: np.ndarray,
    c_vir: np.ndarray,
) -> np.ndarray:
    """返回 chemist notation 的 ``(ia|jb)`` 四指标积分。"""
    coefficients = (c_occ, c_vir, c_occ, c_vir)
    with_df = getattr(mf, "with_df", None)
    if with_df is not None:
        transformed = with_df.ao2mo(coefficients, compact=False)
    else:
        source = getattr(mf, "_eri", None)
        if source is None:
            source = mf.mol
        transformed = ao2mo.kernel(source, coefficients, compact=False)
    shape = (c_occ.shape[1], c_vir.shape[1]) * 2
    return _as_real(np.asarray(transformed).reshape(shape), name="MP2 (ov|ov) integrals")


def _make_mp2_amplitudes(
    fragment: object,
    c_occ: np.ndarray,
    c_vir: np.ndarray,
    occ_energy: np.ndarray,
    vir_energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """构造 restricted MP2 ``t2`` 以及 first-occupied-index 投影能。"""
    mf = _get_mean_field(fragment)
    eri_iajb = _transform_ovov(mf, c_occ, c_vir)
    gijab = eri_iajb.transpose(0, 2, 1, 3)
    denominator = (
        occ_energy[:, None, None, None]
        - vir_energy[None, None, :, None]
        + occ_energy[None, :, None, None]
        - vir_energy[None, None, None, :]
    )
    if denominator.size and np.any(np.abs(denominator) < 1e-12):
        raise RuntimeError("MP2 energy denominator is zero or numerically singular.")
    t2 = gijab / denominator

    overlap = _get_overlap(fragment, mf)
    c_frag = _as_real(getattr(fragment, "c_frag"), name="fragment orbitals")
    rfrag = metric_overlap(c_occ, overlap, c_frag)
    projected_t2 = np.einsum("ix,ijab->xjab", rfrag, t2, optimize=True)
    projected_eri = np.einsum("ix,ijab->xjab", rfrag, gijab, optimize=True)
    correlation_energy = 2 * np.einsum(
        "xjab,xjab->", projected_t2, projected_eri, optimize=True
    ) - np.einsum(
        "xjab,xjba->", projected_t2, projected_eri, optimize=True
    )
    return t2, gijab, float(correlation_energy)


def _transform_ovov_uhf(
    mf: object,
    c_occ: tuple[np.ndarray, np.ndarray],
    c_vir: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 UHF ``aa/ab/bb`` 的 ``(ia|jb)`` 积分。"""

    coefficient_sets = (
        (c_occ[0], c_vir[0], c_occ[0], c_vir[0]),
        (c_occ[0], c_vir[0], c_occ[1], c_vir[1]),
        (c_occ[1], c_vir[1], c_occ[1], c_vir[1]),
    )
    with_df = getattr(mf, "with_df", None)
    source = getattr(mf, "_eri", None)
    if source is None:
        source = mf.mol
    values = []
    for coeffs in coefficient_sets:
        if with_df is not None:
            transformed = with_df.ao2mo(coeffs, compact=False)
        else:
            transformed = ao2mo.general(source, coeffs, compact=False)
        shape = tuple(value.shape[1] for value in coeffs)
        values.append(
            _as_real(np.asarray(transformed).reshape(shape), name="UMP2 (ov|ov) integrals")
        )
    return tuple(values)


def _make_ump2_amplitudes(
    fragment: object,
    c_occ: tuple[np.ndarray, np.ndarray],
    c_vir: tuple[np.ndarray, np.ndarray],
    occ_energy: tuple[np.ndarray, np.ndarray],
    vir_energy: tuple[np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """构造 Vayesta UMP2-BNO 使用的 ``t2aa/t2ab/t2bb``。"""

    mf = _get_mean_field(fragment)
    eri_aa, eri_ab, eri_bb = _transform_ovov_uhf(mf, c_occ, c_vir)
    gijaa = eri_aa.transpose(0, 2, 1, 3)
    gijab = eri_ab.transpose(0, 2, 1, 3)
    gijbb = eri_bb.transpose(0, 2, 1, 3)
    eia = tuple(
        occ_energy[s][:, None] - vir_energy[s][None, :] for s in range(2)
    )
    den_aa = eia[0][:, None, :, None] + eia[0][None, :, None, :]
    den_ab = eia[0][:, None, :, None] + eia[1][None, :, None, :]
    den_bb = eia[1][:, None, :, None] + eia[1][None, :, None, :]
    for denominator in (den_aa, den_ab, den_bb):
        if denominator.size and np.any(np.abs(denominator) < 1e-12):
            raise RuntimeError("UMP2 energy denominator is zero or numerically singular.")
    t2aa = gijaa / den_aa
    t2aa -= t2aa.transpose(0, 1, 3, 2)
    t2ab = gijab / den_ab
    t2bb = gijbb / den_bb
    t2bb -= t2bb.transpose(0, 1, 3, 2)

    overlap = _get_overlap(fragment, mf)
    rfrag = tuple(
        metric_overlap(c_occ[s], overlap, fragment.c_frag[s]) for s in range(2)
    )
    tpaa = np.einsum("ix,ijab->xjab", rfrag[0], t2aa, optimize=True)
    gpaa = np.einsum("ix,ijab->xjab", rfrag[0], gijaa, optimize=True)
    tpbb = np.einsum("ix,ijab->xjab", rfrag[1], t2bb, optimize=True)
    gpbb = np.einsum("ix,ijab->xjab", rfrag[1], gijbb, optimize=True)
    ecorr = (
        np.einsum("xjab,xjab->", tpaa, gpaa, optimize=True)
        - np.einsum("xjab,xjba->", tpaa, gpaa, optimize=True)
    ) / 4
    ecorr += (
        np.einsum("xjab,xjab->", tpbb, gpbb, optimize=True)
        - np.einsum("xjab,xjba->", tpbb, gpbb, optimize=True)
    ) / 4
    tpab_a = np.einsum("ix,ijab->xjab", rfrag[0], t2ab, optimize=True)
    gpab_a = np.einsum("ix,ijab->xjab", rfrag[0], gijab, optimize=True)
    tpab_b = np.einsum("jx,ijab->ixab", rfrag[1], t2ab, optimize=True)
    gpab_b = np.einsum("jx,ijab->ixab", rfrag[1], gijab, optimize=True)
    ecorr += 0.5 * np.einsum("xjab,xjab->", tpab_a, gpab_a, optimize=True)
    ecorr += 0.5 * np.einsum("ixab,ixab->", tpab_b, gpab_b, optimize=True)
    return (t2aa, t2ab, t2bb), float(ecorr)


def _project_ump2_t2(
    t2: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    occtype: OrbitalType,
    c_occ: tuple[np.ndarray, np.ndarray],
    c_vir: tuple[np.ndarray, np.ndarray],
    c_frag: tuple[np.ndarray, np.ndarray],
    dmet: DMETBath,
    overlap: np.ndarray,
    order: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 UMP2 的同自旋和异自旋块上施加 fragment+DMET projector。"""

    if order == 0:
        return t2
    projectors = []
    for spin in range(2):
        weights = np.concatenate(
            (
                np.ones(c_frag[spin].shape[1]),
                dmet_projector_weights(dmet.occupations[spin], mode),
            )
        )
        c_fragdmet = hstack(c_frag[spin], dmet.c_dmet[spin], nrow=c_frag[spin].shape[0])
        target = c_vir[spin] if occtype == "occupied" else c_occ[spin]
        rotation = metric_overlap(target, overlap, c_fragdmet)
        projectors.append((rotation * weights) @ rotation.T)
    pa, pb = projectors
    aa, ab, bb = t2
    if occtype == "occupied":
        if order == 1:
            aa = np.einsum("xa,ijab->ijxb", pa, aa, optimize=True)
            bb = np.einsum("xa,ijab->ijxb", pb, bb, optimize=True)
            ab = 0.5 * (
                np.einsum("xa,ijab->ijxb", pa, ab, optimize=True)
                + np.einsum("xb,ijab->ijax", pb, ab, optimize=True)
            )
        else:
            aa = np.einsum("xa,yb,ijab->ijxy", pa, pa, aa, optimize=True)
            bb = np.einsum("xa,yb,ijab->ijxy", pb, pb, bb, optimize=True)
            ab = 0.5 * (
                np.einsum("xa,yb,ijab->ijxy", pa, pb, ab, optimize=True)
                + np.einsum("xb,ya,ijab->ijyx", pb, pa, ab, optimize=True)
            )
    else:
        if order == 1:
            aa = np.einsum("xi,ijab->xjab", pa, aa, optimize=True)
            bb = np.einsum("xi,ijab->xjab", pb, bb, optimize=True)
            ab = 0.5 * (
                np.einsum("xi,ijab->xjab", pa, ab, optimize=True)
                + np.einsum("xj,ijab->ixab", pb, ab, optimize=True)
            )
        else:
            aa = np.einsum("xi,yj,ijab->xyab", pa, pa, aa, optimize=True)
            bb = np.einsum("xi,yj,ijab->xyab", pb, pb, bb, optimize=True)
            ab = 0.5 * (
                np.einsum("xi,yj,ijab->xyab", pa, pb, ab, optimize=True)
                + np.einsum("xj,yi,ijab->yxab", pb, pa, ab, optimize=True)
            )
    aa = (aa + aa.transpose(1, 0, 3, 2)) / 2
    bb = (bb + bb.transpose(1, 0, 3, 2)) / 2
    return aa, ab, bb


def _build_mp2_bno_uhf(
    fragment: object,
    dmet: DMETBath,
    *,
    occtype: OrbitalType,
    threshold: float,
    project_dmet_order: int,
    project_dmet_mode: str,
    canonicalize: bool,
) -> MP2BNO:
    mf = _get_mean_field(fragment)
    fock = tuple(_as_real(value, name="UHF AO Fock") for value in mf.get_fock())
    overlap = _get_overlap(fragment, mf)
    nao = fragment.c_frag[0].shape[0]
    if occtype == "occupied":
        target_env = dmet.c_env_occ
        occ_original = tuple(
            hstack(dmet.c_cluster_occ[s], target_env[s], nrow=nao) for s in range(2)
        )
        vir_original = dmet.c_cluster_vir
        ncluster = tuple(value.shape[1] for value in dmet.c_cluster_occ)
    else:
        target_env = dmet.c_env_vir
        occ_original = dmet.c_cluster_occ
        vir_original = tuple(
            hstack(dmet.c_cluster_vir[s], target_env[s], nrow=nao) for s in range(2)
        )
        ncluster = tuple(value.shape[1] for value in dmet.c_cluster_vir)

    occ_data = tuple(
        _canonicalize(occ_original[s], fock[s], enabled=canonicalize) for s in range(2)
    )
    vir_data = tuple(
        _canonicalize(vir_original[s], fock[s], enabled=canonicalize) for s in range(2)
    )
    c_occ = tuple(value[0] for value in occ_data)
    r_occ = tuple(value[1] for value in occ_data)
    occ_energy = tuple(value[2] for value in occ_data)
    c_vir = tuple(value[0] for value in vir_data)
    r_vir = tuple(value[1] for value in vir_data)
    vir_energy = tuple(value[2] for value in vir_data)
    t2, correlation_energy = _make_ump2_amplitudes(
        fragment, c_occ, c_vir, occ_energy, vir_energy
    )
    t2 = _project_ump2_t2(
        t2,
        occtype=occtype,
        c_occ=c_occ,
        c_vir=c_vir,
        c_frag=fragment.c_frag,
        dmet=dmet,
        overlap=overlap,
        order=project_dmet_order,
        mode=project_dmet_mode,
    )
    aa, ab, bb = t2
    if occtype == "occupied":
        densities = (
            0.5 * np.einsum("imef,jmef->ij", aa, aa, optimize=True)
            + np.einsum("imef,jmef->ij", ab, ab, optimize=True),
            0.5 * np.einsum("imef,jmef->ij", bb, bb, optimize=True)
            + np.einsum("mief,mjef->ij", ab, ab, optimize=True),
        )
        densities = tuple(
            np.linalg.multi_dot((r_occ[s], densities[s], r_occ[s].T))
            for s in range(2)
        )
    else:
        densities = (
            0.5 * np.einsum("mnae,mnbe->ba", aa, aa, optimize=True)
            + np.einsum("mnae,mnbe->ba", ab, ab, optimize=True),
            0.5 * np.einsum("mnae,mnbe->ba", bb, bb, optimize=True)
            + np.einsum("mnea,mneb->ba", ab, ab, optimize=True),
        )
        densities = tuple(
            np.linalg.multi_dot((r_vir[s], densities[s], r_vir[s].T))
            for s in range(2)
        )

    all_coeff, kept_coeff, frozen_coeff, occupations = [], [], [], []
    for spin in range(2):
        density = densities[spin]
        if not np.allclose(density, density.T, atol=1e-10, rtol=0):
            raise RuntimeError("UMP2 delta density matrix is not symmetric.")
        delta_env = density[ncluster[spin] :, ncluster[spin] :]
        values, rotation = eigh(delta_env)
        order = np.argsort(values)[::-1]
        values = values[order]
        coeff = fix_orbital_sign(target_env[spin] @ rotation[:, order])
        assert_orthonormal(coeff, overlap, name=f"spin-{spin} {occtype} UMP2-BNO orbitals")
        nkeep = int(np.count_nonzero(values >= threshold))
        all_coeff.append(coeff)
        kept_coeff.append(coeff[:, :nkeep].copy())
        frozen_coeff.append(coeff[:, nkeep:].copy())
        occupations.append(values)
    return MP2BNO(
        c_bath=tuple(kept_coeff),
        c_frozen=tuple(frozen_coeff),
        c_all=tuple(all_coeff),
        occupations=tuple(occupations),
        correlation_energy=correlation_energy,
    )


def build_mp2_bno(
    fragment: object,
    dmet: DMETBath,
    *,
    occtype: OrbitalType,
    threshold: float = 1e-8,
    project_dmet_order: int = 2,
    project_dmet_mode: str = "squared-entropy",
    canonicalize: bool = True,
) -> MP2BNO:
    """构造并按 occupation threshold 截断一种 MP2-BNO bath。"""
    if occtype not in ("occupied", "virtual"):
        raise ValueError("occtype must be 'occupied' or 'virtual'.")
    if threshold < 0:
        raise ValueError("MP2-BNO threshold must be non-negative.")
    if project_dmet_order not in (0, 1, 2):
        raise ValueError("project_dmet_order must be 0, 1, or 2.")
    # 即使 order=0 也验证 mode，避免静默接受拼写错误。UHF 的两个
    # occupation block 可能长度不同，必须逐自旋验证，不能先强制拼成数组。
    occupation_blocks = (
        dmet.occupations
        if isinstance(dmet.occupations, tuple)
        else (dmet.occupations,)
    )
    for occupations in occupation_blocks:
        dmet_projector_weights(occupations, project_dmet_mode)

    if isinstance(getattr(fragment, "c_frag"), tuple):
        return _build_mp2_bno_uhf(
            fragment,
            dmet,
            occtype=occtype,
            threshold=threshold,
            project_dmet_order=project_dmet_order,
            project_dmet_mode=project_dmet_mode,
            canonicalize=canonicalize,
        )

    mf = _get_mean_field(fragment)
    fock = _get_fock(fragment, mf)
    overlap = _get_overlap(fragment, mf)
    c_frag = _as_real(getattr(fragment, "c_frag"), name="fragment orbitals")
    nao = c_frag.shape[0]

    if occtype == "occupied":
        c_target_env = dmet.c_env_occ
        c_occ_original = hstack(dmet.c_cluster_occ, c_target_env, nrow=nao)
        c_vir_original = dmet.c_cluster_vir
        ncluster = dmet.c_cluster_occ.shape[1]
    else:
        c_target_env = dmet.c_env_vir
        c_occ_original = dmet.c_cluster_occ
        c_vir_original = hstack(dmet.c_cluster_vir, c_target_env, nrow=nao)
        ncluster = dmet.c_cluster_vir.shape[1]

    if c_target_env.shape[1] == 0:
        empty = _empty_coeff(nao)
        return MP2BNO(empty, empty, empty, np.zeros(0), None)

    c_occ, r_occ, occ_energy = _canonicalize(
        c_occ_original, fock, enabled=canonicalize
    )
    c_vir, r_vir, vir_energy = _canonicalize(
        c_vir_original, fock, enabled=canonicalize
    )
    t2, _, correlation_energy = _make_mp2_amplitudes(
        fragment,
        c_occ,
        c_vir,
        occ_energy,
        vir_energy,
    )
    t2 = _project_t2(
        t2,
        occtype=occtype,
        c_occ=c_occ,
        c_vir=c_vir,
        c_frag=c_frag,
        dmet=dmet,
        overlap=overlap,
        order=project_dmet_order,
        mode=project_dmet_mode,
    )

    if occtype == "occupied":
        delta_dm = 2 * np.einsum(
            "ikab,jkab->ij", t2, t2, optimize=True
        ) - np.einsum("ikab,jkba->ij", t2, t2, optimize=True)
        delta_dm = np.linalg.multi_dot((r_occ, delta_dm, r_occ.T))
    else:
        delta_dm = 2 * np.einsum(
            "ijac,ijbc->ab", t2, t2, optimize=True
        ) - np.einsum("ijac,ijcb->ab", t2, t2, optimize=True)
        delta_dm = np.linalg.multi_dot((r_vir, delta_dm, r_vir.T))

    if not np.allclose(delta_dm, delta_dm.T, atol=1e-10, rtol=0):
        raise RuntimeError("MP2 delta density matrix is not symmetric.")
    delta_env = delta_dm[ncluster:, ncluster:]
    occupations, rotation = eigh(delta_env)
    order = np.argsort(occupations)[::-1]
    occupations = occupations[order]
    c_all = fix_orbital_sign(c_target_env @ rotation[:, order])
    assert_orthonormal(c_all, overlap, name=f"{occtype} MP2-BNO orbitals")

    nkeep = int(np.count_nonzero(occupations >= threshold))
    c_bath = c_all[:, :nkeep].copy()
    c_frozen = c_all[:, nkeep:].copy()
    return MP2BNO(
        c_bath=c_bath,
        c_frozen=c_frozen,
        c_all=c_all,
        occupations=occupations,
        correlation_energy=correlation_energy,
    )


def build_bath(
    fragment: object,
    *,
    bathtype: str = "mp2",
    threshold: float = 1e-8,
    dmet_threshold: float = 1e-8,
    occupation_tolerance: float = 1e-6,
    canonicalize: bool = True,
    project_dmet_order: int = 2,
    project_dmet_mode: str = "squared-entropy",
) -> BathData:
    """构造 ``dmet``、``mp2`` 或 ``full`` bath，并组装 ``BathData``。"""
    if bathtype not in ("dmet", "mp2", "full"):
        raise ValueError("bathtype must be 'dmet', 'mp2', or 'full'.")
    if not isinstance(canonicalize, (bool, np.bool_)):
        raise TypeError("canonicalize must be a boolean.")
    # 上游该开关控制最终 active cluster；DMET 与 MP2-BNO 构造自身始终
    # canonicalize。这里接收它以便 driver 传递统一的 bath_options。

    if isinstance(getattr(fragment, "c_frag"), tuple):
        return _build_bath_uhf(
            fragment,
            bathtype=bathtype,
            threshold=threshold,
            dmet_threshold=dmet_threshold,
            occupation_tolerance=occupation_tolerance,
            canonicalize=canonicalize,
            project_dmet_order=project_dmet_order,
            project_dmet_mode=project_dmet_mode,
        )

    dmet = build_dmet_bath(
        fragment,
        dmet_threshold=dmet_threshold,
        occupation_tolerance=occupation_tolerance,
    )
    nao = dmet.c_cluster_occ.shape[0]
    empty = _empty_coeff(nao)
    empty_occupations = np.zeros(0)

    if bathtype == "dmet":
        c_extra_occ = empty
        c_extra_vir = empty
        c_frozen_occ = dmet.c_env_occ
        c_frozen_vir = dmet.c_env_vir
        occ_occupations = empty_occupations
        vir_occupations = empty_occupations
        occ_energy = None
        vir_energy = None
    elif bathtype == "full":
        # BathData 沿用 c_mp2_* 字段表示 DMET 之外的附加 bath。
        c_extra_occ = dmet.c_env_occ
        c_extra_vir = dmet.c_env_vir
        c_frozen_occ = empty
        c_frozen_vir = empty
        occ_occupations = empty_occupations
        vir_occupations = empty_occupations
        occ_energy = None
        vir_energy = None
    else:
        occupied = build_mp2_bno(
            fragment,
            dmet,
            occtype="occupied",
            threshold=threshold,
            project_dmet_order=project_dmet_order,
            project_dmet_mode=project_dmet_mode,
            # 与 Vayesta 一致：该总开关只控制最终 cluster 的 canonicalization；
            # BNO 振幅分母始终使用 canonical occupied/virtual 轨道。
            canonicalize=True,
        )
        virtual = build_mp2_bno(
            fragment,
            dmet,
            occtype="virtual",
            threshold=threshold,
            project_dmet_order=project_dmet_order,
            project_dmet_mode=project_dmet_mode,
            canonicalize=True,
        )
        c_extra_occ = occupied.c_bath
        c_extra_vir = virtual.c_bath
        c_frozen_occ = occupied.c_frozen
        c_frozen_vir = virtual.c_frozen
        occ_occupations = occupied.occupations
        vir_occupations = virtual.occupations
        occ_energy = occupied.correlation_energy
        vir_energy = virtual.correlation_energy

    overlap = _get_overlap(fragment, _get_mean_field(fragment))
    complete = hstack(
        dmet.c_cluster_occ,
        dmet.c_cluster_vir,
        c_extra_occ,
        c_extra_vir,
        c_frozen_occ,
        c_frozen_vir,
        nrow=nao,
    )
    assert_orthonormal(complete, overlap, name=f"assembled {bathtype} bath")
    complete_error = orthonormality_error(complete, overlap)
    n_mp2_occ_kept = int(c_extra_occ.shape[1]) if bathtype == "mp2" else 0
    n_mp2_vir_kept = int(c_extra_vir.shape[1]) if bathtype == "mp2" else 0
    diagnostics = {
        "assembled_orthonormality_error": complete_error,
        "n_dmet_bath": int(dmet.c_dmet.shape[1]),
        "n_cluster_occ": int(dmet.c_cluster_occ.shape[1]),
        "n_cluster_vir": int(dmet.c_cluster_vir.shape[1]),
        "n_mp2_occ_candidates": int(occ_occupations.size),
        "n_mp2_vir_candidates": int(vir_occupations.size),
        "n_mp2_occ_kept": n_mp2_occ_kept,
        "n_mp2_vir_kept": n_mp2_vir_kept,
        "n_frozen_occ": int(c_frozen_occ.shape[1]),
        "n_frozen_vir": int(c_frozen_vir.shape[1]),
    }
    return BathData(
        c_dmet=dmet.c_dmet,
        dmet_occupations=dmet.occupations,
        c_cluster_occ=dmet.c_cluster_occ,
        c_cluster_vir=dmet.c_cluster_vir,
        c_env_occ=dmet.c_env_occ,
        c_env_vir=dmet.c_env_vir,
        c_mp2_occ=c_extra_occ,
        c_mp2_vir=c_extra_vir,
        c_frozen_occ=c_frozen_occ,
        c_frozen_vir=c_frozen_vir,
        mp2_occ_occupations=occ_occupations,
        mp2_vir_occupations=vir_occupations,
        mp2_occ_correlation_energy=occ_energy,
        mp2_vir_correlation_energy=vir_energy,
        bathtype=bathtype,
        threshold=float(threshold),
        dmet_threshold=float(dmet_threshold),
        canonicalize=bool(canonicalize),
        project_dmet_order=int(project_dmet_order),
        project_dmet_mode=str(project_dmet_mode),
        n_mp2_occ_kept=n_mp2_occ_kept,
        n_mp2_vir_kept=n_mp2_vir_kept,
        diagnostics=diagnostics,
    )


def _build_bath_uhf(
    fragment: object,
    *,
    bathtype: str,
    threshold: float,
    dmet_threshold: float,
    occupation_tolerance: float,
    canonicalize: bool,
    project_dmet_order: int,
    project_dmet_mode: str,
) -> BathData:
    """组装 spin-resolved DMET/UMP2/full bath，字段顺序与 RHF 一致。"""

    dmet = build_dmet_bath(
        fragment,
        dmet_threshold=dmet_threshold,
        occupation_tolerance=occupation_tolerance,
    )
    nao = dmet.c_cluster_occ[0].shape[0]
    empty = (_empty_coeff(nao), _empty_coeff(nao))
    empty_occupations = (np.zeros(0), np.zeros(0))
    if bathtype == "dmet":
        c_extra_occ = c_extra_vir = empty
        c_frozen_occ, c_frozen_vir = dmet.c_env_occ, dmet.c_env_vir
        occ_occupations = vir_occupations = empty_occupations
        occ_energy = vir_energy = None
    elif bathtype == "full":
        c_extra_occ, c_extra_vir = dmet.c_env_occ, dmet.c_env_vir
        c_frozen_occ = c_frozen_vir = empty
        occ_occupations = vir_occupations = empty_occupations
        occ_energy = vir_energy = None
    else:
        occupied = build_mp2_bno(
            fragment,
            dmet,
            occtype="occupied",
            threshold=threshold,
            project_dmet_order=project_dmet_order,
            project_dmet_mode=project_dmet_mode,
            canonicalize=True,
        )
        virtual = build_mp2_bno(
            fragment,
            dmet,
            occtype="virtual",
            threshold=threshold,
            project_dmet_order=project_dmet_order,
            project_dmet_mode=project_dmet_mode,
            canonicalize=True,
        )
        c_extra_occ, c_extra_vir = occupied.c_bath, virtual.c_bath
        c_frozen_occ, c_frozen_vir = occupied.c_frozen, virtual.c_frozen
        occ_occupations, vir_occupations = occupied.occupations, virtual.occupations
        occ_energy = occupied.correlation_energy
        vir_energy = virtual.correlation_energy

    overlap = _get_overlap(fragment, _get_mean_field(fragment))
    errors = []
    for spin in range(2):
        complete = hstack(
            dmet.c_cluster_occ[spin],
            dmet.c_cluster_vir[spin],
            c_extra_occ[spin],
            c_extra_vir[spin],
            c_frozen_occ[spin],
            c_frozen_vir[spin],
            nrow=nao,
        )
        assert_orthonormal(complete, overlap, name=f"spin-{spin} assembled {bathtype} bath")
        errors.append(orthonormality_error(complete, overlap))
    kept_occ = tuple(int(value.shape[1]) for value in c_extra_occ)
    kept_vir = tuple(int(value.shape[1]) for value in c_extra_vir)
    if bathtype != "mp2":
        kept_occ = (0, 0)
        kept_vir = (0, 0)
    diagnostics = {
        "assembled_orthonormality_error": max(errors),
        "n_dmet_bath": tuple(int(value.shape[1]) for value in dmet.c_dmet),
        "n_cluster_occ": tuple(int(value.shape[1]) for value in dmet.c_cluster_occ),
        "n_cluster_vir": tuple(int(value.shape[1]) for value in dmet.c_cluster_vir),
        "n_mp2_occ_candidates": tuple(int(value.size) for value in occ_occupations),
        "n_mp2_vir_candidates": tuple(int(value.size) for value in vir_occupations),
        "n_mp2_occ_kept": kept_occ,
        "n_mp2_vir_kept": kept_vir,
        "n_frozen_occ": tuple(int(value.shape[1]) for value in c_frozen_occ),
        "n_frozen_vir": tuple(int(value.shape[1]) for value in c_frozen_vir),
    }
    return BathData(
        c_dmet=dmet.c_dmet,
        dmet_occupations=dmet.occupations,
        c_cluster_occ=dmet.c_cluster_occ,
        c_cluster_vir=dmet.c_cluster_vir,
        c_env_occ=dmet.c_env_occ,
        c_env_vir=dmet.c_env_vir,
        c_mp2_occ=c_extra_occ,
        c_mp2_vir=c_extra_vir,
        c_frozen_occ=c_frozen_occ,
        c_frozen_vir=c_frozen_vir,
        mp2_occ_occupations=occ_occupations,
        mp2_vir_occupations=vir_occupations,
        mp2_occ_correlation_energy=occ_energy,
        mp2_vir_correlation_energy=vir_energy,
        bathtype=bathtype,
        threshold=float(threshold),
        dmet_threshold=float(dmet_threshold),
        canonicalize=bool(canonicalize),
        project_dmet_order=int(project_dmet_order),
        project_dmet_mode=str(project_dmet_mode),
        n_mp2_occ_kept=sum(kept_occ),
        n_mp2_vir_kept=sum(kept_vir),
        diagnostics=diagnostics,
    )


# 语义更明确的别名，供 driver 或测试按需使用。
assemble_bath = build_bath


__all__ = [
    "DMETBath",
    "MP2BNO",
    "assemble_bath",
    "build_bath",
    "build_dmet_bath",
    "build_mp2_bno",
    "dmet_projector_weights",
]
