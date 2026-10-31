"""组合片段与浴轨道，生成 RHF/UHF 嵌入簇及 Hamiltonian。"""

from __future__ import annotations

import numpy as np
from pyscf import ao2mo

from .linalg import assert_orthonormal, canonicalize_subspace, hstack
from .types import BathData, Cluster


def build_cluster(
    mf,
    bath: BathData,
    *,
    canonicalize: bool = True,
) -> Cluster:
    """由 DMET cluster 和截断后的 BNO bath 构造最终 active cluster。"""
    if isinstance(bath.c_cluster_occ, tuple):
        return _build_cluster_uhf(mf, bath, canonicalize=canonicalize)

    nao = int(mf.mo_coeff.shape[0])
    overlap = np.asarray(mf.get_ovlp(), dtype=float)
    fock_ao = np.asarray(mf.get_fock(), dtype=float)

    c_active_occ = hstack(bath.c_cluster_occ, bath.c_mp2_occ, nrow=nao)
    c_active_vir = hstack(bath.c_cluster_vir, bath.c_mp2_vir, nrow=nao)
    if canonicalize:
        c_active_occ = canonicalize_subspace(c_active_occ, fock_ao)[0]
        c_active_vir = canonicalize_subspace(c_active_vir, fock_ao)[0]

    c_active = hstack(c_active_occ, c_active_vir, nrow=nao)
    c_total = hstack(
        bath.c_frozen_occ,
        c_active_occ,
        c_active_vir,
        bath.c_frozen_vir,
        nrow=nao,
    )
    assert_orthonormal(c_active, overlap, name="Active cluster orbitals")
    assert_orthonormal(c_total, overlap, name="Active+frozen cluster orbitals")
    if c_total.shape[1] != mf.mo_coeff.shape[1]:
        raise RuntimeError(
            "Active and frozen cluster spaces do not cover the molecular MO space: "
            f"{c_total.shape[1]} != {mf.mo_coeff.shape[1]}."
        )

    norb = int(c_active.shape[1])
    fock = c_active.T @ fock_ao @ c_active
    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, c_active), norb)
    else:
        # density-fitted RHF 的 Fock 与 bath 已使用 DF 积分；这里也用同一
        # auxiliary basis 生成稠密 cluster ERI，避免半 DF/半 exact Hamiltonian。
        eri = np.asarray(
            with_df.ao2mo((c_active, c_active, c_active, c_active), compact=False)
        ).reshape((norb,) * 4)

    # 完整分子 Fock 已含环境平均场；扣除 active occupied 产生的 2J-K，
    # 得到 cluster solver 应使用的有效一体积分。
    nocc = int(c_active_occ.shape[1])
    occ = np.s_[:nocc]
    active_potential = 2 * np.einsum(
        "iipq->pq", eri[occ, occ, :, :], optimize=True
    ) - np.einsum("iqpi->pq", eri[occ, :, :, occ], optimize=True)
    h_eff = fock - active_potential

    if not np.allclose(h_eff, h_eff.T, atol=1e-10, rtol=0):
        raise RuntimeError("Embedded one-electron Hamiltonian is not symmetric.")
    return Cluster(
        c_active_occ=np.asarray(c_active_occ, dtype=float),
        c_active_vir=np.asarray(c_active_vir, dtype=float),
        c_frozen_occ=np.asarray(bath.c_frozen_occ, dtype=float),
        c_frozen_vir=np.asarray(bath.c_frozen_vir, dtype=float),
        fock=np.asarray(fock, dtype=float),
        h_eff=np.asarray(h_eff, dtype=float),
        eri=np.asarray(eri, dtype=float),
    )


def _transform_eri_uhf(
    mf,
    c_active: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 AO ERI 转到不同的 alpha/beta cluster 轨道基。"""

    ca, cb = c_active
    coefficient_sets = (
        (ca, ca, ca, ca),
        (ca, ca, cb, cb),
        (cb, cb, cb, cb),
    )
    with_df = getattr(mf, "with_df", None)
    source = getattr(mf, "_eri", None)
    if source is None:
        source = mf.mol
    blocks = []
    for coefficients in coefficient_sets:
        if with_df is None:
            transformed = ao2mo.general(source, coefficients, compact=False)
        else:
            transformed = with_df.ao2mo(coefficients, compact=False)
        shape = tuple(value.shape[1] for value in coefficients)
        blocks.append(np.asarray(transformed, dtype=float).reshape(shape))
    return tuple(blocks)


def _pad_square(array: np.ndarray, norb: int, *, diagonal: float | None = None) -> np.ndarray:
    """把物理 spin channel 末尾补成公共轨道数。"""

    result = np.zeros((norb,) * array.ndim, dtype=float)
    slices = tuple(slice(0, size) for size in array.shape)
    result[slices] = array
    if diagonal is not None:
        for index in range(array.shape[0], norb):
            result[(index,) * array.ndim] = diagonal
    return result


def _build_cluster_uhf(mf, bath: BathData, *, canonicalize: bool) -> Cluster:
    """构造 spin-resolved cluster，并用末尾 dummy virtual 对齐维数。"""

    nao = int(np.asarray(mf.mo_coeff).shape[-2])
    overlap = np.asarray(mf.get_ovlp(), dtype=float)
    fock_ao = tuple(np.asarray(value, dtype=float) for value in mf.get_fock())
    active_occ, active_vir, frozen_occ, frozen_vir = [], [], [], []
    for spin in range(2):
        occ = hstack(bath.c_cluster_occ[spin], bath.c_mp2_occ[spin], nrow=nao)
        vir = hstack(bath.c_cluster_vir[spin], bath.c_mp2_vir[spin], nrow=nao)
        if canonicalize:
            occ = canonicalize_subspace(occ, fock_ao[spin])[0]
            vir = canonicalize_subspace(vir, fock_ao[spin])[0]
        total = hstack(
            bath.c_frozen_occ[spin],
            occ,
            vir,
            bath.c_frozen_vir[spin],
            nrow=nao,
        )
        assert_orthonormal(hstack(occ, vir, nrow=nao), overlap, name=f"spin-{spin} active cluster orbitals")
        assert_orthonormal(total, overlap, name=f"spin-{spin} active+frozen cluster orbitals")
        nmo = int(np.asarray(mf.mo_coeff[spin]).shape[1])
        if total.shape[1] != nmo:
            raise RuntimeError(
                "Active and frozen UHF cluster spaces do not cover the molecular MO space: "
                f"spin={spin}, {total.shape[1]} != {nmo}."
            )
        active_occ.append(occ)
        active_vir.append(vir)
        frozen_occ.append(np.asarray(bath.c_frozen_occ[spin], dtype=float))
        frozen_vir.append(np.asarray(bath.c_frozen_vir[spin], dtype=float))

    c_active = tuple(
        hstack(active_occ[s], active_vir[s], nrow=nao) for s in range(2)
    )
    fock_physical = tuple(
        c_active[s].T @ fock_ao[s] @ c_active[s] for s in range(2)
    )
    eri_physical = _transform_eri_uhf(mf, c_active)
    gaa, gab, gbb = eri_physical
    nocca, noccb = active_occ[0].shape[1], active_occ[1].shape[1]
    oa, ob = np.s_[:nocca], np.s_[:noccb]
    potential_a = (
        np.einsum("iipq->pq", gaa[oa, oa, :, :], optimize=True)
        - np.einsum("ipqi->pq", gaa[oa, :, :, oa], optimize=True)
        + np.einsum("pqii->pq", gab[:, :, ob, ob], optimize=True)
    )
    potential_b = (
        np.einsum("iipq->pq", gbb[ob, ob, :, :], optimize=True)
        - np.einsum("ipqi->pq", gbb[ob, :, :, ob], optimize=True)
        + np.einsum("iipq->pq", gab[oa, oa, :, :], optimize=True)
    )
    h_eff_physical = (
        fock_physical[0] - potential_a,
        fock_physical[1] - potential_b,
    )
    for value in h_eff_physical:
        if not np.allclose(value, value.T, atol=1e-10, rtol=0):
            raise RuntimeError("Embedded UHF one-electron Hamiltonian is not symmetric.")

    physical_norb = tuple(value.shape[1] for value in c_active)
    norb = max(physical_norb)
    dummy_energy = 10.0 + max(float(np.max(np.diag(value))) for value in fock_physical)
    fock = tuple(
        _pad_square(value, norb, diagonal=dummy_energy) for value in fock_physical
    )
    h_eff = tuple(
        _pad_square(value, norb, diagonal=dummy_energy) for value in h_eff_physical
    )
    eri = tuple(_pad_square(value, norb) for value in eri_physical)
    spin_overlap = np.zeros((norb, norb), dtype=float)
    spin_overlap[: physical_norb[0], : physical_norb[1]] = (
        c_active[0].T @ overlap @ c_active[1]
    )
    return Cluster(
        c_active_occ=tuple(active_occ),
        c_active_vir=tuple(active_vir),
        c_frozen_occ=tuple(frozen_occ),
        c_frozen_vir=tuple(frozen_vir),
        fock=fock,
        h_eff=h_eff,
        eri=eri,
        spin_orbital_overlap=spin_overlap,
    )


build_cluster_hamiltonian = build_cluster
