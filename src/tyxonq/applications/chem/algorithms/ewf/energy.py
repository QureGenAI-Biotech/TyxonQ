"""计算 RHF/UHF 嵌入波函数的分片投影能量和 DMET 总能量。"""

from __future__ import annotations

from collections.abc import Iterable
import warnings

import numpy as np
from pyscf import cc, ci, scf

from .solver import build_cluster_rhf, build_cluster_uhf
from .types import Cluster, FragmentResult, SolverResult

FragmentEnergyInput = tuple[Cluster, SolverResult, np.ndarray]


def _real_scalar(value: complex | float, name: str) -> float:
    """把数值噪声范围内的复数转为实数。"""

    result = np.real_if_close(value)
    if np.iscomplexobj(result):
        raise ValueError(f"{name} has a non-negligible imaginary component: {value!r}.")
    return float(result)


def ccsd_to_cisd(result: SolverResult):
    """把 restricted/unrestricted CCSD 振幅转成中间归一化 CISD 系数。"""

    if result.method.upper() != "CCSD" or result.t1 is None or result.t2 is None:
        raise TypeError("A CCSD SolverResult with t1 and t2 is required.")
    if isinstance(result.t1, tuple):
        t1a, t1b = (np.asarray(value) for value in result.t1)
        t2aa, t2ab, t2bb = (np.asarray(value) for value in result.t2)
        c1 = (np.array(t1a, copy=True), np.array(t1b, copy=True))
        c2aa = t2aa + np.einsum("ia,jb->ijab", t1a, t1a, optimize=True) - np.einsum(
            "ib,ja->ijab", t1a, t1a, optimize=True
        )
        c2ab = t2ab + np.einsum("ia,jb->ijab", t1a, t1b, optimize=True)
        c2bb = t2bb + np.einsum("ia,jb->ijab", t1b, t1b, optimize=True) - np.einsum(
            "ib,ja->ijab", t1b, t1b, optimize=True
        )
        return 1.0, c1, (c2aa, c2ab, c2bb)
    t1 = np.asarray(result.t1)
    t2 = np.asarray(result.t2)
    c1 = np.array(t1, copy=True)
    c2 = t2 + np.einsum("ia,jb->ijab", t1, t1, optimize=True)
    return 1.0, c1, c2


def fci_to_cisd(
    result: SolverResult,
    cluster: Cluster,
    *,
    zero_threshold: float = 1e-12,
    warning_threshold: float = 0.1,
):
    """从 PySCF determinant ordering 提取中间归一化 C1/C2。

    对二体 Hamiltonian，参考行 Schrödinger 方程只需要参考、单激发与双激发
    系数，因此无需把高激发截断误差引入能量表达式。
    """

    if result.method.upper() != "FCI" or result.ci is None:
        raise TypeError("An FCI SolverResult with a CI vector is required.")
    ci_vector = np.asarray(result.ci)
    reference = ci_vector[0, 0]
    if abs(reference) <= zero_threshold:
        raise ValueError(
            "The FCI reference amplitude is too small for intermediate normalization: "
            f"{reference!r}."
        )
    if abs(reference) < warning_threshold:
        warnings.warn(
            "The FCI reference amplitude is small; intermediate-normalized C1/C2 "
            "may be numerically unstable.",
            RuntimeWarning,
            stacklevel=2,
        )

    if cluster.is_unrestricted:
        return _ufci_to_cisd(ci_vector, reference, cluster)

    addresses, signs = ci.cisd.t1strs(cluster.norb, cluster.nocc)
    addresses = np.asarray(addresses, dtype=int)
    signs = np.asarray(signs)
    c1 = (ci_vector[0, addresses] * signs).reshape(cluster.nocc, cluster.nvir)
    c2 = np.einsum(
        "i,j,ij->ij",
        signs,
        signs,
        ci_vector[addresses[:, None], addresses],
        optimize=True,
    )
    c2 = c2.reshape(cluster.nocc, cluster.nvir, cluster.nocc, cluster.nvir)
    c2 = c2.transpose(0, 2, 1, 3)
    c1 = c1 / reference
    c2 = c2 / reference
    return 1.0, c1, c2


def _ufci_to_cisd(ci_vector: np.ndarray, reference: float, cluster: Cluster):
    """从 PySCF UHF determinant ordering 提取 α/β C1 与 aa/ab/bb C2。"""

    nocca, noccb = cluster.nocc
    nvira, nvirb = cluster.nvir
    t1addra, t1signa = ci.cisd.tn_addrs_signs(cluster.norb, nocca, 1)
    t1addrb, t1signb = ci.cisd.tn_addrs_signs(cluster.norb, noccb, 1)
    t2addra, t2signa = ci.cisd.tn_addrs_signs(cluster.norb, nocca, 2)
    t2addrb, t2signb = ci.cisd.tn_addrs_signs(cluster.norb, noccb, 2)
    t1addra, t1addrb = np.asarray(t1addra, dtype=int), np.asarray(t1addrb, dtype=int)
    c1a = (ci_vector[t1addra, 0] * t1signa).reshape(nocca, nvira)
    c1b = (ci_vector[0, t1addrb] * t1signb).reshape(noccb, nvirb)
    nooa, noob = nocca * (nocca - 1) // 2, noccb * (noccb - 1) // 2
    nvva, nvvb = nvira * (nvira - 1) // 2, nvirb * (nvirb - 1) // 2
    c2aa_compact = (ci_vector[np.asarray(t2addra, dtype=int), 0] * t2signa).reshape(nooa, nvva)
    c2bb_compact = (ci_vector[0, np.asarray(t2addrb, dtype=int)] * t2signb).reshape(noob, nvvb)
    c2aa = cc.ccsd._unpack_4fold(c2aa_compact, nocca, nvira)
    c2bb = cc.ccsd._unpack_4fold(c2bb_compact, noccb, nvirb)
    c2ab = np.einsum(
        "i,j,ij->ij", t1signa, t1signb, ci_vector[t1addra[:, None], t1addrb], optimize=True
    ).reshape(nocca, nvira, noccb, nvirb).transpose(0, 2, 1, 3)
    return (
        1.0,
        (c1a / reference, c1b / reference),
        (c2aa / reference, c2ab / reference, c2bb / reference),
    )


def intermediate_normalized_cisd(
    result: SolverResult,
    cluster: Cluster,
) -> tuple[float, np.ndarray, np.ndarray]:
    """统一得到 CCSD、FCI 或 selected-CI 的中间归一化 C0/C1/C2。"""

    method = result.method.upper()
    if method == "CCSD":
        return ccsd_to_cisd(result)
    if method == "FCI":
        return fci_to_cisd(result, cluster)
    if method == "SCI":
        if result.cisd_c1 is None or result.cisd_c2 is None:
            raise TypeError(
                "An SCI SolverResult with explicit intermediate-normalized C1/C2 is required."
            )
        return 1.0, result.cisd_c1, result.cisd_c2
    raise ValueError(f"Unsupported solver result method: {result.method!r}.")


def occupied_fragment_overlap(
    mf: scf.hf.RHF,
    cluster: Cluster,
    c_projector: np.ndarray,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """返回 ``R[x,i]=<projector x|cluster occupied i>``。"""

    overlap = np.asarray(mf.get_ovlp())
    if cluster.is_unrestricted:
        return tuple(
            np.asarray(c_projector[s]).T @ overlap @ cluster.c_active_occ[s]
            for s in range(2)
        )
    c_projector = np.asarray(c_projector)
    return c_projector.T @ overlap @ cluster.c_active_occ


def fragment_wf_correlation_energy(
    mf: scf.hf.RHF,
    cluster: Cluster,
    result: SolverResult,
    c_projector: np.ndarray,
) -> tuple[float, float, float]:
    """计算 Vayesta first-occupied-index 分片关联能。

    ``R`` 在系数投影与能量收缩中各出现一次，等价于 ``P=R.T@R`` 只作用
    在 C1/C2 的第一个 occupied index。
    """

    _, c1, c2 = intermediate_normalized_cisd(result, cluster)
    if cluster.is_unrestricted:
        return _fragment_wf_correlation_energy_uhf(mf, cluster, c1, c2, c_projector)
    r_frag = occupied_fragment_overlap(mf, cluster, c_projector)
    projector = r_frag.T @ r_frag
    c1_projected = projector @ c1
    c2_projected = np.tensordot(projector, c2, axes=1)

    fock = np.asarray(mf.get_fock())
    fov = cluster.c_active_occ.T @ fock @ cluster.c_active_vir
    nocc = cluster.nocc
    eri_ovvo = np.asarray(cluster.eri)[:nocc, nocc:, nocc:, :nocc]
    e_singles = 2.0 * np.einsum("ia,ia->", fov, c1_projected, optimize=True)
    e_doubles = 2.0 * np.einsum(
        "ijab,iabj->", c2_projected, eri_ovvo, optimize=True
    ) - np.einsum("ijab,ibaj->", c2_projected, eri_ovvo, optimize=True)
    e_singles = _real_scalar(e_singles, "fragment singles energy")
    e_doubles = _real_scalar(e_doubles, "fragment doubles energy")
    return e_singles, e_doubles, e_singles + e_doubles


def _fragment_wf_correlation_energy_uhf(
    mf,
    cluster: Cluster,
    c1,
    c2,
    c_projector,
) -> tuple[float, float, float]:
    """Vayesta first-occupied-index 的 UHF 分片关联能。"""

    overlaps = occupied_fragment_overlap(mf, cluster, c_projector)
    pa, pb = (value.T @ value for value in overlaps)
    nocca, noccb = cluster.nocc
    nvira, nvirb = cluster.physical_nvir
    c1a, c1b = c1[0][:, :nvira], c1[1][:, :nvirb]
    caa = c2[0][:, :, :nvira, :nvira]
    cab = c2[1][:, :, :nvira, :nvirb]
    cbb = c2[2][:, :, :nvirb, :nvirb]
    caa = np.einsum("ik,kjab->ijab", pa, caa, optimize=True)
    cbb = np.einsum("ik,kjab->ijab", pb, cbb, optimize=True)
    cab_a = np.einsum("ik,kjab->ijab", pa, cab, optimize=True)
    cab_b = np.einsum("jl,ilab->ijab", pb, cab, optimize=True)
    fock = tuple(np.asarray(value) for value in mf.get_fock())
    fova = cluster.c_active_occ[0].T @ fock[0] @ cluster.c_active_vir[0]
    fovb = cluster.c_active_occ[1].T @ fock[1] @ cluster.c_active_vir[1]
    e_singles = np.einsum("ia,ia->", fova, pa @ c1a, optimize=True)
    e_singles += np.einsum("ia,ia->", fovb, pb @ c1b, optimize=True)
    gaa, gab, gbb = cluster.eri
    gaa = gaa[:nocca, nocca : nocca + nvira, :nocca, nocca : nocca + nvira]
    gab = gab[:nocca, nocca : nocca + nvira, :noccb, noccb : noccb + nvirb]
    gbb = gbb[:noccb, noccb : noccb + nvirb, :noccb, noccb : noccb + nvirb]
    e_doubles = (
        np.einsum("ijab,iajb->", caa, gaa, optimize=True)
        - np.einsum("ijab,ibja->", caa, gaa, optimize=True)
    ) / 4
    e_doubles += (
        np.einsum("ijab,iajb->", cbb, gbb, optimize=True)
        - np.einsum("ijab,ibja->", cbb, gbb, optimize=True)
    ) / 4
    e_doubles += 0.5 * np.einsum("ijab,iajb->", cab_a, gab, optimize=True)
    e_doubles += 0.5 * np.einsum("ijab,iajb->", cab_b, gab, optimize=True)
    e_singles = _real_scalar(e_singles, "UHF fragment singles energy")
    e_doubles = _real_scalar(e_doubles, "UHF fragment doubles energy")
    return e_singles, e_doubles, e_singles + e_doubles


def compute_wf_fragment_energy(
    mf: scf.hf.RHF,
    fragment: object,
    cluster: Cluster,
    result: SolverResult,
) -> float:
    """供 EWF driver 使用的单值 wavefunction 能量接口。"""

    try:
        c_projector = np.asarray(getattr(fragment, "c_proj"))
    except AttributeError as exc:
        raise TypeError("fragment must provide active projector coefficients in c_proj.") from exc
    return fragment_wf_correlation_energy(
        mf, cluster, result, c_projector
    )[2]


def cumulant_rdm2(
    result: SolverResult,
    cluster: Cluster,
    *,
    approx_cumulant: bool = True,
    reference_mf: scf.hf.RHF | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由完整 spin-traced RDM 构造 Vayesta 的近似或严格 cumulant。"""

    if cluster.is_unrestricted:
        return _cumulant_rdm2_uhf(
            result,
            cluster,
            approx_cumulant=approx_cumulant,
            reference_mf=reference_mf,
        )

    if result.method.upper() == "CCSD":
        if reference_mf is None:
            raise TypeError("reference_mf is required for a CCSD cumulant RDM.")
        if any(value is None for value in (result.t1, result.t2, result.l1, result.l2)):
            raise TypeError("CCSD t1/t2/l1/l2 amplitudes are required for a cumulant RDM.")

        # PySCF 2.13 的公开接口与 Vayesta 的 approximate-cumulant 后端一致。
        from pyscf import cc

        cluster_solver = cc.RCCSD(build_cluster_rhf(cluster, reference_mf))
        dm2 = cluster_solver.make_rdm2(
            t1=result.t1,
            t2=result.t2,
            l1=result.l1,
            l2=result.l2,
            ao_repr=False,
            with_frozen=False,
            with_dm1=False,
        )
        dm2 = np.asarray(dm2)
        if approx_cumulant:
            return dm2

        dm1_correlation = np.array(result.rdm1, copy=True)
        dm1_correlation[np.diag_indices(cluster.nocc)] -= 2.0
        dm2 -= np.einsum(
            "ij,kl->ijkl", dm1_correlation, dm1_correlation, optimize=True
        )
        dm2 += 0.5 * np.einsum(
            "ij,kl->iklj", dm1_correlation, dm1_correlation, optimize=True
        )
        return dm2

    if result.method.upper() not in {"FCI", "SCI"}:
        raise ValueError(f"Unsupported solver result method: {result.method!r}.")

    dm1 = np.array(result.rdm1, copy=True)
    dm2 = np.array(result.rdm2, copy=True)
    if not approx_cumulant:
        dm2 -= np.einsum("ij,kl->ijkl", dm1, dm1, optimize=True)
        dm2 += 0.5 * np.einsum("ij,kl->iklj", dm1, dm1, optimize=True)
        return dm2

    # 该操作等价于 PySCF CCSD make_rdm2(with_dm1=False)，适用于完整或稀疏 CI 的 RDM。
    delta_half = dm1.copy()
    delta_half[np.diag_indices(cluster.nocc)] -= 1.0
    for occupied in range(cluster.nocc):
        dm2[occupied, occupied, :, :] -= 2.0 * delta_half
        dm2[:, :, occupied, occupied] -= 2.0 * delta_half
        dm2[:, occupied, occupied, :] += delta_half
        dm2[occupied, :, :, occupied] += delta_half
    return dm2


def _cumulant_rdm2_uhf(
    result: SolverResult,
    cluster: Cluster,
    *,
    approx_cumulant: bool,
    reference_mf,
):
    """构造 Vayesta 约定的 UHF ``aa/ab/bb`` cumulant。"""

    method = result.method.upper()
    if method == "CCSD":
        if reference_mf is None:
            raise TypeError("reference_mf is required for a UCCSD cumulant RDM.")
        if any(value is None for value in (result.t1, result.t2, result.l1, result.l2)):
            raise TypeError("UCCSD t1/t2/l1/l2 amplitudes are required for a cumulant RDM.")
        cluster_solver = cc.UCCSD(build_cluster_uhf(cluster, reference_mf))
        dm2 = tuple(
            np.asarray(value)
            for value in cluster_solver.make_rdm2(
                t1=result.t1,
                t2=result.t2,
                l1=result.l1,
                l2=result.l2,
                ao_repr=False,
                with_frozen=False,
                with_dm1=False,
            )
        )
        if approx_cumulant:
            return dm2
        dm1a, dm1b = (np.array(value, copy=True) for value in result.rdm1)
        dm1a[np.diag_indices(cluster.nocc[0])] -= 1.0
        dm1b[np.diag_indices(cluster.nocc[1])] -= 1.0
        aa, ab, bb = (np.array(value, copy=True) for value in dm2)
        aa -= np.einsum("ij,kl->ijkl", dm1a, dm1a, optimize=True)
        aa += np.einsum("ij,kl->iklj", dm1a, dm1a, optimize=True)
        bb -= np.einsum("ij,kl->ijkl", dm1b, dm1b, optimize=True)
        bb += np.einsum("ij,kl->iklj", dm1b, dm1b, optimize=True)
        ab -= np.einsum("ij,kl->ijkl", dm1a, dm1b, optimize=True)
        return aa, ab, bb

    if method not in {"FCI", "SCI"}:
        raise ValueError(f"Unsupported solver result method: {result.method!r}.")
    dm1a, dm1b = (np.array(value, copy=True) for value in result.rdm1)
    aa, ab, bb = (np.array(value, copy=True) for value in result.rdm2)
    if not approx_cumulant:
        aa -= np.einsum("ij,kl->ijkl", dm1a, dm1a, optimize=True)
        aa += np.einsum("ij,kl->iklj", dm1a, dm1a, optimize=True)
        bb -= np.einsum("ij,kl->ijkl", dm1b, dm1b, optimize=True)
        bb += np.einsum("ij,kl->iklj", dm1b, dm1b, optimize=True)
        ab -= np.einsum("ij,kl->ijkl", dm1a, dm1b, optimize=True)
        return aa, ab, bb

    dm1a[np.diag_indices(cluster.nocc[0])] -= 0.5
    dm1b[np.diag_indices(cluster.nocc[1])] -= 0.5
    for occupied in range(cluster.nocc[0]):
        aa[occupied, occupied, :, :] -= dm1a
        aa[:, :, occupied, occupied] -= dm1a
        aa[:, occupied, occupied, :] += dm1a
        aa[occupied, :, :, occupied] += dm1a
        ab[occupied, occupied, :, :] -= dm1b
    for occupied in range(cluster.nocc[1]):
        bb[occupied, occupied, :, :] -= dm1b
        bb[:, :, occupied, occupied] -= dm1b
        bb[:, occupied, occupied, :] += dm1b
        bb[occupied, :, :, occupied] += dm1b
        ab[:, :, occupied, occupied] -= dm1a
    return aa, ab, bb


def cluster_fragment_projector(
    mf: scf.hf.RHF,
    cluster: Cluster,
    c_projector: np.ndarray,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """在完整 active cluster 空间中构造分片 projector。"""

    overlap = np.asarray(mf.get_ovlp())
    if cluster.is_unrestricted:
        projectors = []
        for spin in range(2):
            rotation = cluster.c_active[spin].T @ overlap @ np.asarray(c_projector[spin])
            projector = rotation @ rotation.T
            projectors.append(_pad_projector(projector, cluster.norb))
        return tuple(projectors)
    rotation = cluster.c_active.T @ overlap @ np.asarray(c_projector)
    return rotation @ rotation.T


def _pad_projector(value: np.ndarray, norb: int) -> np.ndarray:
    result = np.zeros((norb, norb), dtype=value.dtype)
    result[: value.shape[0], : value.shape[1]] = value
    return result


def fragment_dmet_energy(
    mf: scf.hf.RHF,
    cluster: Cluster,
    result: SolverResult,
    c_projector: np.ndarray,
    *,
    part_cumulant: bool = True,
    approx_cumulant: bool = True,
) -> float:
    """计算一个分片的 electronic DMET energy contribution。"""

    if cluster.is_unrestricted:
        return _fragment_dmet_energy_uhf(
            mf,
            cluster,
            result,
            c_projector,
            part_cumulant=part_cumulant,
            approx_cumulant=approx_cumulant,
        )

    dm1 = np.asarray(result.rdm1)
    eri = np.asarray(cluster.eri)
    if part_cumulant:
        dm2 = cumulant_rdm2(
            result,
            cluster,
            approx_cumulant=approx_cumulant,
            reference_mf=mf,
        )
        h1e_eff = cluster.c_active.T @ np.asarray(mf.get_hcore()) @ cluster.c_active
    else:
        dm2 = np.asarray(result.rdm2)
        hcore_with_half_veff = np.asarray(mf.get_hcore()) + 0.5 * np.asarray(mf.get_veff())
        h1e_eff = cluster.c_active.T @ hcore_with_half_veff @ cluster.c_active
        occupied = slice(0, cluster.nocc)
        v_active_half = np.einsum(
            "iipq->pq", eri[occupied, occupied, :, :], optimize=True
        ) - 0.5 * np.einsum(
            "iqpi->pq", eri[occupied, :, :, occupied], optimize=True
        )
        h1e_eff -= v_active_half

    projector = cluster_fragment_projector(mf, cluster, c_projector)
    e_one = np.einsum("xj,xi,ij->", h1e_eff, projector, dm1, optimize=True)
    e_two = 0.5 * np.einsum(
        "xjkl,xi,ijkl->", eri, projector, dm2, optimize=True
    )
    return _real_scalar(e_one + e_two, "fragment DMET energy")


def _fragment_dmet_energy_uhf(
    mf,
    cluster: Cluster,
    result: SolverResult,
    c_projector,
    *,
    part_cumulant: bool,
    approx_cumulant: bool,
) -> float:
    dm1a, dm1b = (np.asarray(value) for value in result.rdm1)
    gaa, gab, gbb = (np.asarray(value) for value in cluster.eri)
    if part_cumulant:
        dm2aa, dm2ab, dm2bb = cumulant_rdm2(
            result,
            cluster,
            approx_cumulant=approx_cumulant,
            reference_mf=mf,
        )
        hcore = np.asarray(mf.get_hcore())
        h1a_physical = cluster.c_active[0].T @ hcore @ cluster.c_active[0]
        h1b_physical = cluster.c_active[1].T @ hcore @ cluster.c_active[1]
        h1a, h1b = _pad_projector(h1a_physical, cluster.norb), _pad_projector(h1b_physical, cluster.norb)
    else:
        dm2aa, dm2ab, dm2bb = (np.asarray(value) for value in result.rdm2)
        hcore = np.asarray(mf.get_hcore())
        veff = tuple(np.asarray(value) for value in mf.get_veff())
        h1a_physical = cluster.c_active[0].T @ (hcore + 0.5 * veff[0]) @ cluster.c_active[0]
        h1b_physical = cluster.c_active[1].T @ (hcore + 0.5 * veff[1]) @ cluster.c_active[1]
        h1a, h1b = _pad_projector(h1a_physical, cluster.norb), _pad_projector(h1b_physical, cluster.norb)
        oa, ob = np.s_[: cluster.nocc[0]], np.s_[: cluster.nocc[1]]
        va = 0.5 * (
            np.einsum("iipq->pq", gaa[oa, oa], optimize=True)
            + np.einsum("pqii->pq", gab[:, :, ob, ob], optimize=True)
            - np.einsum("ipqi->pq", gaa[oa, :, :, oa], optimize=True)
        )
        vb = 0.5 * (
            np.einsum("iipq->pq", gbb[ob, ob], optimize=True)
            + np.einsum("iipq->pq", gab[oa, oa], optimize=True)
            - np.einsum("ipqi->pq", gbb[ob, :, :, ob], optimize=True)
        )
        h1a, h1b = h1a - va, h1b - vb

    pa, pb = cluster_fragment_projector(mf, cluster, c_projector)
    e_one = np.einsum("xj,xi,ij->", h1a, pa, dm1a, optimize=True)
    e_one += np.einsum("xj,xi,ij->", h1b, pb, dm1b, optimize=True)
    e_two = (
        np.einsum("xjkl,xi,ijkl->", gaa, pa, dm2aa, optimize=True)
        + np.einsum("xjkl,xi,ijkl->", gbb, pb, dm2bb, optimize=True)
        + np.einsum("xjkl,xi,ijkl->", gab, pa, dm2ab, optimize=True)
        + np.einsum("ijxl,xk,ijkl->", gab, pb, dm2ab, optimize=True)
    ) / 2
    return _real_scalar(e_one + e_two, "UHF fragment DMET energy")


def democratic_rdm1(
    mf: scf.hf.RHF,
    fragments: Iterable[FragmentEnergyInput],
    *,
    ao_basis: bool = True,
    with_mf: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """组装 first-index democratically partitioned spin-traced 1RDM。"""

    fragment_list = list(fragments)
    if fragment_list and fragment_list[0][0].is_unrestricted:
        return _democratic_rdm1_uhf(
            mf,
            fragment_list,
            ao_basis=ao_basis,
            with_mf=with_mf,
        )
    mo_coeff = np.asarray(mf.mo_coeff)
    overlap = np.asarray(mf.get_ovlp())
    nmo = mo_coeff.shape[1]
    dm1_mo = np.zeros((nmo, nmo))
    for cluster, result, c_projector in fragment_list:
        dm1_correlation = np.array(result.rdm1, copy=True)
        dm1_correlation[np.diag_indices(cluster.nocc)] -= 2.0
        projector = cluster_fragment_projector(mf, cluster, c_projector)
        rotation = mo_coeff.T @ overlap @ cluster.c_active
        dm1_mo += rotation @ projector @ dm1_correlation @ rotation.T

    if with_mf:
        dm1_mo += np.diag(np.asarray(mf.mo_occ))
    dm1_mo = 0.5 * (dm1_mo + dm1_mo.T)
    if ao_basis:
        return mo_coeff @ dm1_mo @ mo_coeff.T
    return dm1_mo


def _democratic_rdm1_uhf(
    mf,
    fragments,
    *,
    ao_basis: bool,
    with_mf: bool,
):
    mo_coeff = tuple(np.asarray(value) for value in mf.mo_coeff)
    overlap = np.asarray(mf.get_ovlp())
    dm1_mo = [np.zeros((value.shape[1], value.shape[1])) for value in mo_coeff]
    for cluster, result, c_projector in fragments:
        projectors = cluster_fragment_projector(mf, cluster, c_projector)
        for spin in range(2):
            dm_corr = np.array(result.rdm1[spin], copy=True)
            dm_corr[np.diag_indices(cluster.nocc[spin])] -= 1.0
            nphysical = cluster.physical_norb[spin]
            rotation = mo_coeff[spin].T @ overlap @ cluster.c_active[spin]
            dm1_mo[spin] += (
                rotation
                @ projectors[spin][:nphysical, :nphysical]
                @ dm_corr[:nphysical, :nphysical]
                @ rotation.T
            )
    if with_mf:
        for spin in range(2):
            dm1_mo[spin] += np.diag(np.asarray(mf.mo_occ[spin]))
    dm1_mo = tuple(0.5 * (value + value.T) for value in dm1_mo)
    if ao_basis:
        return tuple(
            mo_coeff[s] @ dm1_mo[s] @ mo_coeff[s].T for s in range(2)
        )
    return dm1_mo


def dmet_total_energy(
    mf: scf.hf.RHF,
    fragments: Iterable[FragmentEnergyInput],
    *,
    part_cumulant: bool = True,
    approx_cumulant: bool = True,
) -> float:
    """求和分片贡献并补齐 Vayesta DMET 的全局非 cumulant 项。"""

    fragment_list = list(fragments)
    if not fragment_list:
        raise ValueError("At least one fragment is required for DMET energy reconstruction.")
    electronic = sum(
        fragment_dmet_energy(
            mf,
            cluster,
            result,
            c_projector,
            part_cumulant=part_cumulant,
            approx_cumulant=approx_cumulant,
        )
        for cluster, result, c_projector in fragment_list
    )

    unrestricted = fragment_list[0][0].is_unrestricted
    if part_cumulant:
        dm1 = democratic_rdm1(mf, fragment_list, ao_basis=True, with_mf=True)
        if unrestricted:
            if approx_cumulant:
                mf_dm1 = mf.make_rdm1()
                dm1_for_veff = tuple(2.0 * dm1[s] - mf_dm1[s] for s in range(2))
                veff = tuple(np.asarray(value) for value in mf.get_veff())
            else:
                dm1_for_veff = dm1
                veff = tuple(np.asarray(value) for value in mf.get_veff(dm=dm1))
            electronic += 0.5 * sum(
                np.einsum("ij,ij->", veff[s], dm1_for_veff[s], optimize=True)
                for s in range(2)
            )
        elif approx_cumulant:
            dm1_for_veff = 2.0 * dm1 - np.asarray(mf.make_rdm1())
            veff = np.asarray(mf.get_veff())
        else:
            dm1_for_veff = dm1
            veff = np.asarray(mf.get_veff(dm=dm1))
        if not unrestricted:
            electronic += 0.5 * np.einsum(
                "ij,ij->", veff, dm1_for_veff, optimize=True
            )

    total = electronic + float(mf.mol.energy_nuc())
    return _real_scalar(total, "total DMET energy")


def compute_dmet_energy(
    mf: scf.hf.RHF,
    fragment_results: Iterable[tuple[object, FragmentResult]],
    *,
    part_cumulant: bool = True,
    approx_cumulant: bool = True,
    return_fragment_energies: bool = False,
) -> float | tuple[float, tuple[float, ...]]:
    """把 driver 的 ``(Fragment, FragmentResult)`` 转成最小能量输入。"""

    paired = list(fragment_results)
    entries: list[FragmentEnergyInput] = []
    for fragment, result in paired:
        try:
            c_projector = np.asarray(getattr(fragment, "c_proj"))
        except AttributeError as exc:
            raise TypeError(
                "fragment must provide active projector coefficients in c_proj."
            ) from exc
        entries.append((result.cluster, result.solver, c_projector))

    total = dmet_total_energy(
        mf,
        entries,
        part_cumulant=part_cumulant,
        approx_cumulant=approx_cumulant,
    )
    if not return_fragment_energies:
        return total
    per_fragment = tuple(
        fragment_dmet_energy(
            mf,
            cluster,
            result,
            c_projector,
            part_cumulant=part_cumulant,
            approx_cumulant=approx_cumulant,
        )
        for cluster, result, c_projector in entries
    )
    return total, per_fragment


__all__ = [
    "FragmentEnergyInput",
    "ccsd_to_cisd",
    "cluster_fragment_projector",
    "compute_dmet_energy",
    "compute_wf_fragment_energy",
    "cumulant_rdm2",
    "democratic_rdm1",
    "dmet_total_energy",
    "fci_to_cisd",
    "fragment_dmet_energy",
    "fragment_wf_correlation_energy",
    "intermediate_normalized_cisd",
    "occupied_fragment_overlap",
]
