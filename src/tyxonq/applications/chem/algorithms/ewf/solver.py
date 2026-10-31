"""调用 PySCF CCSD/FCI 求解 RHF/UHF 嵌入簇并整理振幅及密度。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pyscf import cc, fci, gto, scf
from pyscf.cc import uccsd

from .types import Cluster, SolverResult

_CCSD_DEFAULTS: dict[str, Any] = {
    # Vayesta EWF 的全局 solver_options 会覆盖 RCCSD 类自身的 100 步默认值。
    "max_cycle": 300,
    "conv_tol": None,
    "conv_tol_normt": None,
    "diis_space": None,
    "diis_start_cycle": None,
    "iterative_damping": None,
    "level_shift": None,
    "solve_lambda": True,
}

_FCI_DEFAULTS: dict[str, Any] = {
    "max_cycle": 300,
    "conv_tol": 1e-12,
    "lindep": None,
    "fix_spin": None,
    "fix_spin_penalty": 1.0,
    "davidson_only": True,
    "threads": 1,
}


def _validated_options(
    method: str,
    options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """合并求解器选项，并拒绝尚未迁移的 Vayesta 选项。"""

    defaults = _CCSD_DEFAULTS if method == "CCSD" else _FCI_DEFAULTS
    supplied = dict(options or {})
    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        allowed = ", ".join(sorted(defaults))
        raise ValueError(
            f"Unsupported {method} solver option(s): {unknown}. "
            f"Allowed options are: {allowed}."
        )
    merged = defaults.copy()
    merged.update(supplied)
    return merged


def _validate_reference_mf(mf: scf.hf.RHF | scf.uhf.UHF) -> None:
    """确认输入是已收敛、实数的分子 RHF/UHF。"""

    if not isinstance(mf, (scf.hf.RHF, scf.uhf.UHF)) or isinstance(mf, scf.rohf.ROHF):
        raise TypeError("EWF cluster solvers require a molecular RHF or UHF object.")
    if not getattr(mf, "converged", False):
        raise ValueError("The molecular RHF calculation must be converged.")
    is_uhf = isinstance(mf, scf.uhf.UHF)
    if not is_uhf and getattr(mf.mol, "spin", 0) != 0:
        raise ValueError("The molecular RHF reference must have spin=0.")
    mo_coeff = np.asarray(mf.mo_coeff)
    mo_occ = np.asarray(mf.mo_occ)
    if (mo_coeff.ndim, mo_occ.ndim) != ((3, 2) if is_uhf else (2, 1)):
        raise TypeError("Only molecular RHF/UHF orbitals are supported.")
    if np.iscomplexobj(mo_coeff):
        raise TypeError("Complex molecular orbitals are not supported by the minimal RHF EWF path.")
    occupied_value = 1.0 if is_uhf else 2.0
    if not np.all(np.isclose(mo_occ, 0.0) | np.isclose(mo_occ, occupied_value)):
        raise ValueError(f"The molecular occupations must contain only 0 or {occupied_value:g}.")


def _validate_cluster(cluster: Cluster) -> None:
    """检查 cluster 积分及 occupied/virtual 排列。"""

    norb = cluster.norb
    if norb == 0 or (not cluster.is_unrestricted and cluster.nocc == 0):
        raise ValueError("The cluster must contain at least one occupied orbital.")
    if cluster.is_unrestricted:
        if sum(cluster.nocc) == 0:
            raise ValueError("The UHF cluster must contain at least one occupied orbital.")
        for name, value in (("fock", cluster.fock), ("h_eff", cluster.h_eff)):
            if not isinstance(value, tuple) or any(np.asarray(x).shape != (norb, norb) for x in value):
                raise ValueError(f"cluster.{name} UHF blocks must have shape {(norb, norb)}.")
            if any(np.iscomplexobj(x) for x in value):
                raise TypeError(f"Complex cluster {name} is not supported.")
        if not isinstance(cluster.eri, tuple) or len(cluster.eri) != 3:
            raise ValueError("cluster.eri must be an (aa, ab, bb) tuple for UHF.")
        if any(np.asarray(x).shape != (norb,) * 4 for x in cluster.eri):
            raise ValueError(f"cluster.eri UHF blocks must have shape {(norb,) * 4}.")
        if any(np.iscomplexobj(x) for x in cluster.eri):
            raise TypeError("Complex cluster eri is not supported.")
        return
    expected2 = (norb, norb)
    if np.asarray(cluster.fock).shape != expected2:
        raise ValueError(f"cluster.fock must have shape {expected2}.")
    if np.asarray(cluster.h_eff).shape != expected2:
        raise ValueError(f"cluster.h_eff must have shape {expected2}.")
    if np.asarray(cluster.eri).shape != (norb,) * 4:
        raise ValueError(f"cluster.eri must have shape {(norb,) * 4}.")
    for name, value in (
        ("fock", cluster.fock),
        ("h_eff", cluster.h_eff),
        ("eri", cluster.eri),
    ):
        if np.iscomplexobj(value):
            raise TypeError(f"Complex cluster {name} is not supported.")


def build_cluster_rhf(cluster: Cluster, reference_mf: scf.hf.RHF) -> scf.hf.RHF:
    """把 EWF cluster Hamiltonian 包装成标准 PySCF RHF 对象。

    ``h_eff`` 已包含 frozen/environment mean-field 势。返回对象的核排斥能为
    零，因此它的 ``e_tot`` 只是 cluster 电子 Hamiltonian 能量，不能直接当作
    分子总能量。
    """

    _validate_reference_mf(reference_mf)
    _validate_cluster(cluster)

    norb = cluster.norb
    mol = gto.Mole()
    mol.verbose = reference_mf.mol.verbose
    mol.max_memory = reference_mf.mol.max_memory
    mol.nelec = cluster.nelec
    # Vayesta 使用无原子的 dummy Mole；轨道维数由后续恒等 mo_coeff 给出。
    mol.nao = norb
    mol.incore_anyway = True
    mol.build()

    cluster_mf = scf.RHF(mol)
    cluster_mf.verbose = reference_mf.verbose
    cluster_mf.max_memory = reference_mf.max_memory
    cluster_mf.mo_coeff = np.eye(norb)
    cluster_mf.mo_occ = np.zeros(norb)
    cluster_mf.mo_occ[: cluster.nocc] = 2.0
    cluster_mf.mo_energy = np.diag(np.asarray(cluster.fock)).copy()
    cluster_mf.get_ovlp = lambda *args, **kwargs: np.eye(norb)
    cluster_mf.get_hcore = lambda *args, **kwargs: np.asarray(cluster.h_eff)
    cluster_mf._eri = np.asarray(cluster.eri)
    cluster_mf.converged = True

    # PySCF 的 post-HF e_tot 以 mean-field e_tot 为零点；显式设置该值。
    dm0 = cluster_mf.make_rdm1()
    cluster_mf.e_tot = float(cluster_mf.energy_tot(dm=dm0))
    return cluster_mf


def build_cluster_uhf(cluster: Cluster, reference_mf: scf.uhf.UHF) -> scf.uhf.UHF:
    """把 spin-dependent cluster Hamiltonian 包装成 PySCF UHF 对象。"""

    _validate_reference_mf(reference_mf)
    _validate_cluster(cluster)
    if not cluster.is_unrestricted:
        raise TypeError("build_cluster_uhf requires an unrestricted Cluster.")
    norb = cluster.norb
    mol = gto.Mole()
    mol.verbose = reference_mf.mol.verbose
    mol.max_memory = reference_mf.mol.max_memory
    mol.nelec = cluster.nelec
    mol.nao = norb
    mol.incore_anyway = True
    mol.build()

    cluster_mf = scf.UHF(mol)
    cluster_mf.verbose = reference_mf.verbose
    cluster_mf.max_memory = reference_mf.max_memory
    cluster_mf.mo_coeff = np.asarray((np.eye(norb), np.eye(norb)))
    cluster_mf.mo_occ = np.zeros((2, norb))
    cluster_mf.mo_occ[0, : cluster.nocc[0]] = 1.0
    cluster_mf.mo_occ[1, : cluster.nocc[1]] = 1.0
    cluster_mf.mo_energy = np.asarray(tuple(np.diag(value) for value in cluster.fock))
    cluster_mf.get_ovlp = lambda *args, **kwargs: np.eye(norb)
    cluster_mf.get_hcore = lambda *args, **kwargs: np.asarray(cluster.h_eff)
    cluster_mf.get_fock = lambda *args, **kwargs: np.asarray(cluster.fock)
    cluster_mf.get_veff = lambda *args, **kwargs: np.asarray(cluster.fock) - np.asarray(cluster.h_eff)
    cluster_mf._eri = cluster.eri
    cluster_mf.converged = True
    cluster_mf.e_tot = float(cluster_mf.energy_tot(dm=cluster_mf.make_rdm1()))
    return cluster_mf


def _pad_matrix(value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.asarray(value).dtype)
    result[tuple(slice(0, size) for size in value.shape)] = value
    return result


def _pad_uccsd_amplitudes(
    cluster: Cluster,
    t1,
    t2,
):
    """只在 virtual 末尾补零，使 UCCSD 振幅直接匹配公共 ``norb``。"""

    nocca, noccb = cluster.nocc
    nvira, nvirb = cluster.norb - nocca, cluster.norb - noccb
    t1a = _pad_matrix(np.asarray(t1[0]), (nocca, nvira))
    t1b = _pad_matrix(np.asarray(t1[1]), (noccb, nvirb))
    t2aa = _pad_matrix(np.asarray(t2[0]), (nocca, nocca, nvira, nvira))
    t2ab = _pad_matrix(np.asarray(t2[1]), (nocca, noccb, nvira, nvirb))
    t2bb = _pad_matrix(np.asarray(t2[2]), (noccb, noccb, nvirb, nvirb))
    return (t1a, t1b), (t2aa, t2ab, t2bb)


def _solve_uccsd(
    cluster: Cluster,
    reference_mf: scf.uhf.UHF,
    opts: Mapping[str, Any],
) -> SolverResult:
    cluster_mf = build_cluster_uhf(cluster, reference_mf)
    frozen = [list(value) for value in cluster.dummy_orbitals]
    solver = cc.UCCSD(cluster_mf, frozen=frozen if any(frozen) else None)

    # PySCF 默认假定 alpha/beta 共用 AO ERI。这里按固定 aa,bb,ab 调用顺序
    # 注入各自的物理积分，避免把不同 spin orbital basis 混为一谈。
    na, nb = cluster.physical_norb
    gaa, gab, gbb = cluster.eri

    def make_eris(mo_coeff=None):
        blocks = iter((gaa[:na, :na, :na, :na], gbb[:nb, :nb, :nb, :nb], gab[:na, :na, :nb, :nb]))
        return uccsd._make_eris_incore(
            solver,
            mo_coeff,
            ao2mofn=lambda coefficients: next(blocks),
        )

    solver.ao2mo = make_eris
    if opts["max_cycle"] is not None:
        solver.max_cycle = int(opts["max_cycle"])
    if opts["conv_tol"] is not None:
        solver.conv_tol = float(opts["conv_tol"])
    if opts["conv_tol_normt"] is not None:
        solver.conv_tol_normt = float(opts["conv_tol_normt"])
    for option in ("diis_space", "diis_start_cycle", "iterative_damping", "level_shift"):
        if opts[option] is not None:
            setattr(solver, option, opts[option])

    e_corr, t1_raw, t2_raw = solver.kernel()
    converged = bool(solver.converged)
    if bool(opts["solve_lambda"]):
        l1_raw, l2_raw = solver.solve_lambda(t1=t1_raw, t2=t2_raw)
        converged = converged and bool(getattr(solver, "converged_lambda", True))
    else:
        l1_raw = tuple(np.array(value, copy=True) for value in t1_raw)
        l2_raw = tuple(np.array(value, copy=True) for value in t2_raw)
    rdm1_raw = solver.make_rdm1(t1=t1_raw, t2=t2_raw, l1=l1_raw, l2=l2_raw, ao_repr=False)
    rdm2_raw = solver.make_rdm2(t1=t1_raw, t2=t2_raw, l1=l1_raw, l2=l2_raw, ao_repr=False)
    t1, t2 = _pad_uccsd_amplitudes(cluster, t1_raw, t2_raw)
    l1, l2 = _pad_uccsd_amplitudes(cluster, l1_raw, l2_raw)
    rdm1 = tuple(_pad_matrix(np.asarray(value), (cluster.norb,) * 2) for value in rdm1_raw)
    rdm2 = tuple(_pad_matrix(np.asarray(value), (cluster.norb,) * 4) for value in rdm2_raw)
    return SolverResult(
        method="CCSD",
        converged=converged,
        energy=float(cluster_mf.e_tot + e_corr),
        rdm1=rdm1,
        rdm2=rdm2,
        t1=t1,
        t2=t2,
        l1=l1,
        l2=l2,
    )


def solve_ccsd(
    cluster: Cluster,
    reference_mf: scf.hf.RHF,
    options: Mapping[str, Any] | None = None,
) -> SolverResult:
    """按 cluster 类型使用 PySCF RCCSD 或 UCCSD。"""

    opts = _validated_options("CCSD", options)
    if cluster.is_unrestricted:
        return _solve_uccsd(cluster, reference_mf, opts)
    cluster_mf = build_cluster_rhf(cluster, reference_mf)
    solver = cc.RCCSD(cluster_mf)
    if opts["max_cycle"] is not None:
        solver.max_cycle = int(opts["max_cycle"])
    if opts["conv_tol"] is not None:
        solver.conv_tol = float(opts["conv_tol"])
    if opts["conv_tol_normt"] is not None:
        solver.conv_tol_normt = float(opts["conv_tol_normt"])
    for option in (
        "diis_space",
        "diis_start_cycle",
        "iterative_damping",
        "level_shift",
    ):
        if opts[option] is not None:
            setattr(solver, option, opts[option])

    e_corr, t1, t2 = solver.kernel()
    converged = bool(solver.converged)
    if bool(opts["solve_lambda"]):
        l1, l2 = solver.solve_lambda()
        converged = converged and bool(solver.converged_lambda)
    else:
        # 沿用 Vayesta 的 Lambda=T 近似。
        l1, l2 = np.array(t1, copy=True), np.array(t2, copy=True)

    rdm1 = solver.make_rdm1(t1=t1, t2=t2, l1=l1, l2=l2, ao_repr=False)
    rdm2 = solver.make_rdm2(t1=t1, t2=t2, l1=l1, l2=l2, ao_repr=False)
    return SolverResult(
        method="CCSD",
        converged=converged,
        energy=float(cluster_mf.e_tot + e_corr),
        rdm1=np.asarray(rdm1),
        rdm2=np.asarray(rdm2),
        t1=np.asarray(t1),
        t2=np.asarray(t2),
        l1=np.asarray(l1),
        l2=np.asarray(l2),
    )


def solve_fci(
    cluster: Cluster,
    reference_mf: scf.hf.RHF,
    options: Mapping[str, Any] | None = None,
) -> SolverResult:
    """按 cluster 类型使用 PySCF ``direct_spin1`` 或 ``direct_uhf``。"""

    _validate_reference_mf(reference_mf)
    _validate_cluster(cluster)
    opts = _validated_options("FCI", options)

    unrestricted = cluster.is_unrestricted
    solver: Any = (
        fci.direct_uhf.FCISolver(reference_mf.mol)
        if unrestricted
        else fci.direct_spin1.FCISolver(reference_mf.mol)
    )
    if opts["max_cycle"] is not None:
        solver.max_cycle = int(opts["max_cycle"])
    if opts["conv_tol"] is not None:
        solver.conv_tol = float(opts["conv_tol"])
    if opts["davidson_only"] is not None:
        solver.davidson_only = bool(opts["davidson_only"])
    if opts["threads"] is not None:
        solver.threads = int(opts["threads"])
    if opts["lindep"] is not None:
        solver.lindep = float(opts["lindep"])
    if opts["fix_spin"] is not None:
        if unrestricted:
            raise NotImplementedError(
                "fix_spin is not available for UHF clusters with spin-dependent orbital bases."
            )
        # 与上游 0.0 被当作 False 的行为不同：显式传入 0.0 即强制 singlet。
        solver = fci.addons.fix_spin_(
            solver,
            shift=float(opts["fix_spin_penalty"]),
            ss=float(opts["fix_spin"]),
        )

    h_eff = tuple(np.asarray(value) for value in cluster.h_eff) if unrestricted else np.asarray(cluster.h_eff)
    eri = tuple(np.asarray(value) for value in cluster.eri) if unrestricted else np.asarray(cluster.eri)
    energy, ci = solver.kernel(
        h_eff,
        eri,
        cluster.norb,
        cluster.nelec,
    )
    if unrestricted:
        rdm1, rdm2 = fci.direct_uhf.make_rdm12s(ci, cluster.norb, cluster.nelec)
        rdm1 = tuple(np.asarray(value) for value in rdm1)
        rdm2 = tuple(np.asarray(value) for value in rdm2)
    else:
        rdm1, rdm2 = fci.direct_spin1.make_rdm12(ci, cluster.norb, cluster.nelec)
    reference_amplitude = float(np.asarray(ci)[0, 0])
    return SolverResult(
        method="FCI",
        converged=bool(solver.converged),
        energy=float(energy),
        rdm1=rdm1 if unrestricted else np.asarray(rdm1),
        rdm2=rdm2 if unrestricted else np.asarray(rdm2),
        ci=np.asarray(ci),
        reference_amplitude=reference_amplitude,
    )


def solve_cluster(
    cluster: Cluster,
    reference_mf: scf.hf.RHF,
    method: str = "CCSD",
    options: Mapping[str, Any] | None = None,
) -> SolverResult:
    """按名称分派 CCSD 或 FCI cluster solver。"""

    normalized = method.upper()
    if normalized == "CCSD":
        return solve_ccsd(cluster, reference_mf, options)
    if normalized == "FCI":
        return solve_fci(cluster, reference_mf, options)
    raise ValueError("solver must be either 'CCSD' or 'FCI'.")


__all__ = [
    "build_cluster_rhf",
    "build_cluster_uhf",
    "solve_ccsd",
    "solve_cluster",
    "solve_fci",
]
