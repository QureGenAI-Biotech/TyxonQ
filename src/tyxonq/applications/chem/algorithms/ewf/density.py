"""从闭壳层片段 RDM 组装有效振幅、全局密度及线性密度能量。

支持直接使用有效 T 或进一步求解 Λ；有效 T 不等同于全局 CCSD 解。
``dm2_corr`` 是 PySCF ``with_dm1=False`` 返回的相关部分，不是从最终
1-RDM 扣除直积得到的精确累积量。"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from pyscf import ao2mo, cc, scf
from pyscf.cc import ccsd_rdm

from .types import Cluster, SolverResult


@dataclass(frozen=True)
class DensityResult:
    """全局 MO 表示的密度、有效振幅与近似方法诊断。"""

    energy: float
    rdm1: np.ndarray
    dm2_corr: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    diagnostics: dict[str, Any]


def _peak_memory_mb() -> float | None:
    """报告进程生命周期的峰值 RSS；不冒充单次组装独占内存。"""

    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak / (1024**2 if sys.platform == "darwin" else 1024))


def _validate_reference(mf) -> tuple[np.ndarray, int]:
    """仅支持已收敛的标准分子、实数、规范 RHF 参考态。"""

    if not isinstance(mf, scf.hf.RHF) or hasattr(mf, "xc"):
        raise NotImplementedError("Density assembly requires a molecular RHF reference")
    if getattr(mf.mol, "dimension", 0) or mf.mol.spin != 0:
        raise NotImplementedError("Density assembly requires a closed-shell molecule")
    if not mf.converged:
        raise RuntimeError("RHF reference did not converge")
    for name in ("with_df", "with_solvent", "with_x2c", "mm_mol"):
        if getattr(mf, name, None) is not None:
            raise NotImplementedError(f"Density assembly does not support {name}")
    coeff = np.asarray(mf.mo_coeff)
    occ = np.asarray(mf.mo_occ)
    nocc = mf.mol.nelectron // 2
    expected = np.zeros(coeff.shape[1])
    expected[:nocc] = 2
    if coeff.ndim != 2 or np.iscomplexobj(coeff) or not np.array_equal(occ, expected):
        raise ValueError("RHF orbitals must be real and ordered occupied then virtual")
    if not np.all(np.isfinite(coeff)):
        raise ValueError("RHF orbitals must be finite")
    if not np.allclose(coeff.T @ mf.get_ovlp() @ coeff, np.eye(coeff.shape[1]), atol=1e-8):
        raise ValueError("RHF orbitals must be orthonormal")
    fock = coeff.T @ mf.get_fock() @ coeff
    if not np.allclose(fock, np.diag(mf.mo_energy), atol=1e-5, rtol=0):
        raise ValueError("Density assembly requires canonical RHF orbitals")
    return coeff, nocc


def effective_rdm2(rdm1: np.ndarray, dm2_corr: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
    """构造线性密度能量对应的有效 2-RDM，不包含 Δγ 的二次直积。"""

    hf = np.diag(mo_occ)
    delta = rdm1 - hf
    result = np.array(dm2_corr, copy=True)
    for left, right in ((hf, hf), (delta, hf), (hf, delta)):
        result += np.einsum("ij,kl->ijkl", left, right)
        result -= np.einsum("ij,kl->iklj", left, right) / 2
    return result


def assemble_density(
    mf,
    inputs: Iterable[tuple[Cluster, SolverResult, np.ndarray]],
    *,
    assembly: str = "rdm_t_lambda",
) -> DensityResult:
    """将片段完整 RDM 投影为全局有效 T，构造密度并计算总能量。

    ``inputs`` 的每项为 ``(cluster, solver_result, c_proj)``；``c_proj``
    是 AO 基中的片段投影轨道。``rdm_t`` 取 Λ=T；``rdm_t_lambda`` 固定
    注入的 T 求解 Λ，绝不调用 CCSD 振幅优化。两者均不是变分 EWF 波函数。
    """

    started = perf_counter()
    if assembly not in {"rdm_t", "rdm_t_lambda"}:
        raise ValueError("density assembly must be 'rdm_t' or 'rdm_t_lambda'")
    coeff, nocc = _validate_reference(mf)
    nmo = coeff.shape[1]
    nvir = nmo - nocc
    overlap = mf.get_ovlp()
    t1 = np.zeros((nocc, nvir))
    t2 = np.zeros((nocc, nocc, nvir, nvir))
    nfragments = 0
    for cluster, solution, c_proj in inputs:
        nfragments += 1
        if cluster.is_unrestricted or isinstance(solution.rdm1, tuple):
            raise NotImplementedError("Density assembly supports restricted clusters only")
        if not solution.converged:
            raise RuntimeError("A cluster solver did not converge")
        dm1 = np.asarray(solution.rdm1)
        dm2 = np.asarray(solution.rdm2)
        norb, no = cluster.norb, cluster.nocc
        if dm1.shape != (norb, norb) or dm2.shape != (norb,) * 4:
            raise ValueError("Cluster RDM shapes do not match its active orbitals")
        if any(np.iscomplexobj(x) or not np.all(np.isfinite(x)) for x in (dm1, dm2)):
            raise ValueError("Cluster RDMs must be real and finite")
        if not np.allclose(dm1, dm1.T, atol=1e-8, rtol=0):
            raise ValueError("Cluster 1-RDM must be symmetric")
        if not np.isclose(np.trace(dm1), 2 * no, atol=1e-7, rtol=0):
            raise ValueError("Cluster 1-RDM particle number does not match its electron count")
        if not np.allclose(np.einsum("pqrr->pq", dm2), (2 * no - 1) * dm1, atol=1e-6, rtol=0):
            raise ValueError("Cluster 2-RDM contraction violates the PySCF particle-number convention")
        c_occ, c_vir = cluster.c_active_occ, cluster.c_active_vir
        c_proj = np.asarray(c_proj)
        if c_proj.ndim != 2 or c_proj.shape[0] != coeff.shape[0] or np.iscomplexobj(c_proj):
            raise ValueError("Fragment projector orbitals must be real AO coefficient columns")
        # 这里才是从完整片段 RDM 扣除直积得到的精确累积量。
        cumulant = dm2 - np.einsum("ij,kl->ijkl", dm1, dm1)
        cumulant += np.einsum("ij,kl->iklj", dm1, dm1) / 2
        p_occ = c_occ.T @ overlap @ c_proj
        p_occ = p_occ @ p_occ.T
        t1_local = p_occ @ dm1[:no, no:]
        t2_local = np.einsum("xi,ijab->xjab", p_occ, cumulant[:no, :no, no:, no:])
        t2_local = (t2_local + t2_local.transpose(1, 0, 3, 2)) / 2
        ro = coeff[:, :nocc].T @ overlap @ c_occ
        rv = coeff[:, nocc:].T @ overlap @ c_vir
        t1 += np.einsum("Ii,Aa,ia->IA", ro, rv, t1_local, optimize=True)
        t2 += np.einsum("Ii,Jj,Aa,Bb,ijab->IJAB", ro, ro, rv, rv, t2_local, optimize=True)
    if nfragments == 0:
        raise ValueError("At least one fragment is required for density assembly")
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    assembly_seconds = perf_counter() - started

    mycc = cc.CCSD(mf)
    mycc.verbose = 0
    mycc.t1, mycc.t2 = t1, t2
    lambda_started = perf_counter()
    if assembly == "rdm_t_lambda" and nocc and nvir:
        eris = mycc.ao2mo(coeff)
        l1, l2 = mycc.solve_lambda(t1, t2, eris=eris)
        if not mycc.converged_lambda:
            raise RuntimeError("Global fixed-T CCSD Lambda equations did not converge")
        if not all(np.all(np.isfinite(x)) for x in (l1, l2)):
            raise RuntimeError("Global Lambda amplitudes are non-finite")
    else:
        l1, l2 = t1, t2
    lambda_seconds = perf_counter() - lambda_started
    if nocc and nvir:
        dm1 = ccsd_rdm.make_rdm1(mycc, t1, t2, l1, l2, with_frozen=False, ao_repr=False)
        dm2_corr = ccsd_rdm.make_rdm2(
            mycc, t1, t2, l1, l2, with_dm1=False, with_frozen=False, ao_repr=False
        )
    else:
        # 无可激发轨道时是严格 HF 极限；绕过 PySCF 零长度 HDF5 分块限制。
        dm1 = np.diag(mf.mo_occ)
        dm2_corr = np.zeros((nmo,) * 4)
    dm1 = (dm1 + dm1.T) / 2
    dm2_corr = (dm2_corr + dm2_corr.transpose(1, 0, 3, 2)) / 2
    fock = coeff.T @ mf.get_fock() @ coeff
    eri = ao2mo.kernel(mf.mol, coeff, compact=False).reshape((nmo,) * 4)
    delta = dm1 - np.diag(mf.mo_occ)
    energy = float(mf.e_tot + np.einsum("pq,pq", fock, delta)
                   + 0.5 * np.einsum("pqrs,pqrs", eri, dm2_corr))
    if not np.isfinite(energy) or not np.all(np.isfinite(dm1)) or not np.all(np.isfinite(dm2_corr)):
        raise RuntimeError("Global density or its energy is non-finite")
    diagnostics = {
        "assembly": assembly,
        "nfragments": nfragments,
        "lambda_converged": True if assembly == "rdm_t_lambda" else None,
        "amplitudes_optimized": False,
        "gradient_approximation": "frozen_bath_fixed_effective_amplitudes",
        "bath_projector_response": False,
        "electron_number": float(np.trace(dm1)),
        "assembly_seconds": assembly_seconds,
        "lambda_seconds": lambda_seconds,
        "total_seconds": perf_counter() - started,
        "rdm_bytes": int(dm1.nbytes + dm2_corr.nbytes),
        "process_peak_memory_mb": _peak_memory_mb(),
    }
    return DensityResult(energy, dm1, dm2_corr, t1, t2, diagnostics)
