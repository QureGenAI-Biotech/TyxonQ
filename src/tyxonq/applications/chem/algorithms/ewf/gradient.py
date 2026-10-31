"""计算闭壳层线性密度能量的 frozen-bath（固定浴）近似核梯度。

包含积分、重叠和 RHF 轨道响应；不包含片段浴及投影的完整响应，
因此不是每个位移点重建 EWF 后的完整能量严格导数。"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from pyscf import ao2mo, lib
from pyscf.grad import rhf as rhf_grad
from pyscf.grad.mp2 import _shell_prange
from pyscf.scf import cphf
from pyscf.scf.hf import get_veff

from .density import DensityResult, _peak_memory_mb, _validate_reference, effective_rdm2


def nuclear_gradient(mf, density: DensityResult) -> np.ndarray:
    """返回含核排斥项的近似梯度，形状 ``(natm, 3)``、单位 Ha/Bohr。

    全空间相当于 CASCI 梯度的 ``ncore=0, ncas=nmo``，因此仅有占据—
    虚轨道的 CPHF 响应。全局 Λ 即使收敛，也没有补齐 EWF 构造的导数。
    """

    started = perf_counter()
    coeff, nocc = _validate_reference(mf)
    if density.diagnostics.get("assembly") == "rdm_t_lambda" and not density.diagnostics.get("lambda_converged"):
        raise RuntimeError("Cannot evaluate gradients with unconverged global Lambda")
    mol = mf.mol
    nao, nmo = coeff.shape
    dm1 = np.asarray(density.rdm1)
    if dm1.shape != (nmo, nmo) or density.dm2_corr.shape != (nmo,) * 4:
        raise ValueError("Global density dimensions do not match the RHF orbitals")
    dm2 = effective_rdm2(dm1, density.dm2_corr, mf.mo_occ)
    if not np.all(np.isfinite(dm1)) or not np.all(np.isfinite(dm2)):
        raise ValueError("Global densities must be finite")
    hcore = mf.get_hcore()
    hf_dm1 = mf.make_rdm1()
    dm_ao = coeff @ dm1 @ coeff.T
    eri = ao2mo.kernel(mol, coeff, compact=False).reshape((nmo,) * 4)
    # 轨道拉格朗日矩阵使用线性能量对应的有效 2-RDM。
    imat = coeff.T @ hcore @ coeff @ dm1
    imat += lib.einsum("uviw,vuwt->it", eri, dm2)
    del eri
    occupied, virtual = coeff[:, :nocc], coeff[:, nocc:]
    xvo = imat[nocc:, :nocc] - imat[:nocc, nocc:].T

    def fvind(x):
        """RHF 占据—虚轨道响应的库仑/交换线性算符。"""

        x = x.reshape(xvo.shape)
        delta = virtual @ x @ occupied.T
        potential = get_veff(mol, delta + delta.T)
        return 2 * (virtual.T @ potential @ occupied)

    zvec = np.zeros_like(imat)
    residual_max = 0.0
    initial_residual_max = 0.0
    refinement_steps = 0
    if xvo.size:
        # PySCF Krylov 的绝对线性相关阈值可能截断很小的驱动项。
        # 归一化右端后求解；预条件器的轨道能隙仍可能放大残差，
        # 因此用原方程残差作最多两次迭代修正，不放宽验收阈值。
        response = np.zeros_like(xvo)
        residual = xvo.copy()
        gap = mf.mo_energy[nocc:, None] - mf.mo_energy[:nocc]
        for iteration in range(3):
            drive_scale = float(np.max(np.abs(residual)))
            if not np.isfinite(drive_scale):
                raise RuntimeError("CPHF response is non-finite")
            if drive_scale < 1e-14:
                break
            response += cphf.solve(
                fvind, mf.mo_energy, mf.mo_occ, residual / drive_scale,
                max_cycle=50, tol=1e-13,
            )[0] * drive_scale
            residual = gap * response + fvind(response) + xvo
            residual_max = float(np.max(np.abs(residual)))
            if iteration == 0:
                initial_residual_max = residual_max
            refinement_steps = iteration
            if residual_max < 1e-9:
                break
        residual_max = float(np.max(np.abs(residual)))
        if not np.isfinite(residual_max) or residual_max > 1e-7:
            raise RuntimeError(f"CPHF response did not converge; residual={residual_max:.3e}")
        zvec[nocc:, :nocc] = response
    zeta = coeff @ (zvec * mf.mo_energy[None, :]) @ coeff.T
    zvec_ao = coeff @ (zvec + zvec.T) @ coeff.T
    occ_projector = occupied @ occupied.T
    vhf_s1occ = occ_projector @ get_veff(mol, zvec_ao) @ occ_projector
    imat[nocc:, :nocc] = imat[:nocc, nocc:].T
    imat_ao = coeff @ imat @ coeff.T

    # 对最后两个 AO 指标作三角压缩，避免同时存放完整 AO 四阶密度。
    nao_pair = nao * (nao + 1) // 2
    diag = np.arange(nao)
    diag = diag * (diag + 1) // 2 + diag
    sym_dm2 = dm2 + dm2.transpose(0, 1, 3, 2)
    dm2buf = ao2mo._ao2mo.nr_e2(
        sym_dm2.reshape(nmo**2, nmo**2), coeff.T, (0, nao, 0, nao)
    ).reshape(nmo**2, nao, nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:, diag] *= 0.5
    dm2buf = dm2buf.reshape(nmo, nmo, nao_pair)
    del sym_dm2, dm2

    mf_grad = mf.nuc_grad_method()
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = rhf_grad.get_ovlp(mol)
    aoslices = mol.aoslice_by_atom()
    available = mf.max_memory - lib.current_memory()[0]
    blocksize = int(available * 0.9e6 / 8 / ((aoslices[:, 3] - aoslices[:, 2]).max() * nao_pair))
    blocksize = min(nao, max(2, blocksize))
    gradient = np.zeros((mol.natm, 3))
    for atom, (shl0, shl1, p0, p1) in enumerate(aoslices):
        h1 = hcore_deriv(atom)
        gradient[atom] += np.einsum("xij,ij->x", h1, dm_ao + zvec_ao)
        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blocksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum("ijw,pi,qj->pqw", dm2buf, coeff[p0:p1], coeff[q0:q1])
            eri1 = mol.intor(
                "int2e_ip1", comp=3, aosym="s2kl",
                shls_slice=(shl0, shl1, b0, b1, 0, mol.nbas, 0, mol.nbas),
            ).reshape(3, p1 - p0, nf, nao_pair)
            gradient[atom] -= 2 * np.einsum("xijw,ijw->x", eri1, dm2_ao)
            for axis in range(3):
                deriv = lib.unpack_tril(eri1[axis].reshape((p1 - p0) * nf, -1))
                deriv = deriv.reshape(p1 - p0, nf, nao, nao)
                # HF 密度与轨道响应的交叉项。
                gradient[atom, axis] -= 2 * np.einsum("ijkl,ij,kl", deriv, hf_dm1[p0:p1, q0:q1], zvec_ao)
                gradient[atom, axis] -= 2 * np.einsum("ijkl,kl,ij", deriv, hf_dm1, zvec_ao[p0:p1, q0:q1])
                gradient[atom, axis] += np.einsum("ijkl,il,kj", deriv, hf_dm1[p0:p1], zvec_ao[:, q0:q1])
                gradient[atom, axis] += np.einsum("ijkl,jk,il", deriv, hf_dm1[q0:q1], zvec_ao[p0:p1])
        # 重叠导数含能量加权密度及 CPHF 修正。
        for matrix, factor in ((imat_ao, 1), (zeta, 2), (vhf_s1occ, 2)):
            gradient[atom] -= factor * np.einsum("xij,ij->x", s1[:, p0:p1], matrix[p0:p1])
            gradient[atom] -= factor * np.einsum("xij,ji->x", s1[:, p0:p1], matrix[:, p0:p1])
    gradient += mf_grad.grad_nuc(mol)
    if not np.all(np.isfinite(gradient)):
        raise RuntimeError("Nuclear gradient is non-finite")
    density.diagnostics.update({
        "gradient_seconds": perf_counter() - started,
        "gradient_units": "Hartree/Bohr",
        "cphf_residual_max": residual_max,
        "cphf_initial_residual_max": initial_residual_max,
        "cphf_refinement_steps": refinement_steps,
        "gradient_translation_residual": float(np.max(np.abs(gradient.sum(axis=0)))),
        "process_peak_memory_mb": _peak_memory_mb(),
    })
    return gradient
