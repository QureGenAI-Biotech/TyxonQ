"""在固定行列式子空间求解实数 UHF Hamiltonian 的最低本征态。

按 aa、ab、bb 三个自旋块进行 Hamiltonian 收缩，并计算自旋分辨 RDM 和能量。"""

from __future__ import annotations

import ctypes

import numpy as np
from pyscf import ao2mo, lib
from pyscf.fci import cistring, direct_spin1, direct_uhf, selected_ci
from pyscf.lib import logger


OneBodyUHF = tuple[np.ndarray, np.ndarray]
TwoBodyUHF = tuple[np.ndarray, np.ndarray, np.ndarray]


def kernel_fixed_space_uhf(
    myci,
    h1e: OneBodyUHF,
    eri: TwoBodyUHF,
    norb: int,
    nelec: tuple[int, int],
    ci_strs: tuple[np.ndarray, np.ndarray],
    ci0=None,
    tol=None,
    lindep=None,
    max_cycle=None,
    max_space=None,
    nroots=None,
    davidson_only=None,
    max_memory=None,
    verbose=None,
    ecore: float = 0.0,
    **kwargs,
):
    """在给定 alpha/beta determinant 笛卡尔积中对角化 UHF Hamiltonian。"""
    del davidson_only  # 与 PySCF fixed-space 入口保持兼容；实际由 ``myci.eig`` 决定。

    log = logger.new_logger(myci, verbose)
    if tol is None:
        tol = myci.conv_tol
    if lindep is None:
        lindep = myci.lindep
    if max_cycle is None:
        max_cycle = myci.max_cycle
    if max_space is None:
        max_space = myci.max_space
    if max_memory is None:
        max_memory = myci.max_memory
    if nroots is None:
        nroots = myci.nroots
    if myci.verbose >= logger.WARN:
        myci.check_sanity()

    nelec = direct_spin1._unpack_nelec(nelec, myci.spin)
    ci_strs = (np.asarray(ci_strs[0]), np.asarray(ci_strs[1]))
    na = len(ci_strs[0])
    nb = len(ci_strs[1])
    if na == 0 or nb == 0:
        raise ValueError("The selected-CI determinant spaces must not be empty.")

    # direct_uhf 的收缩接口把一体项吸收到三组二体张量中。
    h2e = direct_uhf.absorb_h1e(h1e, eri, norb, nelec, 0.5)
    link_index = selected_ci._all_linkstr_index(ci_strs, norb, nelec)
    hdiag = make_hdiag_uhf(h1e, eri, ci_strs, norb, nelec)

    if isinstance(ci0, selected_ci.SCIvector):
        if ci0.size == na * nb:
            ci0 = [ci0.ravel()]
        else:
            ci0 = [root.ravel() for root in ci0]
    elif ci0 is None:
        ci0 = myci.get_init_guess(ci_strs, norb, nelec, nroots, hdiag)
    else:
        ci0_array = np.asarray(ci0)
        if ci0_array.size == na * nb:
            ci0 = [ci0_array.ravel()]
        else:
            ci0 = [np.asarray(root).ravel() for root in ci0]

    cpu0 = [logger.process_clock(), logger.perf_counter()]

    def hop(coefficients):
        nonlocal cpu0
        sci_vector = selected_ci._as_SCIvector(coefficients, ci_strs)
        contracted = contract_2e_uhf(h2e, sci_vector, norb, nelec, link_index)
        cpu0 = list(log.timer_debug1("contract_2e_uhf", *cpu0))
        return contracted.reshape(-1)

    def precondition(residual, energy, *args):
        del args
        return residual / (hdiag - energy + 1e-4)

    energy, coefficients = myci.eig(
        hop,
        ci0,
        precondition,
        tol=tol,
        lindep=lindep,
        max_cycle=max_cycle,
        max_space=max_space,
        nroots=nroots,
        max_memory=max_memory,
        verbose=log,
        **kwargs,
    )
    if nroots > 1:
        states = [
            selected_ci._as_SCIvector(root.reshape(na, nb), ci_strs)
            for root in coefficients
        ]
        return energy + ecore, states
    state = selected_ci._as_SCIvector(coefficients.reshape(na, nb), ci_strs)
    return energy + ecore, state


def contract_2e_uhf(
    eri: TwoBodyUHF,
    civec_strs,
    norb: int,
    nelec: tuple[int, int],
    link_index=None,
):
    """收缩已吸收一体项的 UHF Hamiltonian 与 selected-CI 系数。"""
    ci_coeff, nelec, ci_strs = selected_ci._unpack(civec_strs, nelec)
    if np.iscomplexobj(ci_coeff) or any(np.iscomplexobj(block) for block in eri):
        raise NotImplementedError("Unrestricted selected-CI currently supports real tensors only.")
    if link_index is None:
        link_index = selected_ci._all_linkstr_index(ci_strs, norb, nelec)

    cd_indexa, dd_indexa, cd_indexb, dd_indexb = link_index
    na = len(ci_strs[0])
    nb = len(ci_strs[1])
    fcivec = np.asarray(ci_coeff, dtype=np.float64, order="C").reshape(na, nb)
    ci1 = np.zeros_like(fcivec)

    h2aa = np.asarray(ao2mo.restore(1, eri[0], norb), dtype=np.float64, order="C")
    h2ab = np.asarray(ao2mo.restore(1, eri[1], norb), dtype=np.float64, order="C")
    h2bb = np.asarray(ao2mo.restore(1, eri[2], norb), dtype=np.float64, order="C")

    # 同自旋块先反对称化，再调用 selected-CI 的 double-annihilation kernel。
    if nelec[0] > 1:
        ci1 += _contract_same_spin(2.0 * h2aa, fcivec, dd_indexa, norb)
    if nelec[1] > 1:
        fcivec_t = np.asarray(fcivec.T, order="C")
        ci1_t = _contract_same_spin(2.0 * h2bb, fcivec_t, dd_indexb, norb)
        ci1 += ci1_t.T

    # ``h2ab`` 的小步长 pair 属于 beta，因此这里需要在 pair 空间转置。
    if nelec[0] and nelec[1]:
        eri_ab_pair = np.asarray(2.0 * ao2mo.restore(4, h2ab, norb).T, order="C")
        selected_ci.libfci.SCIcontract_2e_bbaa(
            eri_ab_pair.ctypes.data_as(ctypes.c_void_p),
            fcivec.ctypes.data_as(ctypes.c_void_p),
            ci1.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(norb),
            ctypes.c_int(na),
            ctypes.c_int(nb),
            ctypes.c_int(cd_indexa.shape[1]),
            ctypes.c_int(cd_indexb.shape[1]),
            cd_indexa.ctypes.data_as(ctypes.c_void_p),
            cd_indexb.ctypes.data_as(ctypes.c_void_p),
        )

    # aaaa kernel 计算真正的双激发；补回 E_pq E_rs 恒等式中的 delta(qr) E_ps。
    if nelec[0]:
        correction_a = np.einsum("pqqs->ps", h2aa)
        _contract_one_spin(correction_a, fcivec, ci1, cd_indexa, cd_indexb, norb, True)
    if nelec[1]:
        correction_b = np.einsum("pqqs->ps", h2bb)
        _contract_one_spin(correction_b, fcivec, ci1, cd_indexa, cd_indexb, norb, False)

    return selected_ci._as_SCIvector(ci1.reshape(ci_coeff.shape), ci_strs)


def make_hdiag_uhf(
    h1e: OneBodyUHF,
    eri: TwoBodyUHF,
    ci_strs: tuple[np.ndarray, np.ndarray],
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    """只为 selected determinant 构造 UHF Hamiltonian 对角元。"""
    if any(np.iscomplexobj(block) for block in (*h1e, *eri)):
        raise NotImplementedError("Unrestricted selected-CI currently supports real tensors only.")
    h1a = np.asarray(h1e[0], dtype=np.float64, order="C")
    h1b = np.asarray(h1e[1], dtype=np.float64, order="C")
    gaa = np.asarray(ao2mo.restore(1, eri[0], norb), dtype=np.float64, order="C")
    gab = np.asarray(ao2mo.restore(1, eri[1], norb), dtype=np.float64, order="C")
    gbb = np.asarray(ao2mo.restore(1, eri[2], norb), dtype=np.float64, order="C")
    jdiag_aa = np.asarray(np.einsum("iijj->ij", gaa), order="C")
    jdiag_ab = np.asarray(np.einsum("iijj->ij", gab), order="C")
    jdiag_bb = np.asarray(np.einsum("iijj->ij", gbb), order="C")
    kdiag_aa = np.asarray(np.einsum("ijji->ij", gaa), order="C")
    kdiag_bb = np.asarray(np.einsum("ijji->ij", gbb), order="C")

    strings_a = np.asarray(ci_strs[0], dtype=np.int64)
    strings_b = np.asarray(ci_strs[1], dtype=np.int64)
    occs_a = np.asarray(cistring._strs2occslst(strings_a, norb), dtype=np.int32, order="C")
    occs_b = np.asarray(cistring._strs2occslst(strings_b, norb), dtype=np.int32, order="C")
    hdiag = np.empty(len(strings_a) * len(strings_b), dtype=np.float64)

    selected_ci.libfci.FCImake_hdiag_uhf(
        hdiag.ctypes.data_as(ctypes.c_void_p),
        h1a.ctypes.data_as(ctypes.c_void_p),
        h1b.ctypes.data_as(ctypes.c_void_p),
        jdiag_aa.ctypes.data_as(ctypes.c_void_p),
        jdiag_ab.ctypes.data_as(ctypes.c_void_p),
        jdiag_bb.ctypes.data_as(ctypes.c_void_p),
        kdiag_aa.ctypes.data_as(ctypes.c_void_p),
        kdiag_bb.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(norb),
        ctypes.c_int(len(strings_a)),
        ctypes.c_int(len(strings_b)),
        ctypes.c_int(nelec[0]),
        ctypes.c_int(nelec[1]),
        occs_a.ctypes.data_as(ctypes.c_void_p),
        occs_b.ctypes.data_as(ctypes.c_void_p),
    )
    return hdiag


def _contract_same_spin(
    eri: np.ndarray,
    fcivec: np.ndarray,
    dd_index: np.ndarray,
    norb: int,
) -> np.ndarray:
    """收缩一个 aa 或 bb 自旋块。"""
    eri_as = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
    idx, idy = np.tril_indices(norb, -1)
    pair_indices = idx * norb + idy
    eri_as = np.asarray(
        lib.take_2d(eri_as.reshape(norb**2, -1), pair_indices, pair_indices),
        order="C",
    )
    n_strings, n_other = fcivec.shape
    output = np.zeros_like(fcivec)
    selected_ci.libfci.SCIcontract_2e_aaaa(
        eri_as.ctypes.data_as(ctypes.c_void_p),
        fcivec.ctypes.data_as(ctypes.c_void_p),
        output.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(norb),
        ctypes.c_int(n_strings),
        ctypes.c_int(n_other),
        ctypes.c_int(dd_index.shape[0]),
        ctypes.c_int(dd_index.shape[1]),
        dd_index.ctypes.data_as(ctypes.c_void_p),
    )
    return output


def _contract_one_spin(
    h1e: np.ndarray,
    fcivec: np.ndarray,
    output: np.ndarray,
    cd_indexa: np.ndarray,
    cd_indexb: np.ndarray,
    norb: int,
    alpha: bool,
) -> None:
    """用 selected link 补充一个 alpha 或 beta 一体收缩。"""
    h1e_tril = np.asarray(lib.pack_tril(h1e), dtype=np.float64, order="C")
    contract = (
        selected_ci.libfci.FCIcontract_a_1e
        if alpha
        else selected_ci.libfci.FCIcontract_b_1e
    )
    contract(
        h1e_tril.ctypes.data_as(ctypes.c_void_p),
        fcivec.ctypes.data_as(ctypes.c_void_p),
        output.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(norb),
        ctypes.c_int(fcivec.shape[0]),
        ctypes.c_int(fcivec.shape[1]),
        ctypes.c_int(cd_indexa.shape[1]),
        ctypes.c_int(cd_indexb.shape[1]),
        cd_indexa.ctypes.data_as(ctypes.c_void_p),
        cd_indexb.ctypes.data_as(ctypes.c_void_p),
    )
