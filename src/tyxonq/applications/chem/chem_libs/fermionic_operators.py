"""实 restricted 积分与占据整数的公共合同，不改变 JW 模式索引。"""

from __future__ import annotations

from operator import index

import numpy as np
from openfermion import FermionOperator, normal_ordered


def validate_integrals(h1e, eri, norb: int, nelec: tuple[int, int]):
    """检查 PySCF/chemist 积分；模式为 alpha 在前、beta 在后。"""
    norb = index(norb)
    if norb < 1 or len(nelec) != 2:
        raise ValueError("norb must be positive and nelec must be (nalpha, nbeta)")
    nelec = tuple(index(n) for n in nelec)
    if any(n < 0 or n > norb for n in nelec):
        raise ValueError("each electron count must be between 0 and norb")
    h, g = np.asarray(h1e), np.asarray(eri)
    if h.shape != (norb, norb) or g.shape != (norb,) * 4:
        raise ValueError("expected h1e[norb,norb] and eri[norb,norb,norb,norb]")
    if np.iscomplexobj(h) or np.iscomplexobj(g):
        raise ValueError("only real restricted electronic integrals are supported")
    h, g = np.asarray(h, float), np.asarray(g, float)
    if not np.isfinite(h).all() or not np.isfinite(g).all():
        raise ValueError("integrals must be finite")
    if not np.allclose(h, h.T, atol=1e-10, rtol=0):
        raise ValueError("h1e must be symmetric")
    if any(not np.allclose(g, other, atol=1e-10, rtol=0) for other in
           (g.swapaxes(0, 1), g.swapaxes(2, 3), g.transpose(2, 3, 0, 1))):
        raise ValueError("eri must have real chemist eightfold symmetry")
    return h, g, norb, nelec


def electronic_fermion_operator(h1e, eri, *, atol: float = 0.0) -> FermionOperator:
    """构造 Σ h[p,q] p†q + 1/2 Σ (pq|rs) p†r†sq；不含能量常数。"""
    n = len(h1e)
    h, g, _, _ = validate_integrals(h1e, eri, n, (0, 0))
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    op = FermionOperator()
    for p, q in np.argwhere(np.abs(h) > atol):
        for spin in (0, n):
            op += FermionOperator(((int(p + spin), 1), (int(q + spin), 0)), h[p, q])
    for p, q, r, s in np.argwhere(np.abs(g) > atol):
        for a in (0, n):
            for b in (0, n):
                term = ((int(p + a), 1), (int(r + b), 1),
                        (int(s + b), 0), (int(q + a), 0))
                op += FermionOperator(term, 0.5 * g[p, q, r, s])
    return normal_ordered(op)


def apply_fermion_term(determinant: int, term):
    """从右向左作用产生/湮灭算符，返回目标占据整数及反对易符号。"""
    state, phase = int(determinant), 1
    for mode, create in reversed(term):
        occupied = (state >> mode) & 1
        if occupied == create:
            return None
        if (state & ((1 << mode) - 1)).bit_count() % 2:
            phase = -phase
        state ^= 1 << mode
    return state, phase


def validate_determinants(determinants, norb, nelec):
    """占据整数的 bit p 对应 mode p；不做 alpha/beta 笛卡尔积闭包。"""
    if norb < 1 or len(nelec) != 2 or any(n < 0 or n > norb for n in nelec):
        raise ValueError("invalid norb or nelec")
    mask = (1 << norb) - 1
    result = []
    for item in determinants:
        value = index(item)
        if value < 0 or value >= 1 << (2 * norb):
            raise ValueError("determinant exceeds the declared orbital register")
        if ((value & mask).bit_count(), (value >> norb).bit_count()) != tuple(nelec):
            raise ValueError("determinant has incorrect spin-resolved electron counts")
        result.append(value)
    return tuple(sorted(set(result)))
