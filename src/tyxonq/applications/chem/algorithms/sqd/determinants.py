"""在指定行列式集合中扩展组态并进行稀疏 Slater–Condon 求解。

占据整数的 bit p 对应 alpha[p]，bit (norb+p) 对应 beta[p]；
组态空间直接使用给定行列式，不对 α/β 标签取笛卡尔积。"""

from dataclasses import dataclass
from functools import cached_property
from itertools import combinations, product
from operator import index

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from ...chem_libs.fermionic_operators import (
    apply_fermion_term, electronic_fermion_operator, validate_determinants, validate_integrals,
)
from .fermion import spin_square_from_rdms


def expand_determinants(determinants, norb, nelec, *, max_rank=1, max_determinants=100000):
    """从原始 seeds 一次性生成总激发阶数 ≤ rank 的 S/SD/SDT，保留原集合。"""
    rank, limit = index(max_rank), index(max_determinants)
    if rank not in (1, 2, 3) or limit < 1:
        raise ValueError("max_rank must be 1, 2 or 3 and max_determinants must be positive")
    seeds = validate_determinants(determinants, norb, nelec)
    if not seeds:
        raise ValueError("at least one seed determinant is required")
    expanded = set(seeds)
    if len(expanded) > limit:
        raise ValueError("seed subspace exceeds max_determinants")
    for seed in seeds:
        occupied = [[s * norb + p for p in range(norb) if seed >> (s * norb + p) & 1] for s in range(2)]
        empty = [[s * norb + p for p in range(norb) if not seed >> (s * norb + p) & 1] for s in range(2)]
        for total in range(1, rank + 1):
            for ra in range(total + 1):
                rb = total - ra
                # 粒子和空穴都从 seed 选择，不在新组态上递归激发，避免超出 rank。
                for remove_a, add_a, remove_b, add_b in product(
                    combinations(occupied[0], ra), combinations(empty[0], ra),
                    combinations(occupied[1], rb), combinations(empty[1], rb),
                ):
                    changed = seed
                    for p in remove_a + add_a + remove_b + add_b:
                        changed ^= 1 << p
                    expanded.add(changed)
                    if len(expanded) > limit:
                        raise ValueError("expanded subspace exceeds max_determinants; no truncation was applied")
    return tuple(sorted(expanded))


def projected_hamiltonian(h1e, eri, determinants, norb, nelec):
    """按一/二电子激发及反对易符号累加矩阵元，只存储显式子空间内的连接。"""
    h, g, n, nelec = validate_integrals(h1e, eri, norb, nelec)
    determinants = validate_determinants(determinants, n, nelec)
    positions = {d: i for i, d in enumerate(determinants)}
    terms = electronic_fermion_operator(h, g).terms
    rows, cols, data = [], [], []
    for col, determinant in enumerate(determinants):
        for term, coefficient in terms.items():
            transition = apply_fermion_term(determinant, term)
            if transition is None:
                continue
            target, sign = transition
            row = positions.get(target)
            if row is not None:
                rows.append(row)
                cols.append(col)
                data.append(float(np.real(coefficient)) * sign)
    matrix = coo_matrix((data, (rows, cols)), shape=(len(determinants),) * 2).tocsr()
    matrix.sum_duplicates()
    difference = matrix - matrix.T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-10:
        raise ValueError("projected electronic Hamiltonian is not Hermitian")
    return matrix


@dataclass
class DeterminantCIResult:
    """一维 CI 系数与显式组态列表；RDM 延迟计算，避免无需求时分配 n⁴ 张量。"""

    energy: float
    determinants: tuple[int, ...]
    amplitudes: np.ndarray
    norb: int
    nelec: tuple[int, int]
    energy_offset: float = 0.0
    residual_norm: float = 0.0

    @property
    def total_energy(self):
        return self.energy + self.energy_offset

    @cached_property
    def rdm1s(self):
        n = self.norb
        densities = [np.zeros((n, n), complex) for _ in range(2)]
        positions = {d: i for i, d in enumerate(self.determinants)}
        for column, determinant in enumerate(self.determinants):
            for spin in range(2):
                for q in range(spin * n, (spin + 1) * n):
                    if not determinant >> q & 1:
                        continue
                    for p in range(spin * n, (spin + 1) * n):
                        moved = apply_fermion_term(determinant, ((p, 1), (q, 0)))
                        if moved is None or moved[0] not in positions:
                            continue
                        value = self.amplitudes[positions[moved[0]]].conjugate() * self.amplitudes[column] * moved[1]
                        # PySCF 的 dm1[q,p] = <a_p† a_q>。
                        densities[spin][q % n, p % n] += value
        return tuple(np.real_if_close(d) for d in densities)

    @cached_property
    def rdm2s(self):
        n = self.norb
        densities = [np.zeros((n,) * 4, complex) for _ in range(3)]
        positions = {d: i for i, d in enumerate(self.determinants)}
        for column, determinant in enumerate(self.determinants):
            for density, (a, b) in zip(densities, ((0, 0), (0, 1), (1, 1))):
                occ_a = [p for p in range(a * n, (a + 1) * n) if determinant >> p & 1]
                occ_b = [p for p in range(b * n, (b + 1) * n) if determinant >> p & 1]
                for q in occ_a:
                    for s in occ_b:
                        if q == s:
                            continue
                        removed = determinant ^ (1 << q) ^ (1 << s)
                        for p in range(a * n, (a + 1) * n):
                            if removed >> p & 1:
                                continue
                            for r in range(b * n, (b + 1) * n):
                                if p == r or removed >> r & 1:
                                    continue
                                moved = apply_fermion_term(determinant, ((p, 1), (r, 1), (s, 0), (q, 0)))
                                row = positions.get(moved[0])
                                if row is not None:
                                    density[p % n, q % n, r % n, s % n] += (
                                        self.amplitudes[row].conjugate() * self.amplitudes[column] * moved[1])
        return tuple(np.real_if_close(d) for d in densities)

    @property
    def rdm1(self):
        return self.rdm1s[0] + self.rdm1s[1]

    @property
    def rdm2(self):
        aa, ab, bb = self.rdm2s
        return aa + ab + ab.transpose(2, 3, 0, 1) + bb

    @property
    def orbital_occupancies(self):
        return tuple(np.diag(d) for d in self.rdm1s)

    def spin_square(self):
        return spin_square_from_rdms(self.rdm1s, self.rdm2s)


def solve_determinant_subspace(h1e, eri, determinants, norb, nelec, *,
                              energy_offset=0.0, max_determinants=100000, tolerance=1e-10):
    """稀疏矩阵 + eigsh 求最低本征对；不构造完整 2^(2*norb) Hamiltonian。"""
    h, g, n, nelec = validate_integrals(h1e, eri, norb, nelec)
    determinants = validate_determinants(determinants, n, nelec)
    if not determinants or len(determinants) > max_determinants:
        raise ValueError("determinant count must be between 1 and max_determinants")
    if not np.isfinite(energy_offset) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("energy_offset must be finite and tolerance positive")
    matrix = projected_hamiltonian(h, g, determinants, n, nelec)
    if len(determinants) <= 3:
        energies, vectors = eigh(matrix.toarray())
    else:
        start = np.random.default_rng(24).normal(size=len(determinants))
        energies, vectors = eigsh(matrix, k=1, which="SA", tol=tolerance, v0=start)
    energy, vector = float(energies[0]), vectors[:, 0]
    residual = float(np.linalg.norm(matrix @ vector - energy * vector))
    return DeterminantCIResult(energy, determinants, vector, n, nelec,
                               float(energy_offset), residual)


def solve_expanded_subspace(h1e, eri, determinants, norb, nelec, *,
                            max_rank=1, max_determinants=100000, energy_offset=0.0):
    """严格 S/SD/SDT 扩展后求解；所有阶数都相对于输入 seeds 定义。"""
    expanded = expand_determinants(determinants, norb, nelec,
                                  max_rank=max_rank, max_determinants=max_determinants)
    return solve_determinant_subspace(h1e, eri, expanded, norb, nelec,
                                     max_determinants=max_determinants, energy_offset=energy_offset)
