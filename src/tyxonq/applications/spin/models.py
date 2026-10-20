"""最近邻自旋链；所有表示共用 S = sigma / 2 和 q0 最左位约定。"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


PAULIS = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def site_operator(n: int, site: int, axis: str) -> sparse.csr_matrix:
    """构造单个位点的 Pauli 算符，不额外乘自旋的 1/2。"""
    axis = axis.upper()
    if axis not in PAULIS or not 0 <= site < n:
        raise ValueError("axis must be X/Y/Z and site must be inside the chain")
    out = sparse.csr_matrix([[1.0 + 0j]])
    for q in range(n):
        out = sparse.kron(out, PAULIS[axis] if q == site else np.eye(2), format="csr")
    return out


@dataclass(frozen=True)
class SpinChainModel:
    """偶数位、开边界 XXZ 链，H = sum[Jxy(XX+YY)+Jz ZZ]/4。"""

    n: int
    jxy: float = 1.0
    jz: float = 1.0

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(self.n, (int, np.integer)) or self.n < 2 or self.n % 2:
            raise ValueError("n must be an even integer >= 2")
        if not np.isfinite([self.jxy, self.jz]).all():
            raise ValueError("couplings must be finite")

    def pauli_terms(self) -> list[tuple[float, tuple[tuple[str, int], ...]]]:
        """以显式位点编号返回 Pauli 项，避免字符串端序歧义。"""
        return [
            (float(j) / 4, ((axis, i), (axis, i + 1)))
            for i in range(self.n - 1)
            for axis, j in (("X", self.jxy), ("Y", self.jxy), ("Z", self.jz))
            if j != 0
        ]

    def sparse_matrix(self) -> sparse.csr_matrix:
        """用于小体系独立精确解的稀疏矩阵。"""
        out = sparse.csr_matrix((2**self.n, 2**self.n), dtype=complex)
        for coeff, term in self.pauli_terms():
            matrices = {q: PAULIS[axis] for axis, q in term}
            product = sparse.csr_matrix([[1.0 + 0j]])
            for q in range(self.n):
                product = sparse.kron(product, matrices.get(q, np.eye(2)), format="csr")
            out += coeff * product
        return out

    def mpo(self):
        """quimb 的 SpinHam1D 使用自旋算符，耦合系数不再除以 4。"""
        import quimb.tensor as qtn

        builder = qtn.SpinHam1D(S=0.5, cyclic=False)
        # SxSx + SySy = (S+S- + S-S+)/2，使用实数 MPO 提高 DMRG 效率。
        builder += self.jxy / 2, "+", "-"
        builder += self.jxy / 2, "-", "+"
        builder += self.jz, "Z", "Z"
        return builder.build_mpo(self.n)
