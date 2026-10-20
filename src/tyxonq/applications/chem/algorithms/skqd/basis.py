"""局部自然轨道及 restricted 积分变基。旋转后必须在新基底重新采样。"""

import numpy as np

from ...chem_libs.fermionic_operators import validate_integrals


def build_local_natural_orbitals(rdm1, blocks):
    """按给定互不重叠轨道块对角化 1-RDM；列为新轨道，占据数从大到小。"""
    density = np.asarray(rdm1)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("rdm1 must be a square matrix")
    if (not np.isfinite(density).all() or np.max(np.abs(density.imag)) > 1e-10 or
        not np.allclose(density, density.conj().T, atol=1e-10, rtol=0)):
        raise ValueError("local natural orbitals currently require a real Hermitian rdm1")
    density = density.real
    n = len(density)
    rotation, seen = np.eye(n), set()
    for block in blocks:
        modes = tuple(int(p) for p in block)
        if not modes or len(set(modes)) != len(modes) or seen.intersection(modes):
            raise ValueError("natural orbital blocks must be non-empty and disjoint")
        if any(p < 0 or p >= n for p in modes):
            raise ValueError("natural orbital index out of range")
        seen.update(modes)
        values, vectors = np.linalg.eigh(density[np.ix_(modes, modes)])
        vectors = vectors[:, np.argsort(-values, kind="stable")]
        # 固定各本征向量的符号，避免一次等价分解产生难追踪的符号差别。
        pivots = np.argmax(np.abs(vectors), axis=0)
        vectors *= np.where(vectors[pivots, np.arange(len(modes))] < 0, -1, 1)
        rotation[np.ix_(modes, modes)] = vectors
    return rotation


def rotate_integrals(h1e, eri, rotation):
    """C[old,new] 定义 h_new=C.T h_old C，eri 四个指标按同一 C 旋转。"""
    n = len(h1e)
    h, g, _, _ = validate_integrals(h1e, eri, n, (0, 0))
    c = np.asarray(rotation)
    if (c.shape != (n, n) or np.iscomplexobj(c) or
        not np.allclose(c.T @ c, np.eye(n), atol=1e-10, rtol=0)):
        raise ValueError("rotation must be real orthogonal")
    return c.T @ h @ c, np.einsum("ap,bq,cr,ds,abcd->pqrs", c, c, c, c, g, optimize=True)


def correlations_from_rdms(rdm1s, rdm2s):
    """返回 <n_i n_j>、连通密度、<Sz_i Sz_j> 和 <S_i·S_j>。"""
    a, b = map(np.asarray, rdm1s)
    aa, ab, bb = map(np.asarray, rdm2s)
    n = a.shape[0]
    if a.shape != (n, n) or b.shape != (n, n) or any(x.shape != (n,) * 4 for x in (aa, ab, bb)):
        raise ValueError("inconsistent spin-resolved RDM shapes")
    occupancy = np.diag(a + b)
    same = np.einsum("iijj->ij", aa + bb)
    opposite = np.einsum("iijj->ij", ab)
    number = same + opposite + opposite.T + np.diag(occupancy)
    zz = (same - opposite - opposite.T + np.diag(occupancy)) / 4
    flip = np.einsum("ijji->ij", ab)
    dot = zz - (flip + flip.T) / 2 + np.diag(occupancy) / 2
    magnetization = np.diag(a - b) / 2
    same_connected = (same + np.diag(occupancy) - np.outer(np.diag(a), np.diag(a))
                      - np.outer(np.diag(b), np.diag(b)))
    return {"density": np.real_if_close(number),
            "connected_density": np.real_if_close(number - np.outer(occupancy, occupancy)),
            "same_spin_connected_density": np.real_if_close(same_connected),
            "spin_z": np.real_if_close(zz), "spin_dot": np.real_if_close(dot),
            "connected_spin_dot": np.real_if_close(dot - np.outer(magnetization, magnetization))}


def rdms_to_original_basis(rdm1s, rdm2s, rotation):
    """C[old,new] 将新轨道基底的 RDM 还原到原始基底，用于空间关联函数。"""
    c = np.asarray(rotation)
    n = len(c)
    if (c.shape != (n, n) or np.iscomplexobj(c) or
        not np.allclose(c.T @ c, np.eye(n), atol=1e-10, rtol=0)):
        raise ValueError("rotation must be real orthogonal")
    one = tuple(c @ d @ c.T for d in rdm1s)
    two = tuple(np.einsum("ap,bq,cr,ds,pqrs->abcd", c, c, c, c, d, optimize=True) for d in rdm2s)
    return one, two
