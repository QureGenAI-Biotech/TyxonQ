"""在实数 selected-CI 乘积空间，以自旋字符串连接直接构造 FP64 密度矩阵。"""
import numpy as np
from pyscf.fci import selected_ci


def _one_spin(coeff, labels, n, electrons, cd):
    """同自旋算符仅作用于一侧，先迹掉另一自旋，得到字符串密度矩阵。"""
    gram = coeff @ coeff.T
    occ = ((labels[:, None] >> np.arange(n)) & 1).astype(np.float64)
    src, slot = np.nonzero((cd[:, :, 3] != 0) & (cd[:, :, 0] != cd[:, :, 1]))
    p, q, dst, sign = cd[src, slot].T
    orbital = p * n + q
    dm1 = np.bincount(orbital, weights=sign * gram[src, dst], minlength=n*n).reshape(n, n)
    dm1[np.diag_indices(n)] = np.diag(gram) @ occ
    dm2 = np.zeros((n, n, n, n))
    if electrons >= 2:
        # 每个 N-2 电子的中间串共享一组可添回的轨道对，只需枚举一种轨道顺序。
        dd = selected_ci.des_des_linkstr_tril(labels, n, electrons)
        rows, slots = np.nonzero(dd[:, :, 3])
        pairs, _, det, phase = dd[rows, slots].T
        degrees = np.bincount(rows, minlength=len(dd))
        starts = np.cumsum(degrees) - degrees
        npair = n * (n-1) // 2
        density = np.zeros(npair**2)
        for degree in np.unique(degrees):
            if degree == 0:
                continue
            first = starts[degrees == degree]
            # 控制临时矩阵大小，每块最多约五十万个轨道对乘积。
            block = max(1, 500_000 // int(degree)**2)
            for i in range(0, len(first), block):
                k = first[i:i+block, None] + np.arange(degree)
                dets, signs, orbs = det[k], phase[k], pairs[k]
                weights = gram[dets[:, :, None], dets[:, None, :]] * signs[:, :, None] * signs[:, None, :]
                keys = orbs[:, :, None]*npair + orbs[:, None, :]
                density += np.bincount(keys.ravel(), weights=weights.ravel(), minlength=npair**2)
        p, r = np.tril_indices(n, -1)
        matrix = density.reshape(npair, npair)
        # D[p,q,r,s]=<p† r† s q> 对两个产生或两个湮灭指标分别反对称。
        for a, b, sign1 in ((p, r, 1), (r, p, -1)):
            for c, d, sign2 in ((p, r, 1), (r, p, -1)):
                dm2[a[:, None], c, b[:, None], d] = sign1*sign2*matrix
    return dm1, dm2, occ, (src, dst, orbital, sign)


def _mixed(coeff, n, occ_a, alpha, occ_b, beta):
    """分别计算无激发、单侧单激发和两侧同时单激发的异自旋密度。"""
    mixed = np.zeros((n*n, n*n))
    diag = np.arange(n) * (n+1)
    mixed[np.ix_(diag, diag)] = occ_a.T @ (coeff*coeff) @ occ_b
    for c, occ, edges, sink in ((coeff, occ_b, alpha, mixed), (coeff.T, occ_a, beta, mixed.T)):
        src, dst, orb, sign = edges
        acc = np.zeros((n*n, n))
        block = max(1, 500_000 // c.shape[1])
        for i in range(0, len(src), block):
            sl = slice(i, i+block)
            np.add.at(acc, orb[sl], ((c[src[sl]] * c[dst[sl]]) @ occ) * sign[sl, None])
        sink[:, diag] += acc
    a, aa, pa, sa = alpha
    b, bb, pb, sb = beta
    block = max(1, 500_000 // max(1, len(b)))
    for i in range(0, len(a), block):
        sl = slice(i, i+block)
        values = coeff[a[sl, None], b] * coeff[aa[sl, None], bb] * sa[sl, None] * sb
        keys = pa[sl, None]*(n*n) + pb
        mixed += np.bincount(keys.ravel(), weights=values.ravel(), minlength=n**4).reshape(n*n, n*n)
    return mixed.reshape((n,)*4)


def spin_rdms(vector, n, electrons, *, report=None):
    """选择直接连接收缩或标准 PySCF RDM；两条路径都保留所有系数。"""
    labels = tuple(np.ascontiguousarray(s, dtype=np.int64) for s in vector._strs)
    c = np.asarray(vector)
    # 对很大而稠密的字符串图使用 PySCF 分块路径，限制 Gram 矩阵和边积大小。
    use_links = c.dtype == np.float64 and 0 < min(electrons) and n <= 32 and max(c.shape) <= 2048
    if use_links:
        links = [selected_ci.cre_des_linkstr(strings, n, count) for strings, count in zip(labels, electrons)]
        edges = [int(np.count_nonzero(table[:, :, 3])) - len(strings)*count
                 for table, strings, count in zip(links, labels, electrons)]
        # 两侧非对角单激发数的乘积决定异自旋收缩工作量，内存另由分块大小控制。
        use_links = edges[0] * edges[1] <= 32_000_000
    if report is not None:
        report["algorithm"] = "spin_links_fp64" if use_links else "pyscf_spin_blocks"
    if not use_links:
        return selected_ci.make_rdm1s(vector, n, electrons), selected_ci.make_rdm2s(vector, n, electrons)
    a1, aa, occ_a, a = _one_spin(c, labels[0], n, electrons[0], links[0])
    b1, bb, occ_b, b = _one_spin(c.T, labels[1], n, electrons[1], links[1])
    ab = _mixed(c, n, occ_a, a, occ_b, b)
    return (a1, b1), (aa, ab, bb)


def same_spin_rdms(coefficients, labels, n, electrons, tables):
    """仅计算 FP64 单体和同自旋双体块；两侧独立，不创建异自旋密度。"""
    a1, aa, _, _ = _one_spin(coefficients, labels[0], n, electrons[0], tables[0])
    b1, bb, _, _ = _one_spin(coefficients.T, labels[1], n, electrons[1], tables[1])
    return (a1, b1), (aa, bb)


def spin_sum(dm1s, dm2s):
    """按 PySCF 对实数态的指标约定组合自旋块。"""
    aa, ab, bb = dm2s
    total = aa.copy()
    total += bb
    total += ab
    total += ab.transpose(2, 3, 0, 1)
    return dm1s[0] + dm1s[1], total
