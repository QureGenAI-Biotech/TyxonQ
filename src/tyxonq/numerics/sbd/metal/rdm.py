"""以轨道转移分组，在 Apple GPU 上对补偿乘积进行整数归约。"""
import numpy as np

from .solver import _kernels, _mlx, metal_gpu_execution


def check_mixed_rdm_size(shape):
    """约束每个轨道四元组的量化误差，并控制同自旋 Gram 矩阵内存。"""
    if shape[0]*shape[1] > 2**23 or max(shape) > 4096:
        raise ValueError("Metal mixed RDM requires D <= 2**23 and at most 4096 strings per spin; "
                         "use rdm_backend='cpu' for larger spaces.")


def _group(table, norb):
    """同一轨道转移的源串、带符号目标串和偏移连续排列。"""
    mx = _mlx()
    row, slot = np.nonzero(table[:, :, 3])
    p, q, target, sign = table[row, slot].T
    orbital = p*norb + q
    order = np.argsort(orbital, kind="stable")
    active, count = np.unique(orbital, return_counts=True)
    links = np.column_stack((row, (target+1)*sign))[order]
    offsets = np.r_[0, np.cumsum(count)]
    return active, mx.array(links.astype(np.int32)), mx.array(offsets.astype(np.uint32))


@metal_gpu_execution
def mixed_rdm_q60(coefficients, norb, nelec, tables):
    """计算实数单位 CI 的异自旋块，结果以 FP64 返回主机。

    固定轨道转移是一对一映射，故 Σ|c_i c_j| ≤ 1，Q60 累加有 8 倍溢出余量。
    两个 FP32 分量表示每个系数，FMA 恢复乘积的一阶舍入误差。高低乘积
    各舍入到整数一次，最多 D 项的绝对量化误差为 D·2⁻⁶⁰，余下乘法误差
    为 FP32 单位舍入误差的二阶项。支持范围内量化上界不超过 2⁻³⁷。
    """
    mx = _mlx()
    c = np.asarray(coefficients)
    if c.ndim != 2 or np.iscomplexobj(c):
        raise ValueError("Metal mixed RDM requires a real CI coefficient matrix")
    c = c.astype(np.float64, copy=False)
    check_mixed_rdm_size(c.shape)
    norm2 = float(np.vdot(c, c))
    if not np.isfinite(norm2) or abs(norm2-1.) > 1e-8:
        raise ValueError("Metal mixed RDM requires normalized real CI coefficients")
    density = np.zeros((norb*norb, norb*norb))
    if min(nelec) == 0:
        return density.reshape((norb,)*4)
    pa, al, ao = _group(tables[0], norb)
    pb, bl, bo = _group(tables[1], norb)
    high = c.astype(np.float32)
    low = (c-high).astype(np.float32)
    result = _kernels()["mixed_rdm_q60"](
        inputs=[mx.array(high), mx.array(low), al, ao, bl, bo,
                mx.array([c.shape[1], len(pb)], dtype=mx.uint32)],
        grid=(256*len(pa)*len(pb), 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(len(pa), len(pb))], output_dtypes=[mx.int64],
    )[0]
    density[np.ix_(pa,pb)] = np.asarray(result)
    density *= 2.**-60
    return density.reshape((norb,)*4)
