"""响应的离散傅里叶变换；原始数值和用于展示的处理严格分开。"""

import numpy as np


def response_from_kicks(plus, minus):
    """正负 R_beta(pi/2) 的 Pauli 期望差 /4 = -i<[S_alpha(t),S_beta]>。"""
    plus, minus = np.asarray(plus), np.asarray(minus)
    if plus.shape != minus.shape or plus.ndim != 2:
        raise ValueError("kick arrays must have the same (n_times, n_sites) shape")
    return (plus - minus) / 4


def compute_dsf(green, times, center, *, n_momentum=100, n_frequency=100):
    """中心位点近似、零温 S(q,omega)=-Im G^R(q,omega)/pi。

    q0 为最左位；时间积分只用 t>0。返回原始谱，不做镜像平均或截断。
    有限时间窗口会产生旁瓣和负值，增加绘图网格不提高实际物理分辨率。
    """
    green, times = np.asarray(green), np.asarray(times, dtype=float)
    if green.ndim != 2 or times.ndim != 1 or len(times) < 2 or green.shape[0] != len(times):
        raise ValueError("green must have shape (len(times), n_sites), with at least two times")
    dt = times[1] - times[0]
    if not np.isfinite(times).all() or dt <= 0 or not np.isclose(times[0], 0) or not np.allclose(np.diff(times), dt):
        raise ValueError("times must be uniform, increasing, and start at zero")
    if not np.isfinite(green).all() or not 0 <= center < green.shape[1]:
        raise ValueError("green must be finite and center must be a valid site")
    for count in (n_momentum, n_frequency):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 2:
            raise ValueError("grid sizes must be integers >= 2")
    momenta = np.linspace(0, 2 * np.pi, n_momentum, endpoint=False)
    frequencies = np.linspace(0, np.pi / dt, n_frequency, endpoint=False)
    time_phases = np.exp(1j * np.outer(frequencies, times[1:]))
    space_phases = np.exp(-1j * np.outer(np.arange(green.shape[1]) - center, momenta))
    transformed = dt * (time_phases @ green[1:]) @ space_phases
    return momenta, frequencies, -transformed.imag / np.pi


def spectrum_for_plot(spectrum, *, mirror=False, clip_negative=False, normalize=False):
    """生成绘图副本；半开动量网格的反射索引是 (-k) mod Q，不是简单翻转。"""
    out = np.asarray(spectrum, dtype=float).copy()
    if out.ndim != 2 or not np.isfinite(out).all():
        raise ValueError("spectrum must be a finite 2D array")
    if mirror:
        opposite = (-np.arange(out.shape[1])) % out.shape[1]
        out = (out + out[:, opposite]) / 2
    if clip_negative:
        out = np.maximum(out, 0)
    if normalize:
        scale = float(np.max(np.abs(out)))
        if scale > 0:
            out /= scale
    return out


def error_metrics(actual, reference):
    """同时报告绝对误差和相对 L2；零参考信号不伪造有限相对误差。"""
    diff = np.asarray(actual) - np.asarray(reference)
    norm = float(np.linalg.norm(reference))
    return {"max_abs": float(np.max(np.abs(diff))), "rmse": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
            "relative_l2": float(np.linalg.norm(diff) / norm) if norm > 1e-12 else None}
