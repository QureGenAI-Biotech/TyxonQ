"""谱学后处理（窗函数 + FFT），纯函数复制自 ``applications/spin/spectra.py``。

按计划约定：**不修改 spin 包**，只把纯函数复制为 chem/dynamics 的
共享模块并按需泛化（关联函数 C(t) → 谱）；spin 包保持独立。

新增（相对 spin/spectra.py）：

- :func:`apply_window`：Hann/Gaussian/矩形窗；
- :func:`correlation_to_spectrum`：C(t) → S(ω)（FFT，含 dt 与窗语义）。

单位：原子单位 ħ=1；``times`` 均匀网格且含 t=0。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def response_from_kicks(plus, minus):
    """正负 kick 的期望差 /4 = -i<[S_alpha(t),S_beta]>（复制自 spin/spectra.py）。"""
    plus, minus = np.asarray(plus), np.asarray(minus)
    if plus.shape != minus.shape or plus.ndim != 2:
        raise ValueError("kick arrays must have the same (n_times, n_sites) shape")
    return (plus - minus) / 4


def compute_dsf(green, times, center, *, n_momentum=100, n_frequency=100):
    """中心位点近似、零温 S(q,omega)=-Im G^R(q,omega)/pi（复制自 spin/spectra.py）。

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
    """生成绘图副本；半开动量网格的反射索引是 (-k) mod Q（复制自 spin/spectra.py）。"""
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
    """同时报告绝对误差和相对 L2（复制自 spin/spectra.py）。"""
    diff = np.asarray(actual) - np.asarray(reference)
    norm = float(np.linalg.norm(reference))
    return {"max_abs": float(np.max(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(np.abs(diff) ** 2))),
            "relative_l2": float(np.linalg.norm(diff) / norm) if norm > 1e-12 else None}


# ---------------- 新增：关联函数 → 谱 ---------------- #

def apply_window(correlation, times, *, window: str = "hann",
                 gaussian_sigma_scale: float = 0.25):
    """给 C(t) 加窗（t=0 点保持原值；窗仅抑制截断旁瓣，不改物理）。

    window ∈ {``"none"``/``"rect"``（默认不加窗语义的恒同）、``"hann"``、
    ``"gaussian"``}；``gaussian_sigma_scale`` 为 Gaussian σ 相对总时长
    的比例。
    """
    corr = np.asarray(correlation)
    t = np.asarray(times, dtype=float)
    if corr.ndim not in (1, 2) or t.ndim != 1 or corr.shape[0] != t.shape[0]:
        raise ValueError("correlation 须为 (T,) 或 (T, n) 且时间轴长度与 times 一致。")
    name = str(window).lower()
    if name in ("none", "rect", "rectangular", ""):
        return corr
    if name == "hann":
        T = float(t[-1] - t[0])
        w = np.ones_like(t)
        if T > 0:
            w = 0.5 * (1.0 + np.cos(np.pi * (t - t[0]) / T))
            w[t == t[0]] = 1.0  # t=0 保持原值（因果谱权重）
        return corr * (w if corr.ndim == 1 else w[:, None])
    if name == "gaussian":
        T = float(t[-1] - t[0])
        sigma = max(gaussian_sigma_scale * T, 1e-12)
        w = np.exp(-0.5 * ((t - t[0]) / sigma) ** 2)
        return corr * (w if corr.ndim == 1 else w[:, None])
    raise ValueError(f"未知窗类型 {window!r}；可选 none/hann/gaussian。")


def correlation_to_spectrum(correlation, times, *, window: str = "hann",
                            n_frequency: int | None = None):
    """C(t) → S(ω)（一维谱；FFT 单一出处）。

    返回 ``(frequencies, spectrum)``：

    - ``frequencies`` 为非负频率网格（``[0, π/dt)``，与 compute_dsf 口径
      一致，避免负频镜像歧义）；
    - ``spectrum`` 语义按 C(t) 类型自动选择：**实值**（时间对称关联，
      如 ⟨A(t)B+B(t)A⟩）→ 幅值谱 ``|S(ω)|``（Dirichlet 主峰在谱隙处）；
      **复值**（响应函数，纯虚/因果）→ 原样返回复谱（调用方按物理取用，
      例如 -Im G^R 即 compute_dsf 的口径）。
    """
    corr = np.asarray(correlation)
    t = np.asarray(times, dtype=float)
    if corr.ndim != 1 or t.ndim != 1 or corr.shape[0] != t.shape[0] or len(t) < 2:
        raise ValueError("correlation 与 times 必须同长（≥2）。")
    dt = t[1] - t[0]
    if dt <= 0 or not np.isclose(t[0], 0) or not np.allclose(np.diff(t), dt):
        raise ValueError("times 必须均匀、递增、从 0 开始（FFT 前提）。")
    w = apply_window(corr, t, window=window)
    n = int(n_frequency) if n_frequency is not None else len(t)
    if n < 2:
        raise ValueError("n_frequency 必须 ≥ 2。")
    frequencies = np.linspace(0, np.pi / dt, n, endpoint=False)
    # 显式 DFT（等价于 FFT；点数可不同于 len(t)，保持谱网格可控）
    phases = np.exp(1j * np.outer(frequencies, t))
    spec = dt * (phases @ w)
    if np.iscomplexobj(corr):
        return frequencies, spec
    return frequencies, np.abs(spec)


__all__ = [
    "response_from_kicks",
    "compute_dsf",
    "spectrum_for_plot",
    "error_metrics",
    "apply_window",
    "correlation_to_spectrum",
]
