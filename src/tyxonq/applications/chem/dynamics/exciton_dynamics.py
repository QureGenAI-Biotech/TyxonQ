"""激子动力学引擎（OLED Tier 1b：布居演化 → MSD/L_D/k_hop/μ 标量管线）。

物理口径（赛道文档 §8.6 Phase 2）：

- 布居 ``P_n(t) = |⟨n|ψ(t)⟩|²``（单激子子空间）；
- 均方位移 ``MSD(t) = Σ_n P_n(t) (r_n - r_0)²``；
- 跃迁速率 ``k_hop``：二聚体 Rabi 振荡 ``k = 2√(J²+(δ/2)²)``；
- 扩散长度 ``L_D = √MSD(τ)``（τ = 激子寿命，模型侧标量；
  **实验对照按用户 2026-11-12 拍板延后**，当前验收终点为
  解析解/ED 精确对照）；
- 迁移率 ``μ = e D/(k_B T)``（Einstein 关系，模型侧标量）。

执行档（provider 直切，无运行时抽象）：

- 数值精确档：``exact.evolve_exact`` + ``observables.site_populations``；
- 线路档：Trotter 演化 + Z 基测量（全部 Z 项同基，**一条线路**测出
  所有位点布居——``expval_pauli_sum`` 聚合，bit-order 单一出处）。

单位：原子单位 ħ=1；τ 以模型时间单位给出；T 以 K 给出。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .exact import evolve_exact
from .observables import site_populations

#: 结果 schema
EXCITON_RESULT_SCHEMA = "tyxonq.qd.exciton.v1"

#: Einstein 关系换算（原子单位 → cm²/(V·s)）：μ = e D/(k_B T)
_KB_HARTREE_PER_K = 3.166811563e-6  # Boltzmann 常数（Hartree/K）


def population_trajectory(
    model,
    times: Sequence[float],
    *,
    site0: Optional[int] = None,
    provider: Optional[str] = None,
    shots: int = 4096,
    init_bitstring: Optional[str] = None,
    dt_trotter: Optional[float] = None,
    **device_kwargs,
) -> Dict[str, Any]:
    """单激子布居轨迹 ``P_n(t)``（shape (T, n_sites)）。

    provider=None 数值精确档；``"simulator"`` 线路档（Trotter + Z 基
    测量；全部 Z 项同基，一次测量聚合所有位点布居）。
    """
    n = int(model.n_sites)
    ham_items = model.pauli_items()
    t_arr = np.asarray(times, dtype=float)
    if provider is None:
        psi0 = model.initial_state(site0)
        states = evolve_exact(ham_items, psi0, t_arr, n)
        pops = np.stack([site_populations(states[it], n) for it in range(len(t_arr))])
        return {"schema": EXCITON_RESULT_SCHEMA, "mode": "numeric",
                "populations": pops, "times": t_arr, "site0": int(site0) if site0 is not None else n // 2}
    if str(provider) != "simulator":
        raise ValueError(
            f"provider 须为 None（数值）或 'simulator'（线路档），收到 {provider!r}。"
        )
    from .circuits import build_trotter_circuit, measure_pauli_expectation

    if init_bitstring is None:
        init_bitstring = model.initial_bitstring(site0)
    if len(t_arr) < 2:
        raise ValueError("线路档 times 至少两个点（推断 dt）。")
    # Trotter 步长可独立于 times 网格指定（默认从 times 推断，便于控制
    # 一阶 Trotter 误差 ~ t·dt·J²）
    dt = float(dt_trotter) if dt_trotter is not None else float(t_arr[1] - t_arr[0])
    if dt <= 0 or not np.allclose(np.diff(t_arr), float(t_arr[1] - t_arr[0])):
        raise ValueError("线路档 times 必须均匀。")
    # 全部 Z 项同基：一次测量得所有位点 ⟨Z_q⟩，布居 = (1-⟨Z_q⟩)/2
    z_items = [(1.0, [("Z", q)]) for q in range(n)]
    pops = np.zeros((len(t_arr), n), dtype=np.float64)
    for it, t in enumerate(t_arr):
        k = int(round(float(t) / dt))
        circ = build_trotter_circuit(
            ham_items, n, dt, k, init_bitstring=init_bitstring
        )
        for q in range(n):
            expval = measure_pauli_expectation(
                circ, [z_items[q]], n, provider="simulator",
                shots=shots, **device_kwargs
            )
            pops[it, q] = (1.0 - float(expval)) / 2.0
    return {"schema": EXCITON_RESULT_SCHEMA, "mode": "circuits",
            "populations": pops, "times": t_arr,
            "site0": int(site0) if site0 is not None else n // 2}


def mean_squared_displacement(populations, site0: Optional[int] = None) -> np.ndarray:
    """``MSD(t) = Σ_n P_n(t)(n - n0)²``（等距格点 r_n = n）。"""
    pops = np.asarray(populations, dtype=float)
    if pops.ndim != 2:
        raise ValueError("populations 应为 (T, n)。")
    T, n = pops.shape
    origin = int(site0) if site0 is not None else n // 2
    r = np.arange(n, dtype=float)
    return np.einsum("tn,n->t", pops, (r - origin) ** 2)


def rabi_frequency_two_site(hopping: float, detuning: float = 0.0) -> float:
    """二聚体 Rabi 解析参照：``k = 2√(J²+(δ/2)²)``。"""
    return 2.0 * float(np.sqrt(hopping ** 2 + (detuning / 2.0) ** 2))


def ballistic_velocity(populations, times, site0: Optional[int] = None) -> float:
    """短时弹道速度 ``v = √MSD(t)/t``（t>0 中位点估计）。"""
    msd = mean_squared_displacement(populations, site0)
    t = np.asarray(times, dtype=float)
    mask = t > 0
    if not np.any(mask):
        raise ValueError("times 需含 t>0 点。")
    v = np.sqrt(msd[mask]) / t[mask]
    return float(np.median(v))


def diffusion_length(populations, times, tau: float, site0: Optional[int] = None) -> Dict[str, Any]:
    """``L_D = √MSD(τ)``（τ = 激子寿命；模型侧标量）。

    返回 ``{"L_D", "tau", "msd", "times"}``；若 τ 超出时间网格则线性
    外推并标注 ``extrapolated=True``。
    """
    t = np.asarray(times, dtype=float)
    msd = mean_squared_displacement(populations, site0)
    tau_f = float(tau)
    extrapolated = tau_f > float(t[-1])
    if extrapolated and len(t) >= 2:
        # 线性外推（短时间行为：ballistic MSD∝t² 时用二次外推更合理，
        # 但保守线性 + 显式标注）
        slope = (msd[-1] - msd[-2]) / (t[-1] - t[-2])
        msd_tau = msd[-1] + slope * (tau_f - t[-1])
    else:
        msd_tau = float(np.interp(tau_f, t, msd))
    return {
        "L_D": float(np.sqrt(max(msd_tau, 0.0))),
        "tau": tau_f,
        "msd": msd,
        "times": t,
        "extrapolated": extrapolated,
    }


def mobility_from_diffusivity(diffusivity: float, temperature_K: float) -> float:
    """Einstein 关系 ``μ = D/(k_B T)``（原子单位：μ 单位 1/(Hartree·time)）。

    实验对照延后：本函数只做单位换算，不与 nbsdc/文献值比较。
    """
    if temperature_K <= 0:
        raise ValueError("温度必须为正。")
    return float(diffusivity) / (_KB_HARTREE_PER_K * float(temperature_K))


def diffusivity_from_msd(populations, times, site0: Optional[int] = None) -> float:
    """扩散系数 ``D = MSD(t)/(2t)``（长时间扩散区；末段线性拟合）。

    封闭链短时 ballistic 不适用——本函数仅当 MSD(t) 呈线性增长时
    有意义（需声子/耗散）；纯激子链返回的斜率仅作诊断参考。
    """
    t = np.asarray(times, dtype=float)
    msd = mean_squared_displacement(populations, site0)
    # 末段 1/3 线性拟合 MSD ≈ 2 D t
    n = len(t)
    seg = slice(2 * n // 3, n)
    if n < 3:
        raise ValueError("times 至少三个点。")
    slope = float(np.polyfit(t[seg], msd[seg], 1)[0])
    return slope / 2.0


__all__ = [
    "population_trajectory",
    "mean_squared_displacement",
    "rabi_frequency_two_site",
    "ballistic_velocity",
    "diffusion_length",
    "diffusivity_from_msd",
    "mobility_from_diffusivity",
    "EXCITON_RESULT_SCHEMA",
]
