"""Ehrenfest 平均场非绝热动力学（Tier2：经典核 + 量子电子混合）。

物理（赛道文档 Tier2 / QD_RESEARCH 轨道 A 降级保留）：

- 核：经典 velocity-Verlet，平均场力 F = -Tr(rho dV/dx)；
- 电子：透热基含时薛定谔 i dc/dt = V c（2x2 矩阵指数传播）；
- 绝热布居：|c_adiabatic|^2（投影到当前绝热本征矢）。

单位：原子单位（Hartree/Bohr/原子时间）；核质量默认 1。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import expm

from .models.pes import TullyModel


@dataclass
class EhrenfestResult:
    times: np.ndarray
    x: np.ndarray
    v: np.ndarray
    forces: np.ndarray
    adiabatic_populations: np.ndarray  # (T, 2)
    diabatic_populations: np.ndarray   # (T, 2)
    energies: np.ndarray               # (T,) 总能量（核动能 + 电子平均能）


def run_ehrenfest(
    pes: TullyModel,
    x0: float,
    v0: float,
    dt: float,
    n_steps: int,
    *,
    mass: float = 1.0,
    init_diabatic: Optional[np.ndarray] = None,
) -> EhrenfestResult:
    """Ehrenfest 平均场轨迹（经典核 + 量子电子）。"""
    x, v = float(x0), float(v0)
    if init_diabatic is None:
        c = np.array([1.0, 0.0], dtype=np.complex128)
    else:
        c = np.asarray(init_diabatic, dtype=np.complex128)
        c = c / np.linalg.norm(c)
    if abs(np.linalg.norm(c) - 1.0) > 1e-12:
        raise ValueError("电子系数必须归一化。")

    T = int(n_steps) + 1
    times = np.arange(T) * float(dt)
    xs = np.zeros(T)
    vs = np.zeros(T)
    forces = np.zeros(T)
    ad_pops = np.zeros((T, 2))
    di_pops = np.zeros((T, 2))
    energies = np.zeros(T)

    for it in range(T):
        V = pes.diabatic(x)
        dV = pes.diabatic_gradient(x)
        evals, evecs = np.linalg.eigh(V)
        force = -float(np.real(np.vdot(c, dV @ c))) / mass
        xs[it], vs[it], forces[it] = x, v, force
        di_pops[it] = np.abs(c) ** 2
        c_ad = evecs.conj().T @ c
        ad_pops[it] = np.abs(c_ad) ** 2
        e_elec = float(np.real(np.vdot(c, V @ c)))
        energies[it] = 0.5 * mass * v * v + e_elec
        if it == T - 1:
            break
        # velocity-Verlet（半步速度 + 全步位置 + 新力半步速度）
        v_half = v + 0.5 * force * dt
        x_new = x + v_half * dt
        V_new = pes.diabatic(x_new)
        dV_new = pes.diabatic_gradient(x_new)
        force_new = -float(np.real(np.vdot(c, dV_new @ c))) / mass
        v_new = v_half + 0.5 * force_new * dt
        # 电子含时演化（透热基，新点 V）
        c = expm(-1j * V_new * dt) @ c
        x, v = x_new, v_new

    return EhrenfestResult(
        times=times, x=xs, v=vs, forces=forces,
        adiabatic_populations=ad_pops, diabatic_populations=di_pops,
        energies=energies,
    )


__all__ = ["run_ehrenfest", "EhrenfestResult"]
