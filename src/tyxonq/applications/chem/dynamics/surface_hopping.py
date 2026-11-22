"""FSSH 少开关面跳跃（Tier2：Tully 1990 fewest-switches + 退相干）。

物理（Tully, J. Chem. Phys. 93, 1061 (1990)；QD_RESEARCH 轨道 A）：

- 当前绝热面经典演化（velocity-Verlet，Hellmann-Feynman 力）；
- 电子绝热基演化（含 NAC 耦合 -i v d_ij）；
- 跳跃概率 g_{i->j} = max(0, b_ij dt / |c_i|^2)，
  b_ij = -2 Re(c_i* c_j v d_ij)；
- 速度沿 NAC 方向 rescale 保持能量守恒；不足则 frustrated hop；
- 退相干（Granucci-Persico energy-based）：非当前面振幅按
  tau ~ 1/|E_i - E_j| 衰减。

单位：原子单位。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .models.pes import TullyModel


@dataclass
class FSSHResult:
    times: np.ndarray
    x: np.ndarray
    v: np.ndarray
    active_surface: np.ndarray          # (T,) 0/1
    adiabatic_populations: np.ndarray   # (T, 2)
    energies: np.ndarray                # (T,) 总能量（核动能 + 当前面能）
    hops: List[dict] = field(default_factory=list)


def run_fssh(
    pes: TullyModel,
    x0: float,
    v0: float,
    surface0: int,
    dt: float,
    n_steps: int,
    *,
    mass: float = 1.0,
    seed: Optional[int] = None,
    decoherence: bool = True,
) -> FSSHResult:
    """FSSH 单条轨迹（随机种子可复现）。"""
    rng = np.random.default_rng(seed)
    x, v = float(x0), float(v0)
    surface = int(surface0)
    if surface not in (0, 1):
        raise ValueError("surface0 须为 0 或 1。")
    evals, evecs = pes.adiabatic(x)
    c = evecs[:, surface].astype(np.complex128)

    T = int(n_steps) + 1
    times = np.arange(T) * float(dt)
    xs, vs = np.zeros(T), np.zeros(T)
    surf = np.zeros(T, dtype=int)
    ad_pops = np.zeros((T, 2))
    energies = np.zeros(T)
    hops: List[dict] = []

    for it in range(T):
        evals, evecs = pes.adiabatic(x)
        grad = pes.adiabatic_gradient(x)
        d12 = pes.nac(x)
        force = -float(grad[surface]) / mass
        xs[it], vs[it], surf[it] = x, v, surface
        ad_pops[it] = np.abs(c) ** 2
        energies[it] = 0.5 * mass * v * v + float(evals[surface])
        if it == T - 1:
            break

        # 电子绝热基演化：i dc_i/dt = E_i c_i - i v sum_j d_ij c_j
        dc = -1j * evals * c - v * np.array([d12 * c[1], -d12 * c[0]])
        c = c + dc * dt
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm

        # 跳跃概率（fewest switches）
        b = -2.0 * np.real(np.conj(c) * c[::-1] * v * np.array([d12, -d12]))
        g = np.maximum(0.0, b * dt / np.maximum(np.abs(c) ** 2, 1e-12))
        other = 1 - surface
        if rng.random() < g[other]:
            dE = float(evals[other] - evals[surface])
            ke = 0.5 * mass * v * v
            if ke >= dE:
                v_new = np.sign(v) * np.sqrt(max(0.0, 2.0 * (ke - dE) / mass))
                hops.append({"step": it, "from": surface, "to": other,
                             "x": x, "frustrated": False})
                surface = other
                v = v_new
                c = evecs[:, surface].astype(np.complex128) * np.exp(1j * np.angle(c[surface]))
            else:
                hops.append({"step": it, "from": surface, "to": other,
                             "x": x, "frustrated": True})
                v = -v  # frustrated hop 反弹

        # 退相干（energy-based）：非当前面振幅衰减
        if decoherence:
            dE = abs(float(evals[1] - evals[0]))
            if dE > 1e-12:
                tau = 1.0 / dE
                decay = np.exp(-dt / tau)
                other = 1 - surface
                c[other] *= decay
                norm = np.linalg.norm(c)
                if norm > 0:
                    c = c / norm

        # 核 velocity-Verlet（当前面）
        v_half = v + 0.5 * force * dt
        x = x + v_half * dt
        grad_new = pes.adiabatic_gradient(x)
        force_new = -float(grad_new[surface]) / mass
        v = v_half + 0.5 * force_new * dt

    return FSSHResult(
        times=times, x=xs, v=vs, active_surface=surf,
        adiabatic_populations=ad_pops, energies=energies, hops=hops,
    )


__all__ = ["run_fssh", "FSSHResult"]
