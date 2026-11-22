"""Frenkel 激子模型（OLED Tier 1b：Frenkel-Holstein + 静态无序）。

物理（赛道文档 §2.A / §8.6 Phase 2）：

- Frenkel 激子：单激子子空间 ``H = Σ_n ε_n |n⟩⟨n| + Σ_{nm} J_nm |n⟩⟨m|``；
- 每 site 一个 qubit（硬芯玻色子编码，单激子子空间内演化）：
  ``|n⟩⟨n| → (I - Z_n)/2``、``|n⟩⟨m| + h.c. → (X_nX_m + Y_nY_m)/2``；
- 静态无序：``ε_n = ε + δ_n``，``δ_n ~ N(0, σ²)``（Anderson 无序，
  种子可复现）；
- 声子项（Holstein）为第二级扩展：第一版输出纯激子部分（Pauli
  编码），声子自由度由 ``evolution.py`` 的 renormalizer MPS 路径承载。

单位：原子单位 ħ=1；ε/J 为无量纲模型参数（对照实验时经
``benchmarks`` 层统一换算）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..observables import PauliItems, bitstring_to_state


@dataclass
class ExcitonModel:
    """Frenkel 激子链（单激子子空间；位点能 + 跃迁 + 静态无序）。"""

    n_sites: int
    #: 位点能 ε_n（长度 n；含无序涨落）
    site_energies: np.ndarray
    #: 跃迁 J_{n,n+1}（长度 n-1；开边界链）
    hoppings: np.ndarray
    #: 无序标准差（0 = 纯净链）
    disorder_sigma: float = 0.0
    #: 随机种子（无序可复现）
    seed: Optional[int] = None

    def pauli_items(self) -> PauliItems:
        """Frenkel 激子 H → Pauli items（单激子子空间内的硬芯玻色子编码）。"""
        items: PauliItems = []
        for n in range(self.n_sites):
            eps = float(self.site_energies[n])
            if abs(eps) > 0:
                # ε_n |n⟩⟨n| → ε_n (I - Z_n)/2；identity 项以空 ops 表示
                items.append((eps / 2.0, []))
                items.append((-eps / 2.0, [("Z", n)]))
        for n in range(self.n_sites - 1):
            J = float(self.hoppings[n])
            if abs(J) > 0:
                items.append((J / 2.0, [("X", n), ("X", n + 1)]))
                items.append((J / 2.0, [("Y", n), ("Y", n + 1)]))
        return items

    def initial_state(self, site: Optional[int] = None):
        """单激子初态（默认居中位点激发）。"""
        return bitstring_to_state(self.initial_bitstring(site), self.n_sites)

    def initial_bitstring(self, site: Optional[int] = None) -> str:
        """单位点位串（little-endian：右端 = qubit 0 = site 0）。"""
        idx = int(site) if site is not None else self.n_sites // 2
        if not 0 <= idx < self.n_sites:
            raise ValueError(f"激发位点 {idx} 越界（n={self.n_sites}）。")
        s = ["0"] * self.n_sites
        s[idx] = "1"
        # 位串右端 = qubit 0：反转使 site idx 落在正确比特位
        return "".join(reversed(s))


def frenkel_chain(
    n_sites: int,
    hopping: float = 1.0,
    epsilon: float = 0.0,
    disorder_sigma: float = 0.0,
    *,
    seed: Optional[int] = None,
) -> ExcitonModel:
    """均匀 Frenkel 链构造器（可选 Anderson 无序，种子可复现）。"""
    n = int(n_sites)
    if n < 2:
        raise ValueError("链长至少 2。")
    rng = np.random.default_rng(seed)
    energies = np.full(n, float(epsilon), dtype=np.float64)
    if disorder_sigma > 0:
        energies = energies + rng.normal(0.0, float(disorder_sigma), size=n)
    hoppings = np.full(n - 1, float(hopping), dtype=np.float64)
    return ExcitonModel(n_sites=n, site_energies=energies, hoppings=hoppings,
                        disorder_sigma=float(disorder_sigma), seed=seed)


def two_site_dimer(hopping: float = 1.0, detuning: float = 0.0) -> ExcitonModel:
    """两位点二聚体（Rabi 解析参照：布居振荡频率 2√(J²+(δ/2)²)）。"""
    return ExcitonModel(
        n_sites=2,
        site_energies=np.array([detuning / 2.0, -detuning / 2.0]),
        hoppings=np.array([float(hopping)]),
    )


__all__ = ["ExcitonModel", "frenkel_chain", "two_site_dimer"]
