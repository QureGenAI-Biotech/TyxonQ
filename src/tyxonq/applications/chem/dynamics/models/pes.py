"""势能面供给（Tier2 非绝热：Tully 三模型解析 PES + NAC）。

Tully 三模型（J. Chem. Phys. 93, 1061 (1990)，FSSH 标准验证集）：

- Model A（single avoided crossing）：双曲线避免交叉；
- Model B（dual avoided crossing）：高斯耦合的双避免交叉；
- Model C（extended coupling region）：扩展耦合区（反射基准）。

每个模型给出：透热势能矩阵 V(x)、绝热本征值/梯度、非绝热耦合
NAC d_12 = <phi1|d/dx|phi2>（2 态解析公式）。

单位：原子单位（Hartree/Bohr）；核坐标 x 一维。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class TullyModel:
    """Tully 标准模型（kind ∈ {"A","B","C"}）。"""

    kind: str = "A"

    def diabatic(self, x: float) -> np.ndarray:
        """透热势能矩阵 V(x)（2x2）。"""
        x = float(x)
        if self.kind == "A":
            A, B, C, D = 0.01, 1.6, 0.005, 1.0
            v11 = A * (1.0 - np.exp(-B * x)) if x > 0 else -A * (1.0 - np.exp(B * x))
            return np.array([[v11, C * np.exp(-D * x * x)],
                             [C * np.exp(-D * x * x), -v11]])
        if self.kind == "B":
            A, C, D = 0.1, 0.015, 0.06
            v12 = C * np.exp(-D * x * x)
            return np.array([[0.0, v12], [v12, -A]])
        if self.kind == "C":
            A, B, C = 6e-4, 0.1, 0.9
            v12 = B * np.exp(C * x) if x < 0 else B * (2.0 - np.exp(-C * x))
            return np.array([[A, v12], [v12, -A]])
        raise ValueError(f"未知 Tully 模型 {self.kind!r}（A/B/C）。")

    def diabatic_gradient(self, x: float) -> np.ndarray:
        """dV/dx（解析导数，2x2）。"""
        x = float(x)
        if self.kind == "A":
            A, B, C, D = 0.01, 1.6, 0.005, 1.0
            dv11 = A * B * np.exp(-B * x) if x > 0 else A * B * np.exp(B * x)
            dv12 = -2.0 * D * x * C * np.exp(-D * x * x)
            return np.array([[dv11, dv12], [dv12, -dv11]])
        if self.kind == "B":
            A, C, D = 0.1, 0.015, 0.06
            dv12 = -2.0 * D * x * C * np.exp(-D * x * x)
            return np.array([[0.0, dv12], [dv12, 0.0]])
        if self.kind == "C":
            A, B, C = 6e-4, 0.1, 0.9
            dv12 = B * C * np.exp(C * x) if x < 0 else B * C * np.exp(-C * x)
            return np.array([[0.0, dv12], [dv12, 0.0]])
        raise ValueError(self.kind)

    def adiabatic(self, x: float) -> Tuple[np.ndarray, np.ndarray]:
        """绝热本征值 (E0, E1) 与本征矢矩阵（列 = 绝热态）。"""
        V = self.diabatic(x)
        evals, evecs = np.linalg.eigh(V)
        return evals, evecs

    def adiabatic_gradient(self, x: float) -> np.ndarray:
        """绝热面梯度 dE_i/dx = <phi_i|dV/dx|phi_i>（Hellmann-Feynman）。"""
        evals, evecs = self.adiabatic(x)
        dV = self.diabatic_gradient(x)
        return np.array([evecs[:, i] @ dV @ evecs[:, i] for i in range(2)])

    def nac(self, x: float) -> float:
        """NAC d_12 = <phi1|d/dx|phi2>（2 态解析公式）。

        d_12 = [(V11'-V22') V12 - (V11-V22) V12'] / [(V11-V22)^2 + 4 V12^2]
        """
        V = self.diabatic(x)
        dV = self.diabatic_gradient(x)
        v11, v22, v12 = V[0, 0], V[1, 1], V[0, 1]
        dv11, dv22, dv12 = dV[0, 0], dV[1, 1], dV[0, 1]
        denom = (v11 - v22) ** 2 + 4.0 * v12 ** 2
        if denom < 1e-24:
            return 0.0
        return float(((dv11 - dv22) * v12 - (v11 - v22) * dv12) / denom)


def tully_model(kind: str = "A") -> TullyModel:
    """构造 Tully 标准模型。"""
    if kind not in ("A", "B", "C"):
        raise ValueError(f"未知 Tully 模型 {kind!r}。")
    return TullyModel(kind=kind)


__all__ = ["TullyModel", "tully_model"]
