"""自旋晶格模型（磁材料 Tier 1a：Heisenberg/XXZ+DMI+场的 Pauli 编码）。

物理（赛道文档 §8.5 Phase 1 / QD_RESEARCH §1）：

- Heisenberg/XXZ：``H = Σ J_xy(X_iX_j+Y_iY_j) + Σ J_z Z_iZ_j + Σ h Z_i``；
- DMI（螺旋序）：``Σ D(X_iY_j - Y_iX_j)``（面内手征耦合，YIG/赤铁矿
  语境的标准项）；
- 输出统一 Pauli items（chem 惯例 ``(coeff, [(sym, idx), ...])``），
  直接供给 ``exact`` / ``circuits`` / ``sqw`` 引擎与 ``pulse_dynamics``。

单位：原子单位 ħ=1；耦合/场系数为无量纲模型参数（1D 链演示与
YIG/CoFeB 锚点对照时由调用方携带物理换算）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..observables import PauliItems, bitstring_to_state


@dataclass
class SpinLatticeModel:
    """通用自旋晶格（1D 链首版；耦合列表显式，扩展晶格由构造器生成）。"""

    n_qubits: int
    #: (kind, i, j, coeff) 列表；kind ∈ {"xy", "z", "dmi"}
    couplings: List[Tuple[str, int, int, float]] = field(default_factory=list)
    #: (i, coeff) 纵场列表（+h Z_i）
    fields: List[Tuple[int, float]] = field(default_factory=list)
    #: 格点坐标（默认 i；sqw 的 e^{iq r} 用）
    positions: Optional[Sequence[float]] = None

    # ---------------- 编码输出 ---------------- #
    def pauli_items(self) -> PauliItems:
        """模型 → Pauli sum（chem 惯例格式）。"""
        items: PauliItems = []
        for kind, i, j, c in self.couplings:
            if kind == "xy":
                items.append((float(c), [("X", i), ("X", j)]))
                items.append((float(c), [("Y", i), ("Y", j)]))
            elif kind == "z":
                items.append((float(c), [("Z", i), ("Z", j)]))
            elif kind == "dmi":
                items.append((float(+c), [("X", i), ("Y", j)]))
                items.append((float(-c), [("Y", i), ("X", j)]))
            else:
                raise ValueError(f"未知耦合类型 {kind!r}（xy/z/dmi）。")
        for q, h in self.fields:
            items.append((float(h), [("Z", q)]))
        return items

    # ---------------- 初态 ---------------- #
    def initial_state(self, kind: str = "afm"):
        """常用基矢初态（数值档态矢；线路档用 ``initial_bitstring``）。"""
        return bitstring_to_state(self.initial_bitstring(kind), self.n_qubits)

    def initial_bitstring(self, kind: str = "afm") -> str:
        """初态位串（little-endian：右端 = qubit 0，与 counts 约定一致）。"""
        n = self.n_qubits
        if kind == "fm":
            return "1" * n
        if kind == "afm":
            # q0=1, q1=0, q2=1, ...（Neel 交替），位串右端 = qubit 0
            bits = ["1" if q % 2 == 0 else "0" for q in range(n)]
            return "".join(reversed(bits))
        if kind == "dw":
            # 中间位点（q = n//2）单自旋翻转
            s = ["0"] * n
            s[n // 2] = "1"
            return "".join(reversed(s))
        raise ValueError(f"未知初态 {kind!r}（fm/afm/dw）。")

    # ---------------- 脉冲程序（演示级 DAQC 映射）---------------- #
    def to_pulse_program(self, duration: int = 160, *, amp_scale: float = 0.1):
        """H → ``PulseProgram`` 波形序列（演示级 DAQC 映射，第一版）。

        映射规则（文档化近似，非严格编译）：

        - 纵场 ``h Z_i`` → 比特 i 的 z 通道 Constant 波形（局域频率牵引）；
        - XY 耦合 ``J(X_iX_j+Y_iY_j)`` → 比特 i 的 xy Gaussian 驱动
          （幅度 ∝ J，交换相互作用的跨克尔近似）；
        - DMI 项并入 xy 驱动相位（第一版省略，显式声明）。

        真机物理正确性需经校准（defcal）与定标实验（M2）确认——本方法
        输出供 ``pulse_dynamics.evolve_with_pulses`` 数值档先行验证波形
        语义，真机档逐次用户确认后才提交。
        """
        from tyxonq.core.ir.pulse import PulseProgram

        prog = PulseProgram(int(self.n_qubits))
        for q, h in self.fields:
            prog.constant(q, amp=abs(h) * amp_scale, duration=int(duration),
                           channel="z")
        for kind, i, j, c in self.couplings:
            if kind == "xy":
                prog.gaussian(i, amp=abs(c) * amp_scale, duration=int(duration),
                              sigma=float(duration) / 4.0)
        return prog


def heisenberg_chain(n_qubits: int, j_xy: float = 1.0, j_z: float = 1.0,
                     h: float = 0.0, dmi: float = 0.0, *,
                     pbc: bool = False) -> SpinLatticeModel:
    """均匀 1D XXZ 链（+DMI/纵场）构造器。``pbc=True`` 闭合链。"""
    n = int(n_qubits)
    if n < 2:
        raise ValueError("链长至少 2。")
    bonds = [(q, (q + 1) % n) for q in range(n - (0 if pbc else 1))]
    couplings: List[Tuple[str, int, int, float]] = []
    for i, j in bonds:
        couplings.append(("xy", i, j, float(j_xy)))
        couplings.append(("z", i, j, float(j_z)))
        if dmi:
            couplings.append(("dmi", i, j, float(dmi)))
    fields = [(q, float(h)) for q in range(n)] if h else []
    return SpinLatticeModel(n_qubits=n, couplings=couplings, fields=fields,
                            positions=list(range(n)))


__all__ = ["SpinLatticeModel", "heisenberg_chain"]
