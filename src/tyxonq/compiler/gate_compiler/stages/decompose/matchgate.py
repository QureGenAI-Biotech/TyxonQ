"""2-qubit 数守恒 matchgate（fSim 家族）到原生门的分解 —— native compiler 一等特色。

物理依据
--------
fSim 门可分解为 iSWAP 型相互作用 + CPhase：

    fSim(θ, φ) = exp(-i·θ/2·(XX+YY)) · diag(1, 1, 1, e^{-iφ})

前因子只作用于单激发子空间 {|01>,|10>}（iSWAP 型），后因子只作用于 |11>（CPhase），
两者作用在正交态空间上、彼此对易。

LUCJ/SQD 的 ``two_mode_fock_matrix(v)`` 正是 fSim 家族的 2-qubit matchgate：4x4 结构为
|00>→1、单激发子空间 {|01>,|10>} 上作用 W=XvX（任意 U(2)）、|11>→det(v)。它比裸 fSim
稍一般（单激发块是任意 U(2)、|11> 相位被绑定成 det(v)），但多出自由度可用两侧单比特
Rz 吸收，故能精确降到原生 {rz, rx, h, cx}：

    two_mode_fock(v) = (Rz⊗Rz) · fSim(θ, φ) · (Rz⊗Rz)   （忽略全局相位）

单一真相源
----------
本模块是这套分解的**唯一实现**，供各方言复用，避免重复：
  - ``decompose_two_qubit_matchgate`` 纯函数返回 TyxonQ IR op 词汇（引擎无关）；
  - ``MatchgateDecompose`` 把它包装成核心编译器的一个分解步骤（由类型化参数
    ``decompose=DecomposeSpec(matchgate=True)`` 装配，默认开启）；
  - lqcloud 等方言 **经核心委托** 复用本分解（platforms/lqcloud/compiler.py 先跑 gate_compiler
    核心完成 matchgate 分解，再把编译后的 IR 直译为 SDK 对象，lowering 不再重复实现分解）。

正确性由离线数值校验兜底：对随机 U(2)、Givens 旋转及退化情形逐例比对分解前后
unitary（允许全局相位差），保真度 1.0（见 tests_core_module/test_decompose_matchgate.py）。
"""

from __future__ import annotations

import math
from typing import Any, List, Tuple, TYPE_CHECKING

import numpy as np

from ..._circuit_utils import rebuild_circuit

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule


def is_number_conserving_matchgate(matrix: Any, tol: float = 1e-8) -> bool:
    """判断 4x4 矩阵是否为数守恒 matchgate（two_mode_fock/fSim 家族）。

    结构要求：|00>、|11> 与单激发子空间 {|01>,|10>} 之间无耦合（粒子数守恒），
    即所有跨粒子数子空间的矩阵元为零。这是 fSim=iSWAP+CPhase 分解成立的前提。
    """
    mat = np.asarray(
        matrix.tolist() if hasattr(matrix, "tolist") else matrix, dtype=complex
    )
    if mat.shape != (4, 4):
        return False
    leakage = [
        mat[0, 1], mat[0, 2], mat[0, 3],
        mat[1, 0], mat[2, 0], mat[3, 0],
        mat[1, 3], mat[2, 3], mat[3, 1], mat[3, 2],
    ]
    return max(abs(x) for x in leakage) <= tol


def decompose_two_qubit_matchgate(
    matrix: Any, q0: int, q1: int
) -> List[Tuple[Any, ...]]:
    """把 2-qubit 数守恒 matchgate 分解成原生门序列（TyxonQ IR op 词汇）。

    分解原理（忽略全局相位，全部用原生 rz/rx/h/cx 实现）：
      1. |11> 相位 det(v)=e^{2iμ} 由 e^{iμN}（N=总粒子数）承载 = 两比特各一个 Rz(μ)；
      2. 去掉该相位后 S=e^{-iμ}W∈SU(2)，在单激发子空间按 ZXZ 欧拉角分解 S=Rz(a)Rx(b)Rz(c)；
      3. 子空间 Rz = 两比特 Rz 之差（单激发基 (|01>,|10>) 下 σ_z^sub=(Z0-Z1)/2）；子空间
         Rx(b)=exp(-i(b/2)X_sub) = partial-iSWAP = RXX(b/2)·RYY(b/2)（XX 与 YY 对易），
         再用 RXX=(H⊗H)RZZ(H⊗H)、RYY=(Rx⊗Rx)RZZ(Rx†⊗Rx†)、RZZ=cx·rz·cx 落到原生门。

    返回按“时间顺序”（先作用者在前）排列的 op 列表，可回放给任意方言 lowering 或
    TyxonQ Circuit。零角度块会被剪枝（b≈0 时跳过整段 partial-iSWAP），以省真机门数。

    Raises:
        ValueError: 矩阵不是 4x4。
        NotImplementedError: 矩阵不是数守恒 matchgate（存在跨粒子数子空间耦合）。
    """
    mat = np.asarray(
        matrix.tolist() if hasattr(matrix, "tolist") else matrix, dtype=complex
    )
    if mat.shape != (4, 4):
        raise ValueError("2-qubit unitary 必须是 4x4 矩阵。")
    if not is_number_conserving_matchgate(mat):
        raise NotImplementedError(
            "2-qubit unitary 分解仅支持数守恒 matchgate（two_mode_fock/fSim 家族）；"
            "检测到跨粒子数子空间耦合，无法按 fSim=iSWAP+CPhase 分解为原生门。"
        )

    # |00> 相位归一到 1（全局相位，物理无关），使分解对一般对角 A 也成立。
    a00 = complex(mat[0, 0])
    if abs(a00) > 1e-8:
        mat = mat * (abs(a00) / a00)

    w_block = mat[1:3, 1:3]
    delta = complex(mat[3, 3])
    mu = float(np.angle(delta)) / 2.0
    # S = e^{-iμ}·W ∈ SU(2)（det S = e^{-2iμ}·det W = e^{-2iμ}·e^{2iμ} = 1）。
    su2 = np.exp(-1j * mu) * w_block
    s00, s10 = complex(su2[0, 0]), complex(su2[1, 0])

    # ZXZ 欧拉角：S = Rz(a)·Rx(b)·Rz(c)，b∈[0,π]（退化情形未定角乘以~0 项，重建仍精确）。
    b = 2.0 * math.atan2(abs(s10), abs(s00))
    a_plus_c = -2.0 * float(np.angle(s00)) if abs(s00) > 1e-12 else 0.0
    a_minus_c = 2.0 * float(np.angle(s10)) + math.pi if abs(s10) > 1e-12 else 0.0
    a = (a_plus_c + a_minus_c) / 2.0
    c = (a_plus_c - a_minus_c) / 2.0
    half_b = b / 2.0

    ops: List[Tuple[Any, ...]] = []
    eps = 1e-12
    # (1) c 端 ZXZ（最先作用）：exp(-i(c/2)σ_z^sub) = Rz_q0(c/2)·Rz_q1(-c/2)
    #     （单激发子空间基 (|01>,|10>) 下 σ_z^sub = |01><01|-|10><10| = (Z0-Z1)/2）
    if abs(c) > eps:
        ops.append(("rz", q0, c / 2.0))
        ops.append(("rz", q1, -c / 2.0))
    # (2)(3) 子空间 Rx(b) = partial-iSWAP = RYY(b/2)·RXX(b/2)；b≈0 时整块为单位元，
    #        跳过以省真机门数（每个冗余 cx 都会引入噪声）。
    if abs(half_b) > eps:
        # RYY(b/2) = (Rx(-π/2)⊗Rx(-π/2))·RZZ(b/2)·(Rx(π/2)⊗Rx(π/2))
        ops.append(("rx", q0, -math.pi / 2))
        ops.append(("rx", q1, -math.pi / 2))
        ops.append(("cx", q0, q1))
        ops.append(("rz", q1, half_b))
        ops.append(("cx", q0, q1))
        ops.append(("rx", q0, math.pi / 2))
        ops.append(("rx", q1, math.pi / 2))
        # RXX(b/2) = (H⊗H)·RZZ(b/2)·(H⊗H)
        ops.append(("h", q0))
        ops.append(("h", q1))
        ops.append(("cx", q0, q1))
        ops.append(("rz", q1, half_b))
        ops.append(("cx", q0, q1))
        ops.append(("h", q0))
        ops.append(("h", q1))
    # (4) a 端 ZXZ：exp(-i(a/2)σ_z^sub) = Rz_q0(a/2)·Rz_q1(-a/2)
    if abs(a) > eps:
        ops.append(("rz", q0, a / 2.0))
        ops.append(("rz", q1, -a / 2.0))
    # (5) e^{iμN} = Rz_q0(μ)·Rz_q1(μ)（承载 |11> 相位 det(v)，全局相位 e^{iμ} 忽略）
    if abs(mu) > eps:
        ops.append(("rz", q0, mu))
        ops.append(("rz", q1, mu))
    return ops


class MatchgateDecompose:
    """核心编译器分解步骤：2-qubit 数守恒 matchgate → 原生 {rz,rx,h,cx}。

    把 fSim=iSWAP+CPhase 的分解提升为引擎无关的一等编译特色，与 ``decompose/rotations``
    并列。目标 op 是 ``("unitary", q0, q1, mat_key)``，矩阵取自 ``circuit._unitary_cache``。

    行为：
      - 数守恒 matchgate 的 2-qubit unitary → 展开为原生门序列（精确，见模块 docstring）；
      - 非 matchgate 的 2-qubit unitary：默认**原样保留**（可组合，交由后续 stage 或引擎
        lowering 处理）；``matchgate_strict=True`` 时改为 loudly raise（适用于已知全部
        2-qubit unitary 都必须是 matchgate 的场景，如 LUCJ）；
      - 1-qubit unitary 及其它 op 原样保留；``_unitary_cache`` 复制到新线路，避免残留的
        1-qubit unitary 丢失矩阵。
    """

    name = "decompose/matchgate"

    def apply(
        self, circuit: "Circuit", device_rule: "DeviceRule" = None, **opts: Any
    ) -> "Circuit":
        strict = bool(opts.get("matchgate_strict", False))
        cache = dict(getattr(circuit, "_unitary_cache", {}) or {})
        new_ops: List[Any] = []

        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                new_ops.append(op)
                continue
            name = str(op[0]).lower()
            # 2-qubit unitary：("unitary", q0, q1, mat_key)
            if name == "unitary" and len(op) == 4:
                q0, q1, mat_key = int(op[1]), int(op[2]), op[3]
                matrix = cache.get(mat_key)
                if matrix is None:
                    raise ValueError(f"matchgate 分解缺少矩阵缓存: {mat_key}")
                if is_number_conserving_matchgate(matrix):
                    new_ops.extend(decompose_two_qubit_matchgate(matrix, q0, q1))
                    continue
                if strict:
                    # 复用纯函数的 raise（错误信息更完整、口径统一）。
                    decompose_two_qubit_matchgate(matrix, q0, q1)
                new_ops.append(tuple(op))
                continue
            new_ops.append(op)

        return rebuild_circuit(circuit, ops=new_ops)


__all__ = [
    "MatchgateDecompose",
    "decompose_two_qubit_matchgate",
    "is_number_conserving_matchgate",
]
