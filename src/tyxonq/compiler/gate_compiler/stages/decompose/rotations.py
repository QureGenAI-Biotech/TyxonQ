from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

from ..._circuit_utils import rebuild_circuit

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule


class RotationsDecompose:
    """用标准恒等式把参数化旋转门分解到 {H,RZ,CX}。

    支持的模式：
      - RX(θ)  => H · RZ(θ) · H
      - RY(θ)  => S† · H · RZ(θ) · H · S  (S=RZ(π/2))
      - RZZ(θ) => CX · RZ(θ on target) · CX
      - RXX(θ) => H⊗H · CX · RZ(θ on target) · CX · H⊗H
      - RYY(θ) => (S†H)⊗(S†H) · CX · RZ(θ on target) · CX · (HS)⊗(HS)

    重建走 ``rebuild_circuit``：保留 ``_unitary_cache``，避免残留的 1-qubit 自定义酉
    在本步之后的 lowering 丢失矩阵引用。
    """

    name = "decompose/rotations"

    def apply(self, circuit: "Circuit", device_rule: "DeviceRule" = None, **opts: Any) -> "Circuit":
        new_ops: list[Any] = []
        pi_over_2 = math.pi / 2.0

        for op in circuit.ops:
            if not isinstance(op, (list, tuple)) or not op:
                new_ops.append(op)
                continue
            name = str(op[0]).lower()

            if name == "rx":
                q = int(op[1]); theta = float(op[2])
                new_ops += [("h", q), ("rz", q, theta), ("h", q)]

            elif name == "ry":
                q = int(op[1]); theta = float(op[2])
                new_ops += [("rz", q, -pi_over_2), ("h", q), ("rz", q, theta), ("h", q), ("rz", q, pi_over_2)]

            elif name == "rzz":
                q0 = int(op[1]); q1 = int(op[2]); theta = float(op[3]) if len(op) > 3 else float(op[2])
                # Assume (q0,q1,theta) or (q0,q1) with missing theta (fallback)
                if len(op) > 3:
                    new_ops += [("cx", q0, q1), ("rz", q1, theta), ("cx", q0, q1)]
                else:
                    new_ops += [("cx", q0, q1), ("rz", q1, 0.0), ("cx", q0, q1)]
                

            elif name == "rxx":
                q0 = int(op[1]); q1 = int(op[2]); theta = float(op[3]) if len(op) > 3 else float(op[2])
                new_ops += [("h", q0), ("h", q1), ("cx", q0, q1), ("rz", q1, theta), ("cx", q0, q1), ("h", q0), ("h", q1)]

            elif name == "ryy":
                q0 = int(op[1]); q1 = int(op[2]); theta = float(op[3]) if len(op) > 3 else float(op[2])
                new_ops += [
                    ("rz", q0, -pi_over_2), ("rz", q1, -pi_over_2),
                    ("h", q0), ("h", q1),
                    ("cx", q0, q1), ("rz", q1, theta), ("cx", q0, q1),
                    ("h", q0), ("h", q1),
                    ("rz", q0, pi_over_2), ("rz", q1, pi_over_2),
                ]

            else:
                new_ops.append(op)

        return rebuild_circuit(circuit, ops=new_ops)


