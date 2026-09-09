from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from ..._circuit_utils import rebuild_circuit

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule


class GatesTransform:
    """按偏好的 basis_gates 重写门（核心规范化步骤之一）。

    - 优先用传入的 ``basis_gates``；未提供则用默认 ``["h","rx","rz","cx","cz"]``。
    - 最小重写规则：
        x -> rx(pi)（当 "rx" 在门集合内）
        y -> ry(pi)（当 "ry" 在门集合内）
        原样保留：{cx, cz, h, rx, ry, rz}
        rxx/rzz/cy：门集合显式请求时原样保留
        其余门（含 unitary）：原样保留，交由后续分解 / lowering 处理

    本步骤跑在 ``MatchgateDecompose`` **之前**，故重建线路时必须保留 ``_unitary_cache``
    （经 ``rebuild_circuit``），否则后续 matchgate 分解会因缺矩阵缓存而抛错。
    """

    name = "rewrite/gates_transform"

    def apply(self, circuit: "Circuit", basis_gates =["h", "rx", "rz", "cx", "cz"] ,device_rule: "DeviceRule" = None, **opts: Any) -> "Circuit":
        new_ops = []
        for op in getattr(circuit, "ops", []) or []:
            if not (isinstance(op, (list, tuple)) and op):
                new_ops.append(op)
                continue
            name = str(op[0]).lower()
            if name == "x" and "rx" in basis_gates:
                q = int(op[1]); new_ops.append(("rx", q, 3.141592653589793))
            elif name == "y" and "ry" in basis_gates:
                q = int(op[1]); new_ops.append(("ry", q, 3.141592653589793))
            elif name in ("cx", "cz", "h", "rx", "ry", "rz"):
                new_ops.append(tuple(op))
            elif name == "rxx" and "rxx" in basis_gates:
                new_ops.append(tuple(op))
            elif name == "rzz" and "rzz" in basis_gates:
                new_ops.append(tuple(op))
            elif name == "cy" and "cy" in basis_gates:
                new_ops.append(tuple(op))
            else:
                new_ops.append(tuple(op))
        return rebuild_circuit(circuit, ops=new_ops)


