from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

from ..._circuit_utils import rebuild_circuit

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule


class AutoMeasure:
    """线路无显式测量时，在所有比特上补 Z 测量。

    行为：
      - 若线路不含任何显式 ``("measure_z", q)`` op，则在每个比特 [0..num_qubits-1] 上补 ``measure_z``；
      - 发一个非致命警告告知用户；
      - 已存在测量时为空操作（原样返回）。

    重建走 ``rebuild_circuit`` 而非 ``circuit.extended``：后者会丢 ``_unitary_cache``，
    导致无测量 + 含自定义酉的线路在后续 ``MatchgateDecompose`` 报“缺矩阵缓存”。
    """

    name = "rewrite/auto_measure"

    def apply(self, circuit: "Circuit", device_rule: "DeviceRule" = None, **opts: Any) -> "Circuit":
        ops = getattr(circuit, "ops", []) or []
        has_meas = any((op and isinstance(op, (list, tuple)) and str(op[0]).lower() == "measure_z") for op in ops)
        # 测量也可能以 instruction 形式存在：add_measure 把 ("measure", (q,)) 存进 instructions
        # 而非 ops。只看 ops 会对这类线路误判“无测量”，进而给**所有**比特补 measure_z，与
        # instructions 里已有的测量重复（lqcloud 委托核心时尤其明显）。故一并检查 instructions。
        if not has_meas:
            for inst in (getattr(circuit, "instructions", []) or []):
                inst_name = inst[0] if isinstance(inst, (list, tuple)) and inst else None
                if str(inst_name).lower() == "measure":
                    has_meas = True
                    break
        if has_meas:
            return circuit
        nq = int(getattr(circuit, "num_qubits", 0))
        if nq <= 0:
            return circuit
        warnings.warn(
            "线路无显式测量；编译时已在所有比特上自动补 Z 测量。",
            UserWarning,
        )
        return rebuild_circuit(
            circuit, ops=list(circuit.ops) + [("measure_z", q) for q in range(nq)]
        )


