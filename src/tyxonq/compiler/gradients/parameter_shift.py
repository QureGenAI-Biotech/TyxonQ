from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from ...core.ir import Circuit


def generate_shifted_circuits(circuit: Circuit, match_op_name: str) -> Tuple[Circuit, Circuit, Dict[str, Any]]:
    """Generate +/− shifted parameter circuits for a given single-parameter op.

    This minimal implementation scans the first matching op by name and applies
    a ±pi/2 shift to its parameter. It returns two cloned circuits and metadata
    containing the parameter-shift coefficient (0.5 for typical RZ/RX/RY).
    """

    plus = Circuit(num_qubits=circuit.num_qubits, ops=list(circuit.ops), metadata=dict(circuit.metadata))
    minus = Circuit(num_qubits=circuit.num_qubits, ops=list(circuit.ops), metadata=dict(circuit.metadata))

    def _shift_ops(ops, delta):
        new_ops = []
        shifted = False
        for op in ops:
            if not shifted and op[0] == match_op_name and len(op) >= 3:
                # op structure: (name, q, param, ...)
                name, q, param = op[0], op[1], float(op[2])
                new_ops.append((name, q, param + delta) + op[3:])
                shifted = True
            else:
                new_ops.append(op)
        return new_ops

    plus.ops = _shift_ops(plus.ops, math.pi / 2)
    minus.ops = _shift_ops(minus.ops, -math.pi / 2)
    meta: Dict[str, Any] = {"coeff": 0.5}
    return plus, minus, meta


