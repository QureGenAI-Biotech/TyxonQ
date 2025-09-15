from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule


class GatesTransformPass:
    """Rewrite gates according to preferred basis_gates.

    - Uses options["basis_gates"] if provided; else defaults to ["h","rx","rz","cx","cz"].
    - Minimal rules:
        x -> rx(pi) if "rx" allowed
        y -> ry(pi) if "ry" allowed
        keep: {cx, cz, h, rx, ry, rz}
        transparently keep rxx/rzz/cy if requested
        otherwise keep as-is
    """

    name = "rewrite/gates_transform"

    def execute_plan(self, circuit: "Circuit", basis_gates =["h", "rx", "rz", "cx", "cz"] ,device_rule: "DeviceRule" = None, **opts: Any) -> "Circuit":
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
        return type(circuit)(
            circuit.num_qubits,
            ops=new_ops,
            metadata=dict(getattr(circuit, "metadata", {})),
            instructions=list(getattr(circuit, "instructions", [])),
        )


