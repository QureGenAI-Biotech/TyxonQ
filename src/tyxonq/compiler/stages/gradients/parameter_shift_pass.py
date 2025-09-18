from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule

from ...gradients.parameter_shift import generate_shifted_circuits


class ParameterShiftPass:
    """Populate parameter-shift metadata for a target op name.

    Options:
      - grad_op: operation name to match (e.g., "rz")
    """

    def execute_plan(self, circuit: "Circuit", device_rule: "DeviceRule" = None, **opts: Any) -> "Circuit":
        op_name = opts.get("grad_op")
        if not op_name:
            return circuit
        plus, minus, meta = generate_shifted_circuits(circuit, op_name)
        circuit.metadata.setdefault("gradients", {})[op_name] = {
            "plus": plus,
            "minus": minus,
            "meta": meta,
        }
        return circuit


