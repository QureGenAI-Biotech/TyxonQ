from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule


class AutoMeasurePass:
    """Insert Z measurements on all qubits when none are present.

    Behavior:
      - If the circuit contains no explicit `("measure_z", q)` ops, append
        `measure_z` on every qubit [0..num_qubits-1].
      - Emits a non-fatal warning to inform users.
      - No-op when any measurement already exists.
    """

    name = "rewrite/auto_measure"

    def execute_plan(self, circuit: "Circuit", device_rule: "DeviceRule" = None, **opts: Any) -> "Circuit":
        ops = getattr(circuit, "ops", []) or []
        has_meas = any((op and isinstance(op, (list, tuple)) and str(op[0]).lower() == "measure_z") for op in ops)
        if has_meas:
            return circuit
        nq = int(getattr(circuit, "num_qubits", 0))
        if nq <= 0:
            return circuit
        warnings.warn(
            "No explicit measurements found; auto-added Z measurements on all qubits during compilation.",
            UserWarning,
        )
        return circuit.extended([("measure_z", q) for q in range(nq)])


