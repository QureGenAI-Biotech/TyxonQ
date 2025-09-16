from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceRule


class NoOpSchedulingPass:
    """Placeholder scheduling pass that returns the circuit unchanged."""

    def execute_plan(self, circuit: "Circuit", device_rule: "DeviceRule" = None, **opts) -> "Circuit":
        return circuit


