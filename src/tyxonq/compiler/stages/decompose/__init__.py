from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.compiler import Pass
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceCapabilities


class NoOpDecomposePass:
    """Placeholder decompose pass that returns the circuit unchanged."""

    def run(self, circuit: "Circuit", caps: "DeviceCapabilities", **opts) -> "Circuit":
        return circuit


