from __future__ import annotations

from typing import Any, Dict, Protocol, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type-only imports to avoid cycles
    from tyxonq.core.ir import Circuit
    Observable = Any


class DeviceCapabilities(TypedDict, total=False):
    """Declarative device capabilities description.

    Keys are optional to keep forward compatibility. Concrete devices may
    expose additional metadata fields as needed.
    """

    native_gates: set[str]
    max_qubits: int
    connectivity: Any
    supports_shots: bool
    supports_batch: bool


class RunResult(TypedDict, total=False):
    """Structured run result returned by `Device.run`.

    Optional keys allow devices to report varying levels of detail while
    preserving a common contract for downstream processing.
    """

    samples: Any
    expectations: Dict[str, float]
    metadata: Dict[str, Any]


class Device(Protocol):
    """Execution device protocol.

    A device is responsible for running compiled circuits, sampling, and
    computing expectation values.
    """

    name: str
    capabilities: DeviceCapabilities

    def run(self, circuit: "Circuit", shots: int | None = None, **kwargs: Any) -> RunResult: ...
    def expval(self, circuit: "Circuit", obs: "Observable", **kwargs: Any) -> float: ...


