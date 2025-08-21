from __future__ import annotations

from typing import Any, Dict, Protocol, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tyxonq.core.ir import Circuit
    from tyxonq.devices import DeviceCapabilities


class CompileRequest(TypedDict):
    """Structured compile request.

    Fields:
        circuit: IR circuit to compile.
        target: Device capability description for capability negotiation.
        options: Additional compile options (optimization level, layout hints).
    """

    circuit: "Circuit"
    target: "DeviceCapabilities"
    options: Dict[str, Any]


class CompileResult(TypedDict):
    """Result of compilation containing the compiled circuit and metadata."""

    circuit: "Circuit"
    metadata: Dict[str, Any]


class Pass(Protocol):
    """Compilation pass that transforms a circuit for a given target."""

    def run(self, circuit: "Circuit", caps: "DeviceCapabilities", **opts: Any) -> "Circuit": ...


class Compiler(Protocol):
    """Compiler interface that transforms IR for a target device/backend."""

    def compile(self, request: CompileRequest) -> CompileResult: ...


