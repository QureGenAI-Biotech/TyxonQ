from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, TypedDict, TYPE_CHECKING, Union

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


# ---- Facade helpers used by cloud.api and Circuit.run ----
def device_descriptor(
    name: Optional[str] = None,
    *,
    provider: Optional[str] = None,
    id: Optional[str] = None,
    shots: Optional[int] = None,
) -> Dict[str, Any]:
    from .hardware import config as hwcfg

    if name is None:
        prov = provider or hwcfg.get_default_provider()
        dev = id or hwcfg.get_default_device()
        if prov == "simulator" and dev is not None and "::" not in str(dev):
            dev = f"{prov}::{dev}"
    else:
        if "." in name:
            prov, dev = name.split(".", 1)
            dev = f"{prov}::{dev}"
        elif "::" in name:
            prov, dev = name.split("::", 1)
            dev = f"{prov}::{dev}"
        else:
            prov = provider or hwcfg.get_default_provider()
            if name in ("simulator:mps", "simulator_mps", "mps"):
                dev = f"{prov}::matrix_product_state"
            elif name in ("statevector",):
                dev = f"{prov}::statevector"
            elif name in ("density_matrix",):
                dev = f"{prov}::density_matrix"
            else:
                dev = name if name.startswith(prov + "::") else f"{prov}::{name}"
    return {"provider": prov, "device": dev, "shots": shots}


def resolve_driver(provider: str, device: str):
    if provider in ("simulator", "local"):
        from .simulators import driver as drv

        return drv
    if provider == "tyxonq":
        from .hardware.tyxonq import driver as drv

        return drv
    if provider == "ibm":
        from .hardware.ibm import driver as drv

        return drv
    raise ValueError(f"Unsupported provider: {provider}")


def list_all_devices(*, provider: Optional[str] = None, token: Optional[str] = None, **kws: Any) -> List[str]:
    from .hardware import config as hwcfg

    prov = provider or hwcfg.get_default_provider()
    dev = hwcfg.get_default_device()
    tok = token or hwcfg.get_token(provider=prov)

    # Aggregate simulators and provider-specific hardware list
    sim_list = [
        "simulator::matrix_product_state",
        "simulator::statevector",
        "simulator::density_matrix",
    ]
    try:
        drv = resolve_driver(prov, dev)
        hw_list = list(drv.list_devices(tok, **kws))
    except Exception:
        hw_list = []
    return sim_list + hw_list

