from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, TypedDict, TYPE_CHECKING, Union
from dataclasses import dataclass
import time

if TYPE_CHECKING:  # pragma: no cover - type-only imports to avoid cycles
    from tyxonq.core.ir import Circuit
    Observable = Any


# ---- Unified task wrapper ----
@dataclass
class DeviceTask:
    provider: str
    device: str
    handle: Any
    async_result: bool

    def get_result(self, *, wait: bool = False, poll_interval: float = 2.0, timeout: float = 60.0) -> Dict[str, Any]:
        return get_task_details(self,wait=wait, poll_interval=poll_interval, timeout=timeout)

    def cancel(self) -> Any:
        return remove_task(self)


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


def init(*, provider: Optional[str] = None, device: Optional[str] = None, token: Optional[str] = None) -> None:
    """Initialize default provider/device and optionally set token.

    This is a light wrapper around hardware.config helpers.
    """
    from .hardware import config as hwcfg

    if token is not None:
        hwcfg.set_token(token, provider=provider, device=device)
    if provider is not None or device is not None:
        hwcfg.set_default(provider=provider, device=device)


_NOISE_ENABLED: bool = False
_NOISE_CONFIG: Dict[str, Any] | None = None
_DEFAULT_NOISE: Dict[str, Any] = {"type": "depolarizing", "p": 0.0}

def enable_noise(enabled: bool = True, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global _NOISE_ENABLED, _NOISE_CONFIG
    _NOISE_ENABLED = bool(enabled)
    if config is not None:
        _NOISE_CONFIG = dict(config)
    return {"enabled": _NOISE_ENABLED, "config": _NOISE_CONFIG or {}}

def is_noise_enabled() -> bool:
    return _NOISE_ENABLED

def get_noise_config() -> Dict[str, Any]:
    return dict(_NOISE_CONFIG or {})

def run(
    *,
    provider: Optional[str] = None,
    device: Optional[str] = None,
    circuit: Optional["Circuit"] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    shots: Union[int, Sequence[int]] = 1024,
    **opts: Any,
) -> Any:
    """Unified device-level selector to execute circuits or sources.

    Responsibilities:
    - Choose driver via provider/device defaults
    - If `source` provided, submit directly (no compilation here)
    - If `circuit` provided:
      - simulator/local: call simulator driver run
      - hardware: require caller to have compiled to `source`
    - Normalize return: single submission -> single task; batch -> list of tasks

    Returns:
        List[DeviceTask] Unified task-handle wrapper:
        - task.async_result=False (simulator): get_results() returns final result immediately
        - task.async_result=True (hardware): get_results(wait=True) polls until completion
    """
    from .hardware import config as hwcfg

    prov = provider or hwcfg.get_default_provider()
    dev = device or hwcfg.get_default_device()
    tok = hwcfg.get_token(provider=prov, device=dev)

    drv = resolve_driver(prov, dev)

    def _normalize(out: Any) -> List[Any]:
        # Always return a list of task-like objects for uniform handling
        if isinstance(out, list):
            return out
        return [out]

    # Assemble noise settings to pass to simulators if not explicitly set
    def _inject_noise(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        use_noise = bool(kwargs.get("use_noise", _NOISE_ENABLED))
        noise_cfg = kwargs.get("noise")
        if use_noise and noise_cfg is None:
            noise_cfg = _NOISE_CONFIG or _DEFAULT_NOISE
        if use_noise:
            new = dict(kwargs)
            new["use_noise"] = True
            if noise_cfg is not None:
                new["noise"] = noise_cfg
            return new
        return kwargs

    # direct source path (already compiled or raw program)
    if source is not None:
        if prov in ("simulator", "local") and device in ('mps','density_matrix','statevector','matrix_product_state'):
            if circuit is not None:
                raw = _normalize(drv.run(dev, tok, circuit=circuit, source=None, shots=shots, **_inject_noise(opts)))
            else:
                raw = _normalize(drv.run(dev, tok, source=source, shots=shots, **_inject_noise(opts)))
        else:
            raw = _normalize(drv.run(dev, tok, source=source, shots=shots, **opts))
    else:
        # circuit path
        if circuit is None:
            raise ValueError("run requires either circuit or source")
        
        if prov not in ("simulator", "local"):
            # hardware path requires source (compilation should have been done by caller)
            raise ValueError("hardware run without source is not supported at device layer; compile in circuit layer")
        if prov in ("simulator", "local") and device in ('mps','density_matrix','statevector','matrix_product_state'):
            raw = _normalize(drv.run(dev, tok, circuit=circuit, shots=shots, **_inject_noise(opts)))
        else:
            raw = _normalize(drv.run(dev, tok, circuit=circuit, shots=shots, **opts))

    # Wrap into unified DeviceTask objects

    device_task_list = []
    for t in raw:
        try:
            async_result = t.async_result
        except:
            async_result = False
        device_task_list.append(DeviceTask(provider=prov, device=dev, handle=t, async_result=async_result))
    
    return device_task_list


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


# ---- Unified task helpers (polling/wait) ----
def get_task_details(task: Any, *, wait: bool = False, poll_interval: float = 2.0, timeout: float = 15.0) -> Dict[str, Any]:
    """Get task details with optional polling, and unify result format.

    Unified return format:
        {
          'result': Dict[str, int],      # normalized counts like {'00': 51, '11': 49}
          'result_meta': Dict[str, Any], # original driver payload
        }
    """
    if not isinstance(task, DeviceTask):
        raise TypeError("Task handle should be a DeviceTask type")

    drv = resolve_driver(task.provider, task.device)

    def _fetch() -> Dict[str, Any]:
        return drv.get_task_details(task.handle, None)

    def _wrap(info: Dict[str, Any]) -> Dict[str, Any]:
        src = info.get('result') or info.get('results') or {}
        return {'result': src, 'result_meta': info}

    if not wait:
        return _wrap(_fetch())

    start = time.perf_counter()
    while True:
        info = _fetch()
        if not task.async_result:
            return _wrap(info)
        uni_status = str(info.get("uni_status", "completed")).lower()
        if uni_status in ("done", "completed", "success", "finished"):
            return _wrap(info)
        if (time.perf_counter() - start) >= timeout:
            return _wrap(info)
        time.sleep(max(0.05, poll_interval))





def remove_task(task: Any) -> Any:
    if isinstance(task, DeviceTask):
        prov = task.provider
        dev_str = task.device
        handle = task.handle
    else:
        raise TypeError("Task handle should be a DeviceTask type")

    drv = resolve_driver(prov, dev_str)
    if hasattr(drv, "remove_task"):
        return drv.remove_task(handle, None)
    raise NotImplementedError("remove_task not supported for this provider")




