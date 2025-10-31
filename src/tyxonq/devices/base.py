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
        info = get_task_details(self,wait=wait, poll_interval=poll_interval, timeout=timeout)
        # Normalize schema: ensure keys exist for downstream postprocessing
        result = dict(info)
        if 'result' not in result:
            result['result'] = {}
        # Promote provider-specific metadata
        if 'result_meta' in result and 'metadata' not in result:
            result['metadata'] = result['result_meta']
        if 'metadata' not in result:
            result['metadata'] = {}
        # Ensure expectations/probabilities keys exist (optional)
        if 'expectations' not in result:
            result['expectations'] = {}
        if 'probabilities' not in result:
            result['probabilities'] = None
        return result

    def cancel(self) -> Any:
        return remove_task(self)


class DeviceRule(TypedDict, total=False):
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
    device_rule: DeviceRule

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
    """Resolve and return the appropriate device driver for a given provider.

    This function provides a factory interface for accessing different quantum
    device drivers based on the provider and device specification. It abstracts
    the complexity of driver instantiation and enables uniform access to
    simulators, local devices, and cloud-based quantum hardware.

    Args:
        provider (str): The quantum computing provider identifier.
            Supported providers:
            - "simulator" or "local": Local quantum simulators
            - "tyxonq": TyxonQ quantum hardware platform
            - "ibm": IBM Quantum devices and simulators
            
        device (str): Specific device identifier within the provider.
            The device string format depends on the provider:
            - Simulator: "statevector", "density_matrix", "qasm_simulator"
            - TyxonQ: "homebrew_s2", "homebrew_s1", etc.
            - IBM: "ibm_brisbane", "ibm_kyoto", "simulator_mps", etc.

    Returns:
        DeviceDriver: A driver instance that implements the quantum device
            interface for circuit execution, task management, and result retrieval.
            The driver provides methods such as:
            - run_circuit(): Execute quantum circuits
            - get_task_details(): Retrieve execution status
            - cancel_task(): Cancel running jobs
            - get_device_properties(): Hardware specifications

    Raises:
        ValueError: If the specified provider is not supported or available.
        ImportError: If the required driver module cannot be imported
            (e.g., missing dependencies for IBM Quantum).

    Examples:
        >>> # Get local simulator driver
        >>> sim_driver = resolve_driver("simulator", "statevector")
        >>> result = sim_driver.run_circuit(circuit, shots=1000)
        
        >>> # Access TyxonQ hardware
        >>> hw_driver = resolve_driver("tyxonq", "homebrew_s2")
        >>> task = hw_driver.run_circuit(circuit, shots=4096)
        >>> status = hw_driver.get_task_details(task)
        
        >>> # Connect to IBM Quantum
        >>> ibm_driver = resolve_driver("ibm", "ibm_brisbane")
        >>> properties = ibm_driver.get_device_properties()
        >>> print(f"Device qubits: {properties['n_qubits']}")

    Provider Details:
        **Simulator/Local Provider**:
            - Provides high-performance local simulation
            - Supports statevector, density matrix, and sampling methods
            - No network connectivity required
            - Ideal for algorithm development and testing
        
        **TyxonQ Provider**:
            - Access to TyxonQ quantum hardware platform
            - Supports pulse-level control via TQASM
            - Includes device-specific optimization flags
            - Requires valid authentication tokens
        
        **IBM Provider**:
            - Integration with IBM Quantum Network
            - Access to real quantum hardware and cloud simulators
            - Requires IBM Quantum account and qiskit-ibm-provider
            - Supports advanced device features and error mitigation

    Notes:
        - Driver selection affects execution performance and capabilities
        - Some providers require additional authentication setup
        - Device availability may vary based on maintenance schedules
        - Driver interfaces are standardized for cross-provider compatibility
        
    See Also:
        tyxonq.devices.simulators: Local simulator implementations.
        tyxonq.devices.hardware: Hardware device drivers.
        tyxonq.cloud.api: High-level cloud execution interface.
    """
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
    circuit: Optional["Circuit"]| None = None,
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

    # Normalize shots: allow shots==0 only for simulators/local; coerce to >=1 for hardware
    if not isinstance(shots, (list, tuple)):
        try:
            _s = int(shots)
        except Exception:
            _s = 1024
        if _s == 0 and prov not in ("simulator", "local"):
            _s = 1
        shots = _s

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
        # circuit path (simulator/local only). Support single or batched circuits uniformly.
        if circuit is None:
            raise ValueError("run requires either circuit or source")

        if prov not in ("simulator", "local"):
            # hardware path requires source (compilation should have been done by caller)
            raise ValueError("hardware run without source is not supported at device layer; compile in circuit layer")

        # Normalize to list of circuits for unified batch handling
        circuits = list(circuit) if isinstance(circuit, (list, tuple)) else [circuit]

        # shots==0 + observable → use analytic expval path (only valid for single circuit)
        if len(circuits) == 1:
            try:
                _shots_int = int(shots)  # type: ignore[arg-type]
            except Exception:
                _shots_int = 0
            if _shots_int == 0 and ("observable" in opts):
                obs = opts.get("observable")
                e = expval(provider=prov, device=dev, circuit=circuits[0], observable=obs, **_inject_noise(opts))
                try:
                    from .simulators.driver import SimTask  # type: ignore
                    task_like = SimTask(id="expval", device=dev, result={
                        'result': {},
                        'expectations': {'expval': float(e)},
                        'probabilities': None,
                        'statevector': None,
                        'metadata': {'shots': 0, 'backend': 'analytic', 'provider': prov, 'device': dev},
                    })
                    raw = _normalize(task_like)
                except Exception:
                    raw = _normalize(resolve_driver(prov, dev).run(dev, tok, circuit=circuits[0], shots=shots, **_inject_noise(opts)))
            else:
                # Single circuit standard path
                raw = _normalize(drv.run(dev, tok, circuit=circuits[0], shots=shots, **_inject_noise(opts)))
        else:
            # Batched circuits path: order-preserving submission and collection
            # Support shots as scalar or list aligned with circuits
            if isinstance(shots, (list, tuple)):
                shots_list = [int(s) for s in shots]
                if len(shots_list) != len(circuits):
                    raise ValueError("shots list length must match number of circuits")
            else:
                shots_list = [int(shots)] * len(circuits)

            raw = []
            for c_obj, s_val in zip(circuits, shots_list):
                # Always inject noise settings consistently for simulators
                per_opts = _inject_noise(opts)
                raw.extend(_normalize(drv.run(dev, tok, circuit=c_obj, shots=s_val, **per_opts)))

    # Wrap into unified DeviceTask objects

    device_task_list = []
    for t in raw:
        try:
            async_result = t.async_result
        except:
            async_result = False
        device_task_list.append(DeviceTask(provider=prov, device=dev, handle=t, async_result=async_result))
    
    return device_task_list


def expval(
    *,
    provider: Optional[str] = None,
    device: Optional[str] = None,
    circuit: Optional["Circuit"] = None,
    observable: Any = None,
    **opts: Any,
) -> float:
    """Unified analytic expectation for simulator/local (shots==0 fast path).

    Routes to specific simulator driver which calls engine.expval.
    """
    from .hardware import config as hwcfg

    prov = provider or hwcfg.get_default_provider()
    dev = device or hwcfg.get_default_device()
    drv = resolve_driver(prov, dev)
    if prov not in ("simulator", "local"):
        raise ValueError("expval is supported only for simulator/local providers")
    if circuit is None or observable is None:
        raise ValueError("expval requires both circuit and observable")
    return float(drv.expval(dev, hwcfg.get_token(provider=prov, device=dev), circuit=circuit, observable=observable, **opts))


 


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




