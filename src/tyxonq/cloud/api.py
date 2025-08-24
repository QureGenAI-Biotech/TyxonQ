from __future__ import annotations

"""Unified cloud API facade (minimal), per migration plan.

Functions:
- set_token(token, provider=None, device=None)
- set_default(provider=None, device=None)
- device(name|provider.device)
- list_devices(provider=None)
- submit_task(provider=None, device=None, circuit|source, shots, **opts)
- get_task_details(task)

Drivers live under devices.hardware.<vendor>.driver and are selected by provider.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

from ..devices.hardware import config as hwcfg
from ..devices.base import device_descriptor as _device_descriptor, resolve_driver as _resolve_driver, list_all_devices as _list_all_devices


def set_token(token: str, *, provider: Optional[str] = None, device: Optional[str] = None, persist: bool = True) -> Dict[str, str]:
    return hwcfg.set_token(token, provider=provider, device=device, persist=persist)


def set_default(*, provider: Optional[str] = None, device: Optional[str] = None) -> None:
    hwcfg.set_default(provider=provider, device=device)


def device(name: Union[str, None] = None, *, provider: Optional[str] = None, id: Optional[str] = None, shots: Optional[int] = None) -> Dict[str, Any]:
    return _device_descriptor(name, provider=provider, id=id, shots=shots)


def _driver(provider: str, device: str):
    return _resolve_driver(provider, device)


def list_devices(*, provider: Optional[str] = None, token: Optional[str] = None, **kws: Any) -> List[str]:
    return _list_all_devices(provider=provider, token=token, **kws)


def submit_task(
    *,
    provider: Optional[str] = None,
    device: Optional[str] = None,
    circuit: Optional[Union[Any, Sequence[Any]]] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    shots: Union[int, Sequence[int]] = 1024,
    token: Optional[str] = None,
    auto_compile: bool = True,
    **opts: Any,
):
    """Thin wrapper that delegates to Circuit.run semantics when possible.

    Preferred path: if `circuit` is an IR Circuit, use its `.run()` to decide
    compile vs IR submission. Otherwise fall back to driver routing with `source`.
    """

    # If IR Circuit object(s) provided, delegate to Circuit.run for canonical behavior
    def _is_ir(obj: Any) -> bool:
        try:
            from ..core.ir import Circuit as _C

            return isinstance(obj, _C)
        except Exception:
            return False

    if circuit is not None and (_is_ir(circuit) or (isinstance(circuit, (list, tuple)) and any(_is_ir(c) for c in circuit))):
        # Import lazily to avoid cycles
        if isinstance(circuit, (list, tuple)):
            tasks = []
            for c in circuit:
                tasks.append(c.run(provider=provider, device=device, shots=int(shots) if isinstance(shots, int) else 1024, auto_compile=auto_compile, **opts))
            return tasks
        return circuit.run(provider=provider, device=device, shots=int(shots) if isinstance(shots, int) else 1024, auto_compile=auto_compile, **opts)

    prov = provider or hwcfg.get_default_provider()
    dev = device or hwcfg.get_default_device()
    tok = token or hwcfg.get_token(provider=prov, device=dev)
    drv = _driver(prov, dev)
    return drv.run(dev, tok, source=source, shots=shots, **opts)


def get_task_details(task: Any, *, token: Optional[str] = None) -> Dict[str, Any]:
    # Delegate to Circuit helper when available
    from ..core.ir import Circuit as _C

    helper = getattr(_C, "get_task_details", None)
    if callable(helper):
        # call as unbound method with a dummy circuit instance is unnecessary; method is instance-based
        # Instead, re-resolve via device driver if circuit instance not available
        pass
    # Fallback: driver-based resolution
    dev = getattr(task, "device", None)
    if dev is None:
        raise ValueError("Unsupported task handle type")
    dev_str = str(dev)
    prov = (dev_str.split("::", 1)[0]) if "::" in dev_str else "simulator"
    drv = _driver(prov, dev_str)
    return drv.get_task_details(task, token)


def run(
    *,
    provider: Optional[str] = None,
    device: Optional[str] = None,
    circuit: Optional[Union[Any, Sequence[Any]]] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    shots: Union[int, Sequence[int]] = 1024,
    token: Optional[str] = None,
    **opts: Any,
):
    # Pure delegation: if circuit is IR, use Circuit.run; else fallback to submit_task with source
    def _is_ir(obj: Any) -> bool:
        try:
            from ..core.ir import Circuit as _C

            return isinstance(obj, _C)
        except Exception:
            return False

    if circuit is not None and (_is_ir(circuit) or (isinstance(circuit, (list, tuple)) and any(_is_ir(c) for c in circuit))):
        if isinstance(circuit, (list, tuple)):
            return [c.run(provider=provider, device=device, shots=int(shots) if isinstance(shots, int) else 1024, **opts) for c in circuit]
        return circuit.run(provider=provider, device=device, shots=int(shots) if isinstance(shots, int) else 1024, **opts)
    # Fallback to driver path (source)
    prov = provider or hwcfg.get_default_provider()
    dev = device or hwcfg.get_default_device()
    tok = token or hwcfg.get_token(provider=prov, device=dev)
    drv = _driver(prov, dev)
    return drv.run(dev, tok, source=source, shots=shots, **opts)


def result(task: Any, *, token: Optional[str] = None, prettify: bool = False) -> Dict[str, Any]:
    # Delegate to Circuit module-level helper
    from ..core.ir.circuit import get_task_details as _get

    return _get(task, prettify=prettify)


def cancel(task: Any, *, token: Optional[str] = None) -> Any:
    # Delegate to Circuit module-level helper
    from ..core.ir.circuit import cancel_task as _cancel

    return _cancel(task)


__all__ = [
    "set_token",
    "set_default",
    "device",
    "list_devices",
    "submit_task",
    "get_task_details",
    "run",
    "result",
    "cancel",
]


