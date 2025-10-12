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


def set_token(token: str, *, provider: Optional[str] = None, device: Optional[str] = None) -> Dict[str, str]:
    return hwcfg.set_token(token, provider=provider, device=device)


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
    # Delegate to base.run for unified behavior (no extra logic)
    from ..devices import base as device_base

    return device_base.run(
        provider=provider,
        device=device,
        circuit=circuit,
        source=source,
        shots=shots,
        **opts,
    )


def get_task_details(task: Any, *, token: Optional[str] = None, wait: bool = False, poll_interval: float = 2.0, timeout: float = 60.0) -> Dict[str, Any]:
    # Delegate to base unified helper
    from ..devices.base import get_task_details as _get

    # token currently unused in base helper; reserved for future
    return _get(task, wait=wait, poll_interval=poll_interval, timeout=timeout)


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
    """Execute quantum circuit(s) or source code on cloud devices.
    
    This is the main entry point for cloud execution in TyxonQ. It supports
    both circuit objects and pre-compiled source code (OpenQASM, TQASM).
    
    Args:
        provider: Cloud provider name ("tyxonq", "simulator", etc.). 
            Uses global default if not specified.
        device: Device name ("homebrew_s2", "statevector", etc.).
            Uses global default if not specified.
        circuit: Single Circuit object or list of Circuit objects to execute.
            Mutually exclusive with source.
        source: Pre-compiled source code (OpenQASM, TQASM) or list of sources.
            Mutually exclusive with circuit.
        shots: Number of measurement shots to execute.
        token: Optional authentication token. Uses global token if not specified.
        **opts: Additional options passed to the device driver.
        
    Returns:
        For single submission: Task handle
        For batch submission: List of task handles
        
    Examples:
        >>> import tyxonq as tq
        >>> from tyxonq.cloud import api
        >>> 
        >>> # Execute circuit on cloud hardware
        >>> c = tq.Circuit(2).h(0).cx(0,1).measure_all()
        >>> task = api.run(provider="tyxonq", device="homebrew_s2", circuit=c, shots=100)
        >>> result = task.get_result(wait=True)
        >>> 
        >>> # Execute OpenQASM source
        >>> qasm = "OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1]; measure q;"
        >>> task = api.run(provider="tyxonq", device="homebrew_s2", source=qasm, shots=100)
    """
    # Delegate directly to devices.base.run for unified behavior
    from ..devices import base as device_base

    return device_base.run(
        provider=provider,
        device=device,
        circuit=circuit,
        source=source,
        shots=shots,
        **opts,
    )


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


