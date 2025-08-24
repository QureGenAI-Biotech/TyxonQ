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


def set_token(token: str, *, provider: Optional[str] = None, device: Optional[str] = None, persist: bool = True) -> Dict[str, str]:
    return hwcfg.set_token(token, provider=provider, device=device, persist=persist)


def set_default(*, provider: Optional[str] = None, device: Optional[str] = None) -> None:
    hwcfg.set_default(provider=provider, device=device)


def device(name: Union[str, None] = None, *, provider: Optional[str] = None, id: Optional[str] = None, shots: Optional[int] = None) -> Dict[str, Any]:
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
            # accept shorthand for common choices
            if name in ("simulator:mps", "simulator_mps", "mps"):
                dev = f"{prov}::matrix_product_state"
            elif name in ("statevector",):
                dev = f"{prov}::statevector"
            elif name in ("density_matrix",):
                dev = f"{prov}::density_matrix"
            else:
                dev = name if name.startswith(prov + "::") else f"{prov}::{name}"
    return {"provider": prov, "device": dev, "shots": shots}


def _driver(provider: str, device: str):
    # simulator routes to simulators.driver; tyxonq routes to cloud driver
    if provider in ("simulator", "local"):
        from ..devices.simulators import driver as drv
        return drv

    if provider == "tyxonq":
        from ..devices.hardware.tyxonq import driver as drv
        return drv
    raise ValueError(f"Unsupported provider: {provider}")


def list_devices(*, provider: Optional[str] = None, token: Optional[str] = None, **kws: Any) -> List[str]:
    prov = provider or hwcfg.get_default_provider()
    dev = hwcfg.get_default_device()
    drv = _driver(prov, dev)
    tok = token or hwcfg.get_token(provider=prov)
    return drv.list_devices(tok, **kws)


def submit_task(
    *,
    provider: Optional[str] = None,
    device: Optional[str] = None,
    circuit: Optional[Union[Any, Sequence[Any]]] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    shots: Union[int, Sequence[int]] = 1024,
    token: Optional[str] = None,
    **opts: Any,
):
    prov = provider or hwcfg.get_default_provider()
    dev = device or hwcfg.get_default_device()
    drv = _driver(prov, dev)
    tok = token or hwcfg.get_token(provider=prov, device=dev)

    # Simulator uses IR circuit directly; cloud providers expect source (e.g., OpenQASM)
    if prov in ("simulator", "local"):
        return drv.submit_task(dev, tok, circuit=circuit, shots=shots, **opts)

    if source is None and circuit is not None:
        # Minimal IR → OpenQASM: rely on object providing to_openqasm
        if isinstance(circuit, (list, tuple)):
            source = [getattr(c, "to_openqasm")() for c in circuit]  # type: ignore
        else:
            source = getattr(circuit, "to_openqasm")()  # type: ignore
    return drv.submit_task(dev, tok, source=source, shots=shots, **opts)


def get_task_details(task: Any, *, token: Optional[str] = None) -> Dict[str, Any]:
    # Accept vendor task or plain id
    if hasattr(task, "device") and hasattr(task, "id_"):
        dev = getattr(task, "device")
        # Normalize provider for simulator tasks where device might be plain name
        dev_str = str(dev)
        prov = (dev_str.split("::", 1)[0]) if "::" in dev_str else "simulator"
        drv = _driver(prov, dev)
        return drv.get_task_details(task, token)
    raise ValueError("Unsupported task handle type")


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
    return submit_task(
        provider=provider,
        device=device,
        circuit=circuit,
        source=source,
        shots=shots,
        token=token,
        **opts,
    )


def result(task: Any, *, token: Optional[str] = None, prettify: bool = False) -> Dict[str, Any]:
    return get_task_details(task, token=token)


def cancel(task: Any, *, token: Optional[str] = None) -> Any:
    if hasattr(task, "device") and hasattr(task, "id_"):
        dev = getattr(task, "device")
        prov = str(dev).split("::", 1)[0]
        drv = _driver(prov, dev)
        if hasattr(drv, "remove_task"):
            return drv.remove_task(task, token)
    raise NotImplementedError("cancel not supported for this provider/task type")


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


