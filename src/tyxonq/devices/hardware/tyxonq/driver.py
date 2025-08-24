from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
from dataclasses import dataclass
from uuid import uuid4

import requests

from ..config import ENDPOINTS, get_token


@dataclass
class TyxonQTask:
    id_: str
    device: str
    state: str = "pending"
    results_dict: Optional[Dict[str, int]] = None

    def status(self) -> str:
        return self.state

    def results(self) -> Dict[str, Any]:
        return {"results": self.results_dict or {}}

    # Normalize: expose details() for compatibility with examples
    def details(self, token: Optional[str] = None) -> list:
        return get_task_details(self, token)


def _endpoint(cmd: str) -> str:
    base = ENDPOINTS["tyxonq"]["base_url"]
    ver = ENDPOINTS["tyxonq"]["api_version"]
    return f"{base}api/{ver}/{cmd}"


def _headers(token: Optional[str]) -> Dict[str, str]:
    tok = token or get_token(provider="tyxonq") or "ANY;0"
    return {"Authorization": f"Bearer {tok}"}


def list_devices(token: Optional[str] = None, **kws: Any) -> List[str]:
    url = _endpoint("devices/list")
    r = requests.post(url, json=kws, headers=_headers(token), timeout=15)
    r.raise_for_status()
    data = r.json()
    devs = [d["id"] for d in data.get("devices", [])]
    return [f"tyxonq::{d}" for d in devs]


def list_properties(device: str, token: Optional[str] = None) -> Dict[str, Any]:
    url = _endpoint("device/detail")
    r = requests.post(url, json={"id": device.split("::")[-1]}, headers=_headers(token), timeout=15)
    r.raise_for_status()
    data = r.json()
    if "device" not in data:
        raise ValueError(f"No device details for {device}")
    return data["device"]


def run(*args,**kwargs):
    return submit_task(*args,**kwargs)

def submit_task(
    device: str,
    token: Optional[str] = None,
    *,
    source: Optional[Union[str, Sequence[str]]] = None,
    shots: Union[int, Sequence[int]] = 1024,
    lang: str = "OPENQASM",
    **kws: Any,
) -> List[TyxonQTask]:
    # Minimal pass-through; compilation handled elsewhere
    url = _endpoint("tasks/submit_task")
    payload: Any
    dev = device.split("::")[-1]
    if isinstance(source, (list, tuple)):
        if not isinstance(shots, (list, tuple)):
            shots = [shots for _ in source]
        payload = [
            {"device": dev, "shots": int(sh), "source": s, "version": "1", "lang": lang}
            for s, sh in zip(source, shots)
        ]
    else:
        payload = {"device": dev, "shots": int(shots), "source": source, "version": "1", "lang": lang}

    r = requests.post(url, json=payload, headers=_headers(token), timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(source, (list, tuple)):
        # Best-effort: fabricate ids when not returned
        return [TyxonQTask(id_=str(uuid4()), device=device, state="submitted")]
    return [TyxonQTask(id_=data.get("id", str(uuid4())), device=device, state="submitted")]


def get_task_details(task: TyxonQTask, token: Optional[str] = None) -> Dict[str, Any]:
    url = _endpoint("tasks/detail")
    r = requests.post(url, json={"task_id": task.id_}, headers=_headers(token), timeout=15)
    r.raise_for_status()
    data = r.json()
    return data.get("task", {})


def remove_task(task: TyxonQTask, token: Optional[str] = None) -> Dict[str, Any]:
    url = _endpoint("task/remove")
    r = requests.post(url, json={"id": task.id_}, headers=_headers(token), timeout=15)
    r.raise_for_status()
    return r.json()


