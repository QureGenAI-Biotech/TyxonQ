from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
from dataclasses import dataclass
from uuid import uuid4

import requests

from ..config import ENDPOINTS, get_token


@dataclass
class TyxonQTask:
    def __init__(self, id: str, device: str,status: str, task_info: None,async_result: bool):
        self.id = id
        self.device = device
        self._result = None
        self.task_info = None
        self.async_result = True
        self.status = status
        self.result_metadata = None

    # Normalize: expose get_result(wait=...) for compatibility with examples
    def get_result(self, token: Optional[str] = None, *, wait: bool = True, poll_interval: float = 2.0, timeout: float = 60.0) -> Dict[str, Any]:
        if not wait:
            info = get_task_details(self, token)
            return info
        else:
            if bool(self.async_result):
                import time as _t
                start = _t.perf_counter()
                while True:
                    info = get_task_details(self, token)
                    state = str(info.get("state", "")).lower()
                    if state in ("done", "completed", "success", "finished"):
                        break
                    if (_t.perf_counter() - start) >= timeout:
                        break
                    _t.sleep(max(0.05, poll_interval))
                return info
                    
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
    url = _endpoint("devices/list")
    r = requests.post(url, json={}, headers=_headers(token), timeout=15)
    r.raise_for_status()
    data = r.json()
    if "devices" not in data:
        raise ValueError(f"No device details for {device}")
    
    device_list=[]
    for dev in data["devices"]:
        dev.pop('memo')
        device_list.append(dev)
    return device_list


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
    """
    {'id': 'xxxxxx', 'job_name': 'xxxxx', 'status': '处理中', 'user_id': 'xxxxx', 'success': True, 'error': None}
    """
    """
    {'success': False, 'error': 'Submit Task Failed'}
    """
    if data.get('error'):
        # On error, attempt to fetch device properties for diagnostics
        info: Dict[str, Any] = {"error": data, "device": device}
        try:
            props = list_properties(device, token)
            info['device_status'] = props
        except Exception:
            pass
        raise RuntimeError('tyxonq execution error: ' + str(info))
    return [TyxonQTask(id=data.get("id", str(uuid4())), device=device, status=data.get('status','submitted'),task_info=data,async_result=True)]



def get_task_details(task: TyxonQTask, token: Optional[str] = None) -> Dict[str, Any]:
    url = _endpoint("tasks/detail")
    r = requests.post(url, json={"task_id": task.id}, headers=_headers(token), timeout=15)
    r.raise_for_status()
    data = r.json()
    if 'uni_status' not in data:
        task_detail = data.get('task',{})
        data['uni_status'] = task_detail.get('state', 'completed')
        data['result'] = task_detail.get('result',{})
    """
    https://github.com/QureGenAI-Biotech/TyxonQ/blob/main/docs/tyxonq_cloud_api.md
    {
        "success": true,
        "task": {
            "id": "<JOB_ID>",
            "queue": "quregenai.lab",
            "device": "homebrew_s2?o=3",
            "qubits": 2,
            "depth": 3,
            "state": "completed",
            "shots": 100,
            "at": 1754275505649825,
            "ts": {
                "completed": 1754275505649825,
                "pending": 1754275502265270,
                "scheduled": 1754275502260031
            },
            "md5": "f31a82f44a53bc8fa6e08ef0c6a34d53",
            "runAt": 1754275488761744,
            "runDur": 2532053,
            "atChip": 1754275446369691,
            "durChip": 120185,
            "result": {
                "00": 33,
                "01": 2,
                "10": 4,
                "11": 61
            },
            "task_type": "quantum_api"
        }
    }
    """
    return data


def remove_task(task: TyxonQTask, token: Optional[str] = None) -> Dict[str, Any]:
    url = _endpoint("task/remove")
    r = requests.post(url, json={"id": task.id}, headers=_headers(token), timeout=15)
    r.raise_for_status()
    return r.json()


