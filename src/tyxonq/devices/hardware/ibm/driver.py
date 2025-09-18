from __future__ import annotations

"""IBM hardware driver skeleton.

This module provides a minimal interface expected by cloud.api:
- list_devices(token, **kws) -> List[str]
- submit_task(device, token, *, source|circuit, shots, **opts) -> List[Any]
- get_task_details(task, token, prettify=False) -> Dict[str, Any]

Implementation note:
- Wiring to real IBM backends will be added via qiskit adapters in a later step.
"""

from typing import Any, Dict, List, Optional, Sequence, Union


def list_devices(token: Optional[str] = None, **kws: Any) -> List[str]:
    # Skeleton: return empty for now; will populate via qiskit providers.
    return []

def run(*args,**kwargs):
    return submit_task(*args,**kwargs)

def submit_task(
    device: str,
    token: Optional[str] = None,
    *,
    circuit: Optional[Union[Any, Sequence[Any]]] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    shots: Union[int, Sequence[int]] = 1024,
    **opts: Any,
) -> List[Any]:
    raise NotImplementedError("IBM driver is a skeleton; submission not wired yet")


def get_task_details(task: Any, token: Optional[str] = None, prettify: bool = False) -> Dict[str, Any]:
    raise NotImplementedError("IBM driver is a skeleton; query not wired yet")


