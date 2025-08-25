from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol
import time


class JobHandle(Protocol):
    def status(self) -> str: ...
    def results(self) -> Dict[str, Any]: ...


def wait_for(job: JobHandle, *, poll: float = 0.5, timeout: Optional[float] = None) -> Dict[str, Any]:
    start = time.time()
    s = job.status()
    while s not in ("completed", "failed"):
        time.sleep(poll)
        if timeout is not None and (time.time() - start) > timeout:
            break
        s = job.status()
    return job.results()


__all__ = ["JobHandle", "wait_for"]


