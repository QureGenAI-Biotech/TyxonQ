"""TyxonQ driver for the BAQIS Quafu Superconducting Quantum Cloud.

Wraps the vendored REST client in `_vendor_quafu.py` to fit TyxonQ's
provider-driver contract (run / get_task_details / list_devices / cancel).
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from ..config import get_token as _hw_get_token
from ._vendor_quafu import Task as _QuafuTaskMgr


# ---------- Token resolution ----------

_TOKEN_HELP = (
    "Quafu token required. Get one at https://quafu-sqc.baqis.ac.cn/ "
    "(rotates every 30 days) and pass via "
    "tq.set_token(..., provider='quafu'), TYXONQ_QUAFU_TOKEN env, "
    "or token=... kwarg."
)


def _resolve_token(token: Optional[str]) -> str:
    """Four-step precedence chain. See spec §5.

    Order: explicit kwarg → tq.set_token(provider='quafu') →
    TYXONQ_QUAFU_TOKEN env → QPU_API_TOKEN env → RuntimeError.

    Crucially does NOT fall back to TYXONQ_API_KEY — that is the tyxonq-cloud
    token and would be wrong for Quafu.
    """
    if token:
        return token
    tok = _hw_get_token(provider="quafu", env_fallback=False)
    if tok:
        return tok
    tok = os.getenv("TYXONQ_QUAFU_TOKEN") or os.getenv("QPU_API_TOKEN")
    if tok:
        return tok
    raise RuntimeError(_TOKEN_HELP)


# ---------- Task wrapper ----------

@dataclass
class QuafuTask:
    """Handle for a Quafu cloud submission. Mirrors TyxonQTask shape so the
    Chain API's downstream poll loop works without special-casing."""

    id: int
    device: str
    status: str = "submitted"
    _mgr: Any = field(default=None, repr=False)
    async_result: bool = True
    task_info: Optional[Dict[str, Any]] = None


def _build_task_dict(
    *, chip: str, source: str, shots: int, opts: Dict[str, Any]
) -> Dict[str, Any]:
    """Construct the payload Quafu's /task/run endpoint expects.

    Pops keys from `opts` so the caller can pass any leftover kwargs through
    without conflicting with internal options. Whatever the user did not set
    explicitly we leave at upstream's defaults (False / None / []).
    """
    return {
        "chip": chip,
        "name": opts.pop("task_name", "TyxonQJob"),
        "circuit": source,
        "shots": int(shots),
        # Ask the server to skip its own compiler by default; TyxonQ's
        # Circuit.compile() already produced runnable QASM. Users can opt in to
        # server-side recompilation via opts={"compiler": "qsteed"} etc.
        "compile": opts.pop("compile", True),
        "options": {
            "compiler": opts.pop("compiler", None),
            "correct": opts.pop("correct", False),
            "open_dd": opts.pop("open_dd", None),
            "target_qubits": opts.pop("target_qubits", []),
        },
    }


def _validate_qasm(src: Any) -> None:
    if not src:
        raise ValueError("Quafu driver requires a source (OpenQASM 2.0 string)")
    if not isinstance(src, str):
        raise TypeError(
            f"Quafu source must be a str (OpenQASM 2.0), got {type(src).__name__}"
        )
    if "OPENQASM 2.0" not in src:
        raise ValueError(
            "Quafu source must be OpenQASM 2.0 (header 'OPENQASM 2.0;' required)"
        )


def run(
    device: str,
    token: Optional[str] = None,
    *,
    source: Optional[Union[str, Sequence[str]]] = None,
    shots: Union[int, Sequence[int]] = 1024,
    **opts: Any,
) -> List[QuafuTask]:
    """Submit one or more OpenQASM 2.0 circuits to the Quafu cloud.

    Args:
        device: Chip name. May be `"Dongling"` or `"quafu::Dongling"`.
        token: Optional explicit token. If None, falls through the resolution
            chain in `_resolve_token`.
        source: OpenQASM 2.0 string, or a list of such strings for batch
            submission. Each must include the 'OPENQASM 2.0' header.
        shots: Per-task shot count. Quafu requires multiples of 1024; the
            driver warns (does not error) on other values.
        **opts: Options forwarded into the task dict. Recognized keys:
            `task_name`, `compiler` ({None|'quarkcircuit'|'qsteed'|'qiskit'}),
            `correct` (readout error correction, bool), `open_dd` (dynamical
            decoupling, {None|'XY4'|'CPMG'}), `target_qubits` (list[int]),
            `compile` (bool, defaults to True — pass False to bypass the
            server-side recompiler).

    Returns:
        List of QuafuTask handles, one per submitted source.
    """
    chip = device.split("::")[-1] if "::" in device else device
    resolved_token = _resolve_token(token)

    sources = source if isinstance(source, (list, tuple)) else [source]
    for s in sources:
        _validate_qasm(s)

    if isinstance(shots, (list, tuple)):
        shots_list = [int(s) for s in shots]
        if len(shots_list) != len(sources):
            raise ValueError(
                f"shots list length {len(shots_list)} != sources length {len(sources)}"
            )
    else:
        shots_list = [int(shots)] * len(sources)

    for s in shots_list:
        if s <= 0:
            raise ValueError(f"shots must be positive, got {s}")
        if s % 1024 != 0:
            warnings.warn(
                f"Quafu requires shots to be a multiple of 1024 (got {s}); "
                "the server may round or reject the request.",
                UserWarning,
                stacklevel=2,
            )

    mgr = _QuafuTaskMgr(resolved_token)

    tasks: List[QuafuTask] = []
    for src, sh in zip(sources, shots_list):
        # Each submission gets a fresh copy of opts so popped keys don't
        # bleed into the next iteration.
        task_opts = dict(opts)
        task_dict = _build_task_dict(chip=chip, source=src, shots=sh, opts=task_opts)
        tid = mgr.run(task_dict)
        if not isinstance(tid, int):
            raise RuntimeError(f"Quafu submission failed: server returned {tid!r}")
        tasks.append(
            QuafuTask(
                id=tid,
                device=chip,
                status="submitted",
                _mgr=mgr,
                async_result=True,
                task_info=task_dict,
            )
        )

    return tasks


_STATUS_MAP: Dict[str, str] = {
    "Finished": "completed",
    "Failed": "failed",
    "Running": "running",
    "Pending": "queued",
    "Queued": "queued",
}


def _map_status(s: Any) -> str:
    if not isinstance(s, str):
        return "unknown"
    return _STATUS_MAP.get(s, "unknown")


def get_task_details(task: QuafuTask, token: Optional[str] = None) -> Dict[str, Any]:
    """Fetch a task's result and normalize to TyxonQ's unified result shape.

    The unified shape (matches qcos / tyxonq drivers) is:
        {"result": dict[bitstring, count],
         "result_meta": {"shots": int, "device": str, "tid": int, "raw": dict},
         "uni_status": str,
         "error": str}
    """
    if task._mgr is None:
        raise RuntimeError(
            "QuafuTask has no manager; was it constructed via driver.run()?"
        )
    raw = task._mgr.result(task.id)
    if not isinstance(raw, dict):
        return {
            "result": {},
            "result_meta": {"tid": task.id, "device": task.device, "raw": raw},
            "uni_status": "unknown",
            "error": f"unexpected response: {raw!r}",
        }

    counts = raw.get("count", {}) or {}
    return {
        "result": counts,
        "result_meta": {
            "shots": sum(counts.values()) if counts else None,
            "device": task.device,
            "tid": task.id,
            "raw": raw,
        },
        "uni_status": _map_status(raw.get("status")),
        "error": raw.get("error", ""),
    }


logger = logging.getLogger(__name__)


def list_devices(token: Optional[str] = None, **kws: Any) -> List[str]:
    """Return online Quafu chips, prefixed `quafu::`.

    Defensive: if no token is configured, returns [] rather than raising.
    Matches the qcos driver's behavior so this can be called eagerly during
    device discovery without crashing.
    """
    try:
        resolved_token = _resolve_token(token)
    except RuntimeError as e:
        logger.warning(f"Quafu list_devices: {e}")
        return []

    try:
        mgr = _QuafuTaskMgr(resolved_token)
        chips = mgr.status(0)
    except Exception as e:
        logger.warning(f"Quafu list_devices: status() failed: {e}")
        return []

    if not isinstance(chips, dict):
        return []
    return [f"quafu::{name}" for name, depth in chips.items() if depth != "Offline"]


def cancel(task: QuafuTask, token: Optional[str] = None) -> Dict[str, Any]:
    """Cancel a queued Quafu task. Returns the upstream response verbatim."""
    if task._mgr is None:
        raise RuntimeError(
            "QuafuTask has no manager; was it constructed via driver.run()?"
        )
    return task._mgr.cancel(task.id)
