"""TyxonQ driver for the BAQIS Quafu Superconducting Quantum Cloud.

Wraps the vendored REST client in `_vendor_quafu.py` to fit TyxonQ's
provider-driver contract (run / get_task_details / list_devices / cancel).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from ..config import get_token as _hw_get_token


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
