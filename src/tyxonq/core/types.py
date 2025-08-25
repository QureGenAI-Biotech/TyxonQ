from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

@dataclass(frozen=True)
class Problem:
    """Domain problem wrapper for input to app/compilers.

    Fields:
        kind: Problem category (e.g., "hamiltonian", "circuit").
        payload: Arbitrary structured data describing the problem.
    """

    kind: Literal["hamiltonian", "circuit", "pulse", "custom"]
    payload: Dict[str, Any]


