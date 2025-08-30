from __future__ import annotations

from typing import Any
import numpy as np
from ....numerics.api import get_backend, ArrayBackend

def _einsum_backend(backend: Any, spec: str, *ops: Any) -> Any:
    # Assume backend provides required methods; no feature checks
    tops = [backend.asarray(x) for x in ops]
    return backend.einsum(spec, *tops)






