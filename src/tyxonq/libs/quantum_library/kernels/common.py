from __future__ import annotations

from typing import Any, Sequence, Callable
import numpy as np
from ....numerics.api import get_backend, ArrayBackend

def _einsum_backend(backend: Any, spec: str, *ops: Any) -> Any:
    # Assume backend provides required methods; no feature checks
    tops = [backend.asarray(x) for x in ops]
    return backend.einsum(spec, *tops)
def parameter_shift_gradient(energy_fn: Callable[[np.ndarray], float], params: Sequence[float]) -> np.ndarray:
    base = np.asarray(params, dtype=np.float64)
    if base.ndim != 1:
        raise ValueError("params must be 1-D")
    s = 0.5 * np.pi
    g = np.zeros_like(base)
    for i in range(len(base)):
        p_plus = base.copy(); p_plus[i] += s
        p_minus = base.copy(); p_minus[i] -= s
        e_plus = float(energy_fn(p_plus))
        e_minus = float(energy_fn(p_minus))
        g[i] = 0.5 * (e_plus - e_minus)
    return g

__all__ = ["parameter_shift_gradient"]

