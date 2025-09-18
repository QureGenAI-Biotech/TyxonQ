from __future__ import annotations

from functools import wraps
from typing import Any, Callable
import numpy as np
from tyxonq.numerics import NumericBackend as nb


def scipy_opt_wrap(f: Callable[..., Any], gradient: bool = True) -> Callable[..., Any]:
    """Wrap a function to be friendly to SciPy optimizers.

    Ensures inputs/outputs are numpy float64 arrays and supports gradient tuples.
    """

    @wraps(f)
    def _wrap_scipy_opt(_params, *args):
        params64 = np.asarray(_params, dtype=np.float64)
        res = f(nb.asarray(params64), *args)
        if gradient:
            # Expect (value, grad) or iterable of grads
            try:
                value, grad = res
                value = float(np.asarray(value, dtype=np.float64))
                grad = np.asarray(grad, dtype=np.float64)
                return value, grad
            except Exception:
                return [np.asarray(v, dtype=np.float64) for v in res]
        return np.asarray(res, dtype=np.float64)

    return _wrap_scipy_opt


__all__ = ["scipy_opt_wrap"]


