from __future__ import annotations

from typing import Any
import numpy as np

def _einsum_backend(backend: Any, spec: str, *ops: Any) -> np.ndarray:
    einsum = getattr(backend, "einsum", None)
    if callable(einsum):
        try:
            asarray = getattr(backend, "asarray", None)
            to_numpy = getattr(backend, "to_numpy", None)
            tops = [asarray(x) if callable(asarray) else x for x in ops]
            res = einsum(spec, *tops)
            return to_numpy(res) if callable(to_numpy) else np.asarray(res)
        except Exception:
            pass
    return np.einsum(spec, *ops)


def normalize(vec: Any) -> Any:
    try:
        import numpy as _np
        nrm = _np.linalg.norm(vec)
        if nrm > 0:
            return vec / nrm
        return vec
    except Exception:
        return vec


def kron(a: Any, b: Any) -> Any:
    return np.kron(a, b)


def ensure_complex(x: Any, dtype=np.complex128) -> Any:
    arr = np.asarray(x)
    if not np.iscomplexobj(arr):
        arr = arr.astype(dtype)
    return arr


