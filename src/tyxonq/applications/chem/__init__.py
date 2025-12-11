__version__ = "0.1.0"
__author__ = "TyxonQ"

# ReWrite TenCirChem with TyxonQ




import os
import logging
import numpy as np

# for debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


logger = logging.getLogger("tyxonq")
logger.setLevel(logging.FATAL)

os.environ["RENO_LOG_LEVEL"] = "100"

logger = logging.getLogger("tyxonq.chem")
logger.setLevel(logging.WARNING)

# finish logger stuff
del logger


# New algorithms API (device runtime by default)
from .algorithms import HEA, UCC  # noqa: F401

# Re-export numerics backend helpers for tests
try:
    from tyxonq import set_backend  # noqa: F401
except Exception:
    pass

# Legacy static API re-exports for tests during migration
try:
    from .algorithms.uccsd import UCCSD, ROUCCSD  # noqa: F401
except Exception:
    pass
try:
    from .algorithms.ucc import UCC  # noqa: F401
except Exception:
    pass
try:
    from .algorithms.kupccgsd import KUPCCGSD  # noqa: F401
except Exception:
    pass
try:
    from .algorithms.puccd import PUCCD  # noqa: F401
except Exception:
    pass
try:
    from .algorithms.hea import parity  # noqa: F401
except Exception:
    pass

def clear_cache() -> None:
    """Clear chem-level internal caches (no-op fallback).

    Tests call this between runs to ensure deterministic behavior. Implement as
    best-effort: try clearing optional caches if modules expose them; otherwise
    remain a no-op.
    """
    try:
        # example: optional caches in chem submodules
        from .chem_libs.quantum_chem_library import ci_state_mapping as _cism  # type: ignore

        for name in ("_CACHE", "CACHE", "cache"):
            c = getattr(_cism, name, None)
            if hasattr(c, "clear"):
                c.clear()
    except Exception:
        pass

# Extend OpenFermion numeric coefficient acceptance to include NumPy scalars
try:
    from openfermion.ops.operators import symbolic_operator as _sym
    if hasattr(_sym, "COEFFICIENT_TYPES"):
        _numpy_num_types = (np.number, np.integer, np.floating, np.complexfloating)
        try:
            # Merge without duplicates
            _sym.COEFFICIENT_TYPES = tuple({*tuple(_sym.COEFFICIENT_TYPES), *_numpy_num_types})  # type: ignore[attr-defined]
        except Exception:
            _sym.COEFFICIENT_TYPES = tuple(list(_sym.COEFFICIENT_TYPES) + list(_numpy_num_types))  # type: ignore[attr-defined]
except Exception:
    pass


__all__ = [
    "set_backend",
    "clear_cache",
    "HEA",
    "UCC",
    "UCCSD",
    "ROUCCSD",
    "KUPCCGSD",
    "PUCCD",
    "parity",
]
