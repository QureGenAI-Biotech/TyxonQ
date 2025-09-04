__version__ = "0.1.0"
__author__ = "TyxonQ"

# ReWrite TenCirChem with TyxonQ




import os
import logging

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

# Legacy static API re-exports for tests during migration
try:
    from .algorithms.uccsd import UCCSD  # noqa: F401
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

__all__ = [
    "set_backend",
    "HEA",
    "UCC",
    "UCCSD",
    "KUPCCGSD",
    "PUCCD",
    "parity",
]
