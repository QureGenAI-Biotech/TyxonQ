__version__ = "0.1.0"
__author__ = "TyxonQ"

# Forked from https://github.com/tencent-quantum-lab/TenCirChem 
# Reconstruction IS IN PROGRESS


#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.

import os
import logging

os.environ["JAX_ENABLE_X64"] = "True"
# for debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# disable CUDA 11.1 warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = logging.getLogger("tyxonq")
logger.setLevel(logging.FATAL)

os.environ["RENO_LOG_LEVEL"] = "100"

logger = logging.getLogger("tencirchem")
logger.setLevel(logging.WARNING)

# finish logger stuff
del logger

from .utils.backend import set_backend, set_dtype

# by default use float64 rather than float32
set_dtype("complex128")

# static module
# as an external interface
from pyscf import M

from .static.ucc import UCC
from .static.uccsd import UCCSD, ROUCCSD
from .static.kupccgsd import KUPCCGSD
from .static.puccd import PUCCD
from .static.hea import HEA, parity, binary, get_noise_conf

# dynamic module
# as an external interface
from renormalizer import Op, BasisSHO, BasisHalfSpin, BasisSimpleElectron, BasisMultiElectron, Model, Mpo
from renormalizer.model import OpSum

from .utils.misc import get_dense_operator
from .dynamic.time_evolution import TimeEvolution


def clear_cache():
    from .utils.backend import ALL_JIT_LIBS
    from .static.evolve_civector import CI_OPERATOR_CACHE, CI_OPERATOR_BATCH_CACHE

    for l in ALL_JIT_LIBS:
        l.clear()
    CI_OPERATOR_CACHE.clear()
    CI_OPERATOR_BATCH_CACHE.clear()
