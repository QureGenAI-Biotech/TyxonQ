#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from typing import Any, List, Dict
from functools import wraps
import logging

import numpy as np
import tyxonq as tq

try:
    import jax

    JAXIMPORTERROR = None
except ImportError as e:
    jax = None
    JAXIMPORTERROR = e

try:
    import cupy as cp
except ImportError:
    cp = None

Tensor = Any

logger = logging.getLogger(__name__)


ALL_JIT_LIBS: List[Dict] = []


def jit(f, static_argnums=None):
    backend_jit_lib = {}

    @wraps(f)
    def _wrap_jit(*args, **kwargs):
        if tq.backend.name not in backend_jit_lib:
            backend_jit_lib[tq.backend.name] = tq.backend.jit(f, static_argnums=static_argnums)
        return backend_jit_lib[tq.backend.name](*args, **kwargs)

    ALL_JIT_LIBS.append(backend_jit_lib)
    return _wrap_jit


def value_and_grad(f, argnums=0, has_aux=False):
    backend_vg_lib = {}

    @wraps(f)
    def _wrap_value_and_grad(*args, **kwargs):
        if tq.backend.name not in backend_vg_lib:
            try:
                vg_func = tq.backend.value_and_grad(f, argnums=argnums, has_aux=has_aux)
            except NotImplementedError:

                def vg_func(*args, **kwargs):
                    msg = (
                        f"The engine requires auto-differentiation, "
                        f"which is not available for backend {tq.backend.name}"
                    )
                    raise NotImplementedError(msg)

            backend_vg_lib[tq.backend.name] = vg_func
        return backend_vg_lib[tq.backend.name](*args, **kwargs)

    ALL_JIT_LIBS.append(backend_vg_lib)
    return _wrap_value_and_grad


set_backend = tq.set_backend


set_dtype = tq.set_dtype


def tensor_set_elem(tensor, idx, elem):
    if tq.backend.name == "jax":
        return tensor.at[idx].set(elem)
    else:
        assert tq.backend.name == "numpy" or tq.backend.name == "cupy"
        tensor[idx] = elem
        return tensor


def fori_loop(lower, upper, body_fun, init_val):
    if tq.backend.name == "jax":
        if jax is None:
            raise JAXIMPORTERROR
        return jax.lax.fori_loop(lower, upper, body_fun, init_val)

    assert tq.backend.name == "numpy" or tq.backend.name == "cupy"
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def scan(f, init, xs, length=None, reverse=False, unroll=1):
    if tq.backend.name == "jax":
        if jax is None:
            raise JAXIMPORTERROR
        return jax.lax.scan(f, init, xs, length, reverse, unroll)

    assert tq.backend.name == "numpy" or tq.backend.name == "cupy"
    carry = init
    ys = []
    xs = zip(*xs)
    if reverse:
        xs = reversed(list(xs))
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(list(reversed(ys)))


def get_xp(_backend):
    if _backend.name == "cupy":
        return cp
    else:
        return np


def get_uint_type():
    if tq.rdtypestr == "float64":
        return np.uint64
    else:
        assert tq.rdtypestr == "float32"
        return np.uint32
