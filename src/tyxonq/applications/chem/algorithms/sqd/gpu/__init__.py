"""GPU-accelerated selected CI for SQD.

The unified entry points are :func:`solve_sci_gpu` and
:func:`make_gpu_sci_solver`. Legacy callable aliases remain available; the old
``torch_*`` module paths were removed.
"""

from __future__ import annotations

from .api import make_gpu_sci_solver, solve_sci_gpu

__all__ = [
    "make_cuda_sci_solver",
    "make_gpu_sci_solver",
    "make_sbd_sci_solver",
    "solve_sci_cuda",
    "solve_sci_gpu",
    "solve_sci_sbd",
]


def __getattr__(name: str):
    from .compat import (
        make_cuda_sci_solver,
        make_sbd_sci_solver,
        solve_sci_cuda,
        solve_sci_sbd,
    )

    aliases = {
        "make_cuda_sci_solver": make_cuda_sci_solver,
        "make_sbd_sci_solver": make_sbd_sci_solver,
        "solve_sci_cuda": solve_sci_cuda,
        "solve_sci_sbd": solve_sci_sbd,
    }
    if name in aliases:
        return aliases[name]
    raise AttributeError(name)
