"""CUDA selected-CI 求解器；实际求解时才加载 PyTorch。"""

from .batch import make_cuda_sci_solver, make_sbd_sci_solver

__all__ = ["solve_sci_cuda", "make_cuda_sci_solver", "make_sbd_sci_solver"]


def __getattr__(name):
    if name == "solve_sci_cuda":
        from .torch_sci_solver import solve_sci_cuda

        return solve_sci_cuda
    raise AttributeError(name)
