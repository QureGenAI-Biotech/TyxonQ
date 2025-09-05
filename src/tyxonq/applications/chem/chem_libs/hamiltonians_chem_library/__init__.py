"""
chem.hamiltonians_lib

领域系统的哈密顿量与基底定义集合（如 Pyrazine、SBM 等）。
这些模块仅包含参数与构造函数，不引入运行时依赖；
算符到量子比特的编码/变换请使用 `libs.operator_library`。
"""

from . import pyrazine  # noqa: F401
from . import sbm  # noqa: F401
from . import hamiltonian_builders  # noqa: F401
from .hamiltonian_builders import (  # noqa: F401
    get_integral_from_hf,
    get_hop_from_integral,
    qubit_operator,
    get_hop_hcb_from_integral,
    get_h_sparse_from_integral,
    get_h_fcifunc_from_integral,
    get_h_fcifunc_hcb_from_integral,
    get_h_from_integral,
    random_integral,
)

__all__ = [
    "pyrazine",
    "sbm",
    "hamiltonian_builders",
    # direct exports for convenience
    "get_integral_from_hf",
    "get_hop_from_integral",
    "qubit_operator",
    "get_hop_hcb_from_integral",
    "get_h_sparse_from_integral",
    "get_h_fcifunc_from_integral",
    "get_h_fcifunc_hcb_from_integral",
    "get_h_from_integral",
    "random_integral",
]


