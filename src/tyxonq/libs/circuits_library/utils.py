from __future__ import annotations

from typing import List, Tuple, Sequence, Callable

from openfermion import FermionOperator, QubitOperator
from openfermion.utils import hermitian_conjugated
from tyxonq.libs.hamiltonian_encoding.pauli_io import (
    reverse_qop_idx as _reverse_qop_idx,
    ex_op_to_fop as _ex_op_to_fop,
)

try:
    import tyxonq as tq
except Exception:
    tq = None  # type: ignore


def unpack_nelec(n_elec_s):
    if isinstance(n_elec_s, tuple):
        na, nb = n_elec_s
    elif isinstance(n_elec_s, int):
        na, nb = n_elec_s // 2, n_elec_s // 2
    else:
        raise TypeError(f"Unknown electron number specification: {n_elec_s}")
    return na, nb



def evolve_pauli_ops(pauli_string: Tuple[Tuple[int, str], ...], theta: float) -> List[Tuple]:
    ops: List[Tuple] = []
    for idx, symbol in pauli_string:
        if symbol == "X":
            ops.append(("h", idx))
        elif symbol == "Y":
            ops.append(("sdg", idx))
            ops.append(("h", idx))
        elif symbol == "Z":
            continue
        else:
            raise ValueError(f"Invalid Pauli symbol in {pauli_string}")

    for i in range(len(pauli_string) - 1):
        ops.append(("cx", pauli_string[i][0], pauli_string[i + 1][0]))
    ops.append(("rz", pauli_string[-1][0], theta))

    for i in range(len(pauli_string) - 1)[::-1]:
        ops.append(("cx", pauli_string[i][0], pauli_string[i + 1][0]))

    for idx, symbol in pauli_string:
        if symbol == "X":
            ops.append(("h", idx))
        elif symbol == "Y":
            ops.append(("h", idx))
            ops.append(("s", idx))
        elif symbol == "Z":
            continue
        else:
            raise ValueError(f"Invalid Pauli symbol in {pauli_string}")
    return ops


def build_multicontrol_ry_ops(i: int, j: int, k: int, l: int, theta: float, *, prefer_subcircuit: bool = True) -> List[Tuple]:
    """Return ops for multi-control RY on wires (i,j,k,l).

    - If prefer_subcircuit and IR runtime recognizes subcircuit, emit a single
      marker tuple ("subcircuit_multicontrol_ry", i, j, k, l, theta) for later lowering.
    - Otherwise return an inlined decomposition into base gates.
    """
    if prefer_subcircuit and _ir_supports_subcircuit():
        return [("subcircuit_multicontrol_ry", i, j, k, l, theta)]

    ops: List[Tuple] = []
    ops.append(("x", i))
    ops.append(("x", k))

    ops.append(("ry", l, theta / 8))
    ops.append(("h", k))
    ops.append(("cx", l, k))

    ops.append(("ry", l, -theta / 8))
    ops.append(("h", i))
    ops.append(("cx", l, i))

    ops.append(("ry", l, theta / 8))
    ops.append(("cx", l, k))

    ops.append(("ry", l, -theta / 8))
    ops.append(("h", j))
    ops.append(("cx", l, j))

    ops.append(("ry", l, theta / 8))
    ops.append(("cx", l, k))

    ops.append(("ry", l, -theta / 8))
    ops.append(("cx", l, i))

    ops.append(("ry", l, theta / 8))
    ops.append(("h", i))
    ops.append(("cx", l, k))

    ops.append(("ry", l, -theta / 8))
    ops.append(("h", k))
    ops.append(("cx", l, j))
    ops.append(("h", j))

    ops.append(("x", i))
    ops.append(("x", k))
    return ops


def _ir_supports_subcircuit() -> bool:
    # 简单能力探测：当前我们缺省关闭，后续可通过全局配置或 runtime flag 打开
    #TODO IF hardware and compiler support subcircuit  then we can make judgement here and return True
    return False




def multicontrol_ry(theta):
    """Decomposed multi-control RY on 4 wires as a reusable subcircuit.

    This mirrors the implementation used in chem.utils.circuit, but placed here
    as a reusable low-level building block for circuits_library.
    """
    # https://arxiv.org/pdf/2005.14475.pdf
    if tq is None:
        raise RuntimeError("tyxonq not available for building subcircuit")
    c = tq.Circuit(4)
    i, j, k, l = 0, 1, 2, 3

    c.x(i)
    c.x(k)

    c.ry(l, theta=theta / 8)
    c.h(k)
    c.cnot(l, k)

    c.ry(l, theta=-theta / 8)
    c.h(i)
    c.cnot(l, i)

    c.ry(l, theta=theta / 8)
    c.cnot(l, k)

    c.ry(l, theta=-theta / 8)
    c.h(j)
    c.cnot(l, j)

    c.ry(l, theta=theta / 8)
    c.cnot(l, k)

    c.ry(l, theta=-theta / 8)
    c.cnot(l, i)

    c.ry(l, theta=theta / 8)
    c.h(i)
    c.cnot(l, k)

    # typo fixed vs paper
    # there's a typo in the paper
    # https://github.com/dibyendu/uccsd/issues/134
    c.ry(l, theta=-theta / 8)
    c.h(k)
    c.cnot(l, j)
    c.h(j)

    c.x(i)
    c.x(k)
    return c




__all__ = [
    "unpack_nelec",
    "evolve_pauli_ops",
]




