


from functools import partial

import numpy as np
from pyscf.fci import cistring
import tyxonq as tq

from ..utils.backend import jit, tensor_set_elem, get_xp, get_uint_type
from ..utils.misc import unpack_nelec


def get_ci_strings(n_qubits, n_elec_s, mode, strs2addr=False):
    xp = get_xp(tq.backend)
    uint_type = get_uint_type()
    if 2**n_qubits > np.iinfo(uint_type).max:
        raise ValueError(f"Too many qubits: {n_qubits}, try using complex128 datatype")
    na, nb = unpack_nelec(n_elec_s)
    if mode in ["fermion", "qubit"]:
        beta = cistring.make_strings(range(n_qubits // 2), nb)
        beta = xp.array(beta, dtype=uint_type)
        if na == nb:
            alpha = beta
        else:
            alpha = cistring.make_strings(range(n_qubits // 2), na)
            alpha = xp.array(alpha, dtype=uint_type)
        ci_strings = ((alpha << (n_qubits // 2)).reshape(-1, 1) + beta.reshape(1, -1)).ravel()
        if strs2addr:
            if na == nb:
                strs2addr = xp.zeros(2 ** (n_qubits // 2), dtype=uint_type)
                strs2addr[beta] = xp.arange(len(beta))
            else:
                strs2addr = xp.zeros((2, 2 ** (n_qubits // 2)), dtype=uint_type)
                strs2addr[0][alpha] = xp.arange(len(alpha))
                strs2addr[1][beta] = xp.arange(len(beta))
            return ci_strings, strs2addr
    else:
        assert mode == "hcb"
        assert na == nb
        ci_strings = cistring.make_strings(range(n_qubits), na).astype(uint_type)
        if strs2addr:
            strs2addr = xp.zeros(2**n_qubits, dtype=uint_type)
            strs2addr[ci_strings] = xp.arange(len(ci_strings))
            return ci_strings, strs2addr

    return ci_strings


def get_addr(excitation, n_qubits, n_elec_s, strs2addr, mode, num_strings=None):
    if mode == "hcb":
        return strs2addr[excitation]
    assert mode in ["fermion", "qubit"]
    alpha = excitation >> (n_qubits // 2)
    beta = excitation & (2 ** (n_qubits // 2) - 1)
    na, nb = n_elec_s
    if na == nb:
        alpha_addr = strs2addr[alpha]
        beta_addr = strs2addr[beta]
    else:
        alpha_addr = strs2addr[0][alpha]
        beta_addr = strs2addr[1][beta]
    if num_strings is None:
        num_strings = cistring.num_strings(n_qubits // 2, nb)
    return alpha_addr * num_strings + beta_addr


def get_ex_bitstring(n_qubits, n_elec_s, ex_op, mode):
    na, nb = n_elec_s
    if mode in ["fermion", "qubit"]:
        bitstring_basea = ["0"] * (n_qubits // 2 - na) + ["1"] * na
        bitstring_baseb = ["0"] * (n_qubits // 2 - nb) + ["1"] * nb
        bitstring_base = bitstring_basea + bitstring_baseb
    else:
        assert mode == "hcb"
        assert na == nb
        bitstring_base = ["0"] * (n_qubits - na) + ["1"] * na

    bitstring = bitstring_base.copy()[::-1]
    # first annihilation then creation
    if len(ex_op) == 2:
        bitstring[ex_op[1]] = "0"
        bitstring[ex_op[0]] = "1"
    else:
        assert len(ex_op) == 4
        bitstring[ex_op[3]] = "0"
        bitstring[ex_op[2]] = "0"
        bitstring[ex_op[1]] = "1"
        bitstring[ex_op[0]] = "1"

    return "".join(reversed(bitstring))


def civector_to_statevector(civector, n_qubits, ci_strings):
    statevector = tq.backend.zeros(2**n_qubits, dtype=tq.rdtypestr)
    return tensor_set_elem(statevector, ci_strings, civector)


def statevector_to_civector(statevector, ci_strings):
    return statevector[ci_strings]


@partial(jit, static_argnums=[0])
def get_init_civector(len_ci):
    civector = tq.backend.zeros(len_ci, dtype=tq.rdtypestr)
    civector = tensor_set_elem(civector, 0, 1)
    return civector
