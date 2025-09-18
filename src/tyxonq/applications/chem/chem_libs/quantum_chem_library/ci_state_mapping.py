from functools import partial
import numpy as np
from pyscf.fci import cistring
import tyxonq as tq

from tyxonq.numerics import NumericBackend as nb


def jit(f, static_argnums=None):
    return nb.jit(f)


def tensor_set_elem(tensor, idx, elem):
    tensor = nb.asarray(tensor)
    tensor[idx] = elem
    return tensor


def get_xp(_backend):
    # Safe backend-to-arraylib selector with numpy fallback when backend is unset
    try:
        import cupy as cp  # type: ignore
        if getattr(_backend, "name", "") == "cupy":
            return cp
    except Exception:
        pass
    import numpy as _np
    return _np


def get_uint_type():
    import numpy as _np
    return _np.uint64 if getattr(tq, "rdtypestr", "float64") == "float64" else _np.uint32


from tyxonq.libs.circuits_library.utils import unpack_nelec


def get_ci_strings(n_qubits, n_elec_s, mode, strs2addr=False):
    """Return CI basis bitstrings for given particle numbers and mode.

    Parameters
    ----------
    n_qubits: int
        Total number of qubits (spin-orbitals for fermion/qubit; sites for hcb).
    n_elec_s: tuple[int,int] | int
        (na, nb) for fermion/qubit, or n for hcb.
    mode: str | bool
        "fermion" | "qubit" | "hcb"; also accepts a boolean for backward tests
        where True means hcb and False means fermion/qubit.
    strs2addr: bool
        Whether to also return a mapping from bitstring to address.
    """
    # Allow calling without global tq.backend set
    try:
        bk = getattr(tq, "backend")
    except Exception:
        bk = None
    xp = get_xp(bk)
    uint_type = get_uint_type()
    if 2 ** n_qubits > np.iinfo(uint_type).max:
        raise ValueError(f"Too many qubits: {n_qubits}, try using complex128 datatype")
    # Normalize mode: accept boolean flag for tests (True→"hcb", False→non-hcb)
    is_hcb = False
    if isinstance(mode, bool):
        is_hcb = mode
    else:
        is_hcb = (mode == "hcb")

    na, nb = unpack_nelec(n_elec_s)
    if not is_hcb:
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
        # hcb mode (paired/spinless)
        assert na == nb
        ci_strings = cistring.make_strings(range(n_qubits), na).astype(uint_type)
        if strs2addr:
            strs2addr = xp.zeros(2 ** n_qubits, dtype=uint_type)
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
    """Map CI coefficients to full statevector.

    后端未设置时，使用 NumPy 构造，避免依赖 tq.backend/rdtypestr。
    """
    statevector = np.zeros(1 << n_qubits, dtype=np.complex128)
    statevector[np.asarray(ci_strings, dtype=int)] = np.asarray(civector, dtype=np.complex128)
    return statevector


def statevector_to_civector(statevector, ci_strings):
    return statevector[ci_strings]


@partial(jit, static_argnums=[0])
def get_init_civector(len_ci):
    # Robust to missing global backend: fall back to numpy
    import numpy as _np
    civector = _np.zeros(len_ci, dtype=_np.float64)
    civector[0] = 1
    return civector


