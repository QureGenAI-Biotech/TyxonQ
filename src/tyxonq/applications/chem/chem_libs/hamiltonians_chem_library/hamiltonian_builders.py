from __future__ import annotations

from itertools import product
from typing import Tuple, List

import numpy as np
from openfermion import FermionOperator, QubitOperator
from pyscf.scf.hf import RHF
from pyscf.mcscf import CASCI
from pyscf.fci import direct_nosym, direct_spin1, cistring
from pyscf import ao2mo

from tyxonq.libs.hamiltonian_encoding.pauli_io import hcb_to_coo, fop_to_coo, reverse_qop_idx, canonical_mo_coeff
from openfermion.transforms import jordan_wigner as _jw_transform

import tyxonq as tq
from tyxonq.numerics import NumericBackend as nb


class _MPOWrapper:
    """Lightweight wrapper to mimic MPO interface used in tests.

    Provides eval_matrix() that returns dense numpy matrix from a scipy.sparse COO/CSC input.
    """

    def __init__(self, sparse):
        self._sparse = sparse

    def eval_matrix(self):
        from scipy.sparse import issparse
        if issparse(self._sparse):
            return np.asarray(self._sparse.todense())
        return np.asarray(self._sparse)


def mpo_to_quoperator(mpo_like):
    """Compatibility shim expected by tests: return an object exposing eval_matrix().

    If input already has eval_matrix, pass-through; otherwise wrap sparse into _MPOWrapper.
    """
    if hasattr(mpo_like, "eval_matrix") and callable(getattr(mpo_like, "eval_matrix")):
        return mpo_like
    return _MPOWrapper(mpo_like)


def get_integral_from_hf(hf: RHF, active_space: Tuple = None, active_orbital_indices: List[int] = None):
    if not isinstance(hf, RHF):
        raise TypeError(f"hf object must be RHF class, got {type(hf)}")
    m = hf.mol
    assert hf.mo_coeff is not None
    hf.mo_coeff = canonical_mo_coeff(hf.mo_coeff)

    if active_space is None:
        nelecas = m.nelectron
        ncas = m.nao
    else:
        nelecas, ncas = active_space

    casci = CASCI(hf, ncas, nelecas)
    if active_orbital_indices is None:
        int1e, e_core = casci.get_h1eff()
        int2e = ao2mo.restore("s1", casci.get_h2eff(), ncas)
    else:
        mo = casci.sort_mo(active_orbital_indices, base=0)
        int1e, e_core = casci.get_h1eff(mo)
        int2e = ao2mo.restore("s1", casci.get_h2eff(mo), ncas)

    return int1e, int2e, e_core


def get_hop_from_integral(int1e, int2e):
    n_orb = int1e.shape[0]
    if int1e.shape != (n_orb, n_orb):
        raise ValueError(f"Invalid one-boby integral array shape: {int1e.shape}")
    int2e = ao2mo.restore(1, int2e, n_orb)
    assert int2e.shape == (n_orb, n_orb, n_orb, n_orb)
    n_sorb = n_orb * 2

    h1e = np.zeros((n_sorb, n_sorb))
    h2e = np.zeros((n_sorb, n_sorb, n_sorb, n_sorb))

    h1e[:n_orb, :n_orb] = h1e[n_orb:, n_orb:] = int1e

    for p, q, r, s in product(range(n_sorb), repeat=4):
        if ((p < n_orb) == (s < n_orb)) and ((q < n_orb) == (r < n_orb)):
            h2e[p, q, r, s] = int2e[p % n_orb, s % n_orb, q % n_orb, r % n_orb]

    op1e: List[FermionOperator] = []
    for p, q in product(range(n_sorb), repeat=2):
        v = h1e[p, q]
        if np.abs(v) < 1e-12:
            continue
        op = FermionOperator(f"{p}^ {q}", v)
        op1e.append(op)

    op2e: List[FermionOperator] = []
    for q, s in product(range(n_sorb), repeat=2):
        for p, r in product(range(q), range(s)):
            v = h2e[p, q, r, s] - h2e[q, p, r, s]
            if np.abs(v) < 1e-12:
                continue
            op = FermionOperator(f"{p}^ {q}^ {r} {s}", v)
            op2e.append(op)

    ops = FermionOperator()
    for op in op1e + op2e:
        ops += op
    return ops


def qubit_operator(string: str, coeff: float) -> QubitOperator:
    ret: QubitOperator | float = coeff
    terms = string.split(" ")
    for term in terms:
        if term[-1] == "^":
            sign = -1
            term = term[:-1]
        else:
            sign = 1
        idx = int(term)
        ret *= (QubitOperator(f"X{idx}") + sign * 1j * QubitOperator(f"Y{idx}")) / 2
    return ret  # type: ignore[return-value]


def get_hop_hcb_from_integral(int1e, int2e):
    # Hard core boson Hamiltonian
    # https://arxiv.org/pdf/2002.00035.pdf
    n_orb = int1e.shape[0]
    qop = QubitOperator()
    for p in range(n_orb):
        for q in range(p + 1):
            if p == q:
                qop += qubit_operator(f"{p}^ {p}", 2 * int1e[p, p] + int2e[p, p, p, p])
            else:
                qop += qubit_operator(f"{p}^ {q}", int2e[p, q, p, q])
                qop += qubit_operator(f"{q}^ {p}", int2e[q, p, q, p])
                qop += qubit_operator(f"{p}^ {p} {q}^ {q}", 4 * int2e[p, p, q, q] - 2 * int2e[p, q, p, q])
    qop = reverse_qop_idx(qop, n_orb)
    return qop


def get_h_sparse_from_integral(int1e, int2e, *, mode: str = "fermion", discard_eps: float = 1e-12):
    if mode in ["fermion", "qubit"]:
        ops = get_hop_from_integral(int1e, int2e)
    else:
        assert mode == "hcb"
        ops = get_hop_hcb_from_integral(int1e, int2e)
    if mode in ["fermion", "qubit"]:
        h_sparse = fop_to_coo(ops, n_qubits=2 * len(int1e))
    else:
        assert mode == "hcb"
        h_sparse = hcb_to_coo(ops, n_qubits=len(int1e))
    return h_sparse


def get_h_mpo_from_integral(int1e, int2e, *, mode: str = "fermion"):
    """Return an MPO-like object for tests.

    Here we wrap the sparse Hamiltonian with a tiny eval_matrix() shim to satisfy tests.
    """
    sparse = get_h_sparse_from_integral(int1e, int2e, mode=mode)
    return _MPOWrapper(sparse)


def get_h_fcifunc_from_integral(int1e, int2e, n_elec):
    """Return CI-space Hamiltonian apply function using PySCF direct_spin1.

    This matches the alpha/beta-separated CI basis order we construct via get_ci_strings.
    """
    n_orb = len(int1e)
    h2e = direct_nosym.absorb_h1e(int1e, int2e, n_orb, n_elec, 0.5)
    def fci_func(civector):
        civector = np.asarray(civector, dtype=np.float64)
        out = direct_nosym.contract_2e(h2e, civector, norb=n_orb, nelec=n_elec)
        return np.asarray(out, dtype=np.float64)
    return fci_func


def get_h_fcifunc_hcb_from_integral(int1e, int2e, n_elec):
    n_orb = len(int1e)
    ci_strings = cistring.make_strings(range(n_orb), n_elec // 2)

    def fci_func(civector):
        res = np.zeros(len(civector), dtype=np.float64)
        for p in range(n_orb):
            for q in range(p + 1):
                if p == q:
                    bitmask = 1 << p
                    arraymask = (ci_strings & bitmask) == bitmask
                    res += (civector * arraymask) * (2 * int1e[p, p] + int2e[p, p, p, p])
                else:
                    bitmask = (1 << p) + (1 << q)
                    excitation = ci_strings ^ bitmask
                    addr = cistring.strs2addr(n_orb, n_elec // 2, excitation)
                    flip = ci_strings ^ (1 << p)
                    masked_flip = flip & bitmask
                    arraymask = (masked_flip == bitmask) | (masked_flip == 0)
                    res += civector[addr] * arraymask * int2e[p, q, p, q]
                    arraymask = (ci_strings & bitmask) == bitmask
                    res += (civector * arraymask) * (4 * int2e[p, p, q, q] - 2 * int2e[p, q, p, q])
        return res

    return fci_func


def get_h_from_integral(int1e, int2e, n_elec_s, mode: str, htype: str):
    htype = htype.lower()
    if htype == "sparse":
        return get_h_sparse_from_integral(int1e, int2e, mode=mode)
    if htype == "mpo":
        return get_h_mpo_from_integral(int1e, int2e, mode=mode)
    assert htype == "fcifunc"
    if mode in ["fermion", "qubit"]:
        return get_h_fcifunc_from_integral(int1e, int2e, n_elec_s)
    # hcb branch: accept int or (na, nb). If tuple, sum to total electrons.
    n_elec = int(n_elec_s) if isinstance(n_elec_s, int) else int(sum(n_elec_s))
    return get_h_fcifunc_hcb_from_integral(int1e, int2e, n_elec)


def pauli_sum_to_qubit_operator(pauli_items: list[tuple[float, list[tuple[str, int]]]], n_qubits: int) -> QubitOperator:
    """Build OpenFermion QubitOperator from internal pauli sum representation.

    Each item is (coeff, [(P, q), ...]) where P in {'X','Y','Z'} and q is int (big-endian handled upstream).
    """
    qop = QubitOperator()
    for coeff, ops in pauli_items:
        if not ops:
            qop += float(coeff)
            continue
        term = tuple((int(q), str(P).upper()) for (P, q) in ops)
        qop += QubitOperator(term, float(coeff))
    return qop


def get_h_from_hf(hf: RHF, *, mode: str = "fermion", htype: str = "sparse", active_space: Tuple[int, int] | None = None, active_orbital_indices: List[int] | None = None):
    """Thin wrapper to preserve legacy import path used in tests.

    Delegates to get_integral_from_hf + get_h_from_integral.
    """
    int1e, int2e, _ = get_integral_from_hf(hf, active_space=active_space, active_orbital_indices=active_orbital_indices)
    if active_space is None:
        n_elec = int(getattr(hf.mol, "nelectron"))
    else:
        n_elec = int(active_space[0])
    return get_h_from_integral(int1e, int2e, n_elec, mode, htype)


__all__ = [
    "get_integral_from_hf",
    "get_hop_from_integral",
    "qubit_operator",
    "get_hop_hcb_from_integral",
    "get_h_sparse_from_integral",
    "get_h_fcifunc_from_integral",
    "get_h_fcifunc_hcb_from_integral",
    "get_h_from_integral",
    "pauli_sum_to_qubit_operator",
]


def random_integral(nao: int, seed: int = 2077):
    np.random.seed(seed)
    int1e = np.random.uniform(-1, 1, size=(nao, nao))
    int2e = np.random.uniform(-1, 1, size=(nao, nao, nao, nao))
    int1e = 0.5 * (int1e + int1e.T)
    int2e = symmetrize_int2e(int2e)
    return int1e, int2e


def symmetrize_int2e(int2e):
    int2e = 0.25 * (
        int2e
        + int2e.transpose((0, 1, 3, 2))
        + int2e.transpose((1, 0, 2, 3))
        + int2e.transpose((2, 3, 0, 1))
    )
    int2e = 0.5 * (int2e + int2e.transpose(3, 2, 1, 0))
    return int2e

from scipy.sparse import issparse  # type: ignore
from inspect import isfunction

def apply_op(hamiltonian, ket):
    """Apply Hamiltonian to ket (backend-agnostic for NumPy/PyTorch)
    
    Args:
        hamiltonian: Sparse matrix, dense matrix, callable, or object with eval_matrix()
        ket: State vector (NumPy array or PyTorch tensor)
    
    Returns:
        H|ket> in the same backend as ket
    """
    K = tq.get_backend()

    # 1. Handle scipy sparse matrices: convert to dense BEFORE passing to backend
    if issparse(hamiltonian):
        hamiltonian_dense = hamiltonian.toarray()  # Convert to NumPy dense
        # Now use backend's matmul (let backend handle NumPy → PyTorch conversion if needed)
        return K.dot(hamiltonian_dense, ket)
    
    # 2. Handle callable (e.g., CI Hamiltonian functions from PySCF)
    if callable(hamiltonian):
        # Callable expects NumPy array, convert if needed
        ket_np = nb.to_numpy(ket)
        result_np = np.asarray(hamiltonian(ket_np), dtype=ket_np.dtype)
        # Convert back to original backend
        return K.asarray(result_np)
    
    # 3. Handle objects with eval_matrix method (MPO-like wrappers)
    if hasattr(hamiltonian, 'eval_matrix') and callable(getattr(hamiltonian, 'eval_matrix')):
        hamiltonian = K.asarray(hamiltonian.eval_matrix())
        # Fall through to dense matrix case
    

    
    # 4. Handle dense matrices (NumPy array or PyTorch tensor)
    # Let backend handle any necessary type alignment
    return K.dot(hamiltonian, ket)

