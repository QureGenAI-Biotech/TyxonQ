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


def get_integral_from_hf(hf: RHF, active_space: Tuple = None, aslst: List[int] = None):
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
    if aslst is None:
        int1e, e_core = casci.get_h1eff()
        int2e = ao2mo.restore("s1", casci.get_h2eff(), ncas)
    else:
        mo = casci.sort_mo(aslst, base=0)
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


def get_h_fcifunc_from_integral(int1e, int2e, n_elec):
    """Return CI-space Hamiltonian apply function using PySCF direct_spin1.

    This matches the alpha/beta-separated CI basis order we construct via get_ci_strings.
    """
    n_orb = len(int1e)
    # Ensure (na, nb) tuple
    if isinstance(n_elec, int):
        assert n_elec % 2 == 0, "total electron number must be even to split into (na, nb)"
        nelec_ab = (n_elec // 2, n_elec // 2)
    else:
        nelec_ab = (int(n_elec[0]), int(n_elec[1]))

    # Absorb one-electron integrals into two-electron tensor (PySCF handles symmetry internally)
    h2e = direct_spin1.absorb_h1e(int1e, int2e, n_orb, nelec_ab, 0.5)
    na, nb = nelec_ab
    nA = cistring.num_strings(n_orb, na)
    nB = cistring.num_strings(n_orb, nb)

    def fci_func(civector):
        civector = np.asarray(civector, dtype=np.float64).reshape((nA, nB))
        out = direct_spin1.contract_2e(h2e, civector, norb=n_orb, nelec=nelec_ab)
        return np.asarray(out, dtype=np.float64).reshape(-1)

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
    assert htype == "fcifunc"
    if mode in ["fermion", "qubit"]:
        return get_h_fcifunc_from_integral(int1e, int2e, n_elec_s)
    n_elec = sum(n_elec_s)
    return get_h_fcifunc_hcb_from_integral(int1e, int2e, n_elec)


def get_h_from_hf(hf: RHF, *, mode: str = "fermion", htype: str = "sparse", active_space: Tuple[int, int] | None = None, aslst: List[int] | None = None):
    """Thin wrapper to preserve legacy import path used in tests.

    Delegates to get_integral_from_hf + get_h_from_integral.
    """
    int1e, int2e, _ = get_integral_from_hf(hf, active_space=active_space, aslst=aslst)
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



