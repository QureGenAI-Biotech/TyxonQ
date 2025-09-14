"""
Pauli/Operator I/O utilities
===========================

Purpose: tooling around OpenFermion/Qiskit qubit operator I/O, sparse representations,
index reversals, and minor chemistry-adjacent helpers (rdm transform, canonical coeff sign).

This module is intentionally domain-agnostic enough to remain in libs; chemistry-specific
numeric baselines are placed under applications/chem/quantum_chem_library.

TODO:
- Expose stable dataclasses for operator transport (reduce ad-hoc tuples).
- Add optional measurement grouping hooks (remain provider-agnostic).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix

import tyxonq as tq
from tyxonq.numerics import get_backend  # NEW: thin forward to current backend

from openfermion import FermionOperator, QubitOperator
from openfermion.utils import hermitian_conjugated
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator

from qiskit.quantum_info import SparsePauliOp  # type: ignore




def get_uint_type():
    rd = getattr(tq, "rdtypestr", "float64")
    return np.uint64 if rd == "float64" else np.uint32


def csc_to_coo(csc):
    coo = coo_matrix(csc)
    # discard tiny imaginary/real parts
    mask = 0.0 < np.abs(coo.data.real)
    indices = np.array([coo.row[mask], coo.col[mask]]).T
    values = coo.data.real[mask].astype(getattr(tq, "rdtypestr", "float64"))
    # Prefer backend sparse if available via Numeric backend API
    try:
        backend = get_backend(None)
        coo_builder = getattr(backend, "coo_sparse_matrix", None)
        if callable(coo_builder):
            return coo_builder(indices=indices, values=values, shape=coo.shape)
    except Exception:
        pass
    # Fallback to scipy coo
    if len(indices) == 0:
        return coo_matrix(coo.shape)
    row = indices[:, 0]
    col = indices[:, 1]
    return coo_matrix((values, (row, col)), shape=coo.shape)


def fop_to_coo(fop: FermionOperator, n_qubits: int, real: bool = True):
    op = get_sparse_operator(jordan_wigner(reverse_fop_idx(fop, n_qubits)), n_qubits=n_qubits)
    if real:
        op = op.real
    return csc_to_coo(op)


def hcb_to_coo(qop: QubitOperator, n_qubits: int, real: bool = True):
    op = get_sparse_operator(qop, n_qubits)
    if real:
        op = op.real
    return csc_to_coo(op)


def qop_to_qiskit(qop: QubitOperator, n_qubits: int):
    if SparsePauliOp is None:
        raise ImportError("qop_to_qiskit requires qiskit installed")
    sparse_list = []
    for k, v in qop.terms.items():
        s = "".join(kk[1] for kk in k)
        idx = [kk[0] for kk in k]
        sparse_list.append([s, idx, v])
    return SparsePauliOp.from_sparse_list(sparse_list, num_qubits=n_qubits)


def reverse_qop_idx(op: QubitOperator, n_qubits: int) -> QubitOperator:
    ret = QubitOperator()
    for pauli_string, v in op.terms.items():
        # ascending index internally; we flip big-endian <-> little-endian
        pauli_string = tuple(reversed([(n_qubits - 1 - idx, symbol) for idx, symbol in pauli_string]))
        ret.terms[pauli_string] = v
    return ret


def reverse_fop_idx(op: FermionOperator, n_qubits: int) -> FermionOperator:
    ret = FermionOperator()
    for word, v in op.terms.items():
        word = tuple([(n_qubits - 1 - idx, symbol) for idx, symbol in word])
        ret.terms[word] = v
    return ret


def format_ex_op(ex_op: Tuple[int, ...]) -> str:
    if len(ex_op) == 2:
        return f"{ex_op[0]}^ {ex_op[1]}"
    else:
        assert len(ex_op) == 4
        return f"{ex_op[0]}^ {ex_op[1]}^ {ex_op[2]} {ex_op[3]}"


def ex_op_to_fop(ex_op: Tuple[int, ...], with_conjugation: bool = False) -> FermionOperator:
    if len(ex_op) == 2:
        fop = FermionOperator(f"{ex_op[0]}^ {ex_op[1]}")
    else:
        assert len(ex_op) == 4
        fop = FermionOperator(f"{ex_op[0]}^ {ex_op[1]}^ {ex_op[2]} {ex_op[3]}")
    if with_conjugation:
        fop = fop - hermitian_conjugated(fop)
    return fop


__all__ = [
    "csc_to_coo",
    "fop_to_coo",
    "hcb_to_coo",
    "qop_to_qiskit",
    "reverse_qop_idx",
    "reverse_fop_idx",
    "format_ex_op",
    "ex_op_to_fop",
]


# --- Chemistry helpers reused across modules ---

def rdm_mo2ao(rdm: np.ndarray, mo_coeff: np.ndarray):
    c = mo_coeff
    if rdm.ndim == 2:
        return c @ rdm @ c.T
    else:
        assert rdm.ndim == 4
        for _ in range(4):
            rdm = np.tensordot(rdm, c.T, axes=1).transpose(3, 0, 1, 2)
        return rdm


def canonical_mo_coeff(mo_coeff: np.ndarray):
    largest_elem_idx = np.argmax(1e-5 < np.abs(mo_coeff), axis=0)
    largest_elem = mo_coeff[(largest_elem_idx, np.arange(len(largest_elem_idx)))]
    return mo_coeff * np.sign(largest_elem).reshape(1, -1)


def get_n_qubits(vector_or_matrix_or_mpo_func):
    if isinstance(vector_or_matrix_or_mpo_func, list):
        return len(vector_or_matrix_or_mpo_func)
    if hasattr(vector_or_matrix_or_mpo_func, "n_qubit"):
        return int(getattr(vector_or_matrix_or_mpo_func, "n_qubit"))
    return int(round(np.log2(vector_or_matrix_or_mpo_func.shape[0])))




# --- Fermion phase helper (moved from static/evolve_civector) ---

FERMION_PHASE_MASK_CACHE = {}


def get_fermion_phase(f_idx: tuple, n_qubits: int, ci_strings: np.ndarray) -> np.ndarray:
    if f_idx in FERMION_PHASE_MASK_CACHE:
        mask, sign = FERMION_PHASE_MASK_CACHE[(f_idx, n_qubits)]
    else:
        from openfermion import FermionOperator
        # fermion operator index, not sorted
        fop = ex_op_to_fop(f_idx)
        # pauli string index, already sorted
        qop = jordan_wigner(fop)
        mask_str = ["0"] * n_qubits
        for idx, term in next(iter(qop.terms.keys())):
            if term != "Z":
                assert idx in f_idx
                continue
            mask_str[n_qubits - 1 - idx] = "1"
        mask = get_uint_type()(int("".join(mask_str), base=2))

        if sorted(qop.terms.items())[0][1].real > 0:
            sign = -1
        else:
            sign = 1

        FERMION_PHASE_MASK_CACHE[(f_idx, n_qubits)] = mask, sign

    parity = ci_strings & mask
    assert parity.dtype in [np.uint32, np.uint64]
    if parity.dtype == np.uint32:
        mask_bits = 0x11111111
        shift = 28
    else:
        mask_bits = 0x1111111111111111
        shift = 60
    parity ^= parity >> 1
    parity ^= parity >> 2
    parity = (parity & mask_bits) * mask_bits
    parity = (parity >> shift) & 1

    return sign * np.sign(parity - 0.5).astype(np.int8)

