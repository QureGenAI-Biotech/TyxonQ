from __future__ import annotations

import numpy as np
from openfermion import jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion import QubitOperator  # type: ignore
from functools import lru_cache
import tyxonq as tq

from tyxonq.applications.chem.constants import (
    ad_a_hc2,
    adad_aa_hc2,
    ad_a_hc,
    adad_aa_hc,
)
from tyxonq.libs.hamiltonian_encoding.pauli_io import ex_op_to_fop
from .civector_ops import civector as _civector_build
from .ci_state_mapping import get_ci_strings
from .ci_operator_tensors import get_operator_tensors
from math import comb


def apply_excitation_statevector(statevector, n_qubits, n_elec, f_idx, mode):
    # Apply in CI space; if provided n_qubits/n_elec mismatch the civector size, infer active CAS
    def _infer_active_from_len(k: int) -> tuple[int, int]:
        # return (m, na) such that C(m,na)^2 == k with na<=m and closed-shell
        for m in range(1, 16):
            for na in range(0, m + 1):
                if comb(m, na) ** 2 == k:
                    return m, na
        return -1, -1

    civ = np.asarray(statevector, dtype=np.float64)
    # try given
    if isinstance(n_elec, (tuple, list)):
        na, nb = int(n_elec[0]), int(n_elec[1])
    else:
        na = nb = int(n_elec) // 2
    desired = comb(n_qubits // 2, na) * comb(n_qubits // 2, nb)
    if civ.shape[0] != desired:
        m, na_inf = _infer_active_from_len(int(civ.shape[0]))
        if m > 0:
            n_qubits = 2 * m
            na = nb = na_inf
    _, fperm, fphase, _ = get_operator_tensors(n_qubits, (na, nb), [tuple(f_idx)], mode)
    out = civ[fperm[0]] * fphase[0]
    return np.asarray(out, dtype=civ.dtype)




def get_statevector_from_params(
    params: np.ndarray,
    n_qubits: int,
    n_elec_s,
    ex_ops,
    param_ids,
    *,
    mode: str = "fermion",
    init_state=None,
) -> np.ndarray:
    # Build CI vector using the same excitation semantics and embed into full statevector
    civ = _civector_build(np.asarray(params, dtype=np.float64), n_qubits, n_elec_s, list(ex_ops or []), list(param_ids or []), mode=mode, init_state=None)
    ci_strings = np.asarray(get_ci_strings(n_qubits, n_elec_s, mode), dtype=np.uint64)
    psi = np.zeros(1 << n_qubits, dtype=np.complex128)
    psi[ci_strings] = np.asarray(civ, dtype=np.complex128)
    return psi
def _qop_to_key(qop: QubitOperator, n_qubits: int):
    items = []
    for term, coeff in qop.terms.items():
        items.append((tuple((int(i), str(p)) for (i, p) in term), float(getattr(coeff, "real", coeff))))
    items.sort()
    return (int(n_qubits), tuple(items))


def _key_to_qop(key) -> QubitOperator:
    n_qubits, items = key
    qop = QubitOperator()
    for term, coeff in items:
        if len(term) == 0:
            qop += float(coeff)
        else:
            qop += QubitOperator(tuple((int(i), str(p)) for (i, p) in term), float(coeff))
    return qop


@lru_cache(maxsize=64)
def _cached_sparse_from_key(key):
    n_qubits, _ = key
    qop = _key_to_qop(key)
    return get_sparse_operator(qop, n_qubits=int(n_qubits))



def energy_statevector(
    params: np.ndarray,
    h_qubit_op,
    n_qubits: int,
    n_elec_s,
    ex_ops,
    param_ids,
    *,
    mode: str = "fermion",
    init_state=None,
) -> float:
    psi = get_statevector_from_params(params, n_qubits, n_elec_s, ex_ops, param_ids, mode=mode, init_state=init_state)
    try:
        key = _qop_to_key(h_qubit_op, int(n_qubits))
        H = _cached_sparse_from_key(key)
    except Exception:
        H = get_sparse_operator(h_qubit_op, n_qubits=n_qubits)
    e = np.vdot(psi, H.dot(psi))
    return float(np.real(e))


def energy_and_grad_statevector(
    params: np.ndarray,
    h_qubit_op,
    n_qubits: int,
    n_elec_s,
    ex_ops,
    param_ids,
    *,
    mode: str = "fermion",
    init_state=None,
) -> tuple[float, np.ndarray]:
    # Use backend value_and_grad wrapper to match TCC style (torchlib.func.grad_and_value)
    from tyxonq.numerics import NumericBackend as nb

    # Precompute sparse operator once for grad loop (important for numpy backend fallback)
    try:
        key = _qop_to_key(h_qubit_op, int(n_qubits))
        H = _cached_sparse_from_key(key)
    except Exception:
        H = get_sparse_operator(h_qubit_op, n_qubits=n_qubits)

    def _f(p):
        psi = get_statevector_from_params(p, n_qubits, n_elec_s, ex_ops, param_ids, mode=mode, init_state=init_state)
        e = np.vdot(psi, H.dot(psi))
        return float(np.real(e))

    vag = nb.value_and_grad(_f, argnums=0)
    e0, g = vag(np.asarray(params, dtype=np.float64))
    return float(e0), np.asarray(g, dtype=np.float64)
