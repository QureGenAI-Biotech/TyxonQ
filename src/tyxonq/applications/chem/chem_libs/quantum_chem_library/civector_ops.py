from __future__ import annotations

from typing import List, Tuple
from functools import lru_cache

import numpy as np
from openfermion import QubitOperator
from tyxonq.numerics import NumericBackend as nb

from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import apply_op
from .ci_state_mapping import get_ci_strings, get_init_civector
# from .ci_operator_tensors import (
#     get_operator_tensors,
# )
from .ci_state_mapping import get_ci_strings
from openfermion import jordan_wigner


from openfermion import jordan_wigner

from tyxonq.libs.hamiltonian_encoding.pauli_io import ex_op_to_fop
from .ci_state_mapping import get_ci_strings, get_addr, get_uint_type
from tyxonq.libs.hamiltonian_encoding.pauli_io import get_fermion_phase


def _apply_x(psi: np.ndarray, q: int, n_qubits: int) -> np.ndarray:
    stride = 1 << q
    out = psi.copy()
    for i in range(0, 1 << n_qubits, 2 * stride):
        for j in range(stride):
            a = i + j
            b = a + stride
            out[a], out[b] = psi[b], psi[a]
    return out


def _apply_z(psi: np.ndarray, q: int, n_qubits: int) -> np.ndarray:
    stride = 1 << q
    out = psi.copy()
    for i in range(0, 1 << n_qubits, 2 * stride):
        for j in range(stride):
            idx = i + j + stride
            out[idx] = -out[idx]
    return out


def _apply_y(psi: np.ndarray, q: int, n_qubits: int) -> np.ndarray:
    # Y = i|1><0| - i|0><1|
    stride = 1 << q
    out = psi.copy()
    for i in range(0, 1 << n_qubits, 2 * stride):
        for j in range(stride):
            a = i + j
            b = a + stride
            out[a] = -1j * psi[b]
            out[b] = 1j * psi[a]
    return out


def apply_h_qubit_to_ci(
    h_qubit_op: QubitOperator,
    n_qubits: int,
    n_elec_s: tuple[int, int],
    civector: np.ndarray,
    *,
    mode: str = "fermion",
) -> np.ndarray:
    ci_strings = np.asarray(get_ci_strings(n_qubits, n_elec_s, mode), dtype=np.uint64)
    size = len(ci_strings)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for term, coeff in h_qubit_op.terms.items():
        if term == ():
            for i in range(size):
                rows.append(i); cols.append(i); data.append(float(np.real(coeff)))
            continue
        for j, basis_index in enumerate(ci_strings):
            vec = np.zeros(1 << n_qubits, dtype=np.complex128)
            vec[int(basis_index)] = 1.0
            phi = vec
            for q, p in term:
                if p == "X":
                    phi = _apply_x(phi, q, n_qubits)
                elif p == "Y":
                    phi = _apply_y(phi, q, n_qubits)
                else:
                    phi = _apply_z(phi, q, n_qubits)
            amp = phi[ci_strings]
            nz = np.where(np.abs(amp) > 0)[0]
            for k in nz:
                rows.append(int(k)); cols.append(int(j)); data.append(float((coeff * amp[k]).real))

    from scipy.sparse import csr_matrix

    mat = csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(size, size))
    return np.asarray(mat.dot(np.asarray(civector, dtype=np.float64)), dtype=np.float64)

def get_civector(params: np.ndarray, n_qubits: int, n_elec_s, ex_ops: List[Tuple[int, ...]], param_ids: List[int], *, mode: str = "fermion", init_state: np.ndarray | None = None) -> np.ndarray:
    """Build CI vector by evolving excitations (NumPy/NumPy backends only).
    
    ⚠️ AUTOGRAD WARNING:
    - This function uses fancy indexing civ[fperm[j]] which does NOT support PyTorch gradients
    - Use statevector path for PyTorch autograd support
    - Gradients are computed analytically using backward evolution (not autograd)
    
    Args:
        params: Parameter vector (np.ndarray, not torch.Tensor)
        n_qubits, n_elec_s, ex_ops, param_ids: Standard UCC parameters
        mode: Fermionic mode
        init_state: Optional initial CI vector
    
    Returns:
        CI vector (np.ndarray)
    """
    # ⚠️ CRITICAL: Force NumPy conversion for all operations
    # Civector path uses fancy indexing which doesn't work with PyTorch tensors
    params = np.asarray(params, dtype=np.float64)
    
    ci_strings, fperm, fphase, f2phase = get_operator_tensors(n_qubits, n_elec_s, ex_ops, mode)
    fperm = np.asarray(fperm, dtype=np.int64)
    theta_sin, theta_1mcos = get_theta_tensors(params, param_ids)
    # Ensure numpy array for civector to avoid mixing backend tensors (e.g., torch) with numpy indices
    from copy import copy
    if init_state is None:
        civ = np.asarray(get_init_civector(len(ci_strings)), dtype=np.float64)
    else:
        #notice:: TODO  this bug fuck me a lot!
        #refenerce will be infulenced in the energy_and_grad_civector function.
        # when envlove ket  the init_state will chane
        # and final result will be change!
        civ = np.asarray(copy(init_state), dtype=np.float64)
        # when init_state is civector ,the following method will fuck!!!!!!!!
        # civ = np.asarray(init_state, dtype=np.float64)

    civ = evolve_civector_by_tensor(civ, fperm, fphase, f2phase, theta_sin, theta_1mcos)
    return np.asarray(civ, dtype=np.float64).reshape(-1)


def energy_and_grad_civector(
    params: np.ndarray,
    hamiltonian,
    n_qubits: int,
    n_elec_s,
    ex_ops: List[Tuple[int, ...]],
    param_ids: List[int],
    *,
    mode: str = "fermion",
    init_state: np.ndarray | None = None,
    precomputed_tensors: tuple | None = None,
) -> Tuple[float, np.ndarray]:
    """Compute energy and gradient using cached operator tensors.
    
    Args:
        params: Parameter vector
        hamiltonian: Hamiltonian operator
        n_qubits: Number of qubits
        n_elec_s: Number of electrons (alpha, beta)
        ex_ops: List of excitation operators
        param_ids: Parameter IDs for each excitation
        mode: Fermionic mode
        init_state: Initial CI vector state
        precomputed_tensors: Cached (ci_strings, fperm, fphase, f2phase) to avoid recomputation
    
    Returns:
        (energy, gradient) tuple
    """
    # ⚠️ CRITICAL: Force NumPy conversion for civector path
    # Civector uses fancy indexing which doesn't work with PyTorch tensors
    params = np.asarray(params, dtype=np.float64)
    
    if precomputed_tensors:
        ci_strings, fperm, fphase, f2phase = precomputed_tensors
    else:
        ci_strings, fperm, fphase, f2phase = get_operator_tensors(n_qubits, n_elec_s, ex_ops, mode)
    
    fperm = np.asarray(fperm, dtype=np.int64)
    theta_sin, theta_1mcos = get_theta_tensors(params, param_ids)
    # Ensure numpy array for civector to avoid mixing backend tensors with numpy ops
    ket = get_civector(params=params, n_qubits=n_qubits, n_elec_s=n_elec_s, ex_ops=ex_ops, param_ids=param_ids, mode=mode, init_state=init_state)
    bra = apply_op(hamiltonian,ket)
    # ⚠️ Force explicit conversion to NumPy for np.dot() to avoid NumPy 2.0 __array__ issues
    bra = np.asarray(bra, dtype=np.float64)
    ket = np.asarray(ket, dtype=np.float64)
    energy = float(np.dot(bra, ket))
    grads_before: List[float] = []
    b = np.asarray(bra, dtype=np.float64)
    k = np.asarray(ket, dtype=np.float64)
    for j in range(len(fperm) - 1, -1, -1):
        k = k + theta_1mcos[j] * (k * f2phase[j]) - theta_sin[j] * (k[fperm[j]] * fphase[j])
        b = b + theta_1mcos[j] * (b * f2phase[j]) - theta_sin[j] * (b[fperm[j]] * fphase[j])
        fket = k[fperm[j]] * fphase[j]
        grad_j = float(np.dot(b, fket))
        grads_before.append(grad_j)
    grads_before = grads_before[::-1]
    g = np.zeros_like(params)
    for grad, pid in zip(grads_before, param_ids):
        g[pid] += grad
    return energy, 2.0 * g


def apply_excitation_civector(civector: np.ndarray, n_qubits: int, n_elec_s, f_idx: Tuple[int, ...], mode: str) -> np.ndarray:
    """Apply one excitation on a CI vector (cached path)."""
    _, fperm, fphase, _ = get_operator_tensors(n_qubits, n_elec_s, [tuple(f_idx)], mode)
    civ = np.asarray(civector, dtype=np.float64)
    out = civ[fperm[0]] * fphase[0]
    return np.asarray(out, dtype=civ.dtype)


def apply_excitation_civector_nocache(civector: np.ndarray, n_qubits: int, n_elec_s, f_idx: Tuple[int, ...], mode: str) -> np.ndarray:
    """Apply one excitation on a CI vector (nocache path, TCC-style).

    Directly builds the operator action using get_ci_strings + get_operators
    without relying on cached batch tensors.
    """
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, mode, strs2addr=True)
    fperm, fphase, _ = get_operators(n_qubits, n_elec_s, strs2addr, tuple(f_idx), ci_strings, mode)
    civ = np.asarray(civector, dtype=np.float64)
    out = civ[fperm] * fphase
    return np.asarray(out, dtype=civ.dtype)


def get_civector_nocache(
    params: np.ndarray,
    n_qubits: int,
    n_elec_s,
    ex_ops: List[Tuple[int, ...]],
    param_ids: List[int],
    *,
    mode: str = "fermion",
    init_state: np.ndarray | None = None,
) -> np.ndarray:
    # ⚠️ CRITICAL: Force NumPy conversion for civector path
    params = np.asarray(params, dtype=np.float64)
    
    theta = np.asarray([params[i] for i in param_ids], dtype=np.float64)
    theta_sin = np.sin(theta)
    theta_1mcos = 1.0 - np.cos(theta)
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, mode, strs2addr=True)
    if init_state is None:
        civ = np.zeros(len(ci_strings), dtype=np.float64)
        civ[0] = 1.0
    else:
        civ = np.asarray(init_state, dtype=np.float64)
    for t_sin, t_1mcos, f_idx in zip(theta_sin, theta_1mcos, ex_ops):
        fperm, fphase, f2phase = get_operators(n_qubits, n_elec_s, strs2addr, tuple(f_idx), ci_strings, mode)
        fket = civ[fperm] * fphase
        f2ket = civ * f2phase
        civ = civ + t_1mcos * f2ket + t_sin * fket
    return np.asarray(civ, dtype=np.float64).reshape(-1)


def _get_gradients_civector_nocache(
    bra: np.ndarray,
    ket: np.ndarray,
    params: np.ndarray,
    n_qubits: int,
    n_elec_s,
    ex_ops: List[Tuple[int, ...]],
    param_ids: List[int],
    mode: str,
) -> np.ndarray:
    # ⚠️ CRITICAL: Force NumPy conversion for civector path
    params = np.asarray(params, dtype=np.float64)
    bra = np.asarray(bra, dtype=np.float64)
    ket = np.asarray(ket, dtype=np.float64)
    
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, mode, strs2addr=True)
    theta = np.asarray([params[i] for i in param_ids], dtype=np.float64)
    theta_sin = np.sin(theta)
    theta_1mcos = 1.0 - np.cos(theta)
    grads: List[float] = []
    b = bra
    k = ket
    for j in range(len(ex_ops) - 1, -1, -1):
        fperm, fphase, f2phase = get_operators(n_qubits, n_elec_s, strs2addr, tuple(ex_ops[j]), ci_strings, mode)
        k = k + theta_1mcos[j] * (k * f2phase) - theta_sin[j] * (k[fperm] * fphase)
        b = b + theta_1mcos[j] * (b * f2phase) - theta_sin[j] * (b[fperm] * fphase)
        fket = k[fperm] * fphase
        grads.append(float(np.dot(b, fket)))
    grads = grads[::-1]
    return np.asarray(grads, dtype=np.float64)


def energy_and_grad_civector_nocache(
    params: np.ndarray,
    hamiltonian,
    n_qubits: int,
    n_elec_s,
    ex_ops: List[Tuple[int, ...]],
    param_ids: List[int],
    *,
    mode: str = "fermion",
    init_state: np.ndarray | None = None,
    precomputed_tensors: tuple | None = None,
) -> Tuple[float, np.ndarray]:
    """Compute energy and gradient (nocache version) with optional precomputed tensors.
    
    Args:
        params: Parameter vector
        hamiltonian: Hamiltonian operator
        n_qubits: Number of qubits
        n_elec_s: Number of electrons (alpha, beta)
        ex_ops: List of excitation operators
        param_ids: Parameter IDs for each excitation
        mode: Fermionic mode
        init_state: Initial CI vector state
        precomputed_tensors: Currently unused for nocache version (for API consistency)
    
    Returns:
        (energy, gradient) tuple
    """
    # ⚠️ CRITICAL: Force NumPy conversion for civector path
    params = np.asarray(params, dtype=np.float64)
    
    ket = get_civector_nocache(params, n_qubits, n_elec_s, ex_ops, param_ids, mode=mode, init_state=init_state)
    bra = apply_op(hamiltonian,ket)
    # ⚠️ Force explicit conversion to NumPy for np.dot() to avoid NumPy 2.0 __array__ issues
    bra = np.asarray(bra, dtype=np.float64)
    ket = np.asarray(ket, dtype=np.float64)
    energy = float(np.dot(bra, ket))
    gbefore = _get_gradients_civector_nocache(bra, ket, params, n_qubits, n_elec_s, ex_ops, param_ids, mode)
    g = np.zeros_like(params, dtype=np.float64)
    for grad, pid in zip(gbefore, param_ids):
        g[pid] += grad
    return energy, 2.0 * g






def evolve_civector_by_tensor(
    civector, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor, theta_sin, theta_1mcos
):
    """TCC-exact: evolve_civector_by_tensor from evolve_civector.py"""
    def _evolve_excitation(j, _civector):
        _fket_phase = fket_phase_tensor[j]
        _fket_permutation = fket_permutation_tensor[j]
        fket = _civector[_fket_permutation] * _fket_phase
        f2ket = f2ket_phase_tensor[j] * _civector
        _civector += theta_1mcos[j] * f2ket + theta_sin[j] * fket
        return _civector

    # Simple loop implementation without fori_loop for now
    _civector = civector
    for j in range(len(fket_permutation_tensor)):
        _civector = _evolve_excitation(j, _civector)
    return _civector


def get_theta_tensors(params, param_ids):
    """Use θ (not 2θ) to match CI-space UCC evolution (TCC scheme).
    
    ⚠️ CRITICAL: Always return NumPy arrays for civector operations
    """
    theta = np.asarray([params[i] for i in param_ids], dtype=np.float64)
    theta_sin = np.sin(theta)
    theta_1mcos = 1.0 - np.cos(theta)
    return theta_sin, theta_1mcos


def get_civector_citensor(
    params: np.ndarray,
    n_qubits: int,
    n_elec_s,
    ex_ops: List[Tuple[int, ...]],
    param_ids: List[int],
    *,
    mode: str = "fermion",
    init_state: np.ndarray | None = None,
) -> np.ndarray:
    """TCC-exact: get_civector from evolve_civector.py"""
    ci_strings, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor = get_operator_tensors(
        n_qubits, n_elec_s, ex_ops, mode
    )
    theta_sin, theta_1mcos = get_theta_tensors(params, param_ids)

    if init_state is None:
        civector = np.zeros(len(ci_strings), dtype=np.float64)
        civector[0] = 1.0  # HF state
    else:
        civector = np.asarray(init_state, dtype=np.float64)
    civector = evolve_civector_by_tensor(
        civector, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor, theta_sin, theta_1mcos
    )
    return np.asarray(civector, dtype=np.float64).reshape(-1)



# FERMION_PHASE_MASK_CACHE = {}


# def get_fermion_phase(f_idx, n_qubits, ci_strings):
#     """TCC-exact: get_fermion_phase from evolve_civector.py"""
#     if (f_idx, n_qubits) in FERMION_PHASE_MASK_CACHE:
#         mask, sign = FERMION_PHASE_MASK_CACHE[(f_idx, n_qubits)]
#     else:
#         # fermion operator index, not sorted
#         fop = ex_op_to_fop(f_idx)

#         # pauli string index, already sorted
#         qop = jordan_wigner(fop)
#         mask_str = ["0"] * n_qubits
#         for idx, term in next(iter(qop.terms.keys())):
#             if term != "Z":
#                 assert idx in f_idx
#                 continue
#             mask_str[n_qubits - 1 - idx] = "1"
#         mask = get_uint_type()(int("".join(mask_str), base=2))

#         if sorted(qop.terms.items())[0][1].real > 0:
#             sign = -1
#         else:
#             sign = 1

#         FERMION_PHASE_MASK_CACHE[(f_idx, n_qubits)] = mask, sign

#     parity = ci_strings & mask
#     assert parity.dtype in [np.uint32, np.uint64]
#     if parity.dtype == np.uint32:
#         mask = 0x11111111
#         shift = 28
#     else:
#         mask = 0x1111111111111111
#         shift = 60
#     parity ^= parity >> 1
#     parity ^= parity >> 2
#     parity = (parity & mask) * mask
#     parity = (parity >> shift) & 1

#     return sign * np.sign(parity - 0.5).astype(np.int8)


def get_operators(n_qubits, n_elec_s, strs2addr, f_idx, ci_strings, mode):
    """TCC-exact: get_operators from evolve_civector.py"""
    if len(set(f_idx)) != len(f_idx):
        raise ValueError(f"Excitation {f_idx} not supported")
    
    fket_permutation = get_fket_permutation(f_idx, n_qubits, n_elec_s, ci_strings, strs2addr, mode)
    fket_phase = np.zeros(len(ci_strings))
    positive, negative = get_fket_phase(f_idx, ci_strings)
    fket_phase -= positive
    fket_phase += negative
    if mode == "fermion":
        fket_phase *= get_fermion_phase(f_idx, n_qubits, ci_strings)
    f2ket_phase = np.zeros(len(ci_strings))
    f2ket_phase -= positive
    f2ket_phase -= negative

    return fket_permutation, fket_phase, f2ket_phase





def get_fket_permutation(f_idx, n_qubits, n_elec_s, ci_strings, strs2addr, mode):
    """TCC-exact: get_fket_permutation from evolve_civector.py"""
    mask = 0
    for i in f_idx:
        mask += 1 << i
    excitation = ci_strings ^ mask
    return get_addr(excitation, n_qubits, n_elec_s, strs2addr, mode)


def get_fket_phase(f_idx, ci_strings):
    """TCC-exact: get_fket_phase from evolve_civector.py"""
    if len(f_idx) == 2:
        mask1 = 1 << f_idx[0]
        mask2 = 1 << f_idx[1]
    else:
        assert len(f_idx) == 4
        mask1 = (1 << f_idx[0]) + (1 << f_idx[1])
        mask2 = (1 << f_idx[2]) + (1 << f_idx[3])
    flip = ci_strings ^ mask1
    mask = mask1 | mask2
    masked = flip & mask
    positive = masked == mask
    negative = masked == 0
    return positive, negative



def get_operator_tensors(
    n_qubits: int, n_elec_s, ex_ops: List[Tuple[int, ...]], mode: str = "fermion"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TCC-exact: get_operator_tensors from evolve_civector.py with caching.

    Cache key depends on (n_qubits, normalized n_elec_s, ordered ex_ops, mode).
    """

    def _normalize_nelec(nelec):
        if isinstance(nelec, (list, tuple)):
            return (int(nelec[0]), int(nelec[1]))
        return (int(nelec),)

    def _normalize_exops(exops):
        return tuple(tuple(int(x) for x in f) for f in exops)

    ne_key = _normalize_nelec(n_elec_s)
    ex_key = _normalize_exops(ex_ops)

    ci_strings, fperm_t, fphase_t, f2phase_t = _build_operator_tensors_cached(
        int(n_qubits), ne_key, ex_key, str(mode)
    )
    # Return copies to avoid external mutation affecting cache
    return (
        np.array(ci_strings, copy=True),
        np.array(fperm_t, copy=True),
        np.array(fphase_t, copy=True),
        np.array(f2phase_t, copy=True),
    )


@lru_cache(maxsize=256)
def _build_operator_tensors_cached(
    n_qubits: int,
    ne_key: Tuple[int, ...],
    ex_key: Tuple[Tuple[int, ...], ...],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Reconstruct original arguments
    if len(ne_key) == 2:
        n_elec_s = (int(ne_key[0]), int(ne_key[1]))
    else:
        n_elec_s = int(ne_key[0])
    ex_ops = [tuple(f) for f in ex_key]

    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, mode, strs2addr=True)

    fket_permutation_tensor = np.zeros((len(ex_ops), len(ci_strings)), dtype=get_uint_type())
    fket_phase_tensor = np.zeros((len(ex_ops), len(ci_strings)), dtype=np.int8)
    f2ket_phase_tensor = np.zeros((len(ex_ops), len(ci_strings)), dtype=np.int8)

    for i, f_idx in enumerate(ex_ops):
        fket_permutation, fket_phase, f2ket_phase = get_operators(
            n_qubits, n_elec_s, strs2addr, f_idx, ci_strings, mode
        )
        fket_permutation_tensor[i] = fket_permutation
        fket_phase_tensor[i] = fket_phase
        f2ket_phase_tensor[i] = f2ket_phase

    return ci_strings, fket_permutation_tensor, fket_phase_tensor, f2ket_phase_tensor