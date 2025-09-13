from __future__ import annotations

from typing import List, Tuple

import numpy as np
from openfermion import jordan_wigner

from tyxonq.libs.hamiltonian_encoding.pauli_io import ex_op_to_fop
from .ci_state_mapping import get_ci_strings, get_addr, get_uint_type


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


FERMION_PHASE_MASK_CACHE = {}


def get_fermion_phase(f_idx, n_qubits, ci_strings):
    """TCC-exact: get_fermion_phase from evolve_civector.py"""
    if (f_idx, n_qubits) in FERMION_PHASE_MASK_CACHE:
        mask, sign = FERMION_PHASE_MASK_CACHE[(f_idx, n_qubits)]
    else:
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
        mask = 0x11111111
        shift = 28
    else:
        mask = 0x1111111111111111
        shift = 60
    parity ^= parity >> 1
    parity ^= parity >> 2
    parity = (parity & mask) * mask
    parity = (parity >> shift) & 1

    return sign * np.sign(parity - 0.5).astype(np.int8)


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


def get_operator_tensors(
    n_qubits: int, n_elec_s, ex_ops: List[Tuple[int, ...]], mode: str = "fermion"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TCC-exact: get_operator_tensors from evolve_civector.py"""
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, mode, strs2addr=True)
    
    fket_permutation_tensor = np.zeros((len(ex_ops), len(ci_strings)), dtype=get_uint_type())
    # Force numpy arrays to avoid backend tensor mixing (torch/cupy) in numpy ops
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
    """Use θ (not 2θ) to match CI-space UCC evolution (TCC scheme)."""
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


