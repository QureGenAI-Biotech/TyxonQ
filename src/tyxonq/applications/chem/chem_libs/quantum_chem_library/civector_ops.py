from __future__ import annotations

from typing import List, Tuple

import numpy as np
from openfermion import QubitOperator

from .ci_state_mapping import get_ci_strings, get_init_civector
from .ci_operator_tensors import (
    get_operator_tensors,
    get_theta_tensors,
    evolve_civector_by_tensor,
    get_operators,
)
from .ci_state_mapping import get_ci_strings


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


def civector(params: np.ndarray, n_qubits: int, n_elec_s, ex_ops: List[Tuple[int, ...]], param_ids: List[int], *, mode: str = "fermion", init_state: np.ndarray | None = None) -> np.ndarray:
    ci_strings, fperm, fphase, f2phase = get_operator_tensors(n_qubits, n_elec_s, ex_ops, mode)
    fperm = np.asarray(fperm, dtype=np.int64)
    theta_sin, theta_1mcos = get_theta_tensors(params, param_ids)
    # Ensure numpy array for civector to avoid mixing backend tensors (e.g., torch) with numpy indices
    civ = np.asarray(get_init_civector(len(ci_strings)), dtype=np.float64) if init_state is None else np.asarray(init_state, dtype=np.float64)
    civ = evolve_civector_by_tensor(civ, fperm, fphase, f2phase, theta_sin, theta_1mcos)
    return np.asarray(civ, dtype=np.float64).reshape(-1)


def energy_and_grad_civector(
    params: np.ndarray,
    h_qubit_op: QubitOperator,
    n_qubits: int,
    n_elec_s,
    ex_ops: List[Tuple[int, ...]],
    param_ids: List[int],
    *,
    mode: str = "fermion",
    init_state: np.ndarray | None = None,
    ci_apply: callable | None = None,
) -> Tuple[float, np.ndarray]:
    ci_strings, fperm, fphase, f2phase = get_operator_tensors(n_qubits, n_elec_s, ex_ops, mode)
    fperm = np.asarray(fperm, dtype=np.int64)
    theta_sin, theta_1mcos = get_theta_tensors(params, param_ids)
    # Ensure numpy array for civector to avoid mixing backend tensors with numpy ops
    civ = np.asarray(get_init_civector(len(ci_strings)), dtype=np.float64) if init_state is None else np.asarray(init_state, dtype=np.float64)
    ket = evolve_civector_by_tensor(np.asarray(civ, dtype=np.float64), fperm, fphase, f2phase, theta_sin, theta_1mcos)
    if ci_apply is not None:
        bra = np.asarray(ci_apply(ket), dtype=np.float64)
    else:
        bra = apply_h_qubit_to_ci(h_qubit_op, n_qubits, n_elec_s, ket, mode=mode)
    energy = float(np.dot(bra, ket))
    grads_before: List[float] = []
    b = np.asarray(bra, dtype=np.float64)
    k = np.asarray(ket, dtype=np.float64)
    for j in range(fperm.shape[0] - 1, -1, -1):
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
    ci_strings, strs2addr = get_ci_strings(n_qubits, n_elec_s, mode, strs2addr=True)
    theta = np.asarray([params[i] for i in param_ids], dtype=np.float64)
    theta_sin = np.sin(theta)
    theta_1mcos = 1.0 - np.cos(theta)
    grads: List[float] = []
    b = np.asarray(bra, dtype=np.float64)
    k = np.asarray(ket, dtype=np.float64)
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
    h_qubit_op,
    n_qubits: int,
    n_elec_s,
    ex_ops: List[Tuple[int, ...]],
    param_ids: List[int],
    *,
    mode: str = "fermion",
    init_state: np.ndarray | None = None,
    ci_apply: callable | None = None,
) -> Tuple[float, np.ndarray]:
    ket = get_civector_nocache(params, n_qubits, n_elec_s, ex_ops, param_ids, mode=mode, init_state=init_state)
    bra = np.asarray(ci_apply(ket), dtype=np.float64) if ci_apply is not None else apply_h_qubit_to_ci(
        h_qubit_op, n_qubits, n_elec_s, ket, mode=mode
    )
    energy = float(np.dot(bra, ket))
    gbefore = _get_gradients_civector_nocache(bra, ket, params, n_qubits, n_elec_s, ex_ops, param_ids, mode)
    g = np.zeros_like(params, dtype=np.float64)
    for grad, pid in zip(gbefore, param_ids):
        g[pid] += grad
    return energy, 2.0 * g


