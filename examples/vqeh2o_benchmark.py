"""
Time comparison for different evaluation approaches on molecule VQE (H2O minimal example).
- Direct numeric path: quantum_library + pytorch (no shots), summing Pauli string expectations without building dense H
"""

from __future__ import annotations

import time
import numpy as np
import torch
import tyxonq as tq

K = tq.set_backend("pytorch")

# Problem setup via OpenFermion (keep as source of Pauli terms)
from openfermion.chem import MolecularData
from openfermion.transforms import (
    get_fermion_operator,
    binary_code_transform,
    checksum_code,
    reorder,
)
from openfermion.chem import geometry_from_pubchem
from openfermion.utils import up_then_down

multiplicity = 1
basis = "sto-3g"
geometry = geometry_from_pubchem("h2o")
description = "h2o"
molecule = MolecularData(geometry, basis, multiplicity, description=description)
from openfermionpyscf import run_pyscf
molecule = run_pyscf(molecule, run_mp2=False, run_cisd=False, run_ccsd=False, run_fci=True)
mh = molecule.get_molecular_hamiltonian()
fh = get_fermion_operator(mh)
b = binary_code_transform(reorder(fh, up_then_down), 2 * checksum_code(7, 1))

# Convert OpenFermion QubitOperator to (lsb, wb)
from openfermion import QubitOperator  # type: ignore

def qubitop_to_pauli_terms(op: QubitOperator):
    terms = []
    weights = []
    max_idx = -1
    for (term, coeff) in op.terms.items():
        if term:
            max_idx = max(max_idx, max(q for q, _ in term))
        terms.append(term)
        weights.append(coeff)
    n_qubits = max_idx + 1 if max_idx >= 0 else 0
    lsb = []
    wb = []
    for term, coeff in zip(terms, weights):
        codes = [0] * n_qubits
        for q, p in term:
            if p == 'X': codes[q] = 1
            elif p == 'Y': codes[q] = 2
            elif p == 'Z': codes[q] = 3
        lsb.append(codes)
        wb.append(float(np.real(coeff)))
    return lsb, wb, n_qubits

lsb, wb, n = qubitop_to_pauli_terms(b)
print(f"{len(wb)} terms in H2O qubit Hamiltonian, n={n}")

nlayers = 2


# def dense_h_from_ps(lsb, wb):
#     I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
#     X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
#     Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
#     Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
#     H = np.zeros((1 << n, 1 << n), dtype=np.complex128)
#     for codes, coeff in zip(lsb, wb):
#         op = None
#         for q in range(n):
#             code = codes[q]
#             m = I if code == 0 else (X if code == 1 else (Y if code == 2 else Z))
#             op = m if op is None else np.kron(op, m)
#         H = H + coeff * op
#     return H

# H_dense = dense_h_from_ps(lsb, wb)
# H_dense_t = torch.as_tensor(H_dense, dtype=torch.complex128)

def ansatz(param: torch.Tensor) -> torch.Tensor:
    # param shape: [nlayers, n]
    nb = tq.get_backend("pytorch")
    from tyxonq.libs.quantum_library.kernels.statevector import (
        init_statevector, apply_1q_statevector, apply_2q_statevector,
    )
    from tyxonq.libs.quantum_library.kernels.gates import gate_rx, gate_cz_4x4

    psi = init_statevector(n, backend=nb)
    for j in range(nlayers):
        for i in range(n - 1):
            psi = apply_2q_statevector(nb, psi, gate_cz_4x4(), i, i + 1, n)
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rx(param[j, i]), i, n)
    return psi


def exact_energy_terms(param: torch.Tensor) -> torch.Tensor:
    # Sum of Pauli string expectations using basis rotations; no dense H built
    nb = tq.get_backend("pytorch")
    from tyxonq.libs.quantum_library.kernels.statevector import (
        apply_1q_statevector,
    )
    from tyxonq.libs.quantum_library.kernels.gates import gate_h, gate_sd

    psi = ansatz(param)

    # Group terms by rotation pattern to reuse rotated states
    from collections import defaultdict
    groups = defaultdict(list)  # key: tuple(codes) with 0/Z/X->Z/Y->Z markers
    for codes, w in zip(lsb, wb):
        # rotation marker: 0->0, 1->'X', 2->'Y', 3->'Z'
        key = tuple(codes)
        groups[key].append((codes, w))

    total = torch.zeros((), dtype=torch.float64)
    for key, items in groups.items():
        psi_rot = psi
        # apply basis change per qubit once for the group
        for q, code in enumerate(key):
            if code == 1:  # X -> Z via H
                psi_rot = apply_1q_statevector(nb, psi_rot, gate_h(), q, n)
            elif code == 2:  # Y -> Z via S^ H (use S^ then H)
                psi_rot = apply_1q_statevector(nb, psi_rot, gate_sd(), q, n)
                psi_rot = apply_1q_statevector(nb, psi_rot, gate_h(), q, n)
        # probs once
        probs = nb.square(nb.abs(psi_rot)) if hasattr(nb, 'square') else nb.abs(psi_rot) ** 2
        dim = 1 << n
        # evaluate each term in group
        for codes, w in items:
            z_sites = [q for q, code in enumerate(codes) if code != 0]
            if not z_sites:
                continue
            signs = [1.0] * dim
            for k in range(dim):
                s = 1.0
                for q in z_sites:
                    s *= (1.0 if ((k >> (n - 1 - q)) & 1) == 0 else -1.0)
                signs[k] = s
            total = total + float(w) * torch.sum(torch.as_tensor(signs, dtype=torch.float64) * probs)
    return total


def benchmark(fn, *args, tries: int = 1):
    t0 = time.time(); v0 = fn(*args); t1 = time.time()
    for _ in range(max(0, tries)):
        _ = fn(*args)
    t2 = time.time()
    stage = t1 - t0
    run = (t2 - t1) / max(1, tries)
    return v0, (stage, run)


if __name__ == "__main__":
    param0 = torch.zeros([nlayers, n], dtype=torch.float64)
    v_e, (s_e, r_e) = benchmark(lambda p: exact_energy_terms(p).detach(), param0, tries=0)
    print({"exact_energy_terms": float(v_e), "stage_s": s_e, "run_s": r_e})
