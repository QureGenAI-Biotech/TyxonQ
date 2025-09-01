"""
Get molecular Hamiltonian (qubit form) from OpenFermion, convert to Pauli-term list.
This demo constructs (lsb, wb) and saves a sparse file using scipy without tq.quantum.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from scipy import sparse

from openfermion.chem import MolecularData
from openfermion.transforms import (
    get_fermion_operator,
    binary_code_transform,
    reorder,
    checksum_code,
)
from openfermion.utils import up_then_down
from openfermionpyscf import run_pyscf

import tyxonq as tq
from tyxonq.libs.circuits_library import pauli_terms_from_openfermion


n = 4
multiplicity = 1
geometry = [("H", (0, 0, 0.95 * i)) for i in range(n)]
description = "H%s_0.95" % str(n)
basis = "sto-3g"

molecule = MolecularData(geometry, basis, multiplicity, description=description)
molecule = run_pyscf(molecule, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)
print(molecule.fci_energy, molecule.ccsd_energy)

fermion_h = get_fermion_operator(molecule.get_molecular_hamiltonian())
qo = binary_code_transform(reorder(fermion_h, up_then_down), 2 * checksum_code(n, 1))

lsb, wb, n_qubits = pauli_terms_from_openfermion(qo)
print(f"n_qubits={n_qubits}, num_terms={len(wb)}")

# Build a COO sparse from (lsb, wb) with naive kron; for demo only
I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

size = 1 << n_qubits
H = sparse.coo_matrix((size, size), dtype=np.complex128)

def _mat_from_codes(codes):
    op = None
    for q in range(n_qubits):
        code = codes[q]
        m = I if code == 0 else (X if code == 1 else (Y if code == 2 else Z))
        op = m if op is None else np.kron(op, m)
    return op

print("building sparse... this may be slow for large n")
t0 = time.time()
rows = []; cols = []; data = []
for codes, coeff in zip(lsb, wb):
    mat = _mat_from_codes(codes)
    coo = sparse.coo_matrix(mat)
    rows.extend(coo.row.tolist())
    cols.extend(coo.col.tolist())
    data.extend((coeff * coo.data).tolist())
H = sparse.coo_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(size, size))
t1 = time.time()
print("build_sparse_s: ", t1 - t0)

sparse.save_npz(f"./h-{n}-chain.npz", H)
H2 = sparse.load_npz(f"./h-{n}-chain.npz")
print(H2)
