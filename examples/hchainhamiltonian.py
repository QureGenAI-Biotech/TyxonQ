"""
Get molecule qubit format Hamiltonian from openfermion.
"""

import time
import torch
from openfermion.chem import MolecularData
from openfermion.transforms import (
    get_fermion_operator,
    binary_code_transform,
    get_fermion_operator,
    reorder,
    checksum_code,
)
from openfermion.utils import up_then_down
from openfermionpyscf import run_pyscf
from scipy import sparse

import tyxonq as tq


n = 4
multiplicity = 1
geometry = [("H", (0, 0, 0.95 * i)) for i in range(n)]
description = "H%s_0.95" % str(n)
basis = "sto-3g"
molecule = MolecularData(geometry, basis, multiplicity, description=description)
molecule = run_pyscf(molecule, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)
print(molecule.fci_energy, molecule.ccsd_energy)
fermion_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
b = binary_code_transform(
    reorder(fermion_hamiltonian, up_then_down), 2 * checksum_code(n, 1)
)
ls, w = tq.templates.chems.get_ps(b, 2 * n - 2)
time0 = time.time()
m = tq.quantum.PauliStringSum2COO_numpy(
    torch.tensor(ls, dtype=torch.int64), torch.tensor(w, dtype=torch.complex128)
)
time1 = time.time()
print(m)
print("tq takes time: ", time1 - time0)
tq.backend.sparse_dense_matmul(m, tq.backend.ones([2 ** (2 * n - 2)]))
time2 = time.time()
print("tq takes time for mvp: ", time2 - time1)
sparse.save_npz("./h-" + str(n) + "-chain.npz", m)
m2 = sparse.load_npz("./h-" + str(n) + "-chain.npz")
print(m2)

"""
from openfermion.linalg import LinearQubitOperator

# too slow to use
h = LinearQubitOperator(b)
ids = np.eye(2 ** h.n_qubits)
time3 = time.time()
m = h.dot(np.ones([2 ** h.n_qubits]))
time4 = time.time()
print("of takes time for mvp: ", time4 - time3)
"""
