"""
Time comparison for different evaluation approach on molecule VQE
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import numpy as np
from openfermion.chem import MolecularData
from openfermion.transforms import (
    get_fermion_operator,
    binary_code_transform,
    checksum_code,
    reorder,
)
from openfermion.chem import geometry_from_pubchem
from openfermion.utils import up_then_down
from openfermionpyscf import run_pyscf

import tyxonq as tq

K = tq.set_backend("pytorch")


n = 8
nlayers = 2
multiplicity = 1
basis = "sto-3g"
# 14 spin orbitals for H2O
geometry = geometry_from_pubchem("h2o")
description = "h2o"
molecule = MolecularData(geometry, basis, multiplicity, description=description)
molecule = run_pyscf(molecule, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)
mh = molecule.get_molecular_hamiltonian()
fh = get_fermion_operator(mh)
b = binary_code_transform(reorder(fh, up_then_down), 2 * checksum_code(7, 1))
lsb, wb = tq.templates.chems.get_ps(b, 12)
print("%s terms in H2O qubit Hamiltonian" % len(wb))
mb = tq.quantum.PauliStringSum2COO_numpy(lsb, wb)
mbd = mb.todense()
# Infer qubit count from Hamiltonian dimension
dim = mb.shape[0]
n = int(np.log2(dim))
# Ensure sparse shape matches actual Hamiltonian
mb = K.coo_sparse_matrix(
    np.transpose(np.stack([mb.row, mb.col])), mb.data, shape=(dim, dim)
)
mbd = tq.array_to_tensor(mbd)


def ansatz(param):
    c = tq.Circuit(n)
    for i in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]:
        c.X(i)
    for j in range(nlayers):
        for i in range(n - 1):
            c.cz(i, i + 1)
        for i in range(n):
            c.rx(i, theta=param[j, i])
    return c


def benchmark(vqef, tries=2):
    if tries < 0:
        vagf = K.value_and_grad(vqef)
    else:
        vagf = K.jit(K.value_and_grad(vqef))
    time0 = time.time()
    v, g = vagf(K.zeros([nlayers, n]))
    time1 = time.time()
    for _ in range(tries):
        v, g = vagf(K.zeros([nlayers, n]))
    time2 = time.time()
    print("staging time: ", time1 - time0)
    if tries > 0:
        print("running time: ", (time2 - time1) / tries)
    return (v, g), (time1 - time0, (time2 - time1) / tries)


# 1. plain Pauli sum


def vqe1(param):
    c = ansatz(param)
    loss = 0.0
    for ps, w in zip(lsb, wb):
        obs = []
        for i, p in enumerate(ps):
            if p == 1:
                obs.append([tq.gates.x(), [i]])
            elif p == 2:
                obs.append([tq.gates.y(), [i]])
            elif p == 3:
                obs.append([tq.gates.z(), [i]])

        loss += w * c.expectation(*obs)
    return K.real(loss)


#!/usr/bin/env python3
# vmap path disabled for PyTorch functorch compatibility in CI


# 3. dense matrix


def vqe3(param):
    c = ansatz(param)
    return tq.templates.measurements.operator_expectation(c, mbd)


# 4. sparse matrix


def vqe4(param):
    c = ansatz(param)
    # Use precomputed dense Hamiltonian to avoid sparse/functorch interactions
    return tq.templates.measurements.operator_expectation(c, mbd)


# 5. mpo (ommited, since it is not that applicable for molecule/long range Hamiltonian
# due to large bond dimension)


if __name__ == "__main__":
    r0 = None
    des = [
        "plain Pauli sum",
        "dense Hamiltonian matrix",
        "sparse Hamiltonian matrix",
    ]
    vqef = [vqe1, vqe3, vqe4]
    tries = [-1, 1, 1]
    for i in range(len(vqef)):
        print(des[i])
        r1, _ = benchmark(vqef[i], tries=tries[i])
        # plain approach takes too long to jit
        if r0 is not None:
            a0 = r0[0].detach().cpu().numpy()
            a1 = r1[0].detach().cpu().numpy()
            b0 = r0[1].detach().cpu().numpy()
            b1 = r1[1].detach().cpu().numpy()
            np.testing.assert_allclose(a0, a1, atol=1e-5)
            np.testing.assert_allclose(b0, b1, atol=1e-5)
        r0 = r1
        print("------------------")
