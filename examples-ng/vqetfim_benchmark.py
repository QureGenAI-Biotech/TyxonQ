"""
Time comparison for different evaluation approach on spin VQE
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
from functools import partial
import numpy as np
import tensornetwork as tn

import tyxonq as tq

K = tq.set_backend("pytorch")


n = 10
nlayers = 1
j, h = 1, -1
xx = tq.gates._xx_matrix  # xx gate matrix to be utilized
enable_dense = False


def ansatz(param):
    c = tq.Circuit(n)
    for j in range(nlayers):
        for i in range(n - 1):
            c.exp1(i, i + 1, unitary=xx, theta=param[2 * j, i])
        for i in range(n):
            c.rz(i, theta=param[2 * j + 1, i])
    return c


def benchmark(vqef, tries=3):
    if tries < 0:
        vagf = K.value_and_grad(vqef)
    else:
        # warning pytorch might be unable to do this exactly
        vagf = K.jit(K.value_and_grad(vqef))
    time0 = time.time()
    v, g = vagf(K.zeros([2 * nlayers, n]))
    time1 = time.time()
    if tries > 0:
        for _ in range(tries):
            v, g = vagf(K.zeros([2 * nlayers, n]))
        time2 = time.time()
        running = (time2 - time1) / tries
    else:
        running = 0.0
    print("staging time: ", time1 - time0)
    if tries > 0:
        print("running time: ", running)
    return (v, g), (time1 - time0, running)


# 1. plain Pauli sum


def vqe1(param):
    c = ansatz(param)
    e = 0.0
    for i in range(n):
        e += h * c.expectation_ps(z=[i])  # <Z_i>
    for i in range(n - 1):  # OBC
        e += j * c.expectation_ps(x=[i, i + 1])  # <X_iX_{i+1}>
    return K.real(e)


# 2. vmap the Pauli sum


def measurement_batch(state, structures):
    # Manual batching to avoid functorch interactions with gate tensors
    c = tq.Circuit(n, inputs=state)
    structures_int = K.cast(structures, dtype="int32")
    results = []
    for t in range(structures_int.shape[0]):
        row = structures_int[t]
        obs = []
        for i in range(n):
            code = int(row[i].item())
            if code == 0:
                continue
            if code == 1:
                obs.append((tq.gates.x(), [i]))
            elif code == 2:
                obs.append((tq.gates.y(), [i]))
            elif code == 3:
                obs.append((tq.gates.z(), [i]))
        if len(obs) == 0:
            results.append(K.ones([]))
        else:
            results.append(c.expectation(*obs, reuse=True))
    return K.real(K.stack(results))


# Avoid jit/vmap here; manual batching above

structures = []
for i in range(n - 1):
    s = [0 for _ in range(n)]
    s[i] = 1
    s[i + 1] = 1
    structures.append(s)
for i in range(n):
    s = [0 for _ in range(n)]
    s[i] = 3
    structures.append(s)

structures = tq.array_to_tensor(structures)
weights = tq.array_to_tensor(
    np.array([1.0 for _ in range(n - 1)] + [-1.0 for _ in range(n)])
)


def vqe2(param):
    c = ansatz(param)
    s = c.state()
    ms = measurement_batch(s, structures)
    return K.sum(ms * K.real(weights))


# 3. dense matrix


def vqe_template(param, op):
    c = ansatz(param)
    e = tq.templates.measurements.operator_expectation(c, op)
    # in operator_expectation, the "hamiltonian" can be sparse matrix, dense matrix or mpo
    return e


hamiltonian_sparse_numpy = tq.quantum.PauliStringSum2COO_numpy(structures, weights)
# Cache COO components for manual quadratic form to avoid sparse mm under functorch
coo_rows = K.convert_to_tensor(hamiltonian_sparse_numpy.row)
coo_cols = K.convert_to_tensor(hamiltonian_sparse_numpy.col)
coo_vals = K.convert_to_tensor(hamiltonian_sparse_numpy.data)
hamiltonian_sparse = K.coo_sparse_matrix(
    np.transpose(np.stack([hamiltonian_sparse_numpy.row, hamiltonian_sparse_numpy.col])),
    hamiltonian_sparse_numpy.data,
    shape=(2**n, 2**n),
)

if enable_dense is True:
    hamiltonian_dense = K.to_dense(hamiltonian_sparse)

    vqe3 = partial(vqe_template, op=hamiltonian_dense)

else:
    vqe3 = vqe1

# 4. sparse matrix (manual quadratic form using cached COO to avoid sparse mm)


def vqe4(param):
    c = ansatz(param)
    state = c.wavefunction(form="ket").squeeze(-1)
    # sum_i conj(s[row_i]) * val_i * s[col_i]
    contrib = K.conj(state[coo_rows]) * coo_vals * state[coo_cols]
    return K.real(K.sum(contrib))


# 5. mpo

# generate the corresponding MPO by converting the MPO in tensornetwork package

Jx = np.array([1.0 for _ in range(n - 1)])  # strength of xx interaction (OBC)
Bz = np.array([1.0 for _ in range(n)])  # strength of transverse field
# Note the convention for the sign of Bz
hamiltonian_mpo = tn.matrixproductstates.mpo.FiniteTFI(
    Jx, Bz, dtype=np.complex64
)  # matrix product operator in TensorNetwork
hamiltonian_mpo = tq.quantum.tn2qop(hamiltonian_mpo)  # QuOperator in TyxonQ

vqe5 = partial(vqe_template, op=hamiltonian_mpo)


if __name__ == "__main__":
    r0 = None
    des = [
        "plain Pauli sum",
        "vmap Pauli sum",
        "dense Hamiltonian matrix",
        "sparse Hamiltonian matrix",
        "mpo Hamiltonian",
    ]
    vqef = [vqe1, vqe2, vqe3, vqe4, vqe5]
    tries = [0, 0, 1, 1, 1]
    if enable_dense:
        tests = [i for i in range(5)]
    else:
        tests = [0, 1, 3, 4]
    for i in tests:
        # we omit dense matrix rep here since the memory cost is unaffordable for large qubit counts
        print(des[i])
        r1, _ = benchmark(vqef[i], tries=tries[i])
        # plain approach takes too long to jit
        if r0 is not None:
            # PyTorch tensors -> numpy
            a0 = r0[0].detach().cpu().numpy()
            a1 = r1[0].detach().cpu().numpy()
            b0 = r0[1].detach().cpu().numpy()
            b1 = r1[1].detach().cpu().numpy()
            np.testing.assert_allclose(a0, a1, atol=1e-5)
            np.testing.assert_allclose(b0, b1, atol=1e-5)
        r0 = r1
        print("------------------")
