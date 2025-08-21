"""
Schrodinger-Heisenberg quantum variational eigensolver (SHVQE) with DQAS-style optimization.

DQAS part is modified from: examples/clifford_optimization.py
"""

import sys
import os

sys.path.insert(0, "../")

import numpy as np
import torch

import tyxonq as tq
from tyxonq.applications.vqes import construct_matrix_v3

ctype, rtype = tq.set_dtype("complex64")
K = tq.set_backend("pytorch")

# n will be inferred from Hamiltonian below
n = 6  # default, will be overwritten
ncz = 1  # number of cz layers in Schrodinger circuit
nlayersq = ncz + 1  # Schrodinger parameter layers

# training setup (shortened for quick validation)
epochs = 10
batch = 16

# Hamiltonian
# Use robust path so it works when run from a temp directory
_DATA_PATH = os.path.join(os.path.dirname(__file__), "h6_hamiltonian.npy")
h6h = np.load(_DATA_PATH)  # reported in 0.99 A
hamiltonian = construct_matrix_v3(h6h.tolist())
# Avoid functorch + sparse issues by densifying once
try:
    if K.is_sparse(hamiltonian):
        hamiltonian = K.to_dense(hamiltonian)
except Exception:
    pass
# Infer number of qubits from Hamiltonian dimension
try:
    dim = hamiltonian.shape[0]
    n = int(np.log2(dim))
    assert 2 ** n == dim, "Hamiltonian dimension must be a power of 2"
except Exception:
    n = n  # keep default
nlayersq = ncz + 1


def hybrid_ansatz(structure, paramq, preprocess="direct", train=True):
    """_summary_

    Parameters
    ----------
    structure : K.Tensor, (n//2, 2)
        parameters to decide graph structure of Clifford circuits
    paramq : K.Tensor, (nlayersq, n, 3)
        parameters in quantum variational circuits, the last layer for Heisenberg circuits
    preprocess : str, optional
        preprocess, by default "direct"

    Returns
    -------
    K.Tensor, [1,]
        loss value
    """
    c = tq.Circuit(n)
    if preprocess == "softmax":
        structure = K.softmax(structure, axis=-1)
    elif preprocess == "most":
        structure = K.onehot(K.argmax(structure, axis=-1), num=2)
    elif preprocess == "direct":
        pass

    structure = K.cast(structure, ctype)
    structure = torch.reshape(structure, shape=[n // 2, 2])

    # quantum variational in Schrodinger part, first consider a ring topol
    for j in range(nlayersq):
        if j != 0 and j != nlayersq - 1:
            for i in range(j % 2, n, 2):
                c.cz(i, (i + 1) % n)
        for i in range(n):
            c.rx(i, theta=paramq[j, i, 0])
            c.ry(i, theta=paramq[j, i, 1])
            c.rz(i, theta=paramq[j, i, 2])

    # Clifford part, which is actually virtual
    if train:
        for j in range(0, n // 2 - 1):
            dis = j + 1
            for i in range(0, n):
                c.unitary(
                    i,
                    (i + dis) % n,
                    unitary=structure[j, 0] * tq.gates.ii().tensor
                    + structure[j, 1] * tq.gates.cz().tensor,
                )

        for i in range(0, n // 2):
            c.unitary(
                i,
                i + n // 2,
                unitary=structure[n // 2 - 1, 0] * tq.gates.ii().tensor
                + structure[n // 2 - 1, 1] * tq.gates.cz().tensor,
            )
    else:  # if not for training, we just put nontrivial gates
        for j in range(0, n // 2 - 1):
            dis = j + 1
            for i in range(0, n):
                if structure[j, 1] == 1:
                    c.cz(i, (i + dis) % n)

        for i in range(0, n // 2):
            if structure[j, 1] == 1:
                c.cz(i, i + n // 2)

    return c


def hybrid_vqe(structure, paramq, preprocess="direct"):
    """_summary_

    Parameters
    ----------
    structure : K.Tensor, (n//2, 2)
        parameters to decide graph structure of Clifford circuits
    paramq : K.Tensor, (nlayersq, n, 3)
        parameters in quantum variational circuits, the last layer for Heisenberg circuits
    preprocess : str, optional
        preprocess, by default "direct"

    Returns
    -------
    K.Tensor, [1,]
        loss value
    """
    c = hybrid_ansatz(structure, paramq, preprocess)
    return tq.templates.measurements.operator_expectation(c, hamiltonian)


def sampling_from_structure(structures, batch=1):
    ch = structures.shape[-1]
    prob = K.softmax(K.real(structures), axis=-1)
    prob = K.reshape(prob, [-1, ch])
    p = prob.shape[0]
    r = np.stack(
        np.array(
            [np.random.choice(ch, p=K.numpy(prob[i]), size=[batch]) for i in range(p)]
        )
    )
    return r.transpose()


@K.jit
def best_from_structure(structures):
    return K.argmax(structures, axis=-1)


def nmf_gradient(structures, oh):
    """compute the Monte Carlo gradient with respect of naive mean-field probabilistic model

    Parameters
    ----------
    structures : K.Tensor, (n//2, ch)
        structure parameter for single- or two-qubit gates
    oh : K.Tensor, (n//2, ch), onehot
        a given structure sampled via strcuture parameters (in main function)

    Returns
    -------
    K.Tensor, (n//2 * 2, ch) == (n, ch)
        MC gradients
    """
    choice = K.argmax(oh, axis=-1)
    prob = K.softmax(K.real(structures), axis=-1)
    # Gather along row indices
    row_idx = K.cast(torch.arange(structures.shape[0]), "int64")
    prob_sel = prob[row_idx, choice]
    prob = prob_sel
    prob = K.reshape(prob, [-1, 1])
    prob = K.tile(prob, [1, structures.shape[-1]])

    # Functional one-hot update: result = -prob + one_hot(choice)
    mask = torch.nn.functional.one_hot(choice.long(), num_classes=structures.shape[-1]).to(prob.dtype)
    result = -prob + mask
    return K.real(result)


# vmap for a batch of structures
nmf_gradient_vmap = K.jit(K.vmap(nmf_gradient, vectorized_argnums=1))

# Single-sample value_and_grad to avoid functorch vmap+sparse issues
vag_hybrid = K.jit(
    K.value_and_grad(hybrid_vqe, argnums=(1,)),
    static_argnums=(2,),
)


def train_hybrid(
    stddev=0.05, lr=None, epochs=2000, debug_step=50, batch=256, verbose=False
):
    # params = K.implicit_randn([n//2, 2], stddev=stddev)
    params = K.ones([n // 2, 2], dtype=float)
    paramq = K.implicit_randn([nlayersq, n, 3], stddev=stddev) * 2 * np.pi
    if lr is None:
        lr = 0.2
    structure_opt = torch.optim.Adam([torch.nn.Parameter(params), torch.nn.Parameter(paramq)], lr=lr)

    avcost = 0
    avcost2 = 0
    loss_history = []
    for epoch in range(epochs):  # iteration to update strcuture param
        # random sample some structures
        batched_stucture = K.onehot(
            sampling_from_structure(params, batch=batch),
            num=params.shape[-1],
        )
        # Evaluate values and grads per-sample to keep tensors strided
        vs_list = []
        gq_acc = 0
        for bidx in range(batched_stucture.shape[0]):
            v_b, gq_b = vag_hybrid(batched_stucture[bidx], paramq, "direct")
            vs_list.append(v_b)
            gq_acc = gq_acc + gq_b[0]
        vs = torch.stack(vs_list)
        loss_history.append(np.min(vs.detach().cpu().numpy()))
        gq = gq_acc / batched_stucture.shape[0]
        avcost = K.mean(vs)  # average cost of the batch
        # Monte Carlo policy gradient per-sample then weighted average
        gs_list = []
        for bidx in range(batched_stucture.shape[0]):
            gs_list.append(nmf_gradient(params, batched_stucture[bidx]))
        gs_stack = torch.stack(gs_list)
        gs = K.mean(K.reshape(vs - avcost2, [-1, 1, 1]) * gs_stack, axis=0)
        # avcost2 is averaged cost in the last epoch
        avcost2 = avcost

        # warning: pytorch optimizer usage is different
        structure_opt.zero_grad()
        params.grad = gs
        paramq.grad = gq
        structure_opt.step()
        
        if epoch % debug_step == 0 or epoch == epochs - 1:
            print("----------epoch %s-----------" % epoch)
            print(
                "batched average loss: ",
                vs.detach().mean().item(),
                "minimum candidate loss: ",
                vs.detach().min().item(),
            )

            # max over choices, min over layers and qubits
            minp = torch.min(
                torch.max(torch.softmax(params, dim=-1), dim=-1)[0]
            )
            if minp > 0.5:
                print("probability converged")

            if verbose:
                print("strcuture parameter: \n", params.detach().cpu().numpy())

            cand_preset = best_from_structure(params)
            print(cand_preset)
            print("current recommendation loss: ", hybrid_vqe(params, paramq, "most"))

    loss_history = np.array(loss_history)
    return hybrid_vqe(params, paramq, "most"), params, paramq, loss_history


print("Train hybrid.")
ee, params, paramq, loss_history = train_hybrid(
    epochs=epochs, batch=batch, verbose=True
)
print("Energy:", ee)
