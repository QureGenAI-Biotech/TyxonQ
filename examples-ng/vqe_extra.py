"""
Demonstration of the TFIM VQE on V100 with lager qubit number counts (100+).
"""

import time
import logging
import sys
import numpy as np

logger = logging.getLogger("tyxonq")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

sys.setrecursionlimit(10000)

import torch
import cotengra as ctg

import tyxonq as tq

optr = ctg.ReusableHyperOptimizer(
    methods=["greedy"],
    parallel=False,
    minimize="flops",
    max_time=5,
    max_repeats=64,
    progbar=False,
)
tq.set_contractor("custom", optimizer=optr, preprocessing=True)
# tq.set_contractor("custom_stateful", optimizer=oem.RandomGreedy, max_time=60, max_repeats=128, minimize="size")
K = tq.set_backend("pytorch")
K.set_dtype("complex64")
dtype = np.complex64

nwires, nlayers = 12, 3


def vqe_forward(param, structures):
    structuresc = K.cast(structures, dtype="complex64")
    paramc = K.cast(param, dtype="complex64")
    c = tq.Circuit(nwires)
    for i in range(nwires):
        c.h(i)
    for j in range(nlayers):
        for i in range(0, nwires - 1):
            c.exp1(
                i,
                (i + 1) % nwires,
                theta=paramc[2 * j, i],
                unitary=tq.gates._zz_matrix,
            )

        for i in range(nwires):
            c.rx(i, theta=paramc[2 * j + 1, i])

    # build weighted single-qubit observables fully in torch
    pauli_mats = [
        tq.array_to_tensor(g.tensor, dtype="complex64") for g in tq.gates.pauli_gates
    ]
    obs = []
    for i in range(nwires):
        mat = K.zeros([2, 2], dtype="complex64")
        for k in range(4):
            mat = mat + structuresc[i, k] * pauli_mats[k]
        obs.append([tq.gates.Gate(mat), (i,)])
    loss = c.expectation(*obs, reuse=False)
    return K.real(loss)


slist = []
for i in range(nwires):
    t = np.zeros(nwires)
    t[i] = 1
    slist.append(t)
for i in range(nwires):
    t = np.zeros(nwires)
    t[i] = 3
    t[(i + 1) % nwires] = 3
    slist.append(t)
structures = np.array(slist, dtype=np.int32)
structures = K.onehot(structures, num=4)
structures = K.reshape(structures, [-1, nwires, 4])
print(structures.shape)
time0 = time.time()

batch = 5
tc_vg_single = K.value_and_grad(vqe_forward, argnums=0)
param = torch.nn.Parameter(torch.randn(2 * nlayers, nwires) * 0.1)

val, grad = tc_vg_single(param, structures[0])
print(val, grad.shape)

time1 = time.time()
print("staging time: ", time1 - time0)

try:
    pass
except ValueError as e:
    print(e)  # keras.save_func now has issues to be resolved


def train_step(param):
    # simple two-sample stochastic update for speed
    e1, g1 = tc_vg_single(param, structures[0])
    e2, g2 = tc_vg_single(param, structures[1])
    loss = e1 - e2
    gr = g1 - g2
    return loss, gr


if __name__ == "__main__":
    optimizer = torch.optim.Adam([param], lr=0.02)
    for j in range(20):
        loss, gr = train_step(param)
        optimizer.zero_grad()
        param.grad = gr
        optimizer.step()
        if j % 5 == 0:
            print("loss", loss.detach().cpu().item())
