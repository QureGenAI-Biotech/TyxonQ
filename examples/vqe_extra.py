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
from tyxonq import keras

optr = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel=True,
    minimize="flops",
    max_time=120,
    max_repeats=4096,
    progbar=True,
)
tq.set_contractor("custom", optimizer=optr, preprocessing=True)
# tq.set_contractor("custom_stateful", optimizer=oem.RandomGreedy, max_time=60, max_repeats=128, minimize="size")
K = tq.set_backend("pytorch")
K.set_dtype("complex64")
dtype = np.complex64

nwires, nlayers = 50, 7


def vqe_forward(param, structures):
    split_conf = {
        "max_singular_values": 2,
        "fixed_choice": 1,
    }
    structuresc = K.cast(structures, dtype="complex64")
    paramc = K.cast(param, dtype="complex64")
    c = tq.Circuit(nwires, split=split_conf)
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

    obs = []
    for i in range(nwires):
        obs.append(
            [
                tq.gates.Gate(
                    sum(
                        [
                            structuresc[i, k] * g.tensor
                            for k, g in enumerate(tq.gates.pauli_gates)
                        ]
                    )
                ),
                (i,),
            ]
        )
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

batch = 50
# warning pytorch might be unable to do this exactly
tc_vg = K.vectorized_value_and_grad(vqe_forward, argnums=0, vectorized_argnums=1)
param = torch.nn.Parameter(torch.randn(2 * nlayers, nwires) * 0.1)

print(tc_vg(param, structures[:batch]))

time1 = time.time()
print("staging time: ", time1 - time0)

try:
    keras.save_func(tc_vg, "./funcs/%s_%s_10_tfim" % (nwires, nlayers))
except ValueError as e:
    print(e)  # keras.save_func now has issues to be resolved


def train_step(param):
    vg_list = []
    for i in range(2):
        vg_list.append(tc_vg(param, structures[i * nwires : i * nwires + nwires]))
    loss = K.sum(vg_list[0][0] - vg_list[1][0])
    gr = vg_list[0][1] - vg_list[1][1]
    return loss, gr


if __name__ == "__main__":
    optimizer = torch.optim.Adam([param], lr=0.02)
    for j in range(5000):
        loss, gr = train_step(param)
        optimizer.zero_grad()
        param.grad = gr
        optimizer.step()
        if j % 20 == 0:
            print("loss", loss.detach().cpu().item())
