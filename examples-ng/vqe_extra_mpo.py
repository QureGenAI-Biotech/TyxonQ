"""
Demonstration of TFIM VQE with extra size in MPO formulation
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
import tensornetwork as tn
import cotengra as ctg

import tyxonq as tq

opt = ctg.ReusableHyperOptimizer(
    methods=["greedy"],
    parallel=False,
    minimize="combo",
    max_time=5,
    max_repeats=64,
    progbar=False,
)


def opt_reconf(inputs, output, size, **kws):
    tree = opt.search(inputs, output, size)
    return tree.path()


tq.set_contractor("custom", optimizer=opt_reconf, preprocessing=True)
K = tq.set_backend("pytorch")
K.set_dtype("complex64")
# jax backend is incompatible with keras.save

dtype = np.complex64

nwires, nlayers = 6, 2


Jx = np.array([1.0 for _ in range(nwires - 1)])  # strength of xx interaction (OBC)
Bz = np.array([-1.0 for _ in range(nwires)])  # strength of transverse field
hamiltonian_mpo = tn.matrixproductstates.mpo.FiniteTFI(
    Jx, Bz, dtype=dtype
)  # matrix product operator
hamiltonian_mpo = tq.quantum.tn2qop(hamiltonian_mpo)


def vqe_forward(param):
    print("compiling")
    c = tq.Circuit(nwires)
    for i in range(nwires):
        c.h(i)
    for j in range(nlayers):
        for i in range(0, nwires - 1):
            c.exp1(
                i,
                (i + 1) % nwires,
                theta=param[4 * j, i],
                unitary=tq.gates._xx_matrix,
            )

        for i in range(nwires):
            c.rz(i, theta=param[4 * j + 1, i])
        for i in range(nwires):
            c.ry(i, theta=param[4 * j + 2, i])
        for i in range(nwires):
            c.rz(i, theta=param[4 * j + 3, i])
    return tq.templates.measurements.mpo_expectation(c, hamiltonian_mpo)


if __name__ == "__main__":
    refresh = False

    time0 = time.time()
    if refresh:
        # warning pytorch might be unable to do this exactly
        tc_vg = K.value_and_grad(vqe_forward)
        # tq.keras.save_func(tc_vg, "./funcs/%s_%s_tfim_mpo" % (nwires, nlayers))
        print("Function saved (keras.save_func removed in TyxonQ)")
        time1 = time.time()
        print("staging time: ", time1 - time0)

            # tc_vg_loaded = tq.keras.load_func("./funcs/%s_%s_tfim_mpo" % (nwires, nlayers))
        print("Function loaded (keras.load_func removed in TyxonQ)")
        tc_vg_loaded = tc_vg  # Use the original function instead
    else:
        tc_vg_loaded = K.value_and_grad(vqe_forward)

    lr1 = 0.008
    lr2 = 0.06
    steps = 30
    switch = 15
    debug_steps = 10

    # parameter and optimizers
    param = torch.nn.Parameter(torch.randn(4 * nlayers, nwires) * 0.1)
    optimizer1 = torch.optim.Adam([param], lr=lr1)
    optimizer2 = torch.optim.SGD([param], lr=lr2)

    times = []
    # Recreate optimizers with param now available
    optimizer1 = torch.optim.Adam([param], lr=lr1)
    optimizer2 = torch.optim.SGD([param], lr=lr2)

    for j in range(steps):
        loss, gr = tc_vg_loaded(param)
        if j < switch:
            optimizer1.zero_grad()
            param.grad = gr
            optimizer1.step()
        else:
            if j == switch:
                print("switching the optimizer")
            optimizer2.zero_grad()
            param.grad = gr
            optimizer2.step()
        if j % debug_steps == 0 or j == steps - 1:
            times.append(time.time())
            print("loss", loss.detach().cpu().item())
            if j > 0:
                print("running time:", (times[-1] - times[0]) / j)


"""
# Baseline code: obtained from DMRG using quimb

import quimb

h = quimb.tensor.tensor_gen.MPO_ham_ising(nwires, 4, 2, cyclic=False)
dmrg = quimb.tensor.tensor_dmrg.DMRG2(m, bond_dims=[10, 20, 100, 100, 200], cutoffs=1e-10)
dmrg.solve(tol=1e-9, verbosity=1) # may require repetition of this API
"""
