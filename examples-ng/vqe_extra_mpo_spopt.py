"""
Demonstration of TFIM VQE with extra size in MPO formulation, with highly customizable scipy optimization interface
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import time
import logging
import sys
import numpy as np
from scipy import optimize

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

optc = ctg.ReusableHyperOptimizer(
    methods=["greedy"],
    parallel=False,
    minimize="combo",
    max_time=2,
    max_repeats=16,
    progbar=False,
)


def opt_reconf(inputs, output, size, **kws):
    tree = optc.search(inputs, output, size)
    return tree.path()


tq.set_contractor("custom", optimizer=opt_reconf, preprocessing=True)
K = tq.set_backend("pytorch")
K.set_dtype("complex64")

dtype = np.complex64

nwires, nlayers = 4, 2


Jx = np.array([1.0 for _ in range(nwires - 1)])  # strength of xx interaction (OBC)
Bz = np.array([-1.0 for _ in range(nwires)])  # strength of transverse field
hamiltonian_mpo = tn.matrixproductstates.mpo.FiniteTFI(
    Jx, Bz, dtype=dtype
)  # matrix product operator
hamiltonian_mpo = tq.quantum.tn2qop(hamiltonian_mpo)


def vqe_forward(param):
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


def scipy_optimize(
    f, x0, method, jac=None, tol=1e-4, maxiter=20, step=10, threshold=1e-4
):
    epoch = 0
    loss_prev = 0
    threshold = 1e-6
    count = 0
    times = []
    while epoch < maxiter:
        time0 = time.time()
        r = optimize.minimize(
            f, x0=x0, method=method, tol=tol, jac=jac, options={"maxiter": step}
        )
        time1 = time.time()
        times.append(time1 - time0)
        loss = r["fun"]
        epoch += step
        x0 = r["x"]
        print(epoch, loss)
        print(r["message"])
        if len(times) > 1:
            running_time = np.mean(times[1:]) / step
            staging_time = times[0] - running_time * step
            print("staging time: ", staging_time)
            print("running time: ", running_time)
        if abs(loss - loss_prev) < threshold:
            count += 1
        loss_prev = loss
        if count > 5 + int(2000 / step):
            break
    return loss, x0, epoch


if __name__ == "__main__":
    param = torch.randn(4 * nlayers, nwires) * 0.1
    vqe_ng = tq.interfaces.scipy_optimize_interface(
        vqe_forward, shape=[4 * nlayers, nwires], gradient=False, jit=True
    )
    vqe_g = tq.interfaces.scipy_optimize_interface(
        vqe_forward, shape=[4 * nlayers, nwires], gradient=True, jit=True
    )
    scipy_optimize(
        vqe_ng, param.detach().cpu().numpy().flatten(), method="COBYLA", jac=False, maxiter=20
    )
    scipy_optimize(
        vqe_g,
        param.detach().cpu().numpy().flatten(),
        method="L-BFGS-B",
        jac=True,
        tol=1e-3,
        maxiter=20,
    )
    # for BFGS, large tol is necessary, see
    # https://stackoverflow.com/questions/34663539/scipy-optimize-fmin-l-bfgs-b-returns-abnormal-termination-in-lnsrch
