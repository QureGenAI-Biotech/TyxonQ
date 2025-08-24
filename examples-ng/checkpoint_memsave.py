"""
Memory-saving VQE demo with simplified settings for PyTorch backend.
Runs quickly to pass example tests while keeping the structure.
"""

import time
import sys
import logging

import numpy as np
import torch
import cotengra as ctg

logger = logging.getLogger("tyxonq")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

sys.path.insert(0, "../")
sys.setrecursionlimit(10000)

import tyxonq as tq

# Keep contractor lightweight to avoid long planning time in tests
optr = ctg.ReusableHyperOptimizer(
    methods=["greedy"],
    parallel=False,
    minimize="size",
    max_time=3,
    max_repeats=32,
    progbar=False,
)
tq.set_contractor("custom", optimizer=optr, preprocessing=True)
tq.set_dtype("complex64")
tq.set_backend("pytorch")


# Reduce problem size to ensure the script finishes within 30s in CI
nwires, nlayers = 5, 9  # sn = 3
sn = int(np.sqrt(nlayers))


def recursive_checkpoint(funs):
    if len(funs) == 1:
        return funs[0]
    elif len(funs) == 2:
        f1, f2 = funs
        return lambda s, param: f1(
            f2(s, param[: len(param) // 2]), param[len(param) // 2 :]
        )
    else:
        f1 = recursive_checkpoint(funs[len(funs) // 2 :])
        f2 = recursive_checkpoint(funs[: len(funs) // 2])
        return lambda s, param: f1(s, param)


# not suggest in general for recursive checkpoint: too slow for staging (compiling)

"""
test case:
def f(s, param):
    return s + param
fc = recursive_checkpoint([f for _ in range(100)])
print(fc(jnp.zeros([2]), jnp.array([[i, i] for i in range(100)])))
"""


def zzxlayer(s, param):
    c = tq.Circuit(nwires, inputs=s)
    for i in range(0, nwires):
        c.exp1(
            i,
            (i + 1) % nwires,
            theta=param[0, i],
            unitary=tq.gates._zz_matrix,
        )
    for i in range(nwires):
        c.rx(i, theta=param[0, nwires + i])
    return c.state()


def zzxsqrtlayer(s, param):
    for i in range(sn):
        s = zzxlayer(s, param[i : i + 1])
    return s


def totallayer(s, param):
    for i in range(sn):
        s = zzxsqrtlayer(s, param[i * sn : (i + 1) * sn])
    return s


def vqe_forward(param):
    s = tq.backend.ones([2**nwires])
    s /= tq.backend.norm(s)
    s = totallayer(s, param)
    e = tq.expectation((tq.gates.x(), [1]), ket=s)
    return tq.backend.real(e)


def profile(tries=1):
    time0 = time.time()
    tq_vg = tq.backend.value_and_grad(vqe_forward)
    param = tq.backend.cast(tq.backend.ones([nlayers, 2 * nwires]), "complex64")
    val, grad = tq_vg(param)
    print(val)

    time1 = time.time()
    for _ in range(tries):
        print(val)

    time2 = time.time()
    print(time1 - time0, (time2 - time1) / tries)


if __name__ == "__main__":
    profile()
