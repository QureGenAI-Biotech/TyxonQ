"""
Quantum natural gradient descent demonstration with the TFIM VQE example.
"""

import sys
import time

sys.path.insert(0, "../")

import torch

import tyxonq as tq
from tyxonq import experimental

tq.set_backend("pytorch")  # pytorch backend
tq.backend.set_dtype("complex64")

n, nlayers = 7, 2
g = tq.templates.graphs.Line1D(n)


@tq.backend.jit
def state(params):
    params = tq.backend.reshape(params, [2 * nlayers, n])
    c = tq.Circuit(n)
    c = tq.templates.blocks.example_block(c, params, nlayers=nlayers)
    return c.state()


def energy(params):
    s = state(params)
    c = tq.Circuit(n, inputs=s)
    loss = tq.templates.measurements.heisenberg_measurements(
        c, g, hzz=1, hxx=0, hyy=0, hx=-1
    )
    return tq.backend.real(loss)


vags = tq.backend.jit(tq.backend.value_and_grad(energy))
lr = 1e-2
# warning: pytorch optimizer is different from jax/tensorflow
# We'll create the optimizer later when we have the parameters

qng = tq.backend.jit(experimental.qng(state, mode="fwd"))
qngr = tq.backend.jit(experimental.qng(state, mode="rev"))
qng2 = tq.backend.jit(experimental.qng2(state, mode="fwd"))
qng2r = tq.backend.jit(experimental.qng2(state, mode="rev"))


def train_loop(params, i, qngf=None):
    value, grad = vags(params)
    if qngf is not None:
        qmetric = qngf(params)
        ngrad = tq.backend.solve(qmetric, grad, assume_a="sym")
    else:
        ngrad = grad
    if not hasattr(params, 'grad'):
        params.grad = None
    params.grad = ngrad
    params = params - lr * ngrad
    if i % 10 == 0:
        print(tq.backend.numpy(value))
    return params


def plain_train_loop(params, i):
    value, grad = vags(params)
    # warning: pytorch optimizer usage is different
    if not hasattr(params, 'grad'):
        params.grad = None
    params.grad = grad
    # Manual gradient descent
    params = params - lr * grad
    if i % 10 == 0:
        print(tq.backend.numpy(value))
    return params


def benchmark(f, *args):
    # params = tq.backend.implicit_randn([2 * nlayers * n])
    params = 0.1 * tq.backend.ones([2 * nlayers * n])
    params = tq.backend.cast(params, "float32")
    time0 = time.time()
    params = f(params, 0, *args)
    time1 = time.time()
    for i in range(30):
        params = f(params, i + 1, *args)
    time2 = time.time()

    print("staging time: ", time1 - time0, "running time: ", (time2 - time1) / 100)


if __name__ == "__main__":
    # run a short plain gradient descent to fit in 5–10s
    print("plain gradient descent (short)")
    benchmark(train_loop, None)
