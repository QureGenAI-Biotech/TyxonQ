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

n, nlayers = 7, 2
g = tq.templates.graphs.Line1D(n)


@tq.backend.jit  # warning pytorch might be unable to do this exactly
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


vags = tq.backend.jit(tq.backend.value_and_grad(energy))  # warning pytorch might be unable to do this exactly
lr = 1e-2
# warning: pytorch optimizer is different from jax/tensorflow
opt = torch.optim.SGD([], lr=lr)

qng = tq.backend.jit(experimental.qng(state, mode="fwd"))  # warning pytorch might be unable to do this exactly
qngr = tq.backend.jit(experimental.qng(state, mode="rev"))  # warning pytorch might be unable to do this exactly
qng2 = tq.backend.jit(experimental.qng2(state, mode="fwd"))  # warning pytorch might be unable to do this exactly
qng2r = tq.backend.jit(experimental.qng2(state, mode="rev"))  # warning pytorch might be unable to do this exactly


def train_loop(params, i, qngf):
    qmetric = qngf(params)
    value, grad = vags(params)
    ngrad = tq.backend.solve(qmetric, grad, assume_a="sym")
    # warning: pytorch optimizer usage is different
    opt.zero_grad()
    params.grad = ngrad
    opt.step()
    if i % 10 == 0:
        print(tq.backend.numpy(value))
    return params


def plain_train_loop(params, i):
    value, grad = vags(params)
    # warning: pytorch optimizer usage is different
    opt.zero_grad()
    params.grad = grad
    opt.step()
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
    for i in range(100):
        params = f(params, i + 1, *args)
    time2 = time.time()

    print("staging time: ", time1 - time0, "running time: ", (time2 - time1) / 100)


if __name__ == "__main__":
    print("quantum natural gradient descent 1+f")
    benchmark(train_loop, qng)
    print("quantum natural gradient descent 1+r")
    benchmark(train_loop, qngr)
    print("quantum natural gradient descent 2+f")
    benchmark(train_loop, qng2)
    print("quantum natural gradient descent 2+r")
    benchmark(train_loop, qng2r)
    print("plain gradient descent")
    benchmark(plain_train_loop)
