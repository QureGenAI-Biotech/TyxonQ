"""
Backend agnostic linear regression with gradient descent optimization:
a demonstration on most of core features and paradigm of tyxonq
"""

# this script shows how backend agnostic magic works, no code change is required to switch backend
# we also include jit, vmap and AD features in this pure classical example
# this demonstrates that tyxonq can serve as a solid unified ML library without any "quantumness"

import sys

sys.path.insert(0, "../")
import numpy as np
import tyxonq as tq

# (x, y) data preparation

nsamples = 100
k0 = 1.0
b0 = 0.0

xs0 = np.random.uniform(low=-1, high=1, size=[nsamples])
ys0 = k0 * xs0 + b0 + np.random.normal(scale=0.1, size=[nsamples])


def lr(xs, ys):
    """
    fully ML backend agnostic linear regression implementation
    """

    # construct the loss
    def loss_pointwise(x, y, param):
        k, b = param["k"], param["b"]
        yp = k * x + b
        return (yp - y) ** 2

    # vectorize loss over samples
    loss_vmap = tq.backend.vmap(loss_pointwise, vectorized_argnums=(0, 1))

    # total loss for all data
    def loss(xs, ys, param):
        losses = loss_vmap(xs, ys, param)
        return tq.backend.sum(losses)

    # jitted function to evaluate loss and its derivatives wrt. param
    loss_and_grad = tq.backend.jit(tq.backend.value_and_grad(loss, argnums=2))

    # setup initial values and optimizers
    weight = {"k": tq.backend.implicit_randn(), "b": tq.backend.implicit_randn()}

    import torch

    optimizer = torch.optim.Adam(
        [torch.nn.Parameter(weight["k"]), torch.nn.Parameter(weight["b"])], lr=1e-2
    )

    # gradient descent optimization loop
    maxstep = 200
    for i in range(maxstep):
        loss, grad = loss_and_grad(xs, ys, weight)
        optimizer.zero_grad()
        weight["k"].grad = grad["k"]
        weight["b"].grad = grad["b"]
        optimizer.step()
        if i % 50 == 0 or i == maxstep - 1:
            print("optimized MSE loss after %s round: " % i, tq.backend.numpy(loss))

    return tq.backend.numpy(weight["k"]), tq.backend.numpy(weight["b"])


if __name__ == "__main__":
    with tq.runtime_backend("pytorch"):
        print("~~~~~~~~ using pytorch backend ~~~~~~~~")
        xs_tensor, ys_tensor = tq.array_to_tensor(xs0, ys0, dtype="float32")
        print("predicted coefficient", lr(xs_tensor, ys_tensor))
