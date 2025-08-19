"""
pytorch parallel paradigm for vqe on multiple gpus
"""

import os

# warning: pytorch parallel processing is different from jax pmap
from functools import partial
import torch
import tyxonq as tq

K = tq.set_backend("pytorch")
tq.set_contractor("cotengra")


def vqef(param, measure, n, nlayers):
    c = tq.Circuit(n)
    c.h(range(n))
    for i in range(nlayers):
        c.rzz(range(n - 1), range(1, n), theta=param[i, 0])
        c.rx(range(n), theta=param[i, 1])
    return K.real(
        tq.templates.measurements.parameterized_measurements(c, measure, onehot=True)
    )


def get_tfim_ps(n):
    tfim_ps = []
    for i in range(n):
        tfim_ps.append(tq.quantum.xyz2ps({"x": [i]}, n=n))
    for i in range(n):
        tfim_ps.append(tq.quantum.xyz2ps({"z": [i, (i + 1) % n]}, n=n))
    return tq.array_to_tensor(tfim_ps)


# warning pytorch might be unable to do this exactly
vqg_vgf = K.vmap(K.value_and_grad(vqef), vectorized_argnums=(0, 1))


# warning: pytorch parallel processing is different from jax pmap
def update(param, measure, n, nlayers):
    # Compute the gradients on the given minibatch (individually on each device).
    loss, grads = vqg_vgf(param, measure, n, nlayers)
    grads = K.sum(grads, axis=0)
    loss = K.sum(loss, axis=0)
    return param, loss


if __name__ == "__main__":
    n = 8
    nlayers = 4
    ndevices = 1  # pytorch parallel processing is different
    m = get_tfim_ps(n)
    m = K.reshape(m, [ndevices, m.shape[0] // ndevices] + list(m.shape[1:]))
    param = torch.nn.Parameter(torch.randn(nlayers, 2, n) * 0.1)
    param = K.stack([param] * ndevices)
    optimizer = torch.optim.Adam([param], lr=1e-2)
    for _ in range(100):
        param, loss = update(param, m, n, nlayers)
        print(loss[0].detach().cpu().item())
