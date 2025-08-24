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


def _ps_to_xyz(ps_row):
    xyz = {"x": [], "y": [], "z": []}
    for idx, v in enumerate(ps_row):
        if int(v) == 1:
            xyz["x"].append(idx)
        elif int(v) == 2:
            xyz["y"].append(idx)
        elif int(v) == 3:
            xyz["z"].append(idx)
    return xyz


def vqef(param, measure, n, nlayers):
    c = tq.Circuit(n)
    c.h(range(n))
    for i in range(nlayers):
        c.rzz(range(n - 1), range(1, n), theta=param[i, 0])
        c.rx(range(n), theta=param[i, 1])
    # Build expectation by summing over provided Pauli strings
    total = 0.0
    for ps_row in measure:
        xyz = _ps_to_xyz(ps_row)
        total = total + c.expectation_ps(**xyz)
    return K.real(total)


def get_tfim_ps(n):
    tfim_ps = []
    for i in range(n):
        tfim_ps.append(tq.quantum.xyz2ps({"x": [i]}, n=n))
    for i in range(n):
        tfim_ps.append(tq.quantum.xyz2ps({"z": [i, (i + 1) % n]}, n=n))
    return tq.array_to_tensor(tfim_ps)


vqg_vgf = K.value_and_grad(vqef)


# warning: pytorch parallel processing is different from jax pmap
def update(param, measure, n, nlayers):
    loss, grads = vqg_vgf(param, measure, n, nlayers)
    return grads, loss


if __name__ == "__main__":
    n = 6
    nlayers = 2
    m = get_tfim_ps(n)
    param = torch.nn.Parameter(torch.randn(nlayers, 2, n) * 0.1)
    optimizer = torch.optim.Adam([param], lr=1e-2)
    for _ in range(30):
        grads, loss = update(param, m, n, nlayers)
        optimizer.zero_grad()
        param.grad = grads
        optimizer.step()
        print(loss.detach().cpu().item())
