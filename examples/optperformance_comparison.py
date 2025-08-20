"""
Optimization for performance comparison for different densities of two-qubit gates
(random layouts averaged).
"""

import sys

sys.path.insert(0, "../")
import torch
import numpy as np
import tyxonq as tq

K = tq.set_backend("pytorch")
K.set_dtype("complex64")
# lightweight contractor for speed
tq.set_contractor("greedy")

eye4 = K.eye(4)
cnot = tq.array_to_tensor(tq.gates._cnot_matrix)


def energy_p(params, p, seed, n, nlayers):
    g = tq.templates.graphs.Line1D(n)
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for i in range(nlayers):
        for k in range(n):
            c.ry(k, theta=params[2 * i, k])
            c.rz(k, theta=params[2 * i + 1, k])
            c.ry(k, theta=params[2 * i + 2, k])
        for k in range(n // 2):  # alternating entangler with probability
            c.unitary_kraus(
                [eye4, cnot],
                2 * k + (i % 2),
                (2 * k + (i % 2) + 1) % n,
                prob=[1 - p, p],
                status=seed[i, k],
            )

    e = tq.templates.measurements.heisenberg_measurements(
        c, g, hzz=1, hxx=0, hyy=0, hx=-1, hy=0, hz=0
    )  # TFIM energy from expectation of circuit c defined on lattice given by g
    return e


vagf_single = K.jit(
    K.value_and_grad(energy_p, argnums=0),
    static_argnums=(3, 4),
)

energy_list = []


if __name__ == "__main__":
    n = 6
    nlayers = 4
    nsteps = 8
    sample = 1
    debug = True

    for a in [0.1, 0.5, 0.9]:
        energy_sublist = []
        params = K.implicit_randn(shape=[sample, 3 * nlayers, n])
        seeds = K.implicit_randu(shape=[sample, nlayers, n // 2])
        optimizer = torch.optim.Adam([torch.nn.Parameter(params)], lr=2e-3)
        for i in range(nsteps):
            p = (n * nlayers) ** (a - 1)
            p = tq.array_to_tensor(p, dtype="float32")
            e_accum = tq.array_to_tensor(0.0, dtype=tq.rdtypestr)
            grads_batch = torch.zeros_like(params)
            for s in range(sample):
                e_s, g_s = vagf_single(params[s], p, seeds[s], n, nlayers)
                e_accum = e_accum + e_s
                grads_batch[s] = g_s
            e_mean = e_accum / sample
            optimizer.zero_grad()
            params.grad = grads_batch / sample
            optimizer.step()
            if debug and (i % 5 == 0):
                print(a, i, e_mean)
        energy_list.append(K.numpy(e_mean))

    print(energy_list)
