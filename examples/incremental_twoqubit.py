"""
Optimizing the parameterized circuit with progressively dense two-qubit gates,
as a potential approach to alleviate barren plateau.
"""

import sys

sys.path.insert(0, "../")
import torch
import numpy as np
import tyxonq as tq

K = tq.set_backend("pytorch")

n = 10
nlayers = 3
g = tq.templates.graphs.Line1D(n)


def energy(params, structures, n, nlayers):
    structures = (K.sign(structures) + 1) / 2  # 0 or 1
    structures = K.cast(structures, params.dtype)
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for j in range(nlayers):
        for i in range(n - 1):
            matrix = structures[j, i] * tq.gates._ii_matrix + (
                1.0 - structures[j, i]
            ) * (
                K.cos(params[2 * j + 1, i]) * tq.gates._ii_matrix
                + 1.0j * K.sin(params[2 * j + 1, i]) * tq.gates._zz_matrix
            )
            c.any(
                i,
                i + 1,
                unitary=matrix,
            )
        for i in range(n):
            c.rx(i, theta=params[2 * j, i])

    e = tq.templates.measurements.heisenberg_measurements(
        c, g, hzz=1, hxx=0, hyy=0, hx=-1, hy=0, hz=0
    )  # TFIM energy from expectation of circuit c defined on lattice given by g
    return e


# warning pytorch might be unable to do this exactly
vagf = K.jit(K.value_and_grad(energy, argnums=0), static_argnums=(2, 3))

params = np.random.uniform(size=[2 * nlayers, n])
structures = np.random.uniform(size=[nlayers, n])
params, structures = tq.array_to_tensor(params, structures)

optimizer = torch.optim.Adam([params], lr=1e-2)

for i in range(300):
    if i % 20 == 0:
        structures -= 0.2 * K.ones([nlayers, n])
    # one can change the structures by tune the structure tensor value
    # this specifically equiv to add two qubit gates
    
    # Forward pass
    e = energy(params, structures, n, nlayers)
    
    # Backward pass
    optimizer.zero_grad()
    e.backward()
    optimizer.step()
    
    print(e.detach().cpu().item())
