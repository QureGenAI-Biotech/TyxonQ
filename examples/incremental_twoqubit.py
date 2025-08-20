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

n = 8
nlayers = 3
g = tq.templates.graphs.Line1D(n)


def energy(params, structures, n, nlayers):
    # binarize structures; support complex dtypes by using real part
    structures = K.real(structures)
    structures = (K.sign(structures) + 1) / 2  # 0 or 1
    structures = K.cast(structures, params.dtype)
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for j in range(nlayers):
        for i in range(n - 1):
            # Implement identity when structures[j,i]==1 and exp( i theta ZZ ) when 0
            # by using theta_eff = (1 - structures[j,i]) * theta
            theta_eff = (1.0 - structures[j, i]) * params[2 * j + 1, i]
            c.exp1(i, i + 1, theta=theta_eff, unitary=tq.gates._zz_matrix)
        for i in range(n):
            c.rx(i, theta=params[2 * j, i])

    e = tq.templates.measurements.heisenberg_measurements(
        c, g, hzz=1, hxx=0, hyy=0, hx=-1, hy=0, hz=0
    )  # TFIM energy from expectation of circuit c defined on lattice given by g
    return e


vagf = K.jit(K.value_and_grad(energy, argnums=0), static_argnums=(2, 3))

params = torch.nn.Parameter(torch.from_numpy(np.random.uniform(size=[2 * nlayers, n]).astype(np.float32)))
structures = torch.from_numpy(np.random.uniform(size=[nlayers, n]).astype(np.float32))

optimizer = torch.optim.Adam([params], lr=1e-2)

for i in range(80):
    if i % 20 == 0:
        structures = structures - 0.2 * torch.ones([nlayers, n], dtype=structures.dtype)
    e = energy(params, structures, n, nlayers)
    optimizer.zero_grad()
    e.backward()
    optimizer.step()
    print(e.detach().cpu().item())
