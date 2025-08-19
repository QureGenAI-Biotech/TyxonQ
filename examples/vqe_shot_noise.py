"""
VQE with finite measurement shot noise
"""

from functools import partial
import numpy as np
from scipy import optimize
import torch
import tyxonq as tq
from tyxonq import experimental as E

K = tq.set_backend("pytorch")

n = 6
nlayers = 4

# We use OBC 1D TFIM Hamiltonian in this script

ps = []
for i in range(n):
    l = [0 for _ in range(n)]
    l[i] = 1
    ps.append(l)
    # X_i
for i in range(n - 1):
    l = [0 for _ in range(n)]
    l[i] = 3
    l[i + 1] = 3
    ps.append(l)
    # Z_i Z_i+1
w = [-1.0 for _ in range(n)] + [1.0 for _ in range(n - 1)]


def generate_circuit(param):
    # construct the circuit ansatz
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for j in range(nlayers):
        for i in range(n - 1):
            c.rzz(i, i + 1, theta=param[i, j, 0])
        for i in range(n):
            c.rx(i, theta=param[i, j, 1])
    return c


def ps2xyz(psi):
    # ps2xyz([1, 2, 2, 0]) = {"x": [0], "y": [1, 2], "z": []}
    xyz = {"x": [], "y": [], "z": []}
    for i, j in enumerate(psi):
        if j == 1:
            xyz["x"].append(i)
        if j == 2:
            xyz["y"].append(i)
        if j == 3:
            xyz["z"].append(i)
    return xyz


@partial(K.jit, static_argnums=(1))  # warning pytorch might be unable to do this exactly
def exp_val(param, shots=1024):
    # expectation with shot noise
    # ps, w: H = \sum_i w_i ps_i
    # describing the system Hamiltonian as a weighted sum of Pauli string
    c = generate_circuit(param)
    if isinstance(shots, int):
        shots = [shots for _ in range(len(ps))]
    loss = 0
    for psi, wi, shot in zip(ps, w, shots):
        xyz = ps2xyz(psi)
        loss += wi * c.sample_expectation_ps(**xyz, shots=shot)
    return K.real(loss)


@K.jit  # warning pytorch might be unable to do this exactly
def exp_val_analytical(param):
    c = generate_circuit(param)
    loss = 0
    for psi, wi in zip(ps, w):
        xyz = ps2xyz(psi)
        loss += wi * c.expectation_ps(**xyz)
    return K.real(loss)


# 0. Exact result

hm = tq.quantum.PauliStringSum2COO_numpy(ps, w)
hm = K.to_dense(hm)
e, v = np.linalg.eigh(hm)
print("exact ground state energy: ", e[0])

# 1.1 VQE with numerically exact expectation: gradient free

print("VQE without shot noise")

exp_val_analytical_sp = tq.interfaces.scipy_interface(
    exp_val_analytical, shape=[n, nlayers, 2], gradient=False
)

r = optimize.minimize(
    exp_val_analytical_sp,
    np.zeros([n * nlayers * 2]),
    method="COBYLA",
    options={"maxiter": 5000},
)
print(r)


# 1.2 VQE with numerically exact expectation: gradient based

param = torch.nn.Parameter(torch.randn(n, nlayers, 2) * 0.1)
# warning pytorch might be unable to do this exactly
exp_val_grad_analytical = K.jit(K.value_and_grad(exp_val_analytical))
optimizer = torch.optim.Adam([param], lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
for i in range(1000):
    e, g = exp_val_grad_analytical(param)
    optimizer.zero_grad()
    param.grad = g
    optimizer.step()
    if i % 100 == 99:
        print(e.detach().cpu().item())
    if (i + 1) % 500 == 0:
        scheduler.step()


# 2.1 VQE with finite shot noise: gradient free

print("VQE with shot noise")


def exp_val_wrapper(param):
    return exp_val(param, shots=1024)


exp_val_sp = tq.interfaces.scipy_interface(
    exp_val_wrapper, shape=[n, nlayers, 2], gradient=False
)

r = optimize.minimize(
    exp_val_sp,
    np.random.normal(scale=0.1, size=[n * nlayers * 2]),
    method="COBYLA",
    options={"maxiter": 5000},
)
print(r)

# the real energy position after optimization

print("converged as: ", exp_val_analytical_sp(r["x"]))


# 2.2 VQE with finite shot noise: gradient based

param = torch.nn.Parameter(torch.randn(n, nlayers, 2) * 0.1)
exp_grad = E.parameter_shift_grad_v2(exp_val, argnums=0)
optimizer = torch.optim.Adam([param], lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

for i in range(1000):
    g = exp_grad(param)
    optimizer.zero_grad()
    param.grad = g
    optimizer.step()
    if i % 100 == 99:
        print(exp_val(param, shots=1024).detach().cpu().item())
    if (i + 1) % 500 == 0:
        scheduler.step()

# the real energy position after optimization

print("converged as:", exp_val_analytical(param).detach().cpu().item())
