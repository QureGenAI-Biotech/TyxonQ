"""
Cross check the correctness of the density matrix simulator and the Monte Carlo trajectory state simulator.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# cpu is fast for small scale circuit simulation
import sys

sys.path.insert(0, "../")

from tqdm import tqdm
import torch
import tyxonq as tq

K = tq.set_backend("pytorch")

n = 5
nlayer = 3
mctries = 100  # 100000

print(torch.cuda.device_count())


def template(c):
    # dont jit me!
    for i in range(n):
        c.h(i)
    for i in range(n):
        c.rz(i, theta=tq.num_to_tensor(i))
    for _ in range(nlayer):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=tq.num_to_tensor(i))
        for i in range(n):
            c.apply_general_kraus(tq.channels.phasedampingchannel(0.15), i)
    return c.state()


# warning pytorch might be unable to do this exactly
@K.jit
def answer():
    c = tq.DMCircuit2(n)
    return template(c)


rho0 = answer()

print(rho0)


# warning pytorch might be unable to do this exactly
@K.jit
def f(seed):
    if seed is not None:
        torch.manual_seed(seed)
    c = tq.Circuit(n)
    return template(c)


seed = 42
f(seed)  # build the graph

rho = 0.0

for i in tqdm(range(mctries)):
    psi = f(seed + i)  # [1, 2**n]
    rho += (
        1
        / mctries
        * K.reshape(psi, [-1, 1])
        @ K.conj(K.reshape(psi, [1, -1]))
    )

print(rho)
print("difference\n", K.abs(rho - rho0))
print("difference in total\n", K.sum(K.abs(rho - rho0)))
print("fidelity", tq.quantum.fidelity(rho, rho0))
print("trace distance", tq.quantum.trace_distance(rho, rho0))
