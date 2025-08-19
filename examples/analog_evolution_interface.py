"""
pytorch backend is required, experimental built-in interface for parameterized hamiltonian evolution
# warning: this script requires torchdiffeq, pip install torchdiffeq
"""
import torch
import tyxonq as tq
from tyxonq.experimental import evol_global, evol_local

K = tq.set_backend("pytorch")


def h_fun(t, b):
    return b * tq.gates.x().tensor


hy = tq.quantum.PauliStringSum2COO([[2, 0]])


def h_fun2(t, b):
    return b[2] * K.cos(b[0] * t + b[1]) * hy


# warning pytorch might be unable to do this jit optimization on value_and_grad
# @K.jit
# @K.value_and_grad
def hybrid_evol(params):
    c = tq.Circuit(2)
    c.x([0, 1])
    c = evol_local(c, [1], h_fun, 1.0, params[0])
    c.cx(1, 0)
    c.h(0)
    c = evol_global(c, h_fun2, 1.0, params[1:])
    return K.real(c.expectation_ps(z=[0, 1]))



b = torch.randn([4], requires_grad=True)
opt = torch.optim.Adam([b], lr=0.1)

for _ in range(50):
    opt.zero_grad()
    v = hybrid_evol(b)
    v.backward()
    opt.step()
    print(v.item(), b)
