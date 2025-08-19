"""
pytorch backend analog evolution targeting at minimizing evolution time
# warning: this script requires torchdiffeq, pip install torchdiffeq
"""
import torch
import tyxonq as tq
from tyxonq.experimental import evol_global

K = tq.set_backend("pytorch")

hx = tq.quantum.PauliStringSum2COO([[1]])


def h_fun(t, b):
    return K.sin(b) * hx


def fast_evol(t, b):
    lbd = 0.08
    c = tq.Circuit(1)
    c = evol_global(c, h_fun, t, b)
    loss = K.real(c.expectation_ps(z=[0]))
    return loss + lbd * t**2, loss
    # l2 regularization to minimize t while target z=-1


# warning pytorch might be unable to do this jit optimization on value_and_grad
# vgf = K.jit(K.value_and_grad(fast_evol, argnums=(0, 1), has_aux=True))


b = torch.tensor(0.5, dtype=getattr(torch, K.rdtypestr), requires_grad=True)
t = torch.tensor(1.0, dtype=getattr(torch, K.rdtypestr), requires_grad=True)

opt = torch.optim.Adam([b, t], lr=0.05)


for i in range(500):
    opt.zero_grad()
    v, loss = fast_evol(b, t)
    v.backward()
    opt.step()
    if i % 20 == 0:
        print(v.item(), loss.item(), b, t)
