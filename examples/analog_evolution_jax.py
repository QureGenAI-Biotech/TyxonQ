"""
Parameterized Hamiltonian (Pulse control/Analog simulation) with AD support using pytorch ode solver
# warning: this script requires torchdiffeq, pip install torchdiffeq
"""
import torch
from torchdiffeq import odeint
import tyxonq as tq

K = tq.set_backend("pytorch")
tq.set_dtype("complex128")

hx = tq.quantum.PauliStringSum2COO([[1]])
hz = tq.quantum.PauliStringSum2COO([[3]])


# psi = -i H psi
# we want to optimize the final z expectation over parameters params
# a single qubit example below


def final_z(b):
    def f(t, y):
        h = b[3] * K.sin(b[0] * t + b[1]) * hx + K.cos(b[2]) * hz
        return -1.0j * K.sparse_dense_matmul(h, y)

    y0 = tq.array_to_tensor([1, 0], dtype="complex128")
    y0 = K.reshape(y0, [-1, 1])
    t = tq.array_to_tensor([0.0, 10.0], dtype=K.rdtypestr)
    yf = odeint(f, y0, t)
    c = tq.Circuit(1, inputs=K.reshape(yf[-1], [-1]))
    return K.real(c.expectation_ps(z=[0]))


# warning pytorch might be unable to do this jit optimization on value_and_grad
# vgf = K.jit(K.value_and_grad(final_z))


b = torch.randn([4], requires_grad=True)
opt = torch.optim.Adam([b], lr=0.1)

for _ in range(50):
    opt.zero_grad()
    v = final_z(b)
    v.backward()
    opt.step()
    print(v.item(), b)
