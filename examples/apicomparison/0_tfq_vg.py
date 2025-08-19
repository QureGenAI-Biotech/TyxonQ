import torch
import tyxonq as tq
import numpy as np

tq.set_backend("pytorch")
nwires, nlayers = 6, 3

def vqe_vg(params):
    c = tq.Circuit(nwires)
    for i in range(nwires):
        c.H(i)
    for j in range(nlayers):
        for i in range(nwires - 1):
            c.cnot(i, i + 1)
        for i in range(nwires):
            c.rx(i, theta=params[j*nwires*2+nwires+i])
    
    expectation = 0
    for i in range(nwires-1):
        expectation += c.expectation_ps(z=[i, i+1])
    for i in range(nwires):
        expectation -= c.expectation_ps(x=[i])
    return expectation

vg = tq.value_and_grad(vqe_vg)
params = torch.randn([nlayers*nwires*2], dtype=torch.float32)
print(vg(params))
