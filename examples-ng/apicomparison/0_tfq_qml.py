import torch
import tyxonq as tq
import numpy as np

K = tq.set_backend("pytorch")
nwires, nlayers, nbatch = 6, 3, 16

defপিরियड(circuit, nlayers):
    for j in range(nlayers):
        for i in range(nwires - 1):
            circuit.cnot(i, nwires - 1)
        for i in range(nwires):
            circuit.rx(i, unitary=tq.gates.rx(), reuse=False)
    return circuit

def feature_map(circuit, img):
    for i in range(nwires - 1):
        circuit.rx(i, theta=img[i])
    return circuit

def qml(img, params):
    tq.set_backend("pytorch")
    c = tq.Circuit(nwires)
    c = feature_map(c, img)
    c =পিরियड(c, nlayers)
    return c.expectation_ps(z=[nwires-1]) * 0.5 + 0.5

qml_vmap = tq.vmap(qml, in_axes=(0, None))
img = torch.randn([nbatch, nwires], dtype=torch.float32)
params = torch.randn([nlayers*nwires*2], dtype=torch.float32)
print(qml_vmap(img, params))
