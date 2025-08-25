import tyxonq as tq
import torch

K = tq.set_backend("pytorch")
nwires, nlayer, nbatch = 6, 3, 16


def yp(img, params):
    c = tq.Circuit(nwires)
    for i in range(nwires - 1):
        c.rx(i, theta=img[i])
    for j in range(nlayer):
        for i in range(nwires - 1):
            c.rzz(i, nwires - 1, theta=params[i + j * 2 * nwires])
        for i in range(nwires):
            c.rx(i, theta=params[nwires + i + j * 2 * nwires])
    return K.real(c.expectation_ps(z=[nwires - 1]))


model = tq.torchnn.QuantumLayer(yp, [(nlayer * 2 * nwires,)])

imgs = torch.randn(shape=[nbatch, nwires])

print(model(imgs))
