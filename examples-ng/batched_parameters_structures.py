"""
VQE optimization over different parameter initializations and different circuit structures
"""

import torch
import numpy as np
import tyxonq as tq

K = tq.set_backend("pytorch")

n = 6
lattice = tq.templates.graphs.Line1D(n, pbc=False)
h = tq.quantum.heisenberg_hamiltonian(
    lattice, hzz=1, hxx=0, hyy=0, hx=-1, hy=0, hz=0, sparse=False
)
# Ensure complex dtype for Hamiltonian
h = K.convert_to_tensor(h)
h = K.cast(h, K.dtypestr)


def gate_list(param):
    # Embed single-qubit parameterized gates on the first qubit of a 2-qubit system
    def embed_first_qubit(gate_unitary):
        u = tq.backend.reshapem(gate_unitary)
        iu = tq.backend.kron(u, tq.backend.eye(2))
        iu = tq.backend.reshape2(iu)
        return tq.gates.Gate(iu)

    l = [
        tq.gates.Gate(np.eye(4)),
        tq.gates.Gate(np.kron(tq.gates._x_matrix, np.eye(2))),
        tq.gates.Gate(np.kron(tq.gates._y_matrix, np.eye(2))),
        tq.gates.Gate(np.kron(tq.gates._z_matrix, np.eye(2))),
        tq.gates.Gate(np.kron(tq.gates._h_matrix, np.eye(2))),
        embed_first_qubit(tq.gates.rx_gate(theta=param).tensor),
        embed_first_qubit(tq.gates.ry_gate(theta=param).tensor),
        embed_first_qubit(tq.gates.rz_gate(theta=param).tensor),
        tq.gates.exp1_gate(theta=param, unitary=tq.gates._xx_matrix),
        tq.gates.exp1_gate(theta=param, unitary=tq.gates._yy_matrix),
        tq.gates.exp1_gate(theta=param, unitary=tq.gates._zz_matrix),
    ]
    return [tq.backend.reshape2(m.tensor) for m in l if isinstance(m, tq.gates.Gate)]


def makec(param, structure):
    c = tq.Circuit(n)
    for i in range(structure.shape[0]):
        for j in range(n):
            c.select_gate(structure[i, j], gate_list(param[i, j]), j, (j + 1) % n)
    return c


def vqef(param, structure):
    c = makec(param, structure)
    w = c.state(form="ket")
    # Ensure dtype alignment by casting h to match state dtype
    local_h = K.cast(K.convert_to_tensor(h), dtype=w.dtype)
    e = (K.adjoint(w) @ local_h @ w)[0, 0]
    e = K.real(e)
    return e


def batched_loss(weights, structure):
    # Manual batching to avoid functorch issues
    bs, bw, d, nq = weights.shape
    vals = []
    for s in range(bs):
        structure_s = structure[s]
        for w in range(bw):
            vals.append(vqef(weights[s, w], structure_s))
    return tq.backend.mean(tq.backend.stack(vals))


batch_structure = 2
batch_weights = 8
depth = 2
structure1 = np.array([[0, 1, 0, 5, 0, 6], [6, 0, 6, 0, 6, 0]])
structure2 = np.array([[0, 1, 0, 5, 0, 6], [9, 0, 8, 0, 3, 0]])
structure = tq.backend.stack([structure1, structure2])
structure = tq.backend.cast(structure, "int32")
structure = tq.backend.convert_to_tensor(structure)
weights = torch.randn(
    size=[batch_structure, batch_weights, depth, n], requires_grad=True
)
opt = torch.optim.Adam([weights], lr=1e-2)

for _ in range(10):
    opt.zero_grad()
    loss = batched_loss(weights, structure)
    print("loss:", float(loss.detach()))
    loss.backward()
    opt.step()
