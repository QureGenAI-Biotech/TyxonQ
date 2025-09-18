"""
VQE on 2D square lattice Heisenberg model with size n*m
"""

import torch
import tyxonq as tq

# import cotengra as ctg

# optr = ctg.ReusableHyperOptimizer(
#     methods=["greedy", "kahypar"],
#     parallel=True,
#     minimize="flops",
#     max_time=120,
#     max_repeats=4096,
#     progbar=True,
# )
# tq.set_contractor("custom", optimizer=optr, preprocessing=True)

K = tq.set_backend("pytorch")
K.set_dtype("complex64")


n, m, nlayers = 2, 2, 2
coord = tq.templates.graphs.Grid2DCoord(n, m)


def singlet_init(circuit):  # assert n % 2 == 0
    nq = circuit._nqubits
    for i in range(0, nq - 1, 2):
        j = (i + 1) % nq
        circuit.x(i)
        circuit.h(i)
        circuit.cnot(i, j)
        circuit.x(j)
    return circuit


def vqe_forward(param):
    paramc = K.cast(param, dtype="complex64")
    c = tq.Circuit(n * m)
    c = singlet_init(c)
    for i in range(nlayers):
        c = tq.templates.blocks.Grid2D_entangling(
            c, coord, tq.gates._swap_matrix, paramc[i]
        )
    loss = tq.templates.measurements.heisenberg_measurements(c, coord.lattice_graph())
    return loss


# warning pytorch might be unable to do this exactly
vgf = K.jit(
    K.value_and_grad(vqe_forward),
)
param = torch.nn.Parameter(torch.randn(nlayers, 2 * n * m) * 0.1)


if __name__ == "__main__":
    optimizer = torch.optim.Adam([param], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    for j in range(50):
        loss, gr = vgf(param)
        optimizer.zero_grad()
        param.grad = gr
        optimizer.step()
        if j % 10 == 0:
            print("loss", loss.detach().cpu().item())
        if (j + 1) % 25 == 0:
            scheduler.step()
