"""
A plain QAOA optimization example with given graphs using networkx.
"""

import sys

sys.path.insert(0, "../")
import networkx as nx
import torch
import tyxonq as tq

# expose backend handle for unified programming model
K = tq.set_backend("pytorch")

## 1. define the graph


def dict2graph(d):
    g = nx.to_networkx_graph(d)
    for e in g.edges:
        if not g[e[0]][e[1]].get("weight"):
            g[e[0]][e[1]]["weight"] = 1.0
    return g


# a graph instance

example_graph_dict = {
    0: {1: {"weight": 1.0}, 7: {"weight": 1.0}, 3: {"weight": 1.0}},
    1: {0: {"weight": 1.0}, 2: {"weight": 1.0}, 3: {"weight": 1.0}},
    2: {1: {"weight": 1.0}, 3: {"weight": 1.0}, 5: {"weight": 1.0}},
    4: {7: {"weight": 1.0}, 6: {"weight": 1.0}, 5: {"weight": 1.0}},
    7: {4: {"weight": 1.0}, 6: {"weight": 1.0}, 0: {"weight": 1.0}},
    3: {1: {"weight": 1.0}, 2: {"weight": 1.0}, 0: {"weight": 1.0}},
    6: {7: {"weight": 1.0}, 4: {"weight": 1.0}, 5: {"weight": 1.0}},
    5: {6: {"weight": 1.0}, 4: {"weight": 1.0}, 2: {"weight": 1.0}},
}

example_graph = dict2graph(example_graph_dict)

# 2. define the quantum ansatz

nlayers = 3


def QAOAansatz(gamma, beta, g=example_graph):
    n = len(g.nodes)
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for j in range(nlayers):
        for e in g.edges:
            c.exp1(
                e[0],
                e[1],
                unitary=tq.gates._zz_matrix,
                theta=g[e[0]][e[1]].get("weight", 1.0) * gamma[j],
            )
        for i in range(n):
            c.rx(i, theta=beta[j])

    # calculate the loss function, max cut
    loss = 0.0
    for e in g.edges:
        loss += c.expectation_ps(z=[e[0], e[1]])

    return K.real(loss)


# 3. get compiled function for QAOA ansatz and its gradient

QAOA_vg = K.jit(K.value_and_grad(QAOAansatz, argnums=(0, 1)), static_argnums=2)


# 4. optimization loop

beta = torch.nn.Parameter(torch.randn(nlayers) * 0.1)
gamma = torch.nn.Parameter(torch.randn(nlayers) * 0.1)
optimizer = torch.optim.Adam([gamma, beta], lr=1e-2)

for i in range(100):
    # value and gradients
    loss, (g_gamma, g_beta) = QAOA_vg(gamma, beta, example_graph)
    print(loss.detach().cpu().item())
    optimizer.zero_grad()
    gamma.grad = g_gamma
    beta.grad = g_beta
    optimizer.step()
