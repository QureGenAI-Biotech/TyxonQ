"""
QAOA with finite measurement shot noise
"""

from functools import partial
import numpy as np
from scipy import optimize
import networkx as nx
import torch
import cotengra as ctg
import tyxonq as tq
from tyxonq import experimental as E
from tyxonq.applications.graphdata import maxcut_solution_bruteforce

K = tq.set_backend("pytorch")
# note this script only supports pytorch backend

opt_ctg = ctg.ReusableHyperOptimizer(
    methods=["greedy", "kahypar"],
    parallel="ray",
    minimize="combo",
    max_time=10,
    max_repeats=128,
    progbar=True,
)

tq.set_contractor("custom", optimizer=opt_ctg, preprocessing=True)


def get_graph(n, d, weights=None):
    g = nx.random_regular_graph(d, n)
    if weights is not None:
        i = 0
        for e in g.edges:
            g[e[0]][e[1]]["weight"] = weights[i]
            i += 1
    return g


def get_exact_maxcut_loss(g):
    cut, _ = maxcut_solution_bruteforce(g)
    totalw = 0
    for e in g.edges:
        totalw += g[e[0]][e[1]].get("weight", 1)
    loss = totalw - 2 * cut
    return loss


def get_pauli_string(g):
    n = len(g.nodes)
    pss = []
    ws = []
    for e in g.edges:
        l = [0 for _ in range(n)]
        l[e[0]] = 3
        l[e[1]] = 3
        pss.append(l)
        ws.append(g[e[0]][e[1]].get("weight", 1))
    return pss, ws


def generate_circuit(param, g, n, nlayers):
    # construct the circuit ansatz
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for j in range(nlayers):
        c = tq.templates.blocks.QAOA_block(c, g, param[j, 0], param[j, 1])
    return c


def ps2z(psi):
    # ps2xyz([1, 2, 2, 0]) = {"x": [0], "y": [1, 2], "z": []}
    zs = []  # no x or y for QUBO problem
    for i, j in enumerate(psi):
        if j == 3:
            zs.append(i)
    return zs


def main_benchmark_suite(n, nlayers, d=3, init=None):
    g = get_graph(n, d, weights=np.random.uniform(size=[int(d * n / 2)]))
    loss_exact = get_exact_maxcut_loss(g)
    print("exact minimal loss by max cut bruteforce: ", loss_exact)
    pss, ws = get_pauli_string(g)
    if init is None:
        init = np.random.normal(scale=0.1, size=[nlayers, 2])

    @partial(K.jit, static_argnums=(1))  # warning pytorch might be unable to do this exactly
    def exp_val(param, shots=10000):
        # expectation with shot noise
        # ps, w: H = \sum_i w_i ps_i
        # describing the system Hamiltonian as a weighted sum of Pauli string
        c = generate_circuit(param, g, n, nlayers)
        loss = 0
        s = c.state()
        mc = tq.quantum.measurement_counts(
            s,
            counts=shots,
            format="sample_bin",
            jittable=True,
            is_prob=False,
        )
        for ps, w in zip(pss, ws):
            loss += w * tq.quantum.correlation_from_samples(ps2z(ps), mc, c._nqubits)
        return K.real(loss)

    @K.jit  # warning pytorch might be unable to do this exactly
    def exp_val_analytical(param):
        c = generate_circuit(param, g, n, nlayers)
        loss = 0
        for ps, w in zip(pss, ws):
            loss += w * c.expectation_ps(z=ps2z(ps))
        return K.real(loss)

    # 0. Exact result double check

    hm = tq.quantum.PauliStringSum2COO(
        tq.array_to_tensor(pss), tq.array_to_tensor(ws), numpy=True
    )
    hm = K.to_dense(hm)
    e, _ = np.linalg.eigh(hm)
    print("exact minimal loss via eigenstate: ", e[0])

    # 1.1 QAOA with numerically exact expectation: gradient free

    print("QAOA without shot noise")

    exp_val_analytical_sp = tq.interfaces.scipy_interface(
        exp_val_analytical, shape=[nlayers, 2], gradient=False
    )

    r = optimize.minimize(
        exp_val_analytical_sp,
        init,
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )
    print(r)
    print("double check the value?: ", exp_val_analytical_sp(r["x"]))
    # cobyla seems to have issue to given consistent x and cobyla

    # 1.2 QAOA with numerically exact expectation: gradient based

    # warning: pytorch scheduler is different from optax
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.tensor(init))], lr=1e-2)
    param = tq.array_to_tensor(init, dtype=tq.rdtypestr)
    exp_val_grad_analytical = K.jit(K.value_and_grad(exp_val_analytical))  # warning pytorch might be unable to do this exactly
    for i in range(1000):
        e, gs = exp_val_grad_analytical(param)
        # warning: pytorch optimizer usage is different
        optimizer.zero_grad()
        param.grad = gs
        optimizer.step()
        if i % 100 == 99:
            print(e)
    print("QAOA energy after gradient descent:", e)

    # 2.1 QAOA with finite shot noise: gradient free

    print("QAOA with shot noise")

    def exp_val_wrapper(param):
        return exp_val(param)

    exp_val_sp = tq.interfaces.scipy_interface(
        exp_val_wrapper, shape=[nlayers, 2], gradient=False
    )

    r = optimize.minimize(
        exp_val_sp,
        init,
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )
    print(r)

    # the real energy position after optimization

    print("converged as: ", exp_val_analytical_sp(r["x"]))

    # 2.2 QAOA with finite shot noise: gradient based

    # warning: pytorch scheduler is different from optax
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.tensor(init))], lr=1e-2)
    param = tq.array_to_tensor(init, dtype=tq.rdtypestr)
    exp_grad = E.parameter_shift_grad_v2(
        exp_val, argnums=0, shifts=(0.001, 0.002)
    )
    # parameter shift doesn't directly apply in QAOA case

    for i in range(1000):
        gs = exp_grad(param)
        # warning: pytorch optimizer usage is different
        optimizer.zero_grad()
        param.grad = gs
        optimizer.step()
        if i % 100 == 99:
            print(exp_val(param))

    # the real energy position after optimization

    print("converged as:", exp_val_analytical(param))


if __name__ == "__main__":
    main_benchmark_suite(8, 4)
