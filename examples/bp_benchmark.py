"""
benchmark on barren plateau using tq
"""

import os

import time
import torch
import numpy as np
# PennyLane optional
try:
    import pennylane as qml  # type: ignore
    _PL_AVAILABLE = True
except Exception:
    qml = None
    _PL_AVAILABLE = False
import tyxonq as tq


def benchmark(f, *args, tries=3):
    time0 = time.time()
    f(*args)
    time1 = time.time()
    for _ in range(tries):
        print(f(*args))
    time2 = time.time()
    print("staging time: ", time1 - time0, "running time: ", (time2 - time1) / tries)


K = tq.set_backend("pytorch")
Rx = tq.gates.rx
Ry = tq.gates.ry
Rz = tq.gates.rz


def op_expectation(params, seed, n_qubits, depth):
    paramsc = tq.backend.cast(params, dtype="float32")  # parameters of gates
    seedc = tq.backend.cast(seed, dtype="float32")
    # parameters of circuit structure

    c = tq.Circuit(n_qubits)
    for i in range(n_qubits):
        c.ry(i, theta=np.pi / 4)
    for l in range(depth):
        for i in range(n_qubits):
            c.unitary_kraus(
                [Rx(paramsc[i, l]), Ry(paramsc[i, l]), Rz(paramsc[i, l])],
                i,
                prob=[1 / 3, 1 / 3, 1 / 3],
                status=seedc[i, l],
            )
        for i in range(n_qubits - 1):
            c.cz(i, i + 1)

    return K.real(c.expectation_ps(z=[0, 1]))


# Remove vmap+vvag to avoid functorch one_hot issues
# op_expectation_vmap_vvag = K.vvag(op_expectation, argnums=0, vectorized_argnums=(0, 1))

def pennylane_approach(n_qubits=10, depth=10, n_circuits=100):
    if not _PL_AVAILABLE:
        return None
    try:
        dev = qml.device("lightning.qubit", wires=n_qubits)
        gate_set = [qml.RX, qml.RY, qml.RZ]

        @qml.qnode(dev)
        def rand_circuit(params, status):
            for i in range(n_qubits):
                qml.RY(np.pi / 4, wires=i)

            for j in range(depth):
                for i in range(n_qubits):
                    gate_set[status[i, j]](params[j, i], wires=i)

                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            return qml.expval(
                qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)], True)
            )

        gf = qml.grad(rand_circuit, argnum=0)
        params = np.random.uniform(0, 2 * np.pi, size=[n_circuits, depth, n_qubits])
        status = np.random.choice(3, size=[n_circuits, depth, n_qubits])

        g_results = []
        for i in range(n_circuits):
            g_results.append(gf(params[i], status[i]))
        g_results = np.stack(g_results)
        return np.std(g_results[:, 0, 0])
    except Exception:
        return None


if _PL_AVAILABLE:
    r = pennylane_approach(6, 4, 8)
    if r is not None:
        benchmark(lambda: r)


def tq_approach(n_qubits=10, depth=10, n_circuits=100):
    # Single-circuit setup to avoid functorch batching issues
    seed = tq.array_to_tensor(
        np.random.uniform(low=0.0, high=1.0, size=[n_qubits, depth]),
        dtype="float32",
    )
    params = tq.array_to_tensor(
        np.random.uniform(low=0.0, high=2 * np.pi, size=[n_qubits, depth]),
        dtype="float32",
    )

    def single_loss(p, s):
        return op_expectation(p, s, n_qubits, depth)

    # Manual gradient over the first param
    def first_param_loss(theta):
        p0 = params.clone()
        p0[0, 0] = theta
        return single_loss(p0, seed)

    grad_fn = torch.autograd.grad
    theta0 = params[0, 0]
    theta0 = theta0.clone().detach().requires_grad_(True)
    loss = first_param_loss(theta0)
    (g,) = grad_fn(loss, (theta0,), create_graph=False, allow_unused=False)
    return torch.abs(g)


benchmark(lambda: tq_approach(6, 4, 16))
