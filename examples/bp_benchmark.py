"""
benchmark on barren plateau using tc
"""

import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import time
import torch
import numpy as np
import pennylane as qml
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


# warning pytorch might be unable to do this
op_expectation_vmap_vvag = K.vvag(op_expectation, argnums=0, vectorized_argnums=(0, 1))


def pennylane_approach(n_qubits=10, depth=10, n_circuits=100):
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

        return qml.expval(qml.Hamiltonian([1.0], [qml.PauliZ(0) @ qml.PauliZ(1)], True))

    gf = qml.grad(rand_circuit, argnum=0)
    params = np.random.uniform(0, 2 * np.pi, size=[n_circuits, depth, n_qubits])
    status = np.random.choice(3, size=[n_circuits, depth, n_qubits])

    g_results = []

    for i in range(n_circuits):
        g_results.append(gf(params[i], status[i]))

    g_results = np.stack(g_results)

    return np.std(g_results[:, 0, 0])


benchmark(pennylane_approach)


def tq_approach(n_qubits=10, depth=10, n_circuits=100):
    seed = tq.array_to_tensor(
        np.random.uniform(low=0.0, high=1.0, size=[n_circuits, n_qubits, depth]),
        dtype="float32",
    )
    params = tq.array_to_tensor(
        np.random.uniform(low=0.0, high=2 * np.pi, size=[n_circuits, n_qubits, depth]),
        dtype="float32",
    )

    _, grad = op_expectation_vmap_vvag(params, seed, n_qubits, depth)
    # the gradient variance of the first parameter
    grad_var = torch.std(K.numpy(grad), axis=0)

    return grad_var[0, 0]


benchmark(tq_approach)
