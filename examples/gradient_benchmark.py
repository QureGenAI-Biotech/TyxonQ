"""
Gradient evaluation comparison between qiskit and tyxonq
"""

import time
import json
from functools import reduce
from operator import xor
import numpy as np


from qiskit.opflow import X, StateFn
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.opflow.gradients import Gradient, QFI, Hessian

import tyxonq as tq
from tyxonq import experimental


def benchmark(f, *args, trials=10):
    time0 = time.time()
    r = f(*args)
    time1 = time.time()
    for _ in range(trials):
        r = f(*args)
    time2 = time.time()
    if trials > 0:
        time21 = (time2 - time1) / trials
    else:
        time21 = 0
    ts = (time1 - time0, time21)
    print("staging time: %.6f s" % ts[0])
    if trials > 0:
        print("running time: %.6f s" % ts[1])
    return r, ts


def grad_qiskit(n, l, trials=2):
    hamiltonian = reduce(xor, [X for _ in range(n)])
    wavefunction = QuantumCircuit(n)
    params = ParameterVector("theta", length=3 * n * l)
    for j in range(l):
        for i in range(n - 1):
            wavefunction.cnot(i, i + 1)
        for i in range(n):
            wavefunction.rx(params[3 * n * j + i], i)
        for i in range(n):
            wavefunction.rz(params[3 * n * j + i + n], i)
        for i in range(n):
            wavefunction.rx(params[3 * n * j + i + 2 * n], i)

    # Define the expectation value corresponding to the energy
    op = ~StateFn(hamiltonian) @ StateFn(wavefunction)
    grad = Gradient().convert(operator=op, params=params)

    def get_grad_qiskit(values):
        value_dict = {params: values}
        grad_result = grad.assign_parameters(value_dict).eval()
        return grad_result

    return benchmark(get_grad_qiskit, np.ones([3 * n * l]), trials=trials)


def qfi_qiskit(n, l, trials=0):
    wavefunction = QuantumCircuit(n)
    params = ParameterVector("theta", length=3 * n * l)
    for j in range(l):
        for i in range(n - 1):
            wavefunction.cnot(i, i + 1)
        for i in range(n):
            wavefunction.rx(params[3 * n * j + i], i)
        for i in range(n):
            wavefunction.rz(params[3 * n * j + i + n], i)
        for i in range(n):
            wavefunction.rx(params[3 * n * j + i + 2 * n], i)

    nat_grad = QFI().convert(operator=StateFn(wavefunction), params=params)

    def get_qfi_qiskit(values):
        value_dict = {params: values}
        grad_result = nat_grad.assign_parameters(value_dict).eval()
        return grad_result

    return benchmark(get_qfi_qiskit, np.ones([3 * n * l]), trials=trials)


def hessian_qiskit(n, l, trials=0):
    hamiltonian = reduce(xor, [X for _ in range(n)])
    wavefunction = QuantumCircuit(n)
    params = ParameterVector("theta", length=3 * n * l)
    for j in range(l):
        for i in range(n - 1):
            wavefunction.cnot(i, i + 1)
        for i in range(n):
            wavefunction.rx(params[3 * n * j + i], i)
        for i in range(n):
            wavefunction.rz(params[3 * n * j + i + n], i)
        for i in range(n):
            wavefunction.rx(params[3 * n * j + i + 2 * n], i)

    # Define the expectation value corresponding to the energy
    op = ~StateFn(hamiltonian) @ StateFn(wavefunction)
    grad = Hessian().convert(operator=op, params=params)

    def get_hs_qiskit(values):
        value_dict = {params: values}
        grad_result = grad.assign_parameters(value_dict).eval()
        return grad_result

    return benchmark(get_hs_qiskit, np.ones([3 * n * l]), trials=trials)


def grad_tq(n, l, trials=10):
    def f(params):
        c = tq.Circuit(n)
        for j in range(l):
            for i in range(n - 1):
                c.cnot(i, i + 1)
            for i in range(n):
                c.rx(i, theta=params[3 * n * j + i])
            for i in range(n):
                c.rz(i, theta=params[3 * n * j + i + n])
            for i in range(n):
                c.rx(i, theta=params[3 * n * j + i + 2 * n])
        return tq.backend.real(c.expectation(*[[tq.gates.x(), [i]] for i in range(n)]))

    # warning pytorch might be unable to do this exactly
    get_grad_tq = tq.backend.jit(tq.backend.grad(f))
    return benchmark(get_grad_tq, tq.backend.ones([3 * n * l], dtype="float32"))


def qfi_tq(n, l, trials=10):
    def s(params):
        c = tq.Circuit(n)
        for j in range(l):
            for i in range(n - 1):
                c.cnot(i, i + 1)
            for i in range(n):
                c.rx(i, theta=params[3 * n * j + i])
            for i in range(n):
                c.rz(i, theta=params[3 * n * j + i + n])
            for i in range(n):
                c.rx(i, theta=params[3 * n * j + i + 2 * n])
        return c.state()

    # warning pytorch might be unable to do this exactly
    get_qfi_tq = tq.backend.jit(experimental.qng(s, mode="fwd"))
    return benchmark(get_qfi_tq, tq.backend.ones([3 * n * l], dtype="float32"))


def hessian_tq(n, l, trials=10):
    def f(params):
        c = tq.Circuit(n)
        for j in range(l):
            for i in range(n - 1):
                c.cnot(i, i + 1)
            for i in range(n):
                c.rx(i, theta=params[3 * n * j + i])
            for i in range(n):
                c.rz(i, theta=params[3 * n * j + i + n])
            for i in range(n):
                c.rx(i, theta=params[3 * n * j + i + 2 * n])
        return tq.backend.real(c.expectation(*[[tq.gates.x(), [i]] for i in range(n)]))

    # warning pytorch might be unable to do this exactly
    get_hs_tq = tq.backend.jit(tq.backend.hessian(f))
    return benchmark(get_hs_tq, tq.backend.ones([3 * n * l], dtype="float32"))


results = {}

for n in [4, 6, 8, 10, 12]:
    for l in [2, 4, 6]:
        _, ts = grad_qiskit(n, l)
        results[str(n) + "-" + str(l) + "-" + "grad" + "-qiskit"] = ts[1]
        _, ts = qfi_qiskit(n, l)
        results[str(n) + "-" + str(l) + "-" + "qfi" + "-qiskit"] = ts[0]
        _, ts = hessian_qiskit(n, l)
        results[str(n) + "-" + str(l) + "-" + "hs" + "-qiskit"] = ts[0]
        with tq.runtime_backend("pytorch"):
            _, ts = grad_tq(n, l)
            results[str(n) + "-" + str(l) + "-" + "grad" + "-tq-pytorch"] = ts
            _, ts = qfi_tq(n, l)
            results[str(n) + "-" + str(l) + "-" + "qfi" + "-tq-pytorch"] = ts
            _, ts = hessian_tq(n, l)
            results[str(n) + "-" + str(l) + "-" + "hs" + "-tq-pytorch"] = ts

print(results)

with open("gradient_results.data", "w") as f:
    json.dump(results, f)

with open("gradient_results.data", "r") as f:
    print(json.load(f))
