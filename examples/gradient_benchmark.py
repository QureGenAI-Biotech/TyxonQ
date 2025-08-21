"""
Gradient evaluation comparison between qiskit and tyxonq
"""

import time
import json
from functools import reduce
from operator import xor
import numpy as np


try:
    from qiskit.opflow import X, StateFn  # type: ignore
    from qiskit.circuit import QuantumCircuit, ParameterVector  # type: ignore
    from qiskit.opflow.gradients import Gradient, QFI, Hessian  # type: ignore
    _HAS_QISKIT_OPFLOW = True
except Exception:
    X = StateFn = QuantumCircuit = ParameterVector = Gradient = QFI = Hessian = None  # type: ignore
    _HAS_QISKIT_OPFLOW = False

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


def grad_qiskit(n, l, trials=1):
    if not _HAS_QISKIT_OPFLOW:
        return (None, (0.0, 0.0))
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
    if not _HAS_QISKIT_OPFLOW:
        return (None, (0.0, 0.0))
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
    if not _HAS_QISKIT_OPFLOW:
        return (None, (0.0, 0.0))
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

    get_grad_tq = tq.backend.grad(f)
    return benchmark(get_grad_tq, tq.backend.ones([3 * n * l], dtype="float32"), trials=3)


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

    # Prefer qng2 (rev-mode); fallback to finite-diff if needed
    try:
        get_qfi_tq = experimental.qng2(s, mode="rev")
        return benchmark(get_qfi_tq, tq.backend.ones([3 * n * l], dtype="float32"), trials=1)
    except Exception:
        def finite_diff_qfi(params):
            eps = 1e-3
            params = tq.backend.cast(params, dtype="float32")
            psi0 = s(params)
            dpsi_list = []
            for k in range(params.shape[0]):
                e = tq.backend.zeros([params.shape[0]], dtype="float32")
                e = e + 0.0
                e = e + 0.0  # keep grad-safe
                e = e + 0.0
                e = tq.backend.scatter(e, tq.backend.convert_to_tensor([[k]]), tq.backend.convert_to_tensor([eps]))
                psi_p = s(params + e)
                psi_m = s(params - e)
                dpsi = (psi_p - psi_m) / (2 * eps)
                dpsi_list.append(dpsi)
            J = tq.backend.stack(dpsi_list, axis=0)
            gram = tq.backend.vmap(lambda u, v: tq.backend.tensordot(tq.backend.conj(u), v, 1), vectorized_argnums=(0, 0))(J, J)
            return tq.backend.real(gram)
        return benchmark(finite_diff_qfi, tq.backend.ones([3 * n * l], dtype="float32"), trials=1)


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

    def get_hs_tq(params):
        import torch
        try:
            return torch.autograd.functional.hessian(lambda x: f(x), params)
        except Exception:
            # Finite-difference Hessian via gradient differences
            eps = 1e-3
            size = params.shape[0]
            H = torch.zeros((size, size), dtype=params.dtype)
            def grad_at(p):
                p2 = p.clone().detach().requires_grad_(True)
                y = f(p2)
                (g,) = torch.autograd.grad(y, (p2,), create_graph=False, allow_unused=False)
                return g.detach()
            base = params.clone().detach()
            eye = torch.eye(size, dtype=params.dtype)
            for k in range(size):
                g_plus = grad_at(base + eps * eye[k])
                g_minus = grad_at(base - eps * eye[k])
                H[:, k] = (g_plus - g_minus) / (2 * eps)
            return H

    return benchmark(get_hs_tq, tq.backend.ones([3 * n * l], dtype="float32"), trials=0)


results = {}

# Reduced sweep for CI
for n in [4]:
    for l in [2]:
        try:
            _, ts = grad_qiskit(n, l)
            results[f"{n}-{l}-grad-qiskit"] = ts[1]
            _, ts = qfi_qiskit(n, l)
            results[f"{n}-{l}-qfi-qiskit"] = ts[0]
            _, ts = hessian_qiskit(n, l)
            results[f"{n}-{l}-hs-qiskit"] = ts[0]
        except Exception as e:
            results[f"{n}-{l}-qiskit-error"] = str(e)[:80]
        with tq.runtime_backend("pytorch"):
            _, ts = grad_tq(n, l)
            results[f"{n}-{l}-grad-tq-pytorch"] = ts
            _, ts = qfi_tq(n, l)
            results[f"{n}-{l}-qfi-tq-pytorch"] = ts
            try:
                _, ts = hessian_tq(n, l)
                results[f"{n}-{l}-hs-tq-pytorch"] = ts
            except Exception as e:
                results[f"{n}-{l}-hs-tq-pytorch-error"] = str(e)[:80]

print(results)

with open("gradient_results.data", "w") as f:
    json.dump(results, f)

with open("gradient_results.data", "r") as f:
    print(json.load(f))
