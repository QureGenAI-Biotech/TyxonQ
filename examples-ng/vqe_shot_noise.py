"""
VQE with finite measurement shot noise (counts path) and a direct numeric path for comparison.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
import torch
import tyxonq as tq

K = tq.set_backend("pytorch")

n = 3
nlayers = 2

# Pauli strings for OBC 1D TFIM Hamiltonian (codes: 0-I, 1-X, 2-Y, 3-Z)
ps = []
for i in range(n):  # X_i
    l = [0 for _ in range(n)]
    l[i] = 1
    ps.append(l)
for i in range(n - 1):  # Z_i Z_{i+1}
    l = [0 for _ in range(n)]
    l[i] = 3
    l[i + 1] = 3
    ps.append(l)
# weights: -sum X_i + sum Z_i Z_{i+1}
w = [-1.0 for _ in range(n)] + [1.0 for _ in range(n - 1)]


def generate_circuit(param):
    # construct the circuit ansatz (param shape: [n, nlayers, 2])
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for j in range(nlayers):
        for i in range(n - 1):
            theta = float(param[i, j, 0])
            # ZZ(2*theta) via CX-RZ(2θ)-CX
            c.cx(i, i + 1)
            c.rz(i + 1, theta=2.0 * theta)
            c.cx(i, i + 1)
        for i in range(n):
            c.rx(i, theta=float(param[i, j, 1]))
    return c


def _term_expectation_from_counts(counts: dict[str, int], term: list[int]) -> float:
    total = sum(counts.values()) or 1
    acc = 0.0
    for bitstr, cnt in counts.items():
        val = 1.0
        for q, code in enumerate(term):
            if code == 3:  # Z
                val *= (1.0 if bitstr[q] == '0' else -1.0)
            elif code == 1:  # X measured via H rotation
                val *= (1.0 if bitstr[q] == '0' else -1.0)
        acc += val * cnt
    return acc / total


def exp_val_counts(param, shots=1024) -> float:
    # counts-based expectation with finite shots; param shape: [n, nlayers, 2]
    c_base = generate_circuit(param)
    loss = 0.0
    for term, wi in zip(ps, w):
        c = tq.Circuit(n, ops=list(c_base.ops))
        # rotate X -> Z
        for q, code in enumerate(term):
            if code == 1:
                c.h(q)
        for q in range(n):
            c.measure_z(q)
        out = c.device(provider="simulator", device="statevector", shots=shots).postprocessing(method=None).run()
        counts = out[0]["result"] if isinstance(out, list) else out.get("result", {})
        loss += wi * _term_expectation_from_counts(counts, term)
    return float(loss)


def exp_val_exact(param):
    # exact expectation via statevector (no shots)
    nb = tq.get_backend("pytorch")
    from tyxonq.libs.quantum_library.kernels.statevector import (
        init_statevector,
        apply_1q_statevector,
        apply_2q_statevector,
        expect_z_statevector,
    )
    from tyxonq.libs.quantum_library.kernels.gates import gate_h, gate_rx, gate_rz, gate_cx_4x4

    psi = init_statevector(n, backend=nb)
    for i in range(n):
        psi = apply_1q_statevector(nb, psi, gate_h(), i, n)
    for j in range(nlayers):
        for i in range(n - 1):
            theta = param[i, j, 0]
            from tyxonq.libs.quantum_library.kernels.gates import gate_rzz
            psi = apply_2q_statevector(nb, psi, gate_rzz(2.0 * theta), i, i + 1, n)
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rx(param[i, j, 1]), i, n)

    def _exp_zz(i: int) -> torch.Tensor:
        probs = nb.square(nb.abs(psi)) if hasattr(nb, "square") else nb.abs(psi) ** 2
        dim = 1 << n
        signs = [1.0 if (((k >> (n - 1 - i)) & 1) == ((k >> (n - 2 - i)) & 1)) else -1.0 for k in range(dim)]
        return nb.sum(nb.asarray(signs) * probs)

    e = torch.zeros((), dtype=torch.float64)
    psi_x = psi
    for q in range(n):
        psi_x = apply_1q_statevector(nb, psi_x, gate_h(), q, n)
    for q in range(n):
        e = e + (-1.0) * expect_z_statevector(psi_x, q, n, backend=nb)
    for i in range(n - 1):
        e = e + _exp_zz(i)
    return e


def parameter_shift_grad_counts(shots: int = 64):
    from math import pi
    s = 0.5 * pi
    def grad_fn(param: torch.Tensor) -> torch.Tensor:
        base = param.detach().clone()
        flat = base.reshape(-1)
        g = torch.zeros_like(flat, dtype=torch.float64)
        for k in range(flat.numel()):
            p_plus = flat.clone(); p_plus[k] = flat[k] + s
            p_minus = flat.clone(); p_minus[k] = flat[k] - s
            f_plus = exp_val_counts(p_plus.reshape(n, nlayers, 2), shots=shots)
            f_minus = exp_val_counts(p_minus.reshape(n, nlayers, 2), shots=shots)
            g[k] = 0.5 * (f_plus - f_minus)
        return g.reshape_as(param)
    return grad_fn


# 0. Exact result

def _dense_hamiltonian_from_paulis(n: int, ps_list, weights):
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    H = np.zeros((1 << n, 1 << n), dtype=np.complex128)
    for codes, coeff in zip(ps_list, weights):
        op = None
        for q in range(n):
            code = codes[q]
            m = I if code == 0 else (X if code == 1 else (Y if code == 2 else Z))
            op = m if op is None else np.kron(op, m)
        H = H + coeff * op
    return H

H_dense = _dense_hamiltonian_from_paulis(n, ps, w)
eigvals, _ = np.linalg.eigh(H_dense)
print("exact ground state energy: ", float(eigvals[0]))

# 1.1 VQE without shot noise (direct numeric, gradient-free)
print("VQE without shot noise")

def exact_wrapper(x):
    arr = np.asarray(x, dtype=np.float64).reshape(n, nlayers, 2)
    return float(exp_val_exact(arr).detach())

r = optimize.minimize(
    exact_wrapper,
    np.zeros([n * nlayers * 2], dtype=np.float64),
    method="COBYLA",
    options={"maxiter": 5},
)
print(r)

# 1.2 VQE without shot noise (direct numeric, gradient-based)
param = torch.nn.Parameter(torch.randn(n, nlayers, 2, dtype=torch.float64) * 0.1)
optimizer = torch.optim.Adam([param], lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
for i in range(10):
    optimizer.zero_grad()
    e_t = exp_val_exact(param)
    e_t.backward()
    optimizer.step()
    scheduler.step()
    if i % 5 == 4:
        print(float(e_t.detach()))

# 2.1 VQE with finite shot noise: gradient free
print("VQE with shot noise")

def counts_wrapper(x):
    arr = np.asarray(x, dtype=np.float64).reshape(n, nlayers, 2)
    return float(exp_val_counts(arr, shots=256))

r = optimize.minimize(
    counts_wrapper,
    np.random.normal(scale=0.1, size=[n * nlayers * 2]).astype(np.float64),
    method="COBYLA",
    options={"maxiter": 5},
)
print(r)
print("converged as: ", counts_wrapper(r["x"]))

# 2.2 VQE with finite shot noise: gradient based
param = torch.randn(n, nlayers, 2, dtype=torch.float64) * 0.1
exp_grad = parameter_shift_grad_counts(shots=64)
optimizer = torch.optim.Adam([param], lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
for i in range(10):
    g = exp_grad(param)
    optimizer.zero_grad()
    if isinstance(param, torch.nn.Parameter):
        param.grad = g
    else:
        param = torch.nn.Parameter(param)
        param.grad = g
        optimizer = torch.optim.Adam([param], lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    optimizer.step()
    scheduler.step()
    if i % 5 == 4:
        print(float(exp_val_counts(param.detach(), shots=64)))

print("converged as:", float(exp_val_exact(param).detach()))
