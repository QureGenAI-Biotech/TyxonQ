"""
VQE with finite measurement shot noise (counts path) and a direct numeric path for comparison.

- Counts path: chain API + finite shots, gradient-free (SPSA/Compass) and gradient-based (parameter-shift).
- Direct path: numeric_backend + quantum_library + PyTorch autograd (exact, no shots).
"""

from __future__ import annotations

from functools import partial
import numpy as np
from noisyopt import minimizeCompass, minimizeSPSA
from tabulate import tabulate  # pip install tabulate
import torch
import tyxonq as tq


seed = 42
np.random.seed(seed)

# backend for direct numeric path
K = tq.set_backend("pytorch")

n = 2
nlayers = 2

# initial value of the parameters
initial_value = np.random.uniform(size=[n * nlayers * 2])

result = {
    "Algorithm / Optimization": ["Without Shot Noise", "With Shot Noise"],
    "SPSA (Gradient Free)": [],
    "Compass Search (Gradient Free)": [],
    "Adam (Gradient based)": [],
}

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
            theta = param[i, j, 0]
            # ZZ(2*theta) decomposition via CX-RZ-CX
            c.cx(i, i + 1)
            c.rz(i + 1, theta=2.0 * float(theta))
            c.cx(i, i + 1)
        for i in range(n):
            c.rx(i, theta=float(param[i, j, 1]))
    return c


def _term_expectation_from_counts(counts: dict[str, int], term: list[int]) -> float:
    total = sum(counts.values()) or 1
    acc = 0.0
    for bitstr, cnt in counts.items():
        val = 1.0
        # Only X/Z present in TFIM; we rotate X->Z before sampling
        for q, code in enumerate(term):
            if code == 3:  # Z
                val *= (1.0 if bitstr[q] == '0' else -1.0)
            elif code == 1:  # X measured via H rotation -> Z
                val *= (1.0 if bitstr[q] == '0' else -1.0)
            # code==0 ignored; Y not present in this TFIM
        acc += val * cnt
    return acc / total


def exp_val_counts(param, shots=1024) -> float:
    # counts-based expectation with finite shots; param shape: [n, nlayers, 2]
    c_base = generate_circuit(param)
    if isinstance(shots, int):
        per_term_shots = [shots for _ in range(len(ps))]
    else:
        per_term_shots = list(shots)
    loss = 0.0
    for term, wi, shot in zip(ps, w, per_term_shots):
        # build term-specific readout circuit: rotate X wires by H
        c = tq.Circuit(n, ops=list(c_base.ops))
        for q, code in enumerate(term):
            if code == 1:  # X
                c.h(q)
        for q in range(n):
            c.measure_z(q)
        out = c.device(provider="simulator", device="statevector", shots=shot).postprocessing(method=None).run()
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

    # build state
    psi = init_statevector(n, backend=nb)
    # prepend H on each qubit as in ansatz
    for i in range(n):
        psi = apply_1q_statevector(nb, psi, gate_h(), i, n)
    for j in range(nlayers):
        for i in range(n - 1):
            theta = param[i, j, 0]
            psi = apply_2q_statevector(nb, psi, gate_cx_4x4(), i, i + 1, n)
            psi = apply_1q_statevector(nb, psi, gate_rz(2.0 * theta), i + 1, n)
            psi = apply_2q_statevector(nb, psi, gate_cx_4x4(), i, i + 1, n)
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rx(param[i, j, 1]), i, n)

    # compute expectation for TFIM terms
    def _exp_zz(i: int) -> torch.Tensor:
        probs = nb.square(nb.abs(psi)) if hasattr(nb, "square") else nb.abs(psi) ** 2
        dim = 1 << n
        signs = [1.0 if (((k >> (n - 1 - i)) & 1) == ((k >> (n - 2 - i)) & 1)) else -1.0 for k in range(dim)]
        return nb.sum(nb.asarray(signs) * probs)

    e = torch.zeros((), dtype=torch.float64)
    # -sum X_i
    psi_x = psi
    for q in range(n):
        psi_x = apply_1q_statevector(nb, psi_x, gate_h(), q, n)
    for q in range(n):
        e = e + (-1.0) * expect_z_statevector(psi_x, q, n, backend=nb)
    # +sum Z_i Z_{i+1}
    for i in range(n - 1):
        e = e + _exp_zz(i)
    return e


# local parameter-shift gradient for counts path
from math import pi

def parameter_shift_grad_counts(shots: int = 64):
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


# 0. Exact result by diagonalizing a small dense Hamiltonian (optional reference)
try:
    hm = tq.quantum.PauliStringSum2COO_numpy(ps, w)
    hm = K.to_dense(hm)
    e, v = np.linalg.eigh(hm)
    exact_gs_energy = e[0]
except Exception:
    exact_gs_energy = None

print("==================================================================", flush=True)
print("Exact ground state energy: ", exact_gs_energy, flush=True)
print("==================================================================", flush=True)

# 1.1 VQE without shot noise (direct numeric, gradient-free)
print(">>> VQE without shot noise", flush=True)

def exact_wrapper(x):
    arr = np.asarray(x, dtype=np.float64).reshape(n, nlayers, 2)
    return float(exp_val_exact(arr).detach())

r = minimizeSPSA(
    func=exact_wrapper,
    x0=initial_value.copy(),
    niter=20,
    paired=False,
)
print(r, flush=True)
print(">> SPSA converged as:", exp_val_exact(np.asarray(r.x, dtype=np.float64).reshape(n, nlayers, 2)), flush=True)
result["SPSA (Gradient Free)"].append(float(exp_val_exact(np.asarray(r.x, dtype=np.float64).reshape(n, nlayers, 2)).detach()))

r = minimizeCompass(
    func=exact_wrapper,
    x0=initial_value.copy(),
    deltatol=0.3,
    feps=1e-2,
    paired=False,
)
print(r, flush=True)
print(">> Compass converged as:", exp_val_exact(np.asarray(r.x, dtype=np.float64).reshape(n, nlayers, 2)), flush=True)
result["Compass Search (Gradient Free)"].append(float(exp_val_exact(np.asarray(r.x, dtype=np.float64).reshape(n, nlayers, 2)).detach()))

# 1.2 VQE without shot noise (direct numeric, gradient-based)
param = torch.nn.Parameter(torch.tensor(initial_value.reshape((n, nlayers, 2)), dtype=torch.float64))
optimizer = torch.optim.Adam([param], lr=1e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
for i in range(10):
    optimizer.zero_grad()
    e = exp_val_exact(param)
    # exp_val_exact returns a torch scalar tensor
    e.backward()
    optimizer.step()
    if i % 5 == 4:
        print(f"Expectation value at iteration {i}: {float(e.detach())}", flush=True)
    if (i + 1) % 10 == 0:
        scheduler.step()
print(">> Adam converged as:", float(exp_val_exact(param).detach()), flush=True)
result["Adam (Gradient based)"].append(float(exp_val_exact(param).detach()))

# 2.1 VQE with finite shot noise (counts path, gradient-free)
print("==================================================================", flush=True)
print(">>> VQE with shot noise", flush=True)

def exp_val_counts_wrapper(x):
    arr = np.asarray(x, dtype=np.float64).reshape(n, nlayers, 2)
    return float(exp_val_counts(arr, shots=64))

r = minimizeSPSA(
    func=exp_val_counts_wrapper,
    x0=initial_value.copy(),
    niter=20,
    paired=False,
)
print(r, flush=True)
print(">> SPSA converged as:", float(exp_val_counts_wrapper(r["x"])), flush=True)
result["SPSA (Gradient Free)"].append(float(exp_val_counts_wrapper(r["x"])))

r = minimizeCompass(
    func=exp_val_counts_wrapper,
    x0=initial_value.copy(),
    deltatol=0.3,
    feps=3e-2,
    paired=False,
)
print(r, flush=True)
print(">> Compass converged as:", float(exp_val_counts_wrapper(r["x"])), flush=True)
result["Compass Search (Gradient Free)"].append(float(exp_val_counts_wrapper(r["x"])))

# 2.2 VQE with finite shot noise (counts path, gradient-based via local parameter-shift)
param = torch.tensor(initial_value.reshape((n, nlayers, 2)), dtype=torch.float64)
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
    optimizer.step()
    if i % 5 == 4:
        print(f"Expectation value at iteration {i}: {float(exp_val_counts(param, shots=64))}", flush=True)
    if (i + 1) % 10 == 0:
        scheduler.step()

print(">> Adam converged as:", float(exp_val_exact(param).detach()), flush=True)
result["Adam (Gradient based)"].append(float(exp_val_exact(param).detach()))

print("==================================================================", flush=True)
print(">>> Benchmark", flush=True)
print(">> Exact ground state energy: ", exact_gs_energy, flush=True)
# print(tabulate(result, headers="keys", tablefmt="github"))
print(result, flush=True)
