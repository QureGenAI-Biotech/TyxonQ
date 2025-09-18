"""
Evaluate expectation and gradients for a Pauli-sum observable using chain API.

- Use numeric_backend as `nb` (no K/E)
- Replace legacy expectation_ps/sample_expectation_ps with basis-rotation + counts
"""

from __future__ import annotations

from copy import deepcopy
from typing import List
import tyxonq as tq
from tyxonq.postprocessing.metrics import expectation as counts_expectation


# Backend selection (can be changed to "pytorch" / "cupynumeric" if available)
nb = tq.set_backend("numpy")


# Problem setup
n = 5
nlayers = 4

# Pauli strings encoding per wire: 0=I, 1=X, 2=Y, 3=Z
ps: List[List[int]] = []
for i in range(n):
    row = [0] * n
    row[i] = 1  # X_i
    ps.append(row)
for i in range(n - 1):
    row = [0] * n
    row[i] = 3  # Z_i
    row[i + 1] = 3  # Z_{i+1}
    ps.append(row)

w = [-1.0 for _ in range(n)] + [1.0 for _ in range(n - 1)]


def generate_circuit(param):
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for j in range(nlayers):
        for i in range(n - 1):
            c.rzz(i, i + 1, theta=float(param[i, j, 0]))
        for i in range(n):
            c.rx(i, theta=float(param[i, j, 1]))
    return c


def pauli_term_expectation(c: tq.Circuit, pauli: List[int], shots: int = 4096) -> float:
    # Clone circuit and append basis rotations + measurements per term
    cc = c.extended([])
    for q, p in enumerate(pauli):
        if p == 1:  # X: rotate to Z via H
            cc.h(q)
            cc.measure_z(q)
        elif p == 2:  # Y: rotate to Z via S† then H
            # S† = RZ(-pi/2); we approximate with rz
            cc.rz(q, theta=-1.5707963267948966)
            cc.h(q)
            cc.measure_z(q)
        elif p == 3:  # Z: direct Z measurement
            cc.measure_z(q)
        else:
            # Identity: no measurement needed for this wire
            continue
    res = (
        cc.device(provider="simulator", device="statevector", shots=shots)
          .postprocessing(method=None)
          .run()
    )
    payload = res if isinstance(res, dict) else (res[0] if res else {})
    counts = payload.get("results", {})
    # Build diagonal_op: Z on measured wires of this term, I elsewhere
    diag = []
    for p in pauli:
        if p in (1, 2, 3):
            diag.append([1.0, -1.0])
        else:
            diag.append([1.0, 1.0])
    return float(counts_expectation(counts, diagonal_op=diag))


def exp_val(param, shots: int = 4096) -> float:
    c = generate_circuit(param)
    total = 0.0
    for pauli, wi in zip(ps, w):
        total += float(wi) * pauli_term_expectation(c, pauli, shots=shots)
    return float(total)


def parameter_shift_gradient(param, shots: int = 4096, shift: float = 1.5707963267948966):
    # param shape: [n, nlayers, 2]; 0 -> rzz(i,i+1), 1 -> rx(i)
    p_np = nb.to_numpy(param)
    grad = nb.zeros_like(p_np)
    for i in range(n):
        for j in range(nlayers):
            # rzz theta
            for k in (0, 1):
                p_plus = p_np.copy(); p_minus = p_np.copy()
                p_plus[i, j, k] += shift
                p_minus[i, j, k] -= shift
                f_plus = exp_val(p_plus, shots=shots)
                f_minus = exp_val(p_minus, shots=shots)
                grad[i, j, k] = 0.5 * (f_plus - f_minus)
    return grad


def main():
    import time

    def finite_difference_gradient(param, shots: int = 4096, eps: float = 1e-6):
        p_np = nb.to_numpy(param)
        g = nb.zeros_like(p_np)
        for i in range(n):
            for j in range(nlayers):
                for k in (0, 1):
                    p_plus = p_np.copy(); p_minus = p_np.copy()
                    p_plus[i, j, k] += eps
                    p_minus[i, j, k] -= eps
                    f_plus = exp_val(p_plus, shots=shots)
                    f_minus = exp_val(p_minus, shots=shots)
                    g[i, j, k] = (f_plus - f_minus) / (2.0 * eps)
        return g

    init = nb.ones([n, nlayers, 2], dtype=nb.float32)
    shots = 8192
    print(f"exp (shots={shots}):", exp_val(init, shots=shots))
    t0 = time.perf_counter(); g_ps = parameter_shift_gradient(init, shots=shots); t1 = time.perf_counter()
    t2 = time.perf_counter(); g_fd = finite_difference_gradient(init, shots=shots); t3 = time.perf_counter()
    g_ps_np = nb.to_numpy(g_ps); g_fd_np = nb.to_numpy(g_fd)
    import numpy as _np
    diff_max = float(_np.max(_np.abs(g_ps_np - g_fd_np)))
    diff_l2 = float(_np.linalg.norm((g_ps_np - g_fd_np).reshape(-1)))
    print("grad shape:", g_ps_np.shape)
    print(f"time: param-shift={t1-t0:.4f}s, finite-diff={t3-t2:.4f}s")
    print(f"|grad_ps - grad_fd|_max={diff_max:.4e}, L2={diff_l2:.4e} (increase shots to reduce noise)")


#TODO gradient测试
# print("benchmarking sample expectation")
# tq.utils.benchmark(
#     exp_val, K.ones([n, nlayers, 2], dtype="float32")
# )
# print("benchmarking analytical expectation")
# tq.utils.benchmark(exp_val_analytical, K.ones([n, nlayers, 2], dtype="float32"))
# r1 = exp_val(K.ones([n, nlayers, 2], dtype="float32"))
# r2 = exp_val_analytical(K.ones([n, nlayers, 2], dtype="float32"))
# np.testing.assert_allclose(r1.detach().cpu().numpy(), r2.detach().cpu().numpy(), atol=0.05, rtol=0.01)
# print("correctness check passed for expectation value")
# gradf1 = E.parameter_shift_grad_v2(exp_val, argnums=0)
# gradf2 = K.jit(K.grad(exp_val_analytical))
# print("benchmarking sample gradient")
# tq.utils.benchmark(
#     gradf1, K.ones([n, nlayers, 2], dtype="float32")
# )
# # n=12, nlayers=4, 276s + 0.75s, mac CPU
# print("benchmarking analytical gradient")
# tq.utils.benchmark(gradf2, K.ones([n, nlayers, 2], dtype="float32"))
# r1 = gradf1(K.ones([n, nlayers, 2], dtype="float32"))
# r2 = gradf2(K.ones([n, nlayers, 2], dtype="float32"))
# print("gradient with measurement shot and parameter shift")
# print(r1)
# print(r2)
# np.testing.assert_allclose(r1.detach().cpu().numpy(), r2.detach().cpu().numpy(), atol=0.2, rtol=0.01)
# print("correctness check passed for gradients")
if __name__ == "__main__":
    main()
