"""
Parameter-shift demo (refactored to chain-style API and numeric backend).

- Uses basis rotation + measure_z to evaluate Y-expectation on a target qubit
- Implements parameter-shift gradients for RX and RZZ parameters
"""

from __future__ import annotations

import math
import tyxonq as tq
from tyxonq.postprocessing.metrics import expectation as counts_expectation


# Numeric backend
nb = tq.set_backend("numpy")

# Problem sizes
n = 6
m = 3


def build_layered_circuit_rx(param):
    c = tq.Circuit(n)
    for j in range(m):
        for i in range(n - 1):
            c.cnot(i, i + 1)
        for i in range(n):
            c.rx(i, theta=float(param[i, j]))
    return c


def y_expectation(c: tq.Circuit, q: int, shots: int = 0) -> float:
    # Rotate Y to Z via S† then H, then measure Z
    cc = c.extended([])
    cc.rz(q, theta=-0.5 * math.pi).h(q).measure_z(q)
    if shots and shots > 0:
        res = (
            cc.device(provider="simulator", device="statevector", shots=shots)
              .postprocessing(method=None)
              .run()
        )
        payload = res if isinstance(res, dict) else (res[0] if res else {})
        counts = payload.get("results", {})
        # Diagonal op: Z on q, I elsewhere
        diag = [[1.0, 1.0] for _ in range(n)]
        diag[q] = [1.0, -1.0]
        return float(counts_expectation(counts, diagonal_op=diag))
    else:
        res = (
            cc.device(provider="simulator", device="statevector", shots=0)
              .postprocessing(method=None)
              .run()
        )
        payload = res if isinstance(res, dict) else (res[0] if res else {})
        exps = payload.get("expectations", {})
        return float(exps.get(f"Z{q}", 0.0))


def f1(param, *, shots: int = 0) -> float:
    c = build_layered_circuit_rx(param)
    return y_expectation(c, q=n // 2, shots=shots)


def parameter_shift_grad_f1(param, *, shots: int = 0, shift: float = 0.5 * math.pi):
    p = nb.to_numpy(param)
    grad = nb.zeros_like(p)
    for i in range(n):
        for j in range(m):
            p_plus = p.copy(); p_minus = p.copy()
            p_plus[i, j] += shift
            p_minus[i, j] -= shift
            f_plus = f1(p_plus, shots=shots)
            f_minus = f1(p_minus, shots=shots)
            grad[i, j] = 0.5 * (f_plus - f_minus)
    return grad


def build_layered_circuit_rzz_rx(paramzz, paramx):
    c = tq.Circuit(n)
    for j in range(m):
        for i in range(n - 1):
            c.rzz(i, i + 1, theta=float(paramzz[i, j]))
        for i in range(n):
            c.rx(i, theta=float(paramx[i, j]))
    return c


def f2(paramzz, paramx, *, shots: int = 0) -> float:
    c = build_layered_circuit_rzz_rx(paramzz, paramx)
    return y_expectation(c, q=n // 2, shots=shots)


def parameter_shift_grad_f2(paramzz, paramx, *, shots: int = 0, shift: float = 0.5 * math.pi):
    pz = nb.to_numpy(paramzz)
    px = nb.to_numpy(paramx)
    gz = nb.zeros_like(pz)
    gx = nb.zeros_like(px)
    for i in range(n):
        for j in range(m):
            # grad w.r.t rzz
            pz_plus = pz.copy(); pz_minus = pz.copy()
            pz_plus[i, j] += shift
            pz_minus[i, j] -= shift
            gz[i, j] = 0.5 * (f2(pz_plus, px, shots=shots) - f2(pz_minus, px, shots=shots))
            # grad w.r.t rx
            px_plus = px.copy(); px_minus = px.copy()
            px_plus[i, j] += shift
            px_minus[i, j] -= shift
            gx[i, j] = 0.5 * (f2(pz, px_plus, shots=shots) - f2(pz, px_minus, shots=shots))
    return gz, gx


# --- Finite-difference baselines for testing ---
def finite_difference_grad_f1(param, *, shots: int = 0, eps: float = 1e-6):
    p = nb.to_numpy(param)
    g = nb.zeros_like(p)
    for i in range(n):
        for j in range(m):
            p_plus = p.copy(); p_minus = p.copy()
            p_plus[i, j] += eps
            p_minus[i, j] -= eps
            f_plus = f1(p_plus, shots=shots)
            f_minus = f1(p_minus, shots=shots)
            g[i, j] = (f_plus - f_minus) / (2.0 * eps)
    return g


def finite_difference_grad_f2(paramzz, paramx, *, shots: int = 0, eps: float = 1e-6):
    pz = nb.to_numpy(paramzz); px = nb.to_numpy(paramx)
    gz = nb.zeros_like(pz); gx = nb.zeros_like(px)
    # grad wrt z parameters
    for i in range(n):
        for j in range(m):
            pz_plus = pz.copy(); pz_minus = pz.copy()
            pz_plus[i, j] += eps
            pz_minus[i, j] -= eps
            f_plus = f2(pz_plus, px, shots=shots)
            f_minus = f2(pz_minus, px, shots=shots)
            gz[i, j] = (f_plus - f_minus) / (2.0 * eps)
    # grad wrt x parameters
    for i in range(n):
        for j in range(m):
            px_plus = px.copy(); px_minus = px.copy()
            px_plus[i, j] += eps
            px_minus[i, j] -= eps
            f_plus = f2(pz, px_plus, shots=shots)
            f_minus = f2(pz, px_minus, shots=shots)
            gx[i, j] = (f_plus - f_minus) / (2.0 * eps)
    return gz, gx


def main():
    import time

    print("-- Parameter-shift vs Finite-Diff (shots=0, analytic expectations) --")
    init1 = nb.ones((n, m), dtype=nb.float32)
    t0 = time.perf_counter(); g_ps = parameter_shift_grad_f1(init1, shots=0); t1 = time.perf_counter()
    t2 = time.perf_counter(); g_fd = finite_difference_grad_f1(init1, shots=0); t3 = time.perf_counter()
    print("f1 shapes:", nb.to_numpy(g_ps).shape)
    print(f"f1 time: param-shift={t1-t0:.4f}s, finite-diff={t3-t2:.4f}s")
    # Assert close within tolerance for analytic expectations
    import numpy as _np
    _np.testing.assert_allclose(nb.to_numpy(g_ps), nb.to_numpy(g_fd), atol=1e-4, rtol=1e-2)
    print("f1 gradient check passed")

    initz = nb.ones((n, m), dtype=nb.float32)
    initx = nb.ones((n, m), dtype=nb.float32)
    t0 = time.perf_counter(); gz_ps, gx_ps = parameter_shift_grad_f2(initz, initx, shots=0); t1 = time.perf_counter()
    t2 = time.perf_counter(); gz_fd, gx_fd = finite_difference_grad_f2(initz, initx, shots=0); t3 = time.perf_counter()
    print("f2 shapes:", nb.to_numpy(gz_ps).shape, nb.to_numpy(gx_ps).shape)
    print(f"f2 time: param-shift={t1-t0:.4f}s, finite-diff={t3-t2:.4f}s")
    _np.testing.assert_allclose(nb.to_numpy(gz_ps), nb.to_numpy(gz_fd), atol=1e-4, rtol=1e-2)
    _np.testing.assert_allclose(nb.to_numpy(gx_ps), nb.to_numpy(gx_fd), atol=1e-4, rtol=1e-2)
    print("f2 gradient check passed")

    print("-- Parameter-shift (sampled, shots=8192, smoke test) --")
    g1_s = parameter_shift_grad_f1(init1, shots=8192)
    print("grad f1 (sampled) shape:", nb.to_numpy(g1_s).shape)


if __name__ == "__main__":
    main()
