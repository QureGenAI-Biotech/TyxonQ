"""
Autograd vs Counts: simple 2-qubit ZZ expectation comparison.

- Path A (autograd): numeric backend (pytorch) + quantum_library (statevector & gates)
  computes expectation without sampling and supports autograd.
- Path B (counts): chain API builds circuit with CX-RZ(2*theta)-CX and measures counts.

Compares value and runtime.
"""

from __future__ import annotations

import time
import tyxonq as tq


def build_counts_circuit(theta: float):
    c = tq.Circuit(2)
    # prepare |++>
    c.h(0).h(1)
    # ZZ(theta) via CX-RZ(2*theta)-CX on (0,1)
    c.cx(0, 1)
    c.rz(1, theta=2.0 * float(theta))
    c.cx(0, 1)
    # measure Z on both qubits
    c.measure_z(0).measure_z(1)
    return c


def exp_zz_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values()) or 1
    acc = 0.0
    for bitstr, cnt in counts.items():
        z0 = 1.0 if bitstr[0] == '0' else -1.0
        z1 = 1.0 if bitstr[1] == '0' else -1.0
        acc += (z0 * z1) * cnt
    return acc / total


def path_counts(theta: float, shots: int = 4096) -> tuple[float, float]:
    c = build_counts_circuit(theta)
    t0 = time.perf_counter()
    res = c.device(provider="simulator", device="statevector", shots=shots).postprocessing(method=None).run()
    counts = res[0]["result"] if isinstance(res, list) else res.get("result", {})
    val = exp_zz_from_counts(counts)
    t1 = time.perf_counter()
    return val, (t1 - t0)


def path_autograd(theta0: float) -> tuple[float, float, float]:
    import torch
    nb = tq.set_backend("pytorch")
    from tyxonq.libs.quantum_library.kernels.statevector import (
        init_statevector, apply_1q_statevector, apply_2q_statevector,
    )
    from tyxonq.libs.quantum_library.kernels.gates import gate_h, gate_rz, gate_rzz

    theta = torch.tensor([theta0], dtype=torch.float64, requires_grad=True)
    t0 = time.perf_counter()
    psi = init_statevector(2, backend=nb)
    psi = apply_1q_statevector(nb, psi, gate_h(), 0, 2)
    psi = apply_1q_statevector(nb, psi, gate_h(), 1, 2)
    # ZZ via rank-4 gate exp(-i theta Z⊗Z / 1) with 2*theta to match counts path
    psi = apply_2q_statevector(nb, psi, gate_rzz(2.0 * theta[0]), 0, 1, 2)
    # probs and <Z0 Z1>
    probs = nb.square(nb.abs(psi)) if hasattr(nb, 'square') else nb.abs(psi) ** 2
    # build diag for Z⊗Z = diag(1, -1, -1, 1)
    diag = nb.asarray([1.0, -1.0, -1.0, 1.0])
    val = nb.sum(diag * probs)
    # backprop once
    (-val).backward()
    grad = float(theta.grad.detach()) if theta.grad is not None else 0.0
    t1 = time.perf_counter()
    return float(val.detach()), (t1 - t0), grad


if __name__ == "__main__":
    theta = 0.321

    # Counts path
    v_counts, dt_counts = path_counts(theta, shots=8192)

    # Autograd path
    try:
        v_auto, dt_auto, g_auto = path_autograd(theta)
        print(f"Counts: value={v_counts:.6f}, time={dt_counts*1e3:.1f} ms")
        print(f"Autograd: value={v_auto:.6f}, time={dt_auto*1e3:.1f} ms, d(-value)/dtheta={g_auto:.3e}")
    except Exception as e:
        print(f"Counts: value={v_counts:.6f}, time={dt_counts*1e3:.1f} ms")
        print("Autograd path skipped (torch unavailable):", str(e))
