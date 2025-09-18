"""
Time comparison for different evaluation approaches on spin VQE (TFIM-like).
- Direct numeric path: quantum_library + pytorch autograd (no shots)
- Counts path: chain API with finite shots, measurement-based expectations
"""

from __future__ import annotations

import time
import numpy as np
import torch
import tyxonq as tq

K = tq.set_backend("pytorch")

n = 10
nlayers = 1
Jx, h = 1.0, -1.0


def ansatz_ops_xx_rz(n: int, nlayers: int, param: np.ndarray | torch.Tensor):
    # param shape: [2*nlayers, n]
    ops = []
    t = 0
    for _ in range(nlayers):
        # RXX along chain
        for i in range(n - 1):
            ops.append(("rxx", i, i + 1, float(param[t, i])))
        t += 1
        # RZ on each wire
        for i in range(n):
            ops.append(("rz", i, float(param[t, i])))
        t += 1
    return ops


def counts_energy(param: np.ndarray, shots: int = 1024) -> float:
    # H = h * sum Z_i + Jx * sum X_i X_{i+1}
    c = tq.Circuit(n, ops=ansatz_ops_xx_rz(n, nlayers, param))

    def _term_counts(x_sites: list[int] | None, z_sites: list[int] | None, shots: int) -> float:
        cc = tq.Circuit(n, ops=list(c.ops))
        if x_sites:
            for q in x_sites:
                cc.h(q)
        for q in range(n):
            cc.measure_z(q)
        out = cc.device(provider="simulator", device="statevector", shots=shots).postprocessing(method=None).run()
        counts = out[0]["result"] if isinstance(out, list) else out.get("result", {})
        total = sum(counts.values()) or 1
        acc = 0.0
        sites = (x_sites or []) + (z_sites or [])
        for bitstr, cnt in counts.items():
            val = 1.0
            for q in sites:
                val *= (1.0 if bitstr[q] == '0' else -1.0)
            acc += val * cnt
        return acc / total

    e = 0.0
    # h * sum Z_i
    for i in range(n):
        e += h * _term_counts(None, [i], shots)
    # Jx * sum X_i X_{i+1}
    for i in range(n - 1):
        e += Jx * _term_counts([i, i + 1], None, shots)
    return e


def exact_energy(param: torch.Tensor) -> torch.Tensor:
    nb = tq.get_backend("pytorch")
    from tyxonq.libs.quantum_library.kernels.statevector import (
        init_statevector, apply_1q_statevector, apply_2q_statevector, expect_z_statevector,
    )
    from tyxonq.libs.quantum_library.kernels.gates import gate_h, gate_rxx, gate_rz

    psi = init_statevector(n, backend=nb)
    # build state
    t = 0
    for _ in range(nlayers):
        for i in range(n - 1):
            psi = apply_2q_statevector(nb, psi, gate_rxx(param[t, i]), i, i + 1, n)
        t += 1
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rz(param[t, i]), i, n)
        t += 1

    e = torch.zeros((), dtype=torch.float64)
    # h * sum Z_i
    for i in range(n):
        e = e + h * expect_z_statevector(psi, i, n, backend=nb)
    # Jx * sum X_i X_{i+1}: rotate both by H then <Z Z>
    psi_x = psi
    for q in range(n):
        psi_x = apply_1q_statevector(nb, psi_x, gate_h(), q, n)
    def _zz(i: int) -> torch.Tensor:
        probs = nb.square(nb.abs(psi_x)) if hasattr(nb, 'square') else nb.abs(psi_x) ** 2
        dim = 1 << n
        signs = [1.0 if (((k >> (n - 1 - i)) & 1) == ((k >> (n - 2 - i)) & 1)) else -1.0 for k in range(dim)]
        return torch.sum(torch.as_tensor(signs, dtype=torch.float64) * probs)
    for i in range(n - 1):
        e = e + Jx * _zz(i)
    return e


def benchmark(fn, *args, tries: int = 3):
    t0 = time.time(); v0 = fn(*args); t1 = time.time()
    for _ in range(max(0, tries)):
        _ = fn(*args)
    t2 = time.time()
    stage = t1 - t0
    run = (t2 - t1) / max(1, tries)
    return v0, (stage, run)


if __name__ == "__main__":
    param0 = np.zeros([2 * nlayers, n], dtype=np.float64)
    v_c, (s_c, r_c) = benchmark(counts_energy, param0, 512, tries=2)
    print({"counts_energy": v_c, "stage_s": s_c, "run_s": r_c})

    p_exact = torch.tensor(param0, dtype=torch.float64)
    v_e, (s_e, r_e) = benchmark(lambda p: exact_energy(p).detach(), p_exact, tries=2)
    print({"exact_energy": float(v_e), "stage_s": s_e, "run_s": r_e})
