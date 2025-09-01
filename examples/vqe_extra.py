"""
VQE extra: TFIM-like Hamiltonian with counts-based energy and parameter-shift.

- Uses chain API: circuit.device(...).postprocessing(...).run()
- Builds hardware-efficient ansatz via libs.circuits_library.vqe
- Energy(H) = -J * sum Z_i Z_{i+1} - h * sum X_i
  We estimate expectations with two measurement settings:
    1) Z-basis run for Z_i Z_{i+1}
    2) X-basis run (apply H on all qubits before measure_z) for X_i
- Additionally: a direct numeric_backend + quantum_library + PyTorch autograd path for comparison.
"""

from __future__ import annotations

import time
from typing import List, Tuple, Dict

import tyxonq as tq
from tyxonq.libs.circuits_library import vqe as vqelib


def _counts_z_run(n: int, layers: int, params: List[float], *, shots: int) -> Dict[str, int]:
    c = tq.Circuit(n, ops=vqelib.build_hwe_ansatz_ops(n, layers, params))
    for q in range(n):
        c.measure_z(q)
    out = c.device(provider="simulator", device="statevector", shots=shots).postprocessing(method=None).run()
    return out[0]["result"] if isinstance(out, list) else out.get("result", {})


def _counts_x_run(n: int, layers: int, params: List[float], *, shots: int) -> Dict[str, int]:
    c = tq.Circuit(n, ops=vqelib.build_hwe_ansatz_ops(n, layers, params))
    for q in range(n):
        c.h(q)
        c.measure_z(q)
    out = c.device(provider="simulator", device="statevector", shots=shots).postprocessing(method=None).run()
    return out[0]["result"] if isinstance(out, list) else out.get("result", {})


def energy_tfim(n: int, layers: int, params: List[float], *, shots: int, J: float, h: float) -> float:
    counts_z = _counts_z_run(n, layers, params, shots=shots)
    counts_x = _counts_x_run(n, layers, params, shots=shots)

    Hzz: vqelib.Hamiltonian = [(-J, [("Z", i), ("Z", (i + 1) % n)]) for i in range(n)]
    Hx: vqelib.Hamiltonian = [(-h, [("Z", i)]) for i in range(n)]

    ez = vqelib.energy_from_counts(counts_z, n, Hzz)
    ex = vqelib.energy_from_counts(counts_x, n, Hx)
    return ez + ex


def grad_tfim_ps(n: int, layers: int, params: List[float], *, shots: int, J: float, h: float) -> List[float]:
    layout = vqelib._param_layout(n, layers)
    base = list(params)
    grads: List[float] = []
    from math import pi
    for k, (gname, _) in enumerate(layout):
        s = (0.5 * pi) if gname in ("rx", "rz") else 0.0
        if s == 0.0:
            grads.append(0.0)
            continue
        p_plus = list(base); p_plus[k] = base[k] + s
        p_minus = list(base); p_minus[k] = base[k] - s
        f_plus = energy_tfim(n, layers, p_plus, shots=shots, J=J, h=h)
        f_minus = energy_tfim(n, layers, p_minus, shots=shots, J=J, h=h)
        grads.append(0.5 * (f_plus - f_minus))
    return grads


# Direct path: numeric_backend + quantum_library + PyTorch autograd

def _build_state(nb, n: int, layers: int, params_any) -> any:
    from tyxonq.libs.quantum_library.kernels.statevector import (
        init_statevector,
        apply_1q_statevector,
        apply_2q_statevector,
    )
    from tyxonq.libs.quantum_library.kernels.gates import (
        gate_h,
        gate_rx,
        gate_rz,
        gate_cx_4x4,
    )
    psi = init_statevector(n, backend=nb)
    idx = 0
    for _ in range(layers):
        for i in range(n - 1):
            psi = apply_2q_statevector(nb, psi, gate_cx_4x4(), i, i + 1, n)
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rx(params_any[idx]), i, n); idx += 1
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rz(params_any[idx]), i, n); idx += 1
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rx(params_any[idx]), i, n); idx += 1
    return psi


def _exp_zz_from_state(nb, psi, n: int, u: int, v: int) -> any:
    probs = nb.square(nb.abs(psi)) if hasattr(nb, "square") else nb.abs(psi) ** 2
    dim = 1 << n
    signs = [1.0 if (((k >> (n - 1 - u)) & 1) == ((k >> (n - 1 - v)) & 1)) else -1.0 for k in range(dim)]
    s = nb.asarray(signs)
    return nb.sum(s * probs)


def energy_tfim_state_autograd(n: int, layers: int, params_init: List[float], *, J: float, h: float):
    import torch
    tq.set_backend("pytorch")
    nb = tq.get_backend("pytorch")
    from tyxonq.libs.quantum_library.kernels.statevector import apply_1q_statevector, expect_z_statevector
    from tyxonq.libs.quantum_library.kernels.gates import gate_h

    def objective(vec):
        pv = vec
        psi = _build_state(nb, n, layers, pv)
        ez = 0.0
        for i in range(n):
            ez = ez + (-J) * _exp_zz_from_state(nb, psi, n, i, (i + 1) % n)
        psi_x = psi
        for q in range(n):
            psi_x = apply_1q_statevector(nb, psi_x, gate_h(), q, n)
        ex = 0.0
        for q in range(n):
            ex = ex + (-h) * expect_z_statevector(psi_x, q, n, backend=nb)
        tot = ez + ex
        return tot if isinstance(tot, torch.Tensor) else torch.as_tensor(tot, dtype=vec.dtype)

    x = tq.get_backend("pytorch").asarray(params_init).to(tq.get_backend("pytorch").float64)  # type: ignore
    x = x.clone().detach().requires_grad_(True)  # type: ignore
    val = objective(x)
    g, = __import__("torch").autograd.grad(val, (x,), create_graph=False, allow_unused=False)  # type: ignore
    return val.detach(), g.detach()


def train_tfim_state_autograd(n: int, layers: int, params_init: List[float], *, J: float, h: float, steps: int = 20, lr: float = 0.02):
    """PyTorch autograd training loop for the direct state path (for comparison)."""
    import torch
    tq.set_backend("pytorch")
    nb = tq.get_backend("pytorch")
    from tyxonq.libs.quantum_library.kernels.statevector import apply_1q_statevector, expect_z_statevector
    from tyxonq.libs.quantum_library.kernels.gates import gate_h

    def objective(vec):
        pv = vec
        psi = _build_state(nb, n, layers, pv)
        ez = 0.0
        for i in range(n):
            ez = ez + (-J) * _exp_zz_from_state(nb, psi, n, i, (i + 1) % n)
        psi_x = psi
        for q in range(n):
            psi_x = apply_1q_statevector(nb, psi_x, gate_h(), q, n)
        ex = 0.0
        for q in range(n):
            ex = ex + (-h) * expect_z_statevector(psi_x, q, n, backend=nb)
        tot = ez + ex
        return tot if isinstance(tot, torch.Tensor) else torch.as_tensor(tot, dtype=vec.dtype)

    param = tq.get_backend("pytorch").asarray(params_init).to(tq.get_backend("pytorch").float64)  # type: ignore
    param = param.clone().detach().requires_grad_(True)  # type: ignore
    opt = __import__("torch").optim.Adam([param], lr=lr)  # type: ignore
    history: List[float] = []
    for it in range(steps):
        opt.zero_grad()
        val = objective(param)
        val.backward()
        opt.step()
        if it % 5 == 0:
            history.append(float(val.detach()))
    return float(val.detach()), param.detach(), history


if __name__ == "__main__":
    n, layers = 8, 2
    J, h = 1.0, 0.5
    shots = 2048

    init = [0.1] * (3 * n * layers)

    t0 = time.time()
    e0 = energy_tfim(n, layers, init, shots=shots, J=J, h=h)
    t1 = time.time()
    print({"counts_energy": e0, "stage_ms": (t1 - t0) * 1e3})

    g = grad_tfim_ps(n, layers, init, shots=shots, J=J, h=h)
    lr = 0.05
    nxt = [p - lr * gp for p, gp in zip(init, g)]
    e1 = energy_tfim(n, layers, nxt, shots=shots, J=J, h=h)
    print({"counts_after_step": e1})

    try:
        v_auto, g_auto = energy_tfim_state_autograd(n, layers, init, J=J, h=h)
        print({"state_energy": float(v_auto), "grad_norm": float((g_auto**2).sum().sqrt())})
        v_tr, p_tr, hist = train_tfim_state_autograd(n, layers, init, J=J, h=h, steps=20, lr=0.02)
        print({"state_energy_after_train": v_tr, "history": hist})
    except Exception as e:
        print("state_autograd skipped:", str(e))
