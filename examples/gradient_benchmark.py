"""
Gradient evaluation benchmark: TyxonQ (counts + parameter-shift) vs Qiskit (optional).

- TyxonQ path uses chainable API and counts-based expectation of sum(X_i).
- Gradient computed via parameter-shift over ansatz parameters (Rx/Rz only).
- Also provide direct state-based paths using numeric backend + quantum_library:
  - qfi_tq_fd: finite-difference QFI via statevector
  - hessian_tq_autograd: PyTorch autograd Hessian via statevector
- Qiskit path uses opflow Gradient/QFI/Hessian if qiskit is installed (optional).
"""

from __future__ import annotations

import time
import json
from typing import List, Tuple

try:
    from qiskit.opflow import X, StateFn  # type: ignore
    from qiskit.circuit import QuantumCircuit, ParameterVector  # type: ignore
    from qiskit.opflow.gradients import Gradient, QFI, Hessian  # type: ignore
    _HAS_QISKIT = True
except Exception:  # pragma: no cover
    X = StateFn = QuantumCircuit = ParameterVector = Gradient = QFI = Hessian = None  # type: ignore
    _HAS_QISKIT = False

import tyxonq as tq


def benchmark(fn, *args, trials: int = 5) -> Tuple[float, Tuple[float, float]]:
    t0 = time.time(); _ = fn(*args); t1 = time.time()
    for _ in range(trials):
        _ = fn(*args)
    t2 = time.time()
    stage = t1 - t0
    run = (t2 - t1) / max(1, trials)
    return stage + run, (stage, run)


# ---------- TyxonQ counts path ----------

def _ansatz_ops(n: int, l: int, params: List[float]) -> List[Tuple]:
    ops: List[Tuple] = []
    idx = 0
    for j in range(l):
        for i in range(n - 1):
            ops.append(("cx", i, i + 1))
        for i in range(n):
            ops.append(("rx", i, float(params[idx])))
            idx += 1
        for i in range(n):
            ops.append(("rz", i, float(params[idx])))
            idx += 1
        for i in range(n):
            ops.append(("rx", i, float(params[idx])))
            idx += 1
    # rotate X->Z and measure
    for i in range(n):
        ops.append(("h", i))
        ops.append(("measure_z", i))
    return ops


def _objective_counts(n: int, l: int, params: List[float], shots: int = 4096) -> float:
    c = tq.Circuit(n, ops=_ansatz_ops(n, l, params))
    res = c.device(provider="simulator", device="statevector", shots=shots).postprocessing(method=None).run()
    counts = res[0]["result"] if isinstance(res, list) else res.get("result", {})
    total = sum(counts.values()) or 1
    acc = 0.0
    for bitstr, cnt in counts.items():
        # expectation of sum(X_i): after H, X->Z and bit 0->+1, 1->-1
        val = 0.0
        for i in range(n):
            val += (1.0 if bitstr[i] == '0' else -1.0)
        acc += val * cnt
    return acc / total


def _shift_for_gate(name: str) -> float:
    # Parameter-shift for Rx/Rz
    if name in ("rx", "rz"):
        return 0.5 * 3.141592653589793
    return 0.0


def _param_indices_layout(n: int, l: int) -> List[Tuple[str, int]]:
    layout: List[Tuple[str, int]] = []
    for _ in range(l):
        layout += [("rx", i) for i in range(n)]
        layout += [("rz", i) for i in range(n)]
        layout += [("rx", i) for i in range(n)]
    return layout


def gradient_tq_counts(n: int, l: int, params: List[float], shots: int = 4096) -> List[float]:
    layout = _param_indices_layout(n, l)
    grads: List[float] = []
    base = list(params)
    for k, (gname, _) in enumerate(layout):
        s = _shift_for_gate(gname)
        if s == 0.0:
            grads.append(0.0)
            continue
        p_plus = list(base); p_plus[k] = base[k] + s
        p_minus = list(base); p_minus[k] = base[k] - s
        f_plus = _objective_counts(n, l, p_plus, shots=shots)
        f_minus = _objective_counts(n, l, p_minus, shots=shots)
        grads.append(0.5 * (f_plus - f_minus))
    return grads


# ---------- Direct state paths via numeric backend + quantum_library ----------

def _build_state_statevector(nb, n: int, l: int, params: List[float]):
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
    for _ in range(l):
        for i in range(n - 1):
            psi = apply_2q_statevector(nb, psi, gate_cx_4x4(), i, i + 1, n)
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rx(params[idx]), i, n); idx += 1
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rz(params[idx]), i, n); idx += 1
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rx(params[idx]), i, n); idx += 1
    return psi


def qfi_tq_fd(n: int, l: int, params: List[float], eps: float = 1e-3):
    # Finite-difference QFI: J_ij = Re(<dpsi_i|dpsi_j>) with dpsi_k ≈ (psi(p+e_k)-psi(p-e_k))/(2eps)
    import numpy as np
    tq.set_backend("numpy")  # ensure kernels use numpy-backed tensors
    nb = tq.get_backend("numpy")
    def psi_at(pv: List[float]):
        psi = _build_state_statevector(nb, n, l, pv)
        return np.asarray(psi, dtype=np.complex128)
    base = list(params)
    dim = len(base)
    dpsi = []
    for k in range(dim):
        p_plus = list(base); p_plus[k] = base[k] + eps
        p_minus = list(base); p_minus[k] = base[k] - eps
        dpsi_k = (psi_at(p_plus) - psi_at(p_minus)) / (2.0 * eps)
        dpsi.append(dpsi_k)
    J = np.empty((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            J[i, j] = float(np.real(np.vdot(dpsi[i], dpsi[j])))
    return J


def hessian_tq_autograd(n: int, l: int, params_init: List[float]):
    # Hessian of objective sum(X_i) via PyTorch autograd
    import torch
    tq.set_backend("pytorch")  # ensure kernels use pytorch-backed tensors
    nb = tq.get_backend("pytorch")
    from tyxonq.libs.quantum_library.kernels.statevector import expect_z_statevector, apply_1q_statevector
    from tyxonq.libs.quantum_library.kernels.gates import gate_h

    def objective(vec):
        pv = vec
        psi = _build_state_statevector(nb, n, l, pv)
        # rotate X->Z by H then compute sum(Z_i)
        for q in range(n):
            psi = apply_1q_statevector(nb, psi, gate_h(), q, n)
        total = torch.zeros((), dtype=vec.dtype)
        for q in range(n):
            vq = expect_z_statevector(psi, q, n, backend=nb)
            vq_t = vq if isinstance(vq, torch.Tensor) else torch.as_tensor(vq, dtype=vec.dtype)
            total = total + vq_t
        return total

    x = torch.tensor(params_init, dtype=torch.float64, requires_grad=True)
    H = torch.autograd.functional.hessian(lambda v: objective(v), x, vectorize=True)
    return H.detach()


# ---------- Qiskit optional paths ----------

def gradient_qiskit(n: int, l: int) -> Tuple[float, Tuple[float, float]]:
    if not _HAS_QISKIT:
        return 0.0, (0.0, 0.0)
    hamiltonian = X
    for _ in range(1, n):
        hamiltonian = hamiltonian ^ X  # type: ignore[operator]
    qc = QuantumCircuit(n)
    params = ParameterVector("theta", length=3 * n * l)
    t = 0
    for _ in range(l):
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n):
            qc.rx(params[t + i], i)
        t += n
        for i in range(n):
            qc.rz(params[t + i], i)
        t += n
        for i in range(n):
            qc.rx(params[t + i], i)
        t += n
    op = ~StateFn(hamiltonian) @ StateFn(qc)
    grad = Gradient().convert(operator=op, params=params)

    def eval_grad(values):
        value_dict = {params: values}
        _ = grad.assign_parameters(value_dict).eval()
        return _

    return benchmark(eval_grad, [1.0] * (3 * n * l), trials=1)


def qfi_qiskit(n: int, l: int) -> Tuple[float, Tuple[float, float]]:
    if not _HAS_QISKIT:
        return 0.0, (0.0, 0.0)
    qc = QuantumCircuit(n)
    params = ParameterVector("theta", length=3 * n * l)
    t = 0
    for _ in range(l):
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n):
            qc.rx(params[t + i], i)
        t += n
        for i in range(n):
            qc.rz(params[t + i], i)
        t += n
        for i in range(n):
            qc.rx(params[t + i], i)
        t += n
    nat_grad = QFI().convert(operator=StateFn(qc), params=params)

    def eval_qfi(values):
        value_dict = {params: values}
        _ = nat_grad.assign_parameters(value_dict).eval()
        return _

    return benchmark(eval_qfi, [1.0] * (3 * n * l), trials=1)


def hessian_qiskit(n: int, l: int) -> Tuple[float, Tuple[float, float]]:
    if not _HAS_QISKIT:
        return 0.0, (0.0, 0.0)
    hamiltonian = X
    for _ in range(1, n):
        hamiltonian = hamiltonian ^ X  # type: ignore[operator]
    qc = QuantumCircuit(n)
    params = ParameterVector("theta", length=3 * n * l)
    t = 0
    for _ in range(l):
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n):
            qc.rx(params[t + i], i)
        t += n
        for i in range(n):
            qc.rz(params[t + i], i)
        t += n
        for i in range(n):
            qc.rx(params[t + i], i)
        t += n
    op = ~StateFn(hamiltonian) @ StateFn(qc)
    hs = Hessian().convert(operator=op, params=params)

    def eval_hs(values):
        value_dict = {params: values}
        _ = hs.assign_parameters(value_dict).eval()
        return _

    return benchmark(eval_hs, [1.0] * (3 * n * l), trials=1)


if __name__ == "__main__":
    n, l = 4, 2
    init = [0.1] * (3 * n * l)

    # TyxonQ counts path benchmark (parameter-shift gradient)
    t_val, (stage, run) = benchmark(lambda p: gradient_tq_counts(n, l, p, shots=2048), init, trials=1)
    print({"tq_counts_stage": stage, "tq_counts_run": run})

    # Direct state: QFI (finite-diff, numpy backend)
    t_qfi_fd, (stage_qfi_fd, run_qfi_fd) = benchmark(lambda p: qfi_tq_fd(n, l, p), init, trials=1)
    print({"tq_qfi_fd_stage": stage_qfi_fd, "tq_qfi_fd_run": run_qfi_fd})

    # Direct state: Hessian via PyTorch autograd
    t_hs_tq, (stage_hs_tq, run_hs_tq) = benchmark(lambda p: hessian_tq_autograd(n, l, p), init, trials=1)
    print({"tq_hessian_autograd_stage": stage_hs_tq, "tq_hessian_autograd_run": run_hs_tq})

    # Optional Qiskit comparisons
    if _HAS_QISKIT:
        _, (stage_g, run_g) = gradient_qiskit(n, l)
        print({"qiskit_grad_stage": stage_g, "qiskit_grad_run": run_g})
        _, (stage_fi, run_fi) = qfi_qiskit(n, l)
        print({"qiskit_qfi_stage": stage_fi, "qiskit_qfi_run": run_fi})
        _, (stage_hs, run_hs) = hessian_qiskit(n, l)
        print({"qiskit_hessian_stage": stage_hs, "qiskit_hessian_run": run_hs})

    # Save a minimal report
    out = {"n": n, "l": l, "tq_counts_ms": (stage + run) * 1e3, "tq_qfi_fd_ms": (stage_qfi_fd + run_qfi_fd) * 1e3}
    if _HAS_QISKIT:
        out.update({
            "qiskit_grad_ms": (stage_g + run_g) * 1e3,
            "qiskit_qfi_ms": (stage_fi + run_fi) * 1e3,
            "qiskit_hessian_ms": (stage_hs + run_hs) * 1e3,
        })
    with open("gradient_results.data", "w") as f:
        json.dump(out, f)
    with open("gradient_results.data", "r") as f:
        print(json.load(f))
