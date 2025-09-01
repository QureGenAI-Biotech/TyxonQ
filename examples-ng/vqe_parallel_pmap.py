"""
PyTorch-parallel VQE demo: batch/parallel mapping across parameter sets.
- Direct numeric path: quantum_library + numeric_backend (pytorch) + autograd
- Counts path: chain API sampling per Pauli term (finite shots)
"""

from __future__ import annotations

import math
from typing import List
import torch
import tyxonq as tq

K = tq.set_backend("pytorch")


def _xyz_to_ps(xyz: dict[str, List[int]], n: int) -> List[int]:
    codes = [0] * n
    for q in xyz.get("x", []):
        codes[q] = 1
    for q in xyz.get("y", []):
        codes[q] = 2
    for q in xyz.get("z", []):
        codes[q] = 3
    return codes


def get_tfim_ps(n: int) -> torch.Tensor:
    rows: List[List[int]] = []
    # X_i
    for i in range(n):
        rows.append(_xyz_to_ps({"x": [i]}, n))
    # Z_i Z_{i+1}
    for i in range(n):
        rows.append(_xyz_to_ps({"z": [i, (i + 1) % n]}, n))
    return torch.tensor(rows, dtype=torch.int64)


def vqef_exact(param: torch.Tensor, measure: torch.Tensor, n: int, nlayers: int) -> torch.Tensor:
    """Exact expectation (no shots) via statevector kernels.
    param shape: [nlayers, 2, n]; measure shape: [m, n] with codes (0/1/2/3)
    H = sum over provided Pauli strings in `measure` with unit weights.
    """
    nb = tq.get_backend("pytorch")
    from tyxonq.libs.quantum_library.kernels.statevector import (
        init_statevector, apply_1q_statevector, apply_2q_statevector, expect_z_statevector,
    )
    from tyxonq.libs.quantum_library.kernels.gates import gate_h, gate_rx, gate_rzz

    psi = init_statevector(n, backend=nb)
    # Ansatz: H on all, ZZ(2θ) edges + RX per wire
    for q in range(n):
        psi = apply_1q_statevector(nb, psi, gate_h(), q, n)
    for layer in range(nlayers):
        # ZZ along ring
        for i in range(n):
            theta = param[layer, 0, i % n]
            psi = apply_2q_statevector(nb, psi, gate_rzz(2.0 * theta), i % n, (i + 1) % n, n)
        # RX per wire
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rx(param[layer, 1, i]), i, n)

    def _term_exp(codes_row: torch.Tensor) -> torch.Tensor:
        # Only X/Z present in TFIM
        # rotate X->Z then <Z>
        loc_psi = psi
        for q in range(n):
            if int(codes_row[q]) == 1:
                loc_psi = apply_1q_statevector(nb, loc_psi, gate_h(), q, n)
        val = torch.zeros((), dtype=torch.float64)
        z_sites = [q for q in range(n) if int(codes_row[q]) in (1, 3)]
        if len(z_sites) == 1:
            q = z_sites[0]
            val = expect_z_statevector(loc_psi, q, n, backend=nb)
        elif len(z_sites) == 2:
            probs = nb.square(nb.abs(loc_psi)) if hasattr(nb, "square") else nb.abs(loc_psi) ** 2
            dim = 1 << n
            i, j = z_sites
            signs = [1.0 if (((k >> (n - 1 - i)) & 1) == ((k >> (n - 1 - j)) & 1)) else -1.0 for k in range(dim)]
            val = torch.sum(torch.as_tensor(signs, dtype=torch.float64) * probs)
        return val

    total = torch.zeros((), dtype=torch.float64)
    for row in measure:
        total = total + _term_exp(row)
    return total


def vqef_counts(param: torch.Tensor, measure: torch.Tensor, n: int, nlayers: int, shots: int = 256) -> float:
    """Counts-based finite-shot expectation for provided Pauli strings."""
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for layer in range(nlayers):
        for i in range(n):
            theta = float(param[layer, 0, i % n])
            c.cx(i % n, (i + 1) % n); c.rz((i + 1) % n, theta=2.0 * theta); c.cx(i % n, (i + 1) % n)
        for i in range(n):
            c.rx(i, theta=float(param[layer, 1, i]))

    def _single_term(codes_row: torch.Tensor) -> float:
        cc = tq.Circuit(n, ops=list(c.ops))
        for q in range(n):
            if int(codes_row[q]) == 1:  # X
                cc.h(q)
        for q in range(n):
            cc.measure_z(q)
        out = cc.device(provider="simulator", device="statevector", shots=shots).postprocessing(method=None).run()
        counts = out[0]["result"] if isinstance(out, list) else out.get("result", {})
        total = sum(counts.values()) or 1
        acc = 0.0
        z_sites = [q for q in range(n) if int(codes_row[q]) in (1, 3)]
        for bitstr, cnt in counts.items():
            val = 1.0
            for q in z_sites:
                val *= (1.0 if bitstr[q] == '0' else -1.0)
            acc += val * cnt
        return acc / total

    loss = 0.0
    for row in measure:
        loss += _single_term(row)
    return float(loss)


# Simple parallel/batch mapping across parameter sets using torch.func.grad/vmap
try:
    from torch.func import vmap as torch_vmap  # type: ignore
    from torch.func import grad as torch_grad  # type: ignore
    def batch_update(params_batch: torch.Tensor, measure: torch.Tensor, n: int, nlayers: int):
        def single_val(p):
            return vqef_exact(p, measure, n, nlayers)
        single_grad = torch_grad(single_val)
        vals = torch_vmap(single_val)(params_batch)
        grads = torch_vmap(single_grad)(params_batch)
        return vals, grads
except Exception:
    def batch_update(params_batch: torch.Tensor, measure: torch.Tensor, n: int, nlayers: int):
        vals_list = []; grads_list = []
        for p in params_batch:
            p_req = p.clone().detach().requires_grad_(True)
            val = vqef_exact(p_req, measure, n, nlayers)
            val.backward()
            vals_list.append(val.detach()); grads_list.append(p_req.grad.detach())
        return torch.stack(vals_list), torch.stack(grads_list)


if __name__ == "__main__":
    n = 6; nlayers = 2
    m = get_tfim_ps(n)
    # batch of 4 parameter sets
    B = 4
    params0 = torch.randn(B, nlayers, 2, n, dtype=torch.float64) * 0.1

    # Parallel batch value-and-grad (exact path)
    vals, grads = batch_update(params0, m, n, nlayers)

    grads_flat = grads.reshape(grads.shape[0], -1)
    print({"batch_vals_mean": float(vals.mean()), "grads_norm_mean": float(torch.linalg.norm(grads_flat, dim=1).mean()), "grads_shape": list(grads.shape)})

    # One counts-path evaluation on the first batch member
    v_counts = vqef_counts(params0[0], m, n, nlayers, shots=128)
    print({"counts_value": v_counts})
