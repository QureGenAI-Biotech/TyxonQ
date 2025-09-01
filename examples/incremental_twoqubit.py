"""
Optimizing a parameterized circuit with progressively dense two-qubit interactions
(replacing legacy expectation_ps/exp1 with counts-based TFIM energy estimation),
并新增 PyTorch autograd 直接数值路径，保留结构掩码控制的两比特作用。

- Counts 路径：链式 API + 基变换测量 + counts→期望
- Autograd 路径：quantum_library.statevector + PyTorch 反向传播
"""

from __future__ import annotations

import numpy as np
import tyxonq as tq

n = 8
nlayers = 3
hx = -1.0


def binarize(x):
    # map real to {0,1}
    return ((np.sign(np.real(x)) + 1.0) / 2.0).astype(np.float64)


def build_circuit(n: int, nlayers: int, params: np.ndarray, structures: np.ndarray) -> tq.Circuit:
    # params shape [2*nlayers, n]; structures shape [nlayers, n-1]
    s = binarize(structures)
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for j in range(nlayers):
        # ZZ brick with structure mask (1: identity, 0: apply ZZ)
        for i in range(n - 1):
            theta_eff = (1.0 - s[j, i]) * params[2 * j + 1, i]
            if theta_eff != 0.0:
                # ZZ(theta) = CX (i->i+1) ; RZ(2*theta) on target ; CX
                c.cx(i, i + 1)
                c.rz(i + 1, theta=2.0 * float(theta_eff))
                c.cx(i, i + 1)
        # RX layer
        for i in range(n):
            c.rx(i, theta=float(params[2 * j, i]))
    return c


def counts_from(c: tq.Circuit, shots: int) -> dict:
    r = c.device(provider="simulator", device="statevector", shots=shots).postprocessing(method=None).run()
    return r[0]["result"] if isinstance(r, list) else r.get("result", {})


def expect_z_from_counts(counts: dict, n: int, sites: list[int]) -> float:
    total = sum(counts.values()) or 1
    acc = 0.0
    for bitstr, cnt in counts.items():
        val = 1.0
        for q in sites:
            val *= (1.0 if bitstr[q] == '0' else -1.0)
        acc += val * cnt
    return acc / total


def energy_counts(params: np.ndarray, structures: np.ndarray, shots: int = 4096) -> float:
    c_base = build_circuit(n, nlayers, params, structures)

    # For <Z_i Z_{i+1}>
    cz = tq.Circuit(n, ops=list(c_base.ops))
    for q in range(n):
        cz.measure_z(q)
    counts_z = counts_from(cz, shots)
    e = 0.0
    for i in range(n - 1):
        e += expect_z_from_counts(counts_z, n, [i, i + 1])

    # For <X_i>: rotate with H on qubit i
    for i in range(n):
        cx = tq.Circuit(n, ops=list(c_base.ops))
        cx.h(i)
        for q in range(n):
            cx.measure_z(q)
        counts_x = counts_from(cx, shots)
        e += hx * expect_z_from_counts(counts_x, n, [i])
    return e


# ====== 直接数值（PyTorch autograd）路径 ======

def energy_autograd(params_t, structures_t):
    import torch
    # 确保激活 PyTorch 后端，使 kernels 返回 torch 张量
    tq.set_backend("pytorch")
    nb = tq.get_backend("pytorch")
    from tyxonq.libs.quantum_library.kernels.statevector import (
        init_statevector,
        apply_1q_statevector,
        apply_2q_statevector,
        expect_z_statevector,
    )
    from tyxonq.libs.quantum_library.kernels.gates import gate_rx, gate_rzz, gate_h

    # binarize structures to {0,1}
    s = torch.sign(torch.real(structures_t))
    s = (s + 1.0) / 2.0
    s = s.to(dtype=params_t.dtype)

    # build state
    psi = init_statevector(n, backend=nb)
    # initial H on all qubits
    for i in range(n):
        psi = apply_1q_statevector(nb, psi, gate_h(), i, n)
    for j in range(nlayers):
        # ZZ brick with structure-controlled theta
        for i in range(n - 1):
            theta_eff = (1.0 - s[j, i]) * params_t[2 * j + 1, i]
            psi = apply_2q_statevector(nb, psi, gate_rzz(2.0 * theta_eff), i, i + 1, n)
        # RX layer
        for i in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rx(params_t[2 * j, i]), i, n)

    # Energy: sum <Z_i Z_{i+1}> + hx * sum <X_i>
    e = torch.zeros((), dtype=params_t.dtype)
    # Z Z terms via probabilities signs
    probs = nb.square(nb.abs(psi)) if hasattr(nb, 'square') else nb.abs(psi) ** 2
    dim = 1 << n
    for i in range(n - 1):
        signs = [1.0 if (((k >> (n - 1 - i)) & 1) == ((k >> (n - 2 - i)) & 1)) else -1.0 for k in range(dim)]
        e = e + torch.sum(torch.as_tensor(signs, dtype=params_t.dtype) * probs)
    # X terms by rotating with H on each site and calling <Z>
    for i in range(n):
        psi_x = apply_1q_statevector(nb, psi, gate_h(), i, n)
        e = e + hx * expect_z_statevector(psi_x, i, n, backend=nb)
    return e


if __name__ == "__main__":
    # Counts path quick eval
    rng = np.random.default_rng(42)
    params = rng.uniform(low=0.0, high=2 * np.pi, size=[2 * nlayers, n])
    structures = rng.uniform(low=-1.0, high=1.0, size=[nlayers, n - 1])
    counts_energy_initial = energy_counts(params, structures, shots=2048)
    print({"counts_energy": counts_energy_initial})

    # Autograd path quick training (small steps for demo)
    import torch
    torch.set_num_threads(1)
    tq.set_backend("pytorch")
    p_t = torch.tensor(params, dtype=torch.float64, requires_grad=True)
    s_t = torch.tensor(structures, dtype=torch.float64)
    autograd_energy_initial = float(energy_autograd(p_t, s_t).detach())
    opt = torch.optim.Adam([p_t], lr=1e-2)
    for it in range(10):
        loss = energy_autograd(p_t, s_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print({"it": it, "autograd_energy": float(loss.detach())})
    autograd_energy_final = float(energy_autograd(p_t, s_t).detach())

    # Compare counts energy with trained params
    params_trained = p_t.detach().cpu().numpy()
    counts_energy_final = energy_counts(params_trained, structures, shots=2048)

    print({
        "counts_energy_initial": counts_energy_initial,
        "autograd_energy_initial": autograd_energy_initial,
        "autograd_energy_final": autograd_energy_final,
        "counts_energy_final": counts_energy_final,
    })
