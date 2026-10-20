"""费米子模式排序：MILP 优化 JW 激发跨度，独立于硬件路由。"""

from dataclasses import dataclass
from collections import Counter
from operator import index

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass
class LayoutResult:
    permutation: tuple[int, ...]
    status: str
    objective: float


def optimize_mode_order(excitations, num_modes, *, time_limit=10.0, mix_delta=0.1):
    """excitation 仅包含改变占据的模式；不把 number spectator 算入 Z 链跨度。"""
    n = index(num_modes)
    if n < 1 or not np.isfinite(time_limit) or time_limit <= 0 or not np.isfinite(mix_delta) or mix_delta < 0:
        raise ValueError("invalid layout size, time_limit or mix_delta")
    frequencies = Counter(tuple(sorted(set(index(p) for p in s))) for s in excitations)
    frequencies = Counter({s: count for s, count in frequencies.items() if len(s) > 1})
    supports = sorted(frequencies)
    if any(p < 0 or p >= n for support in supports for p in support):
        raise ValueError("excitation mode out of range")

    def objective(perm):
        spans = [max(perm[p] for p in s) - min(perm[p] for p in s) + 1 for s in supports]
        average = sum(span * frequencies[s] for s, span in zip(supports, spans)) / sum(frequencies.values()) if spans else 0
        return float(max(spans) + mix_delta * average) if spans else 0.0

    identity = tuple(range(n))
    if not supports:
        return LayoutResult(identity, "no_excitations", 0.0)
    m = len(supports)
    # x[old,new] 是二元排列矩阵，随后为每个激发的 low/high 和全局最大跨度。
    size = n * n + 2 * m + 1
    cost = np.zeros(size)
    cost[-1] = 1
    for s in range(m):
        weight = mix_delta * frequencies[supports[s]] / sum(frequencies.values())
        cost[n * n + 2 * s] = -weight
        cost[n * n + 2 * s + 1] = weight
    rows, cols, values, lower, upper = [], [], [], [], []

    def add(entries, lo, hi):
        row = len(lower)
        for col, value in entries:
            rows.append(row)
            cols.append(col)
            values.append(value)
        lower.append(lo)
        upper.append(hi)

    for p in range(n):
        add([(p * n + q, 1) for q in range(n)], 1, 1)
        add([(q * n + p, 1) for q in range(n)], 1, 1)
    for s, support in enumerate(supports):
        low, high = n * n + 2 * s, n * n + 2 * s + 1
        for p in support:
            position = [(p * n + q, -q) for q in range(n)]
            add([(low, 1)] + position, -np.inf, 0)
            add([(high, 1)] + position, 0, np.inf)
        add([(size - 1, 1), (high, -1), (low, 1)], 1, np.inf)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), size)).tocsc()
    upper_bounds = np.r_[np.ones(n * n), np.full(2 * m, n - 1), n]
    result = milp(cost, integrality=np.r_[np.ones(n * n), np.zeros(2 * m + 1)],
                  bounds=Bounds(np.zeros(size), upper_bounds),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"time_limit": float(time_limit)})
    if result.x is not None:
        assignment = result.x[:n * n].reshape(n, n)
        perm = tuple(np.argmax(assignment, axis=1).tolist())
        if sorted(perm) == list(range(n)) and np.allclose(assignment, np.round(assignment), atol=1e-5):
            return LayoutResult(perm, "optimal" if result.success else "feasible_time_limit", objective(perm))
    return LayoutResult(identity, "identity_no_feasible_solution", objective(identity))
