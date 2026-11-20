"""在给定行列式空间中寻找并在完整物理空间验收指定的总自旋。"""

from collections import defaultdict

import numpy as np


def allowed_spin_squares(norb, nelec):
    """列出固定 M_S 下电子数与空穴数允许的 S(S+1)。"""
    twice_min = abs(nelec[0] - nelec[1])
    twice_max = min(sum(nelec), 2 * norb - sum(nelec))
    return [s * (s + 2) / 4 for s in range(twice_min, twice_max + 1, 2)]


def _spin_column(a, b, sz2, target):
    """给出完整空间中 (S²−target)|a,b> 的非零项，采用先 α 后 β 的位序。"""
    single = a ^ b
    yield (a, b), sz2 + single.bit_count() / 2 - target
    ups = a & single
    while ups:
        p = ups & -ups
        ups ^= p
        downs = b & single
        while downs:
            q = downs & -downs
            downs ^= q
            # 交换两个反向自旋。两个单激发的相位取决于端点间的单占轨道数。
            between = max(p, q) - 2 * min(p, q)
            phase = 1 if (single & between).bit_count() % 2 else -1
            flip = p | q
            yield (a ^ flip, b ^ flip), phase


def _spin_blocks(strings):
    """交换自旋不改变空间占据模式，按块处理可限制空间外的临时系数数目。"""
    blocks = defaultdict(list)
    alpha, beta = (list(map(int, x)) for x in strings)
    for i, a in enumerate(alpha):
        for j, b in enumerate(beta):
            blocks[a & b, a ^ b].append((i * len(beta) + j, a, b))
    return blocks.values()


def full_spin_residual(coefficients, blocks, sz2, target):
    """直接累计包括采样空间外分量在内的 ||(S²−target)c||。"""
    norm2 = 0.0
    for block in blocks:
        image = defaultdict(float)
        for index, a, b in block:
            amplitude = coefficients[index]
            if amplitude != 0:
                for pair, weight in _spin_column(a, b, sz2, target):
                    image[pair] += weight * amplitude
        norm2 += sum(value * value for value in image.values())
    return float(np.sqrt(norm2))


def _exact_target_basis(blocks, size, sz2, target):
    """按空间占据块求完整自旋残差算符的零空间，不添加采样空间以外的基矢。"""
    columns = []
    for block in blocks:
        rows = {}
        indices = [entry[0] for entry in block]
        for local, (_, a, b) in enumerate(block):
            for pair, weight in _spin_column(a, b, sz2, target):
                if pair not in rows:
                    rows[pair] = np.zeros(len(block))
                rows[pair][local] += weight
        _, s, vh = np.linalg.svd(np.stack(list(rows.values())), full_matrices=False)
        for vector in vh[s < 1e-10]:
            column = np.zeros(size)
            column[indices] = vector
            columns.append(column)
    if not columns:
        raise RuntimeError("Selected space contains no state with the requested target spin.")
    return np.column_stack(columns)


def solve_target_spin(matvec, bare_matvec, spin_matvec, diagonal, strings, norb, nelec,
                      target, spin_tolerance, initial, solve, energy_tolerance):
    """小空间从精确目标子空间出发，大空间用多个自旋过滤初猜搜索。"""
    import mlx.core as mx

    allowed = allowed_spin_squares(norb, nelec)
    if target not in allowed:
        raise ValueError(f"spin_sq={target} is incompatible with norb/nelec; allowed values: {allowed}")
    blocks = _spin_blocks(strings)
    sz2 = ((nelec[0] - nelec[1]) / 2) ** 2
    size = diagonal.size
    small = size <= 64
    starts = []
    expected = None
    if small:
        basis = _exact_target_basis(blocks, size, sz2, target)
        images = np.column_stack([np.asarray(bare_matvec(mx.array(v, dtype=mx.float32)),
                                              dtype=np.float64) for v in basis.T])
        reduced = basis.T @ images
        values, vectors = np.linalg.eigh((reduced + reduced.T) / 2)
        expected = float(values[0])
        starts.append(mx.array(basis @ vectors[:, 0], dtype=mx.float32))
    else:
        if initial is not None:
            starts.append(initial)
        # 两个固定的全支撑随机初猜覆盖不同的连通分量，避免对角初猜导致错根。
        rng = np.random.default_rng(0)
        diagonal_cpu = np.asarray(diagonal).ravel()
        seeds = []
        # 从最多两个低对角能、能容纳目标自旋的不同空间占据块分别出发。
        # 在对角 Hamiltonian 上，纯随机起点的对角预条件修正容易退化。
        eligible = []
        for block in blocks:
            _, a, b = block[0]
            smax = (a ^ b).bit_count() / 2
            if smax * (smax + 1) >= target:
                index = min((index for index, _, _ in block), key=diagonal_cpu.__getitem__)
                eligible.append((diagonal_cpu[index], index))
        for _, index in sorted(eligible)[:2]:
            seeds.append(mx.zeros(size, dtype=mx.float32).at[index].add(1.0))
        seeds.extend(mx.array(rng.normal(size=size), dtype=mx.float32) for _ in range(2))
        for vector in seeds:
            for other in reversed(allowed):
                if other != target:
                    vector = spin_matvec(vector) - other * vector
                    norm = float(mx.linalg.norm(vector).item())
                    if norm < 1e-12:
                        break
                    vector = vector / norm
            if float(mx.linalg.norm(vector).item()) > 1e-12:
                starts.append(vector)

    chosen = None
    best_energy, best_spin_error = np.inf, np.inf
    total_matvecs = total_cycles = 0
    for guess in starts:
        candidate = solve(matvec, guess)
        total_matvecs += candidate.h_matvecs
        total_cycles += candidate.iterations
        if not candidate.converged:
            continue
        coefficients = np.asarray(candidate.coefficients, dtype=np.float64)
        coefficients = coefficients / np.linalg.norm(coefficients)
        error = full_spin_residual(coefficients, blocks, sz2, target)
        best_spin_error = min(best_spin_error, error)
        if error <= spin_tolerance and candidate.energy < best_energy:
            best_energy, chosen, chosen_error = candidate.energy, candidate, error
    if chosen is None:
        raise RuntimeError(f"No converged root verified for target spin S²={target}; "
                           f"best full spin residual={best_spin_error:.3e}, limit={spin_tolerance:.3e}")
    if small and abs(best_energy - expected) > energy_tolerance:
        raise RuntimeError(f"Metal target root disagrees with the lowest target-subspace energy: "
                           f"{best_energy} vs {expected}")
    return chosen, {
        "target_spin_verified": True, "target_spin_residual": chosen_error,
        "target_spin_tolerance": spin_tolerance,
        "target_search_strategy": "dense_target_subspace" if small else "multistart_spin_filter",
        "target_root_ordering": "lowest_in_dense_target_subspace" if small else "lowest_verified_among_starts",
        "target_search_used_explicit_initial": bool(not small and initial is not None),
        "target_search_trials": len(starts), "target_search_h_matvecs": int(total_matvecs),
        "target_search_iterations": int(total_cycles),
        "target_projection_h_matvecs": int(basis.shape[1]) if small else 0,
    }
