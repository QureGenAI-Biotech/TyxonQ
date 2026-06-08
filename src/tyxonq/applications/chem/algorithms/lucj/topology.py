"""LUCJ 拓扑、qubit 编号和局域相互作用 pair 工具。"""

from __future__ import annotations

from operator import index


SUPPORTED_TOPOLOGIES = ("square", "hex", "linear")


def validate_n_orbitals(n_orbitals: int) -> int:
    """校验空间轨道数 N，并返回普通 `int`。"""
    try:
        n = index(n_orbitals)
    except TypeError as exc:
        raise ValueError("n_orbitals must be a positive integer") from exc
    if n < 1:
        raise ValueError("n_orbitals must be a positive integer")
    return int(n)


def validate_layers(layers: int) -> int:
    """校验 LUCJ 重复层数 L，并返回普通 `int`。"""
    try:
        value = index(layers)
    except TypeError as exc:
        raise ValueError("layers must be a positive integer") from exc
    if value < 1:
        raise ValueError("layers must be a positive integer")
    return int(value)


def normalize_topology(topology: str) -> str:
    """规范化并校验 topology 名称。"""
    name = str(topology).lower()
    if name not in SUPPORTED_TOPOLOGIES:
        allowed = ", ".join(SUPPORTED_TOPOLOGIES)
        raise ValueError(f"Unsupported LUCJ topology {topology!r}; expected one of: {allowed}")
    return name


def alpha_qubit(p: int) -> int:
    """返回第 p 个 alpha spin orbital 的 qubit index。"""
    return int(p)


def beta_qubit(p: int, n_orbitals: int) -> int:
    """返回第 p 个 beta spin orbital 的 qubit index。"""
    return validate_n_orbitals(n_orbitals) + int(p)


def spin_qubit(spin: str, p: int, n_orbitals: int) -> int:
    """按自旋标签返回 qubit index。"""
    if spin == "alpha":
        return alpha_qubit(p)
    if spin == "beta":
        return beta_qubit(p, n_orbitals)
    raise ValueError(f"Unsupported spin label {spin!r}; expected 'alpha' or 'beta'")


def same_spin_orbital_pairs(n_orbitals: int) -> list[tuple[int, int]]:
    """返回 same-spin `Jaa/Jbb` 允许的相邻空间轨道 pair。"""
    n = validate_n_orbitals(n_orbitals)
    return [(p, p + 1) for p in range(n - 1)]


def opposite_spin_orbital_pairs(n_orbitals: int, topology: str) -> list[tuple[int, int]]:
    """返回 opposite-spin `Jab/Jba` 允许的同轨道 pair。

    `square` 保留全部 `(p,p)`，`hex` 只保留偶数 p，`linear` 只保留 p=0。
    """
    n = validate_n_orbitals(n_orbitals)
    name = normalize_topology(topology)
    if name == "square":
        return [(p, p) for p in range(n)]
    if name == "hex":
        return [(p, p) for p in range(0, n, 2)]
    return [(0, 0)]


def opposite_spin_orbital_indices(n_orbitals: int, topology: str) -> list[int]:
    """返回 opposite-spin 同轨道连接中的空间轨道编号。"""
    return [p for p, _ in opposite_spin_orbital_pairs(n_orbitals, topology)]


def opposite_spin_qubit_pairs(n_orbitals: int, topology: str) -> list[tuple[int, int]]:
    """把 opposite-spin 轨道 pair 转换为 qubit pair。"""
    n = validate_n_orbitals(n_orbitals)
    return [(alpha_qubit(p), beta_qubit(q, n)) for p, q in opposite_spin_orbital_pairs(n, topology)]


def interaction_pairs_spin_balanced(
    n_orbitals: int,
    topology: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """返回 spin-balanced UCJ 使用的 `(pairs_aa, pairs_ab)`。"""
    n = validate_n_orbitals(n_orbitals)
    name = normalize_topology(topology)
    return same_spin_orbital_pairs(n), opposite_spin_orbital_pairs(n, name)


def validate_interaction_pairs(
    pairs: list[tuple[int, int]] | None,
    *,
    ordered: bool = False,
) -> None:
    """校验 interaction pair 没有重复，且默认必须是上三角 pair。"""
    if pairs is None:
        return
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"Duplicate interaction pairs encountered: {pairs}.")
    if not ordered:
        for i, j in pairs:
            if i > j:
                raise ValueError(f"Interaction pairs must be upper triangular; got {(i, j)}.")


def allowed_diag_coulomb_indices(n_orbitals: int, topology: str) -> list[tuple[int, int]]:
    """返回 compressed DF 优化中允许非零的单个 `Z_mu` 矩阵指标。"""
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(n_orbitals, topology)
    return sorted(set(pairs_aa + pairs_ab))


def givens_schedule(n_orbitals: int) -> list[tuple[int, int]]:
    """返回相邻 orbital rotation 最多需要的 brickwork pair。"""
    n = validate_n_orbitals(n_orbitals)
    target = n * (n - 1) // 2
    even_pairs = [(p, p + 1) for p in range(0, n - 1, 2)]
    odd_pairs = [(p, p + 1) for p in range(1, n - 1, 2)]
    schedule: list[tuple[int, int]] = []
    while len(schedule) < target:
        for pair in even_pairs:
            schedule.append(pair)
            if len(schedule) == target:
                return schedule
        for pair in odd_pairs:
            schedule.append(pair)
            if len(schedule) == target:
                return schedule
    return schedule
