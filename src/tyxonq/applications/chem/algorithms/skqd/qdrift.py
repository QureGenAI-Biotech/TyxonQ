"""原生费米子 SqDRIFT，默认保留全部 Hamiltonian 项和抽样顺序。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from operator import index

import numpy as np
from openfermion import FermionOperator, get_sparse_operator, jordan_wigner

from ...chem_libs.fermionic_circuits import append_pauli_evolution
from ...chem_libs.fermionic_operators import electronic_fermion_operator, validate_integrals
from .circuit_builder import reference_circuit, validate_schedule
from .ensemble import CircuitRecord, SKQDEnsemble, validate_permutation
from .layout import optimize_mode_order


@dataclass
class FermionicGroup:
    """operator 已除以真实算符范数；weight*operator 恢复原始组。"""

    operator: FermionOperator
    weight: float
    support: tuple[int, ...]
    excitation: tuple[int, ...]
    diagonal: bool


def _commute(a, b):
    right = dict(b)
    return sum(p in right and axis != right[p] for p, axis in a) % 2 == 0


def group_fermionic_hamiltonian(h1e, eri):
    """按激发模式与 number spectator 分组，包含 Hermitian 共轭和相同支持上的对称项。

    对至多四个模式的局部矩阵直接求谱范数，不能用一般组的平均系数替代。
    三模式项还按 spectator 区分，避免把不对易的条件跃迁错误合并。
    """
    hamiltonian = electronic_fermion_operator(h1e, eri)
    groups = {}
    for term, coefficient in hamiltonian.terms.items():
        if not term or coefficient == 0:
            continue
        counts = Counter(p for p, _ in term)
        active = tuple(sorted(p for p, c in counts.items() if c % 2))
        spectator = tuple(sorted(p for p, c in counts.items() if c == 2))
        key = active, spectator
        groups.setdefault(key, FermionOperator())
        groups[key] += FermionOperator(term, coefficient)
    output = []
    for (active, spectator), operator in sorted(groups.items()):
        support = tuple(sorted(set(active + spectator)))
        local_index = {p: i for i, p in enumerate(support)}
        local = FermionOperator()
        for term, coefficient in operator.terms.items():
            local += FermionOperator(tuple((local_index[p], a) for p, a in term), coefficient)
        matrix = get_sparse_operator(local, n_qubits=len(support)).toarray()
        if not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0):
            raise ValueError("electronic group is not Hermitian")
        weight = float(np.max(np.abs(np.linalg.eigvalsh(matrix))))
        if weight == 0:
            continue
        normalized = operator / weight
        paulis = list(jordan_wigner(normalized).terms)
        if any(not _commute(a, b) for i, a in enumerate(paulis) for b in paulis[i + 1:]):
            raise ValueError("electronic group requires non-commuting Pauli synthesis")
        output.append(FermionicGroup(normalized, weight, support, active, not active))
    return output


def _permute_fermionic_state(circuit, permutation):
    """通过相邻 FSWAP 重排参考态；普通 SWAP 会遗漏占据相关的费米子负号。"""
    labels = list(range(len(permutation)))
    desired = np.argsort(permutation)
    fswap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -1]], complex)
    for position, old_mode in enumerate(desired):
        location = labels.index(old_mode)
        while location > position:
            circuit.unitary(location - 1, location, matrix=fswap)
            labels[location - 1], labels[location] = labels[location], labels[location - 1]
            location -= 1


def build_sqdrift_circuits(
    h1e, eri, norb, nelec, *, time_step, krylov_indices=(0, 1, 2, 3),
    num_sampled_groups=20, num_randomizations=10, seed=24, initial_state=None,
    mode_permutation=None, optimize_layout=False, layout_time_limit=10.0,
    filter_diagonal=False, filter_trivial=False, basis_id="input",
):
    """每个 (k,r) 独立抽组；支持重放、F2Q 和显式标记的启发式过滤。"""
    h, g, n, nelec = validate_integrals(h1e, eri, norb, nelec)
    dt, indices = validate_schedule(time_step, krylov_indices)
    length, repeats = index(num_sampled_groups), index(num_randomizations)
    seed = index(seed)
    if length < 1 or repeats < 1:
        raise ValueError("num_sampled_groups and num_randomizations must be positive")
    if optimize_layout and mode_permutation is not None:
        raise ValueError("choose explicit mode_permutation or optimize_layout")
    groups = group_fermionic_hamiltonian(h, g)
    enabled = [i for i, group in enumerate(groups) if not (filter_diagonal and group.diagonal)]
    weights = np.array([groups[i].weight for i in enabled])
    lambd = float(weights.sum())
    identity = tuple(range(2 * n))
    rng = np.random.default_rng(seed)
    circuits, records = [], []
    initial = reference_circuit(initial_state, n, nelec)
    # 对任意叠加态保守地把所有模式标为 uncertain，避免错误裁掉有效激发。
    if all(op[0] == "x" for op in initial.ops):
        occupied0 = set()
        for op in initial.ops:
            occupied0.symmetric_difference_update([int(op[1])])
        unoccupied0 = set(identity) - occupied0
    else:
        occupied0 = unoccupied0 = set(identity)
    synthesized = {}
    for k in indices:
        for r in range(repeats):
            drawn = []
            if k and enabled:
                occupied, unoccupied = set(occupied0), set(unoccupied0)
                failures = 0
                while len(drawn) < length:
                    group_id = int(rng.choice(enabled, p=weights / lambd))
                    support = set(groups[group_id].support)
                    if filter_trivial and not (support & occupied and support & unoccupied):
                        failures += 1
                        if failures > max(1000, 100 * length):
                            raise ValueError("filter_trivial could not draw an active excitation")
                        continue
                    failures = 0
                    drawn.append(group_id)
                    occupied |= support
                    unoccupied |= support
            if optimize_layout and drawn:
                layout = optimize_mode_order([groups[i].excitation for i in drawn], 2 * n,
                                             time_limit=layout_time_limit)
                permutation, layout_status = layout.permutation, layout.status
                layout_objective = layout.objective
            else:
                permutation = validate_permutation(identity if mode_permutation is None else mode_permutation, 2 * n)
                layout_status = "explicit" if mode_permutation is not None else "disabled"
                layout_objective = None
            circuit = reference_circuit(initial, n, nelec)
            if all(op[0] == "x" for op in initial.ops):
                # 单个 determinant 的重排符号只是整体相位，可直接准备映射后的 X。
                circuit.ops = [("x", permutation[int(op[1])]) for op in initial.ops]
            else:
                _permute_fermionic_state(circuit, permutation)
            delta = lambd * k * dt / length
            for group_id in drawn:
                key = group_id, permutation
                if key not in synthesized:
                    remapped = FermionOperator()
                    for term, coefficient in groups[group_id].operator.terms.items():
                        remapped += FermionOperator(tuple((permutation[p], a) for p, a in term), coefficient)
                    synthesized[key] = sorted(jordan_wigner(remapped).terms.items())
                for pauli, coefficient in synthesized[key]:
                    if abs(complex(coefficient).imag) > 1e-10:
                        raise ValueError("Hermitian Pauli coefficients must be real")
                    append_pauli_evolution(circuit, pauli, delta * float(np.real(coefficient)))
            record = CircuitRecord(f"qdrift-k{k}-r{r}", k, k * dt, r, seed,
                                   tuple(drawn), permutation, identity, basis_id,
                                   {"lambda": lambd, "delta": delta, "layout_status": layout_status,
                                    "layout_objective": layout_objective,
                                    "filter_diagonal": bool(filter_diagonal),
                                    "filter_trivial": bool(filter_trivial),
                                    "group_count": len(groups)})
            circuit.metadata["skqd"] = {"circuit_id": record.circuit_id, "k": k,
                                         "time": k * dt, "sampled_group_ids": list(drawn)}
            circuits.append(circuit)
            records.append(record)
    return SKQDEnsemble(circuits, records, n, nelec, "qdrift")
