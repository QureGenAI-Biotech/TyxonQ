"""根据矩阵参数构造闭壳层或开壳层 LUCJ 线路。

按 HF 参考态、轨道旋转 U†、对角 Jastrow 相互作用 J、轨道旋转 U 的顺序构造，
并融合不跨越 Jastrow 层的相邻轨道旋转。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from operator import index
from typing import Any

import numpy as np

from tyxonq.core.ir.circuit import Circuit

from .linalg import (
    decompose_slater_determinant_to_adjacent_rotations,
    decompose_unitary_to_adjacent_rotations,
    two_mode_fock_matrix,
)
from .parameters import (
    lucj_parameter_shapes,
    normalize_excluded_orbitals,
    normalize_lucj_params,
)
from .topology import (
    InteractionPairs,
    alpha_qubit,
    beta_qubit,
    interaction_pairs_spin_balanced,
    normalize_topology,
    spin_qubit,
    validate_layers,
    validate_n_orbitals,
)


LogicalOp = dict[str, Any]
ElectronCount = int | tuple[int, int]
ExcludedOrbitals = Iterable[int] | tuple[Iterable[int], Iterable[int]] | None


class LUCJ:
    """matrix LUCJ 的轻量线路 builder。"""

    def __init__(
        self,
        n_orbitals: int,
        n_electrons: ElectronCount,
        layers: int,
        topology: str = "linear",
        *,
        interaction_pairs: InteractionPairs | None = None,
        excluded_orbitals: ExcludedOrbitals = None,
    ) -> None:
        """保存线路参数；显式 ``interaction_pairs=(pairs_same, pairs_ab)`` 覆盖拓扑。"""
        self.n_orbitals = validate_n_orbitals(n_orbitals)
        self.n_electrons = _validate_electrons(n_electrons, self.n_orbitals)
        self.layers = validate_layers(layers)
        self.topology = normalize_topology(topology)
        self.interaction_pairs = (
            interaction_pairs_spin_balanced(
                self.n_orbitals, self.topology, interaction_pairs,
                ordered_ab=isinstance(self.n_electrons, tuple),
            )
            if interaction_pairs is not None else None
        )
        self.excluded_orbitals = normalize_excluded_orbitals(
            excluded_orbitals,
            self.n_orbitals,
        )
        _validate_reference_exclusions(
            self.n_electrons,
            self.n_orbitals,
            self.excluded_orbitals,
        )

    @property
    def is_spin_unbalanced(self) -> bool:
        """tuple 电子数始终表示 α/β 独立参数模式。"""
        return isinstance(self.n_electrons, tuple)

    @property
    def num_qubits(self) -> int:
        """返回 `[alpha | beta]` JW qubit 顺序下的 qubit 总数。"""
        return 2 * self.n_orbitals

    def parameter_shapes(
        self,
        *,
        with_final_orbital_rotation: bool = False,
    ) -> dict[str, tuple[int, ...]]:
        """返回当前 builder 需要的 matrix LUCJ 参数 shape。"""
        return lucj_parameter_shapes(
            self.n_orbitals,
            self.layers,
            self.topology,
            spin_unbalanced=self.is_spin_unbalanced,
            interaction_pairs=self.interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )

    def get_circuit(
        self,
        params: Mapping[str, Any],
        *,
        with_final_orbital_rotation: bool | None = None,
    ) -> Circuit:
        """用外部 matrix LUCJ 参数生成 TyxonQ `Circuit`。"""
        return build_lucj_circuit(
            self.n_orbitals,
            self.n_electrons,
            self.layers,
            self.topology,
            params,
            excluded_orbitals=self.excluded_orbitals,
            interaction_pairs=self.interaction_pairs,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )


def build_lucj_circuit(
    n_orbitals: int,
    n_electrons: ElectronCount,
    layers: int,
    topology: str,
    params: Mapping[str, Any],
    *,
    interaction_pairs: InteractionPairs | None = None,
    excluded_orbitals: ExcludedOrbitals = None,
    with_final_orbital_rotation: bool | None = None,
) -> Circuit:
    """构建 restricted 或 spin-resolved ffsim 风格 matrix LUCJ 线路。"""
    n = validate_n_orbitals(n_orbitals)
    electrons = _validate_electrons(n_electrons, n)
    layer_count = validate_layers(layers)
    name = normalize_topology(topology)
    excluded = normalize_excluded_orbitals(excluded_orbitals, n)
    _validate_reference_exclusions(electrons, n, excluded)
    operation_plan, use_final_orbital_rotation = _build_lucj_logical_ops(
        n,
        electrons,
        layer_count,
        name,
        params,
        excluded_orbitals=excluded,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )

    circuit = Circuit(2 * n)
    logical_ops: list[LogicalOp] = []
    circuit.metadata["lucj"] = {
        "n_orbitals": n,
        "n_electrons": electrons,
        "spin_unbalanced": isinstance(electrons, tuple),
        "excluded_orbitals": {
            "alpha": excluded[0],
            "beta": excluded[1],
        },
        "layers": layer_count,
        "topology": name,
        "interaction_pairs": interaction_pairs_spin_balanced(
            n, name, interaction_pairs, ordered_ab=isinstance(electrons, tuple)
        ),
        "qubit_order": {
            "alpha": [alpha_qubit(p) for p in range(n)],
            "beta": [beta_qubit(p, n) for p in range(n)],
        },
        "reference_occupations": _reference_occupations(electrons, n),
        "parameter_shapes": lucj_parameter_shapes(
            n,
            layer_count,
            name,
            spin_unbalanced=isinstance(electrons, tuple),
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=use_final_orbital_rotation,
        ),
        "has_final_orbital_rotation": use_final_orbital_rotation,
        "logical_ops": logical_ops,
    }

    for operation in operation_plan:
        _append_logical_op_to_circuit(circuit, logical_ops, operation)
    return circuit


def _build_lucj_logical_ops(
    n_orbitals: int,
    n_electrons: ElectronCount,
    layers: int,
    topology: str,
    params: Mapping[str, Any],
    *,
    interaction_pairs: InteractionPairs | None = None,
    excluded_orbitals: ExcludedOrbitals = None,
    with_final_orbital_rotation: bool | None = None,
) -> tuple[list[LogicalOp], bool]:
    """生成 builder 和硬件导出共同消费的 LUCJ 逻辑门序列。"""
    n = validate_n_orbitals(n_orbitals)
    electrons = _validate_electrons(n_electrons, n)
    layer_count = validate_layers(layers)
    name = normalize_topology(topology)
    excluded = normalize_excluded_orbitals(excluded_orbitals, n)
    _validate_reference_exclusions(electrons, n, excluded)
    spin_unbalanced = isinstance(electrons, tuple)
    normalized = normalize_lucj_params(
        params,
        n,
        layer_count,
        name,
        spin_unbalanced=spin_unbalanced,
        excluded_orbitals=excluded,
        interaction_pairs=interaction_pairs,
    )
    orbital_rotations = normalized["orbital_rotations"]
    diag_coulomb_mats = normalized["diag_coulomb_mats"]
    final_orbital_rotation = normalized["final_orbital_rotation"]
    assert isinstance(orbital_rotations, np.ndarray)
    assert isinstance(diag_coulomb_mats, np.ndarray)
    # 未指定开关时沿用官方行为；显式 False 仍可关闭末尾轨道旋转。
    use_final_orbital_rotation = (with_final_orbital_rotation is None or bool(with_final_orbital_rotation)) and isinstance(
        final_orbital_rotation,
        np.ndarray,
    )

    logical_ops: list[LogicalOp] = []
    first_dagger = np.swapaxes(orbital_rotations[0].conj(), -1, -2)
    _append_slater_determinant(logical_ops, electrons, first_dagger, n)
    for layer in range(layer_count):
        if layer > 0:
            # 线路先执行上一层 U，再执行本层 U†，所以合并矩阵是“后者 @ 前者”。
            current_dagger = np.swapaxes(orbital_rotations[layer].conj(), -1, -2)
            bridge_rotation = current_dagger @ orbital_rotations[layer - 1]
            _append_orbital_rotation(
                logical_ops,
                bridge_rotation,
                n,
                block="orbital_rotation_dagger",
                layer=layer,
            )
        _append_diag_coulomb(
            logical_ops,
            diag_coulomb_mats[layer],
            n,
            name,
            layer=layer,
            interaction_pairs=interaction_pairs,
        )
    terminal_rotation = orbital_rotations[-1]
    if use_final_orbital_rotation:
        assert isinstance(final_orbital_rotation, np.ndarray)
        # final rotation 在线路中后执行，因此左乘最后一层的 U。
        terminal_rotation = final_orbital_rotation @ terminal_rotation
        terminal_block = "final_orbital_rotation"
        terminal_layer = None
    else:
        terminal_block = "orbital_rotation"
        terminal_layer = layer_count - 1
    _append_orbital_rotation(
        logical_ops,
        terminal_rotation,
        n,
        block=terminal_block,
        layer=terminal_layer,
    )
    return logical_ops, use_final_orbital_rotation


def _append_slater_determinant(
    logical_ops: list[LogicalOp],
    n_electrons: ElectronCount,
    orbital_rotation: np.ndarray,
    n_orbitals: int,
) -> None:
    """把 HF reference 与第一段 orbital rotation 合并制备。"""
    _append_reference_state(logical_ops, n_electrons, n_orbitals)
    occupations = _reference_occupations(n_electrons, n_orbitals)
    array = np.asarray(orbital_rotation, dtype=complex)
    spin_matrices = (
        (("alpha", array[0]), ("beta", array[1]))
        if array.ndim == 3
        else (("alpha", array), ("beta", array))
    )
    for spin, matrix in spin_matrices:
        orbital_coeffs = matrix.T[list(occupations[spin])]
        rotations = decompose_slater_determinant_to_adjacent_rotations(orbital_coeffs)
        for rotation_index, rotation in enumerate(rotations):
            if np.allclose(rotation.matrix, np.eye(2), atol=1e-12):
                continue
            p, q = rotation.modes
            logical_ops.append(
                _logical_op(
                    block="orbital_rotation_dagger",
                    layer=0,
                    kind="orbital_givens",
                    spin=spin,
                    orbitals=(p, q),
                    qubits=(
                        spin_qubit(spin, p, n_orbitals),
                        spin_qubit(spin, q, n_orbitals),
                    ),
                    parameter=(
                        f"orbital_rotation_dagger.{spin}.slater[{rotation_index}]"
                    ),
                    matrix=rotation.matrix,
                )
            )


def _append_reference_state(
    logical_ops: list[LogicalOp],
    n_electrons: ElectronCount,
    n_orbitals: int,
) -> None:
    """按 `(nalpha, nbeta)` 追加 Hartree-Fock 参考态逻辑 `x` 门。"""
    for spin, orbitals in _reference_occupations(n_electrons, n_orbitals).items():
        for p in orbitals:
            qubit = spin_qubit(spin, p, n_orbitals)
            logical_ops.append(
                _logical_op(
                    block="reference",
                    layer=None,
                    kind="x",
                    spin=spin,
                    orbitals=(p,),
                    qubits=(qubit,),
                    parameter=f"reference.{spin}[{p}]",
                )
            )


def _append_orbital_rotation(
    logical_ops: list[LogicalOp],
    orbital_rotation: np.ndarray,
    n_orbitals: int,
    *,
    block: str,
    layer: int | None,
) -> None:
    """把共享或 α/β 独立 orbital rotation 分解到两条自旋链。"""
    array = np.asarray(orbital_rotation, dtype=complex)
    spin_matrices = (
        (("alpha", array[0]), ("beta", array[1]))
        if array.ndim == 3
        else (("alpha", array), ("beta", array))
    )
    for spin, matrix in spin_matrices:
        phases, rotations = decompose_unitary_to_adjacent_rotations(matrix)
        for p, phase in enumerate(phases):
            angle = float(np.angle(phase))
            if abs(angle) <= 1e-12:
                continue
            qubit = spin_qubit(spin, p, n_orbitals)
            logical_ops.append(
                _logical_op(
                    block=block,
                    layer=layer,
                    kind="orbital_phase",
                    spin=spin,
                    orbitals=(p,),
                    qubits=(qubit,),
                    theta=angle,
                    parameter=f"{block}.{spin}.phase[{p}]",
                )
            )

        for rotation_index, rotation in enumerate(rotations):
            if np.allclose(rotation.matrix, np.eye(2), atol=1e-12):
                continue
            p, q = rotation.modes
            q0 = spin_qubit(spin, p, n_orbitals)
            q1 = spin_qubit(spin, q, n_orbitals)
            logical_ops.append(
                _logical_op(
                    block=block,
                    layer=layer,
                    kind="orbital_givens",
                    spin=spin,
                    orbitals=(p, q),
                    qubits=(q0, q1),
                    parameter=f"{block}.{spin}.givens[{rotation_index}]",
                    matrix=rotation.matrix,
                )
            )


def _append_diag_coulomb(
    logical_ops: list[LogicalOp],
    diag_coulomb_mats: np.ndarray,
    n_orbitals: int,
    topology: str,
    *,
    layer: int,
    interaction_pairs: InteractionPairs | None = None,
) -> None:
    """追加一层 diagonal Coulomb number-number 逻辑门。

    ffsim 在 UCJ gate 中使用 `time=-1`，因此非零 `J` 参数对应
    `CPhase(J)`；具体线路分解由正常 builder 统一完成。
    """
    mats = np.asarray(diag_coulomb_mats, dtype=float)
    mat_aa = mats[0]
    mat_ab = mats[1]
    mat_bb = mats[2] if mats.shape[0] == 3 else mat_aa
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        n_orbitals, topology, interaction_pairs, ordered_ab=mats.shape[0] == 3
    )
    if mats.shape[0] == 2:
        # 闭壳层 Jab 对称；每个非对角上三角项产生两个不同的 αβ 耦合。
        pairs_ab = pairs_ab + [(q, p) for p, q in pairs_ab if p != q]

    for spin, matrix, channel in (("alpha", mat_aa, 0), ("beta", mat_bb, 2)):
        for p, q in pairs_aa:
            theta = float(matrix[p, q])
            if abs(theta) <= 1e-12:
                continue
            q0 = spin_qubit(spin, p, n_orbitals)
            q1 = spin_qubit(spin, q, n_orbitals)
            _append_cphase(
                logical_ops,
                q0,
                q1,
                theta,
                block="diag_coulomb",
                layer=layer,
                kind="same_spin_cphase",
                spin=spin,
                orbitals=(p, q),
                parameter=(
                    f"diag_coulomb_mats[{layer},{channel},{p},{q}]"
                    if mats.shape[0] == 3
                    else f"diag_coulomb_mats[{layer},0,{p},{q}]"
                ),
            )

    for p, q in pairs_ab:
        theta = float(mat_ab[p, q])
        if abs(theta) <= 1e-12:
            continue
        _append_cphase(
            logical_ops,
            alpha_qubit(p),
            beta_qubit(q, n_orbitals),
            theta,
            block="diag_coulomb",
            layer=layer,
            kind="opposite_spin_cphase",
            spin="alpha-beta",
            orbitals=(p, q),
            parameter=f"diag_coulomb_mats[{layer},1,{p},{q}]",
        )


def _append_cphase(
    logical_ops: list[LogicalOp],
    q0: int,
    q1: int,
    theta: float,
    *,
    block: str,
    layer: int,
    kind: str,
    spin: str,
    orbitals: tuple[int, int],
    parameter: str,
) -> None:
    """追加一条保留语义边界的逻辑 `CPhase(theta)`。"""
    logical_ops.append(
        _logical_op(
            block=block,
            layer=layer,
            kind=kind,
            spin=spin,
            orbitals=orbitals,
            qubits=(q0, q1),
            theta=theta,
            parameter=parameter,
        )
    )


def _append_logical_op_to_circuit(
    circuit: Circuit,
    logical_ops: list[LogicalOp],
    operation: LogicalOp,
) -> None:
    """按现有分解把一条逻辑门追加到 TyxonQ `Circuit`。"""
    kind = operation["kind"]
    qubits = tuple(int(q) for q in operation["qubits"])
    start = len(circuit.ops)

    if kind == "x":
        circuit.x(qubits[0])
    elif kind == "orbital_phase":
        circuit.rz(qubits[0], float(operation["theta"]))
    elif kind == "orbital_givens":
        circuit.unitary(
            qubits[0],
            qubits[1],
            matrix=two_mode_fock_matrix(np.asarray(operation["matrix"], dtype=complex)),
        )
    elif kind in {"same_spin_cphase", "opposite_spin_cphase"}:
        theta = float(operation["theta"])
        circuit.rz(qubits[0], theta / 2.0)
        circuit.rz(qubits[1], theta / 2.0)
        circuit.rzz(qubits[0], qubits[1], -theta / 2.0)
    else:
        raise ValueError(f"Unsupported LUCJ logical op kind: {kind!r}")

    record = dict(operation)
    record["op_indices"] = tuple(range(start, len(circuit.ops)))
    logical_ops.append(record)


def _logical_op(
    *,
    block: str,
    layer: int | None,
    kind: str,
    spin: str,
    orbitals: tuple[int, ...],
    qubits: tuple[int, ...],
    parameter: str,
    theta: float | None = None,
    matrix: np.ndarray | None = None,
    op_indices: tuple[int, ...] = (),
) -> LogicalOp:
    """创建一条用于测试和调试的 LUCJ 逻辑门记录。"""
    record: LogicalOp = {
        "block": block,
        "layer": layer,
        "kind": kind,
        "spin": spin,
        "orbitals": orbitals,
        "qubits": qubits,
        "parameter": parameter,
        "op_indices": op_indices,
    }
    if theta is not None:
        record["theta"] = float(theta)
    if matrix is not None:
        record["matrix"] = np.asarray(matrix, dtype=complex)
    return record


def _validate_closed_shell_electrons(n_electrons: int, n_orbitals: int) -> int:
    """校验 closed-shell RHF 参考态需要的电子数。"""
    try:
        electrons = index(n_electrons)
    except TypeError as exc:
        raise ValueError("n_electrons must be a positive even integer") from exc
    if electrons < 2 or electrons % 2:
        raise ValueError("n_electrons must be a positive even integer")
    if electrons // 2 > validate_n_orbitals(n_orbitals):
        raise ValueError("n_electrons / 2 must be <= n_orbitals for closed-shell LUCJ")
    return int(electrons)


def _validate_electrons(
    n_electrons: ElectronCount,
    n_orbitals: int,
) -> ElectronCount:
    """int 保留 closed-shell 语义；二元 tuple 显式表示 α/β 电子数。"""
    n = validate_n_orbitals(n_orbitals)
    if not isinstance(n_electrons, tuple):
        return _validate_closed_shell_electrons(n_electrons, n)
    if len(n_electrons) != 2:
        raise ValueError("n_electrons tuple must be (nalpha, nbeta)")
    counts: list[int] = []
    for value in n_electrons:
        try:
            count = index(value)
        except TypeError as exc:
            raise ValueError("n_electrons tuple must contain integers") from exc
        if count < 0 or count > n:
            raise ValueError("each spin electron count must satisfy 0 <= count <= n_orbitals")
        counts.append(int(count))
    if sum(counts) < 1:
        raise ValueError("n_electrons must contain at least one electron")
    return counts[0], counts[1]


def _reference_occupations(
    n_electrons: ElectronCount,
    n_orbitals: int,
) -> dict[str, tuple[int, ...]]:
    """返回 α/β Hartree-Fock 参考占据。"""
    electrons = _validate_electrons(n_electrons, n_orbitals)
    if isinstance(electrons, tuple):
        nalpha, nbeta = electrons
        return {
            "alpha": tuple(range(nalpha)),
            "beta": tuple(range(nbeta)),
        }
    return _closed_shell_reference_occupations(electrons, n_orbitals)


def _validate_reference_exclusions(
    n_electrons: ElectronCount,
    n_orbitals: int,
    excluded: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    """dummy 轨道必须保持空占据，不能与参考态重叠。"""
    occupations = _reference_occupations(n_electrons, n_orbitals)
    for spin_index, spin in enumerate(("alpha", "beta")):
        overlap = set(occupations[spin]) & set(excluded[spin_index])
        if overlap:
            raise ValueError(
                f"excluded_orbitals overlap occupied {spin} reference orbitals: "
                f"{sorted(overlap)}"
            )


def _closed_shell_reference_occupations(
    n_electrons: int,
    n_orbitals: int,
) -> dict[str, tuple[int, ...]]:
    """返回 closed-shell RHF 参考态中 alpha/beta 各自占据的空间轨道。"""
    n_occupied = _validate_closed_shell_electrons(n_electrons, n_orbitals) // 2
    occupied = tuple(range(n_occupied))
    return {"alpha": occupied, "beta": occupied}
