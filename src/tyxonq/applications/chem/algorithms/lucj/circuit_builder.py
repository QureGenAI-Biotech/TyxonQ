"""
执行顺序固定为：
`closed-shell RHF reference -> U† -> diagonal Coulomb J -> U`。
这里不做参数初始化和能量计算，只把 matrix UCJ 参数翻译成 TyxonQ 支持的门。
"""

from __future__ import annotations

from collections.abc import Mapping
from operator import index
from typing import Any

import numpy as np

from tyxonq.core.ir.circuit import Circuit

from .linalg import decompose_unitary_to_adjacent_rotations, two_mode_fock_matrix
from .parameters import lucj_parameter_shapes, normalize_lucj_params
from .topology import (
    alpha_qubit,
    beta_qubit,
    interaction_pairs_spin_balanced,
    normalize_topology,
    spin_qubit,
    validate_layers,
    validate_n_orbitals,
)


LogicalOp = dict[str, Any]


class LUCJ:
    """matrix LUCJ 的轻量线路 builder。"""

    def __init__(
        self,
        n_orbitals: int,
        n_electrons: int,
        layers: int,
        topology: str = "square",
    ) -> None:
        """保存构建线路所需的 N、电子数、层数和 topology。"""
        self.n_orbitals = validate_n_orbitals(n_orbitals)
        self.n_electrons = _validate_closed_shell_electrons(n_electrons, self.n_orbitals)
        self.layers = validate_layers(layers)
        self.topology = normalize_topology(topology)

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
            with_final_orbital_rotation=with_final_orbital_rotation,
        )

    def get_circuit(self, params: Mapping[str, Any]) -> Circuit:
        """用外部 matrix LUCJ 参数生成 TyxonQ `Circuit`。"""
        return build_lucj_circuit(
            self.n_orbitals,
            self.n_electrons,
            self.layers,
            self.topology,
            params,
        )


def build_lucj_circuit(
    n_orbitals: int,
    n_electrons: int,
    layers: int,
    topology: str,
    params: Mapping[str, Any],
) -> Circuit:
    """构建 ffsim 风格 spin-balanced matrix LUCJ 线路。"""
    n = validate_n_orbitals(n_orbitals)
    electrons = _validate_closed_shell_electrons(n_electrons, n)
    layer_count = validate_layers(layers)
    name = normalize_topology(topology)
    normalized = normalize_lucj_params(params, n, layer_count, name)
    orbital_rotations = normalized["orbital_rotations"]
    diag_coulomb_mats = normalized["diag_coulomb_mats"]
    final_orbital_rotation = normalized["final_orbital_rotation"]
    assert isinstance(orbital_rotations, np.ndarray)
    assert isinstance(diag_coulomb_mats, np.ndarray)

    circuit = Circuit(2 * n)
    logical_ops: list[LogicalOp] = []
    circuit.metadata["lucj"] = {
        "n_orbitals": n,
        "n_electrons": electrons,
        "layers": layer_count,
        "topology": name,
        "qubit_order": {
            "alpha": [alpha_qubit(p) for p in range(n)],
            "beta": [beta_qubit(p, n) for p in range(n)],
        },
        "reference_occupations": _closed_shell_reference_occupations(electrons, n),
        "parameter_shapes": lucj_parameter_shapes(
            n,
            layer_count,
            name,
            with_final_orbital_rotation=isinstance(final_orbital_rotation, np.ndarray),
        ),
        "has_final_orbital_rotation": final_orbital_rotation is not None,
        "logical_ops": logical_ops,
    }

    _append_reference_state(circuit, logical_ops, electrons, n)
    for layer in range(layer_count):
        _append_orbital_rotation(
            circuit,
            logical_ops,
            orbital_rotations[layer].T.conj(),
            n,
            block="orbital_rotation_dagger",
            layer=layer,
        )
        _append_diag_coulomb(
            circuit,
            logical_ops,
            diag_coulomb_mats[layer],
            n,
            name,
            layer=layer,
        )
        _append_orbital_rotation(
            circuit,
            logical_ops,
            orbital_rotations[layer],
            n,
            block="orbital_rotation",
            layer=layer,
        )

    if isinstance(final_orbital_rotation, np.ndarray):
        _append_orbital_rotation(
            circuit,
            logical_ops,
            final_orbital_rotation,
            n,
            block="final_orbital_rotation",
            layer=None,
        )
    return circuit


def _append_reference_state(
    circuit: Circuit,
    logical_ops: list[LogicalOp],
    n_electrons: int,
    n_orbitals: int,
) -> None:
    """在线路开头追加 closed-shell RHF 参考态的 `x` 门。"""
    for spin, orbitals in _closed_shell_reference_occupations(n_electrons, n_orbitals).items():
        for p in orbitals:
            qubit = spin_qubit(spin, p, n_orbitals)
            start = len(circuit.ops)
            circuit.x(qubit)
            logical_ops.append(
                _logical_op(
                    block="reference",
                    layer=None,
                    kind="x",
                    spin=spin,
                    orbitals=(p,),
                    qubits=(qubit,),
                    parameter=f"reference.{spin}[{p}]",
                    op_indices=tuple(range(start, len(circuit.ops))),
                )
            )


def _append_orbital_rotation(
    circuit: Circuit,
    logical_ops: list[LogicalOp],
    orbital_rotation: np.ndarray,
    n_orbitals: int,
    *,
    block: str,
    layer: int | None,
) -> None:
    """把 full orbital rotation 分解并追加到 alpha/beta 两条自旋链。"""
    phases, rotations = decompose_unitary_to_adjacent_rotations(orbital_rotation)
    for spin in ("alpha", "beta"):
        for p, phase in enumerate(phases):
            angle = float(np.angle(phase))
            if abs(angle) <= 1e-12:
                continue
            qubit = spin_qubit(spin, p, n_orbitals)
            start = len(circuit.ops)
            circuit.rz(qubit, angle)
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
                    op_indices=tuple(range(start, len(circuit.ops))),
                )
            )

        for rotation_index, rotation in enumerate(rotations):
            if np.allclose(rotation.matrix, np.eye(2), atol=1e-12):
                continue
            p, q = rotation.modes
            q0 = spin_qubit(spin, p, n_orbitals)
            q1 = spin_qubit(spin, q, n_orbitals)
            start = len(circuit.ops)
            circuit.unitary(q0, q1, matrix=two_mode_fock_matrix(rotation.matrix))
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
                    op_indices=tuple(range(start, len(circuit.ops))),
                )
            )


def _append_diag_coulomb(
    circuit: Circuit,
    logical_ops: list[LogicalOp],
    diag_coulomb_mats: np.ndarray,
    n_orbitals: int,
    topology: str,
    *,
    layer: int,
) -> None:
    """追加一层 diagonal Coulomb number-number 演化。

    ffsim 在 UCJ gate 中使用 `time=-1`，因此本函数对非零 `J` 参数施加
    `CPhase(J)`。这里通过 `rz/rzz` 分解 `CPhase`，忽略全局相位。
    """
    mat_aa = np.asarray(diag_coulomb_mats[0], dtype=float)
    mat_ab = np.asarray(diag_coulomb_mats[1], dtype=float)
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(n_orbitals, topology)

    for spin in ("alpha", "beta"):
        for p, q in pairs_aa:
            theta = float(mat_aa[p, q])
            if abs(theta) <= 1e-12:
                continue
            q0 = spin_qubit(spin, p, n_orbitals)
            q1 = spin_qubit(spin, q, n_orbitals)
            _append_cphase(
                circuit,
                logical_ops,
                q0,
                q1,
                theta,
                block="diag_coulomb",
                layer=layer,
                kind="same_spin_cphase",
                spin=spin,
                orbitals=(p, q),
                parameter=f"diag_coulomb_mats[{layer},0,{p},{q}]",
            )

    for p, q in pairs_ab:
        theta = float(mat_ab[p, q])
        if abs(theta) <= 1e-12:
            continue
        _append_cphase(
            circuit,
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
    circuit: Circuit,
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
    """把 `CPhase(theta)` 分解成 TyxonQ 支持的 `rz/rzz`。"""
    start = len(circuit.ops)
    circuit.rz(q0, theta / 2.0)
    circuit.rz(q1, theta / 2.0)
    circuit.rzz(q0, q1, -theta / 2.0)
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
            op_indices=tuple(range(start, len(circuit.ops))),
        )
    )


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


def _closed_shell_reference_occupations(
    n_electrons: int,
    n_orbitals: int,
) -> dict[str, tuple[int, ...]]:
    """返回 closed-shell RHF 参考态中 alpha/beta 各自占据的空间轨道。"""
    n_occupied = _validate_closed_shell_electrons(n_electrons, n_orbitals) // 2
    occupied = tuple(range(n_occupied))
    return {"alpha": occupied, "beta": occupied}
