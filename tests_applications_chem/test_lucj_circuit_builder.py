import numpy as np
import pytest

from tyxonq.applications.chem.algorithms.lucj import (
    LUCJ,
    build_lucj_circuit,
    lucj_parameter_count,
)
from tyxonq.applications.chem.algorithms.lucj.circuit_builder import (
    _append_diag_coulomb,
    _append_logical_op_to_circuit,
    _append_orbital_rotation,
    _append_reference_state,
)
from tyxonq.applications.chem.algorithms.lucj.linalg import (
    decompose_slater_determinant_to_adjacent_rotations,
)
from tyxonq.applications.chem.algorithms.lucj.parameters import (
    lucj_parameter_shapes,
    normalize_lucj_params,
)
from tyxonq.applications.chem.algorithms.lucj.topology import alpha_qubit, beta_qubit
from tyxonq.core.ir.circuit import Circuit


def _rotation(theta: float) -> np.ndarray:
    """生成一个简单的 2x2 unitary，方便测试 orbital rotation block。"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _params_for(n_orbitals=4, layers=1, topology="square"):
    """构造符合 matrix UCJ shape 的测试参数。"""
    orbital_rotations = np.tile(np.eye(n_orbitals, dtype=complex), (layers, 1, 1))
    if n_orbitals >= 2:
        orbital_rotations[:, :2, :2] = _rotation(0.2)
    diag_coulomb_mats = np.zeros((layers, 2, n_orbitals, n_orbitals), dtype=float)
    for layer in range(layers):
        for p in range(n_orbitals - 1):
            diag_coulomb_mats[layer, 0, p, p + 1] = 0.1 + 0.01 * p
            diag_coulomb_mats[layer, 0, p + 1, p] = 0.1 + 0.01 * p
        if topology in {"square", "paired7", "paired9", "paired10", "paired12"}:
            indices = range(n_orbitals)
        elif topology == "hex":
            indices = range(0, n_orbitals, 2)
        else:
            indices = [0]
        for p in indices:
            diag_coulomb_mats[layer, 1, p, p] = 0.2 + 0.01 * p
    return {
        "orbital_rotations": orbital_rotations,
        "diag_coulomb_mats": diag_coulomb_mats,
    }


def _open_shell_params(n_orbitals=3, layers=1):
    """构造 α/β 独立旋转及 `aa/ab/bb` 三通道参数。"""
    return {
        "orbital_rotations": np.tile(
            np.eye(n_orbitals, dtype=complex),
            (layers, 2, 1, 1),
        ),
        "diag_coulomb_mats": np.zeros(
            (layers, 3, n_orbitals, n_orbitals),
            dtype=float,
        ),
    }


def _logical_blocks(circuit):
    """提取逻辑 block 名，避免每个测试重复索引 metadata。"""
    return [op["block"] for op in circuit.metadata["lucj"]["logical_ops"]]


def _random_unitary(rng: np.random.Generator, n_orbitals: int) -> np.ndarray:
    """生成确定种子的复 unitary，覆盖一般 orbital rotation。"""
    matrix = rng.normal(size=(n_orbitals, n_orbitals)) + 1j * rng.normal(
        size=(n_orbitals, n_orbitals)
    )
    unitary, _ = np.linalg.qr(matrix)
    return unitary


def _random_params(
    *,
    n_orbitals: int,
    layers: int,
    spin_unbalanced: bool,
    seed: int,
) -> dict[str, np.ndarray]:
    """构造多层、含 final rotation 的一般测试参数。"""
    rng = np.random.default_rng(seed)
    if spin_unbalanced:
        orbital_rotations = np.stack(
            [
                np.stack([_random_unitary(rng, n_orbitals) for _ in range(2)])
                for _ in range(layers)
            ]
        )
        final = np.stack([_random_unitary(rng, n_orbitals) for _ in range(2)])
        n_channels = 3
    else:
        orbital_rotations = np.stack(
            [_random_unitary(rng, n_orbitals) for _ in range(layers)]
        )
        final = _random_unitary(rng, n_orbitals)
        n_channels = 2

    diag_coulomb_mats = rng.normal(
        scale=0.2,
        size=(layers, n_channels, n_orbitals, n_orbitals),
    )
    diag_coulomb_mats = 0.5 * (
        diag_coulomb_mats + np.swapaxes(diag_coulomb_mats, -1, -2)
    )
    return {
        "orbital_rotations": orbital_rotations,
        "diag_coulomb_mats": diag_coulomb_mats,
        "final_orbital_rotation": final,
    }


def _build_unoptimized_circuit_for_test(
    *,
    n_orbitals: int,
    n_electrons: int | tuple[int, int],
    layers: int,
    params: dict[str, np.ndarray],
    excluded_orbitals=None,
    with_final_orbital_rotation: bool,
) -> tuple[Circuit, list[dict]]:
    """仅在测试中复现优化前的 HF -> U† -> J -> U 门序列。"""
    normalized = normalize_lucj_params(
        params,
        n_orbitals,
        layers,
        "linear",
        spin_unbalanced=isinstance(n_electrons, tuple),
        excluded_orbitals=excluded_orbitals,
    )
    orbital_rotations = normalized["orbital_rotations"]
    diag_coulomb_mats = normalized["diag_coulomb_mats"]
    final = normalized["final_orbital_rotation"]
    assert isinstance(orbital_rotations, np.ndarray)
    assert isinstance(diag_coulomb_mats, np.ndarray)

    plan: list[dict] = []
    _append_reference_state(plan, n_electrons, n_orbitals)
    for layer in range(layers):
        _append_orbital_rotation(
            plan,
            np.swapaxes(orbital_rotations[layer].conj(), -1, -2),
            n_orbitals,
            block="orbital_rotation_dagger",
            layer=layer,
        )
        _append_diag_coulomb(
            plan,
            diag_coulomb_mats[layer],
            n_orbitals,
            "linear",
            layer=layer,
        )
        _append_orbital_rotation(
            plan,
            orbital_rotations[layer],
            n_orbitals,
            block="orbital_rotation",
            layer=layer,
        )
    if with_final_orbital_rotation and isinstance(final, np.ndarray):
        _append_orbital_rotation(
            plan,
            final,
            n_orbitals,
            block="final_orbital_rotation",
            layer=None,
        )

    circuit = Circuit(2 * n_orbitals)
    logical_ops: list[dict] = []
    for operation in plan:
        _append_logical_op_to_circuit(circuit, logical_ops, operation)
    return circuit, logical_ops


def _assert_states_equal_up_to_global_phase(
    reference: np.ndarray,
    actual: np.ndarray,
) -> None:
    """比较两个归一化态，允许 Slater 制备带来全局相位。"""
    overlap = np.vdot(reference, actual)
    assert abs(overlap) > 1e-12
    np.testing.assert_allclose(actual, overlap / abs(overlap) * reference, atol=1e-10)


@pytest.mark.parametrize(
    ("topology", "parameter_count"),
    [
        ("square", 23),
        ("hex", 21),
        ("linear", 20),
        ("paired7", 23),
        ("paired9", 23),
        ("paired10", 23),
        ("paired12", 23),
    ],
)
def test_parameter_shapes_and_counts(topology, parameter_count):
    shapes = lucj_parameter_shapes(n_orbitals=4, layers=1, topology=topology)

    assert shapes["orbital_rotations"] == (1, 4, 4)
    assert shapes["diag_coulomb_mats"] == (1, 2, 4, 4)
    assert set(shapes) == {"orbital_rotations", "diag_coulomb_mats"}
    assert lucj_parameter_count(4, 1, topology) == parameter_count


def test_open_shell_parameter_shapes_and_counts():
    lucj = LUCJ(4, (3, 1), 1, "linear")

    assert lucj.parameter_shapes() == {
        "orbital_rotations": (1, 2, 4, 4),
        "diag_coulomb_mats": (1, 3, 4, 4),
    }
    assert lucj.parameter_shapes(with_final_orbital_rotation=True)[
        "final_orbital_rotation"
    ] == (2, 4, 4)
    assert lucj_parameter_count(4, 1, "linear", spin_unbalanced=True) == 39


def test_qubit_mapping_alpha_beta_order():
    assert [alpha_qubit(p) for p in range(4)] == [0, 1, 2, 3]
    assert [beta_qubit(p, n_orbitals=4) for p in range(4)] == [4, 5, 6, 7]


def test_public_lucj_defaults_to_linear():
    lucj = LUCJ(4, 4, 1)

    assert lucj.topology == "linear"
    assert lucj_parameter_count(4, 1) == 20


@pytest.mark.parametrize(
    ("topology", "opposite_count"),
    [
        ("square", 4),
        ("hex", 2),
        ("linear", 1),
        ("paired7", 4),
        ("paired9", 4),
        ("paired10", 4),
        ("paired12", 4),
    ],
)
def test_j_block_topology_counts(topology, opposite_count):
    circuit = build_lucj_circuit(4, 4, 1, topology, _params_for(4, 1, topology))
    j_ops = [
        op
        for op in circuit.metadata["lucj"]["logical_ops"]
        if op["block"] == "diag_coulomb" and op["layer"] == 0
    ]

    assert sum(op["kind"] == "same_spin_cphase" for op in j_ops) == 6
    assert sum(op["kind"] == "opposite_spin_cphase" for op in j_ops) == opposite_count


@pytest.mark.parametrize(("n_orbitals", "expected"), [(6, 6), (12, 10)])
def test_paired10_opposite_spin_pair_count(n_orbitals, expected):
    """小分片保留全部 pair，大分片严格截断到前 10 个 pair。"""
    params = _params_for(n_orbitals, 1, "paired10")
    circuit = build_lucj_circuit(n_orbitals, 2, 1, "paired10", params)
    opposite_ops = [
        op
        for op in circuit.metadata["lucj"]["logical_ops"]
        if op["kind"] == "opposite_spin_cphase"
    ]

    assert len(opposite_ops) == expected
    assert {op["orbitals"] for op in opposite_ops} == {
        (p, p) for p in range(expected)
    }


@pytest.mark.parametrize(("n_orbitals", "expected"), [(6, 6), (12, 7)])
def test_paired7_opposite_spin_pair_count(n_orbitals, expected):
    """小分片保留全部 pair，大分片严格截断到前 7 个 pair。"""

    params = _params_for(n_orbitals, 1, "paired7")
    circuit = build_lucj_circuit(n_orbitals, 2, 1, "paired7", params)
    opposite_ops = [
        op
        for op in circuit.metadata["lucj"]["logical_ops"]
        if op["kind"] == "opposite_spin_cphase"
    ]

    assert len(opposite_ops) == expected
    assert {op["orbitals"] for op in opposite_ops} == {
        (p, p) for p in range(expected)
    }


@pytest.mark.parametrize(("n_orbitals", "expected"), [(6, 6), (12, 9), (37, 9)])
def test_paired9_opposite_spin_pair_count(n_orbitals, expected):
    """paired9 对小分片全保留，对大分片严格停在第 9 对。"""

    params = _params_for(n_orbitals, 1, "paired9")
    circuit = build_lucj_circuit(n_orbitals, 2, 1, "paired9", params)
    opposite_ops = [
        op
        for op in circuit.metadata["lucj"]["logical_ops"]
        if op["kind"] == "opposite_spin_cphase"
    ]

    assert len(opposite_ops) == expected
    assert {op["orbitals"] for op in opposite_ops} == {
        (p, p) for p in range(expected)
    }


def test_paired10_circuit_emits_only_allowed_interaction_pairs():
    """即使矩阵其余位置非零，线路也只能发射 rail 与前 10 个 rung。"""
    n_orbitals = 12
    params = _params_for(n_orbitals, 1, "paired10")
    params["diag_coulomb_mats"][0, 0] = 0.37
    params["diag_coulomb_mats"][0, 1] = 0.41
    circuit = build_lucj_circuit(n_orbitals, 2, 1, "paired10", params)
    j_ops = [
        op
        for op in circuit.metadata["lucj"]["logical_ops"]
        if op["block"] == "diag_coulomb"
    ]

    same_spin = {
        (op["spin"], op["orbitals"])
        for op in j_ops
        if op["kind"] == "same_spin_cphase"
    }
    opposite_spin = {
        op["orbitals"]
        for op in j_ops
        if op["kind"] == "opposite_spin_cphase"
    }
    expected_rail = {(p, p + 1) for p in range(n_orbitals - 1)}
    assert same_spin == {
        (spin, pair) for spin in ("alpha", "beta") for pair in expected_rail
    }
    assert opposite_spin == {(p, p) for p in range(10)}


@pytest.mark.parametrize(
    ("n_orbitals", "expected"),
    [(10, 10), (12, 12), (37, 12)],
)
def test_paired12_opposite_spin_pair_count(n_orbitals, expected):
    """paired12 对小分片全保留，对大分片严格停在第 12 对。"""

    params = _params_for(n_orbitals, 1, "paired12")
    circuit = build_lucj_circuit(n_orbitals, 2, 1, "paired12", params)
    opposite_ops = [
        operation
        for operation in circuit.metadata["lucj"]["logical_ops"]
        if operation["kind"] == "opposite_spin_cphase"
    ]

    assert len(opposite_ops) == expected
    assert {operation["orbitals"] for operation in opposite_ops} == {
        (p, p) for p in range(expected)
    }


def test_reference_state_prepares_closed_shell_rhf():
    circuit = build_lucj_circuit(4, 4, 1, "square", _params_for(4, 1, "square"))
    reference_ops = [
        op for op in circuit.metadata["lucj"]["logical_ops"] if op["block"] == "reference"
    ]

    assert [op["kind"] for op in reference_ops] == ["x", "x", "x", "x"]
    assert [op["qubits"] for op in reference_ops] == [(0,), (1,), (4,), (5,)]
    assert circuit.metadata["lucj"]["reference_occupations"] == {
        "alpha": (0, 1),
        "beta": (0, 1),
    }


def test_reference_state_prepares_open_shell_alpha_beta_counts():
    circuit = build_lucj_circuit(3, (2, 1), 1, "linear", _open_shell_params())
    reference_ops = [
        op for op in circuit.metadata["lucj"]["logical_ops"] if op["block"] == "reference"
    ]

    assert [op["qubits"] for op in reference_ops] == [(0,), (1,), (3,)]
    assert circuit.metadata["lucj"]["reference_occupations"] == {
        "alpha": (0, 1),
        "beta": (0,),
    }
    assert circuit.metadata["lucj"]["spin_unbalanced"] is True


def test_open_shell_uses_independent_rotations_and_three_coulomb_channels():
    params = _open_shell_params()
    params["orbital_rotations"][0, 0, :2, :2] = _rotation(0.2)
    params["orbital_rotations"][0, 1, 1:, 1:] = _rotation(0.3)
    params["diag_coulomb_mats"][0, 0, 0, 1] = 0.11
    params["diag_coulomb_mats"][0, 0, 1, 0] = 0.11
    params["diag_coulomb_mats"][0, 1, 0, 0] = 0.22
    params["diag_coulomb_mats"][0, 2, 1, 2] = 0.33
    params["diag_coulomb_mats"][0, 2, 2, 1] = 0.33

    circuit = build_lucj_circuit(3, (2, 1), 1, "linear", params)
    logical_ops = circuit.metadata["lucj"]["logical_ops"]
    dagger_ops = [op for op in logical_ops if op["block"] == "orbital_rotation_dagger"]
    orbital_ops = [op for op in logical_ops if op["kind"] == "orbital_givens"]
    coulomb_ops = [op for op in logical_ops if op["block"] == "diag_coulomb"]

    # beta 的第一段 rotation 只作用于虚轨道，不改变一电子参考态，因此会被 Slater 制备省略。
    assert {(op["spin"], op["orbitals"]) for op in dagger_ops} == {
        ("alpha", (0, 1))
    }
    assert {(op["spin"], op["orbitals"]) for op in orbital_ops} >= {
        ("alpha", (0, 1)),
        ("beta", (1, 2)),
    }
    assert {(op["spin"], op["theta"]) for op in coulomb_ops} == {
        ("alpha", 0.11),
        ("alpha-beta", 0.22),
        ("beta", 0.33),
    }
    beta_op = next(op for op in coulomb_ops if op["spin"] == "beta")
    assert beta_op["parameter"].startswith("diag_coulomb_mats[0,2")


def test_excluded_orbitals_remain_empty_and_decoupled():
    params = _open_shell_params()
    params["orbital_rotations"][0, 0, 1:, 1:] = _rotation(0.4)
    params["diag_coulomb_mats"][:] = 0.2
    for channel in (0, 2):
        params["diag_coulomb_mats"][0, channel] = (
            params["diag_coulomb_mats"][0, channel]
            + params["diag_coulomb_mats"][0, channel].T
        ) / 2

    circuit = LUCJ(
        3,
        (1, 1),
        1,
        excluded_orbitals=((2,), (2,)),
    ).get_circuit(params)
    logical_ops = circuit.metadata["lucj"]["logical_ops"]

    assert all(2 not in op["qubits"] and 5 not in op["qubits"] for op in logical_ops)
    assert circuit.metadata["lucj"]["excluded_orbitals"] == {
        "alpha": (2,),
        "beta": (2,),
    }


def test_layers_follow_reference_orbital_j_orbital_order():
    circuit = build_lucj_circuit(4, 4, 1, "square", _params_for(4, 1, "square"))
    blocks = [op["block"] for op in circuit.metadata["lucj"]["logical_ops"]]

    first_orbital = blocks.index("orbital_rotation_dagger")
    first_j = blocks.index("diag_coulomb")
    first_forward = blocks.index("orbital_rotation")
    assert all(block == "reference" for block in blocks[:first_orbital])
    assert first_orbital < first_j < first_forward


def test_final_orbital_rotation_can_be_disabled_explicitly():
    params = _params_for(4, 1, "square")
    params["final_orbital_rotation"] = np.eye(4, dtype=complex)
    params["final_orbital_rotation"][:2, :2] = _rotation(0.4)

    circuit = build_lucj_circuit(4, 4, 1, "square", params, with_final_orbital_rotation=False)
    class_circuit = LUCJ(4, 4, 1, "square").get_circuit(params, with_final_orbital_rotation=False)

    assert "final_orbital_rotation" not in _logical_blocks(circuit)
    assert "final_orbital_rotation" not in _logical_blocks(class_circuit)
    assert circuit.metadata["lucj"]["has_final_orbital_rotation"] is False


def test_final_orbital_rotation_can_be_enabled_explicitly():
    params = _params_for(4, 1, "square")
    params["final_orbital_rotation"] = np.eye(4, dtype=complex)
    params["final_orbital_rotation"][:2, :2] = _rotation(0.4)

    circuit = build_lucj_circuit(
        4,
        4,
        1,
        "square",
        params,
        with_final_orbital_rotation=True,
    )

    assert "final_orbital_rotation" in _logical_blocks(circuit)
    assert circuit.metadata["lucj"]["has_final_orbital_rotation"] is True


@pytest.mark.parametrize(
    ("n_electrons", "excluded_orbitals", "spin_unbalanced"),
    [
        (4, None, False),
        ((2, 1), ((3,), (2, 3)), True),
    ],
)
@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_circuit_optimization_preserves_state_and_reduces_two_qubit_gates(
    n_electrons,
    excluded_orbitals,
    spin_unbalanced,
    with_final_orbital_rotation,
):
    """两种优化在 restricted/unrestricted 下都保持理想态并减少门数。"""
    n_orbitals = 4
    layers = 2
    params = _random_params(
        n_orbitals=n_orbitals,
        layers=layers,
        spin_unbalanced=spin_unbalanced,
        seed=20260817,
    )
    optimized = build_lucj_circuit(
        n_orbitals,
        n_electrons,
        layers,
        "linear",
        params,
        excluded_orbitals=excluded_orbitals,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )
    unoptimized, unoptimized_logical_ops = _build_unoptimized_circuit_for_test(
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
        layers=layers,
        params=params,
        excluded_orbitals=excluded_orbitals,
        with_final_orbital_rotation=with_final_orbital_rotation,
    )

    _assert_states_equal_up_to_global_phase(
        unoptimized.state(form="numpy"),
        optimized.state(form="numpy"),
    )
    optimized_logical_ops = optimized.metadata["lucj"]["logical_ops"]
    optimized_givens = sum(
        operation["kind"] == "orbital_givens" for operation in optimized_logical_ops
    )
    unoptimized_givens = sum(
        operation["kind"] == "orbital_givens" for operation in unoptimized_logical_ops
    )
    assert optimized_givens < unoptimized_givens

    j_kinds = {"same_spin_cphase", "opposite_spin_cphase"}
    assert sum(operation["kind"] in j_kinds for operation in optimized_logical_ops) == sum(
        operation["kind"] in j_kinds for operation in unoptimized_logical_ops
    )
    assert sum(operation[0] in {"unitary", "rzz"} for operation in optimized.ops) < sum(
        operation[0] in {"unitary", "rzz"} for operation in unoptimized.ops
    )

    if excluded_orbitals is not None:
        excluded_qubits = {3, 6, 7}
        assert all(
            excluded_qubits.isdisjoint(operation["qubits"])
            for operation in optimized_logical_ops
        )

    if with_final_orbital_rotation:
        assert optimized.metadata["lucj"]["has_final_orbital_rotation"] is True
        assert "final_orbital_rotation" in _logical_blocks(optimized)
    else:
        assert optimized.metadata["lucj"]["has_final_orbital_rotation"] is False
        assert "final_orbital_rotation" not in _logical_blocks(optimized)
        params_without_final = dict(params)
        params_without_final.pop("final_orbital_rotation")
        without_final_data = build_lucj_circuit(
            n_orbitals,
            n_electrons,
            layers,
            "linear",
            params_without_final,
            excluded_orbitals=excluded_orbitals,
        )
        _assert_states_equal_up_to_global_phase(
            without_final_data.state(form="numpy"),
            optimized.state(form="numpy"),
        )


def test_slater_decomposition_handles_empty_and_full_occupation():
    """零电子和满占据自旋块都不需要 Givens 门。"""
    assert decompose_slater_determinant_to_adjacent_rotations(
        np.empty((0, 3), dtype=complex)
    ) == []
    assert decompose_slater_determinant_to_adjacent_rotations(
        _random_unitary(np.random.default_rng(7), 3)
    ) == []


def test_builder_uses_supported_circuit_ops_only():
    circuit = build_lucj_circuit(4, 4, 1, "linear", _params_for(4, 1, "linear"))
    op_names = {op[0] for op in circuit.ops}

    assert op_names <= {"x", "rz", "rzz", "unitary"}
    assert "rxx" not in op_names
    assert "ryy" not in op_names


def test_explicit_linear_builder_preserves_gate_order_and_metadata_contract():
    circuit = build_lucj_circuit(2, 2, 1, "linear", _params_for(2, 1, "linear"))

    assert [op[0] for op in circuit.ops] == [
        "x",
        "x",
        "unitary",
        "unitary",
        "rz",
        "rz",
        "rzz",
        "rz",
        "rz",
        "rzz",
        "rz",
        "rz",
        "rzz",
        "unitary",
        "unitary",
    ]
    logical_ops = circuit.metadata["lucj"]["logical_ops"]
    assert [(op["block"], op["kind"], op["qubits"], op["op_indices"]) for op in logical_ops] == [
        ("reference", "x", (0,), (0,)),
        ("reference", "x", (2,), (1,)),
        ("orbital_rotation_dagger", "orbital_givens", (0, 1), (2,)),
        ("orbital_rotation_dagger", "orbital_givens", (2, 3), (3,)),
        ("diag_coulomb", "same_spin_cphase", (0, 1), (4, 5, 6)),
        ("diag_coulomb", "same_spin_cphase", (2, 3), (7, 8, 9)),
        ("diag_coulomb", "opposite_spin_cphase", (0, 2), (10, 11, 12)),
        ("orbital_rotation", "orbital_givens", (0, 1), (13,)),
        ("orbital_rotation", "orbital_givens", (2, 3), (14,)),
    ]


def test_invalid_parameter_shape_raises_clear_error():
    params = _params_for(n_orbitals=4, layers=1, topology="square")
    params["orbital_rotations"] = np.zeros((1, 3, 3), dtype=complex)

    with pytest.raises(ValueError, match="orbital_rotations"):
        build_lucj_circuit(4, 4, 1, "square", params)


@pytest.mark.parametrize("n_electrons", [0, 3, 10])
def test_invalid_closed_shell_electrons_raise(n_electrons):
    with pytest.raises(ValueError, match="n_electrons"):
        LUCJ(4, n_electrons, 1, "square")


@pytest.mark.parametrize("n_electrons", [(0, 0), (5, 1), (1,), (1, 2, 3)])
def test_invalid_open_shell_electrons_raise(n_electrons):
    with pytest.raises(ValueError, match="n_electrons|electron count"):
        LUCJ(4, n_electrons, 1, "square")


def test_excluded_orbitals_cannot_overlap_reference():
    with pytest.raises(ValueError, match="overlap occupied alpha"):
        LUCJ(3, (2, 1), 1, excluded_orbitals=((1,), ()))


@pytest.mark.parametrize("nelec", [(1, 1), (2, 2)])
def test_explicit_pairs_match_ffsim_state(nelec):
    """以相同矩阵比较态，覆盖非对角 ab、非相邻 aa 及位序转换。"""
    ffsim = pytest.importorskip("ffsim")
    n = 3
    pairs = ([(0, 2)], [(0, 0), (0, 2)])
    params = _random_params(n_orbitals=n, layers=2, spin_unbalanced=False, seed=27)
    original_mats = params["diag_coulomb_mats"].copy()
    normalized = normalize_lucj_params(params, n, 2, interaction_pairs=pairs)
    circuit = LUCJ(n, sum(nelec), 2, interaction_pairs=pairs).get_circuit(params)
    np.testing.assert_array_equal(params["diag_coulomb_mats"], original_mats)
    assert not np.shares_memory(normalized["diag_coulomb_mats"], params["diag_coulomb_mats"])
    op = ffsim.UCJOpSpinBalanced(**normalized)
    reference = ffsim.apply_unitary(ffsim.hartree_fock_state(n, nelec), op, n, nelec)
    # ffsim 的全空间转换遵循 Qiskit 小端位序；TyxonQ state 的 q0 在最高位。
    reference = ffsim.qiskit.ffsim_vec_to_qiskit_vec(reference, n, nelec)
    reference = reference.reshape((2,) * (2 * n)).transpose(tuple(range(2 * n - 1, -1, -1))).ravel()
    _assert_states_equal_up_to_global_phase(reference, circuit.state(form="numpy"))
    assert lucj_parameter_count(n, 2, interaction_pairs=pairs) == 2 * (n**2 + 3)


def test_paper_pairs_override_topology_and_exclusions():
    """论文四条异自旋耦合覆盖 linear 默认；dummy 轨道仍被剔除。"""
    n = 13
    pairs = ([(p, p + 1) for p in range(n - 1)], [(p, p) for p in (0, 4, 8, 12)])
    params = _params_for(n_orbitals=n, topology="square")
    circuit = LUCJ(n, 2, 1, interaction_pairs=pairs, excluded_orbitals=[12]).get_circuit(params)
    operations = circuit.metadata["lucj"]["logical_ops"]
    assert {op["orbitals"] for op in operations if op["kind"] == "opposite_spin_cphase"} == {
        (0, 0), (4, 4), (8, 8)
    }
    assert circuit.metadata["lucj"]["interaction_pairs"] == pairs


@pytest.mark.parametrize("pairs", [([(1, 0)], []), ([(0, 0)], []), ([], [(0, 3)]), ([], [(0, 0), (0, 0)])])
def test_explicit_pairs_reject_invalid_indices(pairs):
    with pytest.raises(ValueError):
        LUCJ(3, 2, 1, interaction_pairs=pairs)


def test_explicit_open_shell_ab_is_ordered():
    """独立自旋轨道下 ab 不隐含反向耦合。"""
    params = _open_shell_params()
    params["diag_coulomb_mats"][0, 1, 2, 0] = 0.2
    circuit = LUCJ(3, (2, 1), 1, interaction_pairs=([], [(2, 0)])).get_circuit(params)
    pairs = [op["orbitals"] for op in circuit.metadata["lucj"]["logical_ops"] if op["kind"] == "opposite_spin_cphase"]
    assert pairs == [(2, 0)]
