# 核心 IR 测试合并：Circuit / Hamiltonian 的构造、字段校验、编辑（compose/remap/
# inverse）、指令与序列化（to_json_str/from_json_str），以及可视化渲染（circuit_to_dot）。
import pytest

from tyxonq.core.ir import Circuit, Hamiltonian
from tyxonq.visualization import circuit_to_dot


def test_circuit_minimal_fields():
    c = Circuit(num_qubits=2, ops=[("rx", 0, 0.3), ("cx", 0, 1)])
    assert c.num_qubits == 2
    assert len(c.ops) == 2
    assert isinstance(c.metadata, dict)


def test_hamiltonian_terms_container():
    ham = Hamiltonian(terms=[("Z", 0, 1.0), ("ZZ", (0, 1), 0.5)])
    assert hasattr(ham, "terms")
    assert len(ham.terms) == 2


def test_circuit_validates_qubits_and_with_metadata():
    c = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1)])
    c2 = c.with_metadata(tag="test")
    assert c2.metadata.get("tag") == "test"

    with pytest.raises(ValueError):
        Circuit(num_qubits=1, ops=[("h", 1)])


def test_gate_count_and_summary_and_extended_and_json():
    c = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1)])
    assert c.gate_count() == 2
    assert c.gate_count(["h"]) == 1
    s = c.gate_summary()
    assert s["h"] == 1 and s["cx"] == 1

    c2 = c.extended([("rz", 1, 0.1)])
    assert c2.gate_count() == 3
    assert c.gate_count() == 2  # no mutation

    js = c2.to_json_str()
    c3 = Circuit.from_json_str(js)
    assert c3.num_qubits == 2
    assert [op[0] for op in c3.ops] == ["h", "cx", "rz"]


def test_compose_and_remap_and_inverse_and_measure_mapping():
    a = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1)])
    b = Circuit(num_qubits=2, ops=[("rz", 1, 0.5), ("measure_z", 1)])

    c = a.compose(b)  # same size
    assert [op[0] for op in c.ops] == ["h", "cx", "rz", "measure_z"]

    # remap b: logical 0->1, 1->0 onto a (2 qubits)
    c2 = a.compose(b, indices=[1, 0])
    assert ("rz", 0, 0.5) in c2.ops

    # remap explicit
    c3 = c2.remap_qubits({0: 1, 1: 0})
    assert ("rz", 1, 0.5) in c3.ops

    inv = a.inverse()
    assert [op[0] for op in inv.ops] == ["cx", "h"]  # inverse order

    # positional->logical mapping from measures
    circ_m = Circuit(num_qubits=3, ops=[("h", 0), ("measure_z", 2), ("measure_z", 0)])
    plm = circ_m.positional_logical_mapping()
    assert plm == {0: 2, 1: 0}


def test_instruction_add_and_mapping_and_json():
    c = Circuit(num_qubits=3, ops=[("h", 0)])
    c2 = c.add_measure(2).add_measure(0).add_barrier(0, 2).add_reset(1)
    plm = c2.positional_logical_mapping()
    assert plm == {0: 2, 1: 0}

    js = c2.to_json_str()
    c3 = Circuit.from_json_str(js)
    assert c3.instructions[:2] == [("measure", (2,)), ("measure", (0,))]


def test_circuit_to_dot_basic_single_and_two_qubit():
    c = Circuit(num_qubits=2, ops=[
        ("h", 0),
        ("cx", 0, 1),
        ("rz", 1, 0.5),
    ])
    dot = circuit_to_dot(c)
    assert dot.startswith("digraph \"")
    assert "q0" in dot and "q1" in dot
    assert "h" in dot and "cx" in dot and "rz" in dot
