from tyxonq.core.ir.circuit import Circuit
from tyxonq.visualization import circuit_to_dot


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


