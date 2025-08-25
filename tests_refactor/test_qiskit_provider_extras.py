import pytest
from tyxonq.core.ir import Circuit
from tyxonq.compiler.providers.qiskit.dialect import to_qiskit, from_qiskit, ir_to_qasm, qasm_to_ir


@pytest.mark.skipif(__import__("importlib").util.find_spec("qiskit") is None, reason="qiskit not installed")
def test_qiskit_roundtrip_with_measures_and_noops():
    # Include ops that may become no-ops in basis (barrier) and explicit measures
    c = Circuit(num_qubits=2, ops=[
        ("h", 0),
        ("cx", 0, 1),
        ("measure_z", 0),
        ("measure_z", 1),
    ])
    qc = to_qiskit(c)
    c2 = from_qiskit(qc)
    assert c2.num_qubits == c.num_qubits
    # Check essential ops preserved
    kinds = [op[0] for op in c2.ops]
    assert kinds.count("h") == 1 and kinds.count("cx") == 1
    assert kinds.count("measure_z") == 2


@pytest.mark.skipif(__import__("importlib").util.find_spec("qiskit") is None, reason="qiskit not installed")
def test_qasm_roundtrip_with_free_pi_comments():
    c = Circuit(num_qubits=1, ops=[("rz", 0, 3.141592653589793/2)])
    qasm = ir_to_qasm(c)
    c2 = qasm_to_ir(qasm)
    assert c2.ops[0][0] == "rz" and pytest.approx(c2.ops[0][2]) == c.ops[0][2]

