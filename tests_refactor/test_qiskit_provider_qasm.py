import pytest

qiskit = pytest.importorskip("qiskit")

from tyxonq.core.ir import Circuit
from tyxonq.compiler.providers.qiskit.dialect import ir_to_qasm, qasm_to_ir


def test_ir_qasm_roundtrip_minimal():
    c = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("rz", 1, 0.25), ("measure_z", 0)])
    qasm = ir_to_qasm(c)
    assert "OPENQASM" in qasm or qasm.strip().startswith("OPENQASM")

    c2 = qasm_to_ir(qasm)
    assert c2.num_qubits == 2
    names = [op[0] for op in c2.ops]
    for name in ("h", "cx", "rz"):
        assert name in names


