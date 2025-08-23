import pytest

qiskit = pytest.importorskip("qiskit")

from tyxonq.core.ir import Circuit
from tyxonq.compiler.providers.qiskit.qiskit_compiler import ir_to_qiskit, qiskit_to_ir


def test_ir_to_qiskit_and_back_minimal_roundtrip():
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("rz", 1, 0.3), ("measure_z", 0)])
    qc = ir_to_qiskit(circ)
    assert qc.num_qubits == 2

    # Round trip back to IR (will include measure_z if present)
    circ2 = qiskit_to_ir(qc)
    assert circ2.num_qubits == 2
    # We don't guarantee identical ordering of measures in qiskit, but ops should be supported subset
    names = [op[0] for op in circ2.ops]
    for name in ("h", "cx", "rz"):
        assert name in names


