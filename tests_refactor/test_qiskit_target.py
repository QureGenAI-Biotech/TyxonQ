from tyxonq.compiler.targets.qiskit import QiskitCompiler
from tyxonq.core.ir import Circuit


def test_qiskit_compiler_minimal_contract():
    comp = QiskitCompiler()
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1)])
    res = comp.compile({"circuit": circ, "target": {"native_gates": set()}, "options": {"opt_level": 1}})  # type: ignore[arg-type]
    assert res["circuit"] is circ
    assert res["metadata"]["target"] == "qiskit"
    assert "options" in res["metadata"]


