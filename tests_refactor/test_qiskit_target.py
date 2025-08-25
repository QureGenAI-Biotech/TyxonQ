import importlib

import pytest

from tyxonq.compiler.providers.qiskit import QiskitCompiler
from tyxonq.core.ir import Circuit


@pytest.mark.skipif(importlib.util.find_spec("qiskit") is None, reason="qiskit not installed")
def test_qiskit_compiler_minimal_contract():
    from qiskit import QuantumCircuit

    comp = QiskitCompiler()
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("measure_z", 1)])
    res = comp.compile({"circuit": circ, "target": {"native_gates": set()}, "options": {"opt_level": 1, "output": "qiskit"}})  # type: ignore[arg-type]
    assert isinstance(res["circuit"], QuantumCircuit)
    assert res["metadata"]["target"] == "qiskit"
    assert "options" in res["metadata"]


