import importlib

import pytest

from tyxonq.compiler.api import compile as compile_ir
from tyxonq.core.ir import Circuit


def test_compile_with_default_provider_returns_ir():
    circ = Circuit(num_qubits=1, ops=[("h", 0), ("measure_z", 0)])
    res = compile_ir(circ, provider="default", output="ir")
    assert res["circuit"] is circ
    assert res["metadata"]["target"] == "tyxonq"


@pytest.mark.skipif(importlib.util.find_spec("qiskit") is None, reason="qiskit not installed")
def test_compile_with_qiskit_provider_returns_qc():
    from qiskit import QuantumCircuit

    circ = Circuit(num_qubits=1, ops=[("h", 0), ("measure_z", 0)])
    res = compile_ir(circ, provider="qiskit", output="qiskit", options={"opt_level": 1})
    assert isinstance(res["circuit"], QuantumCircuit)

