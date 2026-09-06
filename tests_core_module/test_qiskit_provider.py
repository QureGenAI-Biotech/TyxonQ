# qiskit provider 测试合并：QiskitCompiler 契约、IR<->qiskit 与 IR<->QASM 往返、
# 以及 dialect 纯函数（basis/opt 归一化、free_pi、注释生成）。需要 qiskit 的用例通过
# skipif 单独跳过；dialect 纯函数用例不依赖 qiskit，故不加标记。
import importlib

import pytest

from tyxonq.core.ir import Circuit
from tyxonq.compiler.compile_engine.qiskit.dialect import (
    DEFAULT_BASIS_GATES,
    DEFAULT_OPT_LEVEL,
    comment_dict,
    comment_qasm,
    free_pi,
    normalize_transpile_options,
    to_qiskit,
    from_qiskit,
    ir_to_qasm,
    qasm_to_ir,
)

_qiskit_missing = importlib.util.find_spec("qiskit") is None


@pytest.mark.skipif(_qiskit_missing, reason="qiskit not installed")
def test_qiskit_compiler_minimal_contract():
    from qiskit import QuantumCircuit
    from tyxonq.compiler.compile_engine.qiskit import QiskitCompiler

    comp = QiskitCompiler()
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("measure_z", 1)])
    res = comp.compile(circuit = circ, device_rule= {"native_gates": set()}, options =  {"opt_level": 1}, output = "qiskit")  # type: ignore[arg-type]
    assert isinstance(res["circuit"], QuantumCircuit)
    assert "options" in res["metadata"]


@pytest.mark.skipif(_qiskit_missing, reason="qiskit not installed")
def test_ir_qasm_roundtrip_minimal():
    c = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("rz", 1, 0.25), ("measure_z", 0)])
    qasm = ir_to_qasm(c)
    assert "OPENQASM" in qasm or qasm.strip().startswith("OPENQASM")

    c2 = qasm_to_ir(qasm)
    assert c2.num_qubits == 2
    names = [op[0] for op in c2.ops]
    for name in ("h", "cx", "rz"):
        assert name in names


@pytest.mark.skipif(_qiskit_missing, reason="qiskit not installed")
def test_ir_to_qiskit_and_back_minimal_roundtrip():
    from tyxonq.compiler.compile_engine.qiskit.qiskit_compiler import ir_to_qiskit, qiskit_to_ir

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


def test_normalize_transpile_options_defaults():
    opts = normalize_transpile_options(None)
    assert opts["basis_gates"] == DEFAULT_BASIS_GATES
    assert opts["optimization_level"] == DEFAULT_OPT_LEVEL


def test_free_pi_and_comments():
    s = "u(2*pi, pi/2, -pi/2) q[0];"
    out = free_pi(s)
    assert "pi" not in out
    commented = comment_qasm(out)
    assert commented.startswith("//circuit begins")
    mapping = comment_dict({0: 2, 1: 3})
    assert "logical_physical_mapping" in mapping


@pytest.mark.skipif(_qiskit_missing, reason="qiskit not installed")
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


@pytest.mark.skipif(_qiskit_missing, reason="qiskit not installed")
def test_qasm_roundtrip_with_free_pi_comments():
    c = Circuit(num_qubits=1, ops=[("rz", 0, 3.141592653589793/2)])
    qasm = ir_to_qasm(c)
    c2 = qasm_to_ir(qasm)
    assert c2.ops[0][0] == "rz" and pytest.approx(c2.ops[0][2]) == c.ops[0][2]
