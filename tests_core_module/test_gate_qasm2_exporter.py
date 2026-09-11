# 纯 QASM2 exporter（GateQASM2Exporter）直译契约测试。
#
# 覆盖：header/include/寄存器声明、measure 顺序映射、各参数元数（arity）门、名称归一别名、
# barrier/reset，以及非 qelib1 门（iswap/rxx/ryy/rzz）必须抛 ValueError（要求先分解）。
# 这些用例不依赖 qiskit（exporter 为零外部依赖的纯 IR→字符串）；末尾附一条可选的
# qiskit.qasm2.loads 解析校验，缺 qiskit 时自动跳过。
import importlib

import pytest

from tyxonq.core.ir import Circuit
from tyxonq.compiler.gate_compiler.exporters import GateQASM2Exporter

_qiskit_missing = importlib.util.find_spec("qiskit") is None


def _export(ops, num_qubits):
    return GateQASM2Exporter().export(Circuit(num_qubits=num_qubits, ops=ops))


def test_header_and_registers():
    code = _export([("h", 0), ("cx", 0, 1), ("measure_z", 0), ("measure_z", 1)], 2)
    lines = code.splitlines()
    assert lines[0] == "OPENQASM 2.0;"
    assert lines[1] == 'include "qelib1.inc";'
    assert "qreg q[2];" in lines
    # 经典寄存器大小 = measure 数，按出现顺序映射
    assert "creg c[2];" in lines
    assert "measure q[0]->c[0];" in lines
    assert "measure q[1]->c[1];" in lines


def test_no_measure_omits_creg():
    code = _export([("h", 0)], 1)
    assert "qreg q[1];" in code
    assert "creg" not in code


def test_gate_arities():
    ops = [
        ("h", 0),                    # 单比特无参
        ("rz", 0, 0.5),              # 单比特一参
        ("u2", 0, 0.1, 0.2),         # 单比特二参
        ("u3", 0, 0.1, 0.2, 0.3),    # 单比特三参
        ("cx", 0, 1),                # 双比特无参
        ("crx", 0, 1, 0.7),          # 双比特一参（受控旋转）
        ("ccx", 0, 1, 2),            # 三比特无参
    ]
    code = _export(ops, 3)
    assert "h q[0];" in code
    assert "rz(0.5) q[0];" in code
    assert "u2(0.1, 0.2) q[0];" in code
    assert "u3(0.1, 0.2, 0.3) q[0];" in code
    assert "cx q[0],q[1];" in code
    assert "crx(0.7) q[0],q[1];" in code
    assert "ccx q[0],q[1],q[2];" in code


def test_alias_normalization():
    code = _export([("cnot", 0, 1), ("i", 0), ("phase", 0, 0.5), ("toffoli", 0, 1, 2)], 3)
    assert "cx q[0],q[1];" in code
    assert "id q[0];" in code
    assert "p(0.5) q[0];" in code
    assert "ccx q[0],q[1],q[2];" in code


def test_barrier_and_reset():
    code = _export([("barrier", 0, 1), ("reset", 0)], 2)
    assert "barrier q[0],q[1];" in code
    assert "reset q[0];" in code


@pytest.mark.parametrize("gate", ["iswap", "rxx", "ryy", "rzz"])
def test_non_qelib1_gate_raises(gate):
    # 非 qelib1 门需先分解到兼容基，exporter 直接抛 ValueError
    with pytest.raises(ValueError):
        _export([(gate, 0, 1, 0.5)], 2)


@pytest.mark.skipif(_qiskit_missing, reason="qiskit not installed")
def test_output_parses_with_qiskit():
    from qiskit.qasm2 import loads

    code = _export([("h", 0), ("cx", 0, 1), ("rz", 1, 0.25), ("measure_z", 0), ("measure_z", 1)], 2)
    qc = loads(code)
    assert qc.num_qubits == 2
    names = [inst.operation.name for inst in qc.data]
    for name in ("h", "cx", "rz", "measure"):
        assert name in names
