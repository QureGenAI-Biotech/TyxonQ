"""SpinQCompiler 核心委托与无 Qiskit 中间层测试。"""

from __future__ import annotations

import pytest

import tyxonq as tq
from tyxonq.compiler import api as compiler_api
from tyxonq.compiler.platforms.spinq import compiler as spinq_compiler


@pytest.fixture
def capture_lowering(monkeypatch):
    calls = []

    def fake_to_exe(circuit, **kwargs):
        calls.append((circuit, kwargs))
        return "NATIVE_EXE", {
            "target": "spinqit.native.executable",
            "physical_qubits": kwargs.get("physical_qubits") or list(range(circuit.num_qubits)),
            "gate_stats": {"H": 1, "CX": 1},
        }

    monkeypatch.setattr(spinq_compiler, "to_spinq_exe", fake_to_exe)
    return calls


def test_compiler_delegates_core_then_native_lowering(capture_lowering):
    circuit = tq.Circuit(2).h(0).cx(0, 1).measure_z(0).measure_z(1)
    result = compiler_api.compile(circuit, platform="spinq", output="spinq")

    assert result["compiled_source"] == "NATIVE_EXE"
    assert result["metadata"]["conversion"] == "core_compile_then_ir_to_spinqit_native"
    assert result["metadata"]["core_steps"]
    assert capture_lowering[0][1]["optimization_level"] == 0
    assert circuit._compiled_source == "NATIVE_EXE"


def test_compiler_uses_no_qiskit_or_qasm_interop(capture_lowering, monkeypatch):
    from tyxonq.compiler.gate_compiler.interop import qiskit_interop

    def forbidden(*args, **kwargs):
        raise AssertionError("SpinQ 主链路不应调用 Qiskit/QASM 互操作")

    monkeypatch.setattr(qiskit_interop, "to_qiskit", forbidden)
    circuit = tq.Circuit(2).h(0).cx(0, 1).measure_z(0).measure_z(1)
    result = compiler_api.compile(circuit, platform="spinq", output="spinq")
    assert result["metadata"]["target"] == "spinqit.native.executable"


def test_compiler_forwards_layout_and_native_optimization(capture_lowering):
    circuit = tq.Circuit(2).cx(0, 1).measure_z(0).measure_z(1)
    compiler_api.compile(
        circuit,
        platform="spinq",
        output="spinq",
        options={"physical_qubits": [3, 1], "spinq_optimization_level": 2},
    )
    assert capture_lowering[0][1]["physical_qubits"] == [3, 1]
    assert capture_lowering[0][1]["optimization_level"] == 2


def test_runtime_layout_is_not_baked_into_native_lowering(capture_lowering):
    circuit = tq.Circuit(2).cx(0, 1).measure_z(0).measure_z(1)
    compiler_api.compile(
        circuit,
        platform="spinq",
        output="spinq",
        options={"physical_qubits": [15, 24], "spinq_runtime_layout": True},
    )
    assert capture_lowering[0][1]["physical_qubits"] == [15, 24]
    assert capture_lowering[0][1]["apply_physical_mapping"] is False


def test_routed_layout_and_measurement_order_flow_into_native_lowering(capture_lowering):
    circuit = tq.Circuit(3).cx(0, 2).measure_z(0).measure_z(2)
    device_rule = {
        "coupling_map": [[10, 11], [11, 10], [11, 12], [12, 11]],
        "num_qubits": 13,
        "initial_layout": [10, 11, 12],
    }
    result = compiler_api.compile(
        circuit,
        platform="spinq",
        output="spinq",
        route=True,
        device_rule=device_rule,
    )

    routed_ir = capture_lowering[0][0]
    assert capture_lowering[0][1]["physical_qubits"] == [10, 11, 12]
    assert capture_lowering[0][1]["logical_qubit_map"] == [0, 2, 1]
    assert result["metadata"]["routing"]
    measured = [op[1] for op in routed_ir.ops if op[0] == "measure_z"]
    assert measured == [0, 1]
    assert result["metadata"]["routing"]["final_layout"] == [0, 2, 1]
    # 源 IR 不被路由改写；经典位 0/1 仍对应原来的逻辑比特 0/2。
    assert [op[1] for op in circuit.ops if op[0] == "measure_z"] == [0, 2]


def test_invalid_output_is_rejected(capture_lowering):
    circuit = tq.Circuit(1).x(0).measure_z(0)
    with pytest.raises(ValueError, match="output='spinq'"):
        compiler_api.compile(circuit, platform="spinq", output="qasm2")
