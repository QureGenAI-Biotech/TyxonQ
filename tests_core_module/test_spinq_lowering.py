"""SpinQit 原生 IR lowering 测试；不需要安装 SpinQit，也不联网。"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tyxonq as tq
from tyxonq.compiler.platforms.spinq import lowering


class _FakeCircuit:
    def __init__(self):
        self.instructions = []
        self.measurements = []

    def allocateQubits(self, count):
        self.allocated_qubits = count
        self.qubits = list(range(count))
        return self.qubits

    def allocateClbits(self, count):
        self.allocated_clbits = count
        self.clbits = list(range(count))
        return self.clbits

    def __lshift__(self, instruction):
        self.instructions.append(instruction)
        return self

    def measure(self, qubit, clbit):
        self.measurements.append((qubit, clbit))


class _FakeExecutable:
    def __init__(self, circuit, level):
        self.circuit = circuit
        self.level = level


class _FakeNativeCompiler:
    def compile(self, circuit, level):
        return _FakeExecutable(circuit, level)


def _fake_api():
    names = (
        "I H X Y Z S Sd T Td Rx Ry Rz CX CY CZ SWAP CCX CSWAP BARRIER"
    ).split()
    return SimpleNamespace(
        Circuit=_FakeCircuit,
        get_compiler=lambda name: _FakeNativeCompiler(),
        **{name: name for name in names},
    )


@pytest.fixture
def fake_api():
    return _fake_api()


def test_native_gates_measurements_and_compile_time_mapping(fake_api):
    circuit = tq.Circuit(
        3,
        ops=[
            ("h", 0), ("rx", 1, 0.2), ("cx", 0, 2),
            ("rzz", 1, 2, 0.3), ("measure_z", 2), ("measure_z", 0),
        ],
    )
    native, metadata = lowering.to_spinq_circuit(
        circuit, physical_qubits=[2, 0, 1], api=fake_api
    )

    assert native.allocated_qubits == 3
    assert native.allocated_clbits == 0
    assert [item[0] for item in native.instructions] == ["H", "Rx", "CX", "CX", "Rz", "CX"]
    assert native.instructions[0][1] == 2
    assert native.instructions[1][1:] == (0, 0.2)
    assert native.instructions[2][1] == (2, 1)
    assert native.measurements == []
    assert metadata["physical_qubits"] == [2, 0, 1]
    assert metadata["measurement_map"] == [
        {"classical_bit": 0, "ir_qubit": 2, "logical_qubit": 2, "physical_qubit": 1},
        {"classical_bit": 1, "ir_qubit": 0, "logical_qubit": 0, "physical_qubit": 2},
    ]
    assert metadata["mapping_stage"] == "compile_time_lowering"
    assert metadata["bit_order"] == "little_endian"


def test_routed_measurement_metadata_traces_original_logical_qubits(fake_api):
    circuit = tq.Circuit(
        3,
        ops=[("cx", 0, 2), ("measure_z", 0), ("measure_z", 1)],
    )
    _, metadata = lowering.to_spinq_circuit(
        circuit,
        physical_qubits=[10, 11, 12],
        logical_qubit_map=[0, 2, 1],
        api=fake_api,
    )
    assert metadata["measured_qubits"] == [0, 1]
    assert metadata["measured_logical_qubits"] == [0, 2]
    assert [entry["physical_qubit"] for entry in metadata["measurement_map"]] == [10, 11]


def test_runtime_layout_keeps_native_circuit_compact(fake_api):
    circuit = tq.Circuit(
        2, ops=[("cx", 0, 1), ("measure_z", 0), ("measure_z", 1)]
    )
    native, metadata = lowering.to_spinq_circuit(
        circuit,
        physical_qubits=[15, 24],
        apply_physical_mapping=False,
        api=fake_api,
    )
    assert native.allocated_qubits == 2
    assert native.instructions[0][1] == (0, 1)
    assert metadata["physical_qubits"] == [15, 24]
    assert metadata["native_qubits"] == [0, 1]
    assert metadata["cloud_measure_qubits"] == [0, 1]
    assert metadata["runtime_keep_layout"] == [15, 24]
    assert metadata["mapping_stage"] == "runtime_keep_layout"


def test_instruction_measurements_and_barrier(fake_api):
    circuit = tq.Circuit(2).h(0).add_barrier(0, 1).add_measure(1, 0)
    native, metadata = lowering.to_spinq_circuit(circuit, api=fake_api)
    assert native.instructions[0][0] == "H"
    assert native.instructions[1][0] == "BARRIER"
    assert native.measurements == []
    assert metadata["measured_qubits"] == [1, 0]
    assert metadata["cloud_measure_qubits"] == [1, 0]
    assert metadata["measurement_mode"] == "cloud_automatic"


def test_native_exe_uses_native_compiler_only(fake_api):
    circuit = tq.Circuit(2).h(0).cx(0, 1).measure_z(0).measure_z(1)
    executable, metadata = lowering.to_spinq_exe(
        circuit, optimization_level=2, api=fake_api
    )
    assert isinstance(executable, _FakeExecutable)
    assert executable.level == 2
    assert isinstance(executable.circuit, _FakeCircuit)
    assert metadata["target"] == "spinqit.native.executable"


@pytest.mark.parametrize("level", [-1, 4])
def test_invalid_native_optimization_level(fake_api, level):
    circuit = tq.Circuit(1).x(0).measure_z(0)
    with pytest.raises(ValueError, match="optimization_level"):
        lowering.to_spinq_exe(circuit, optimization_level=level, api=fake_api)


def test_missing_measurement_fails(fake_api):
    with pytest.raises(ValueError, match="显式调用"):
        lowering.to_spinq_circuit(tq.Circuit(1).x(0), api=fake_api)


def test_non_terminal_measurement_fails(fake_api):
    circuit = tq.Circuit(1).measure_z(0).x(0)
    with pytest.raises(ValueError, match="线路末尾"):
        lowering.to_spinq_circuit(circuit, api=fake_api)


def test_duplicate_measurement_fails(fake_api):
    circuit = tq.Circuit(1).measure_z(0).measure_z(0)
    with pytest.raises(ValueError, match="重复测量"):
        lowering.to_spinq_circuit(circuit, api=fake_api)


def test_unsupported_gate_and_instruction_fail_loudly(fake_api):
    circuit = tq.Circuit(2).rxx(0, 1, 0.4).measure_z(0).measure_z(1)
    with pytest.raises(NotImplementedError, match="rxx"):
        lowering.to_spinq_circuit(circuit, api=fake_api)
    with pytest.raises(NotImplementedError, match="instruction"):
        lowering.to_spinq_circuit(
            tq.Circuit(1).add_reset(0).add_measure(0), api=fake_api
        )


def test_unitary_requires_core_decomposition(fake_api):
    import numpy as np

    circuit = tq.Circuit(1).unitary(0, matrix=np.eye(2)).measure_z(0)
    with pytest.raises(NotImplementedError, match="核心分解"):
        lowering.to_spinq_circuit(circuit, api=fake_api)


@pytest.mark.parametrize(
    "layout,error",
    [([0], "长度"), ([0, 0], "重复"), ([-1, 0], "负数")],
)
def test_invalid_physical_mapping(fake_api, layout, error):
    circuit = tq.Circuit(2).cx(0, 1).measure_z(0).measure_z(1)
    with pytest.raises(ValueError, match=error):
        lowering.to_spinq_circuit(circuit, physical_qubits=layout, api=fake_api)


def test_example_defaults_to_offline(monkeypatch, capsys):
    monkeypatch.setattr(lowering, "_spinqit_api", _fake_api)
    example = Path(__file__).parents[1] / "examples" / "run_circuit_on_spinq.py"
    monkeypatch.setattr(sys, "argv", [str(example), "--circuit", "bell"])
    runpy.run_path(str(example), run_name="__main__")
    output = capsys.readouterr().out
    assert "core_compile_then_ir_to_spinqit_native" in output
    assert "离线模式完成" in output
    assert "未登录、未查询设备、未提交任务" in output
