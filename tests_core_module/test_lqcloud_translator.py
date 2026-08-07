"""LQCloud 线路转换测试；默认不要求安装官方 SDK。"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tyxonq as tq
from tyxonq.devices.hardware.lqcloud import translator


class _FakeGate:
    def __init__(self, name):
        self.name = name


class _FakeQubit:
    def __init__(self, index):
        self.index = index


class _FakeQuantumCircuit:
    def __init__(self, num_qubits, num_clbits):
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.qubits = [_FakeQubit(i) for i in range(num_qubits)]
        self.data = []

    def _append(self, name, *qubits):
        self.data.append((_FakeGate(name), [self.qubits[q] for q in qubits], []))

    def h(self, q): self._append("h", q)
    def x(self, q): self._append("x", q)
    def y(self, q): self._append("y", q)
    def z(self, q): self._append("z", q)
    def s(self, q): self._append("s", q)
    def sdg(self, q): self._append("sdg", q)
    def t(self, q): self._append("t", q)
    def tdg(self, q): self._append("tdg", q)
    def reset(self, q): self._append("reset", q)
    def rx(self, theta, q): self._append("rx", q)
    def ry(self, theta, q): self._append("ry", q)
    def rz(self, theta, q): self._append("rz", q)
    def cx(self, q0, q1): self._append("cx", q0, q1)
    def cy(self, q0, q1): self._append("cy", q0, q1)
    def cz(self, q0, q1): self._append("cz", q0, q1)
    def swap(self, q0, q1): self._append("swap", q0, q1)
    def iswap(self, q0, q1): self._append("iswap", q0, q1)
    def barrier(self, *qubits): self._append("barrier", *qubits)
    def measure(self, qubit, clbit): self._append("measure", qubit)


@pytest.fixture
def fake_lqcloud(monkeypatch):
    module = SimpleNamespace(QuantumCircuit=_FakeQuantumCircuit)
    monkeypatch.setitem(sys.modules, "lqcloud", module)
    return module


def _names(qc):
    return [item[0].name for item in qc.data]


def test_translates_supported_gates_and_instruction_measurements(fake_lqcloud):
    circuit = tq.Circuit(3)
    circuit.h(0).x(1).y(2).z(0).s(1).sdg(2).t(0).tdg(1)
    circuit.rx(0, 0.1).ry(1, 0.2).rz(2, 0.3)
    circuit.cx(0, 1).cy(1, 2).cz(2, 0).swap(0, 1).iswap(1, 2)
    circuit = circuit.add_reset(0).add_barrier(0, 1, 2).add_measure(2, 0, 1)

    native = translator.to_lqcloud(circuit)

    assert native.num_clbits == 3
    assert _names(native) == [
        "h", "x", "y", "z", "s", "sdg", "t", "tdg",
        "rx", "ry", "rz", "cx", "cy", "cz", "swap", "iswap",
        "reset", "barrier", "barrier", "measure", "measure", "measure",
    ]
    measured = [item[1][0].index for item in native.data if item[0].name == "measure"]
    assert measured == [2, 0, 1]


def test_measure_z_ops_are_supported(fake_lqcloud):
    circuit = tq.Circuit(2).h(0).cx(0, 1).measure_z(0).measure_z(1)
    native = translator.to_lqcloud(circuit)
    assert _names(native)[-2:] == ["measure", "measure"]
    assert _names(native).count("barrier") == 1


def test_missing_measurement_fails_before_submission(fake_lqcloud):
    with pytest.raises(ValueError, match="显式调用"):
        translator.to_lqcloud(tq.Circuit(1).x(0))


def test_non_terminal_measurement_is_rejected(fake_lqcloud):
    circuit = tq.Circuit(1).measure_z(0).x(0)
    with pytest.raises(ValueError, match="线路末尾"):
        translator.to_lqcloud(circuit)


def test_duplicate_measurement_is_rejected(fake_lqcloud):
    circuit = tq.Circuit(1).measure_z(0).measure_z(0)
    with pytest.raises(ValueError, match="重复测量"):
        translator.to_lqcloud(circuit)


def test_unsupported_gate_is_rejected(fake_lqcloud):
    circuit = tq.Circuit(2).rxx(0, 1, 0.4).measure_z(0).measure_z(1)
    with pytest.raises(NotImplementedError, match="rxx"):
        translator.to_lqcloud(circuit)


@pytest.mark.skipif(importlib.util.find_spec("lqcloud") is None, reason="lqcloud extra not installed")
def test_official_sdk_object_can_be_built(monkeypatch):
    monkeypatch.delitem(sys.modules, "lqcloud", raising=False)
    from lqcloud.backend.serialization import serialize_circuit

    circuit = tq.Circuit(4).h(0).ry(1, 0.4).cx(0, 1).cz(2, 3)
    circuit = circuit.add_measure(0, 1, 2, 3)
    native = translator.to_lqcloud(circuit)
    payload = serialize_circuit(native, initial_layout=[0, 1, 2, 3])
    assert native.__class__.__module__.startswith("lqcloud.")
    assert [item[0].name for item in native.data][-4:] == ["measure"] * 4
    assert payload["initial_layout"] == [0, 1, 2, 3]
    assert [item["name"] for item in payload["instructions"]][-4:] == ["measure"] * 4


def test_example_defaults_to_offline_mode(fake_lqcloud, monkeypatch, capsys):
    example = Path(__file__).parents[1] / "examples" / "run_circuit_on_lqcloud.py"
    monkeypatch.setattr(sys, "argv", [str(example), "--physical-qubits", "0,1,2,3"])
    runpy.run_path(str(example), run_name="__main__")
    output = capsys.readouterr().out
    assert "离线模式完成" in output
    assert "未登录、未查询设备、未提交任务" in output


def test_example_extracts_lqcloud_job_metadata():
    example = Path(__file__).parents[1] / "examples" / "run_circuit_on_lqcloud.py"
    namespace = runpy.run_path(str(example))
    result = {"result_meta": {"result_meta": {"job_id": "task-test"}}}
    assert namespace["provider_metadata"](result)["job_id"] == "task-test"
