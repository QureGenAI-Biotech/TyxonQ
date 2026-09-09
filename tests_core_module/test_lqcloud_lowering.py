"""LQCloud 直译降门（lowering）测试；默认不要求安装官方 SDK。

canonical 实现已提升为一等 compile_engine：
``tyxonq.compiler.platforms.lqcloud.lowering.to_lqcloud``（与 guodun 的
lowering 对称）。原设备层 ``devices/hardware/lqcloud/translator.py`` 已删除，
本测试直接针对新位置，行为契约不变。
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tyxonq as tq
from tyxonq.compiler.platforms.lqcloud import to_lqcloud


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
    def crx(self, theta, q0, q1): self._append("crx", q0, q1)
    def cry(self, theta, q0, q1): self._append("cry", q0, q1)
    def crz(self, theta, q0, q1): self._append("crz", q0, q1)
    def su2(self, matrix, q): self._append("su2", q)
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

    native = to_lqcloud(circuit)

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
    native = to_lqcloud(circuit)
    assert _names(native)[-2:] == ["measure", "measure"]
    assert _names(native).count("barrier") == 1


def test_missing_measurement_fails_before_submission(fake_lqcloud):
    with pytest.raises(ValueError, match="显式调用"):
        to_lqcloud(tq.Circuit(1).x(0))


def test_non_terminal_measurement_is_rejected(fake_lqcloud):
    circuit = tq.Circuit(1).measure_z(0).x(0)
    with pytest.raises(ValueError, match="线路末尾"):
        to_lqcloud(circuit)


def test_duplicate_measurement_is_rejected(fake_lqcloud):
    circuit = tq.Circuit(1).measure_z(0).measure_z(0)
    with pytest.raises(ValueError, match="重复测量"):
        to_lqcloud(circuit)


def test_unsupported_gate_is_rejected(fake_lqcloud):
    circuit = tq.Circuit(2).rxx(0, 1, 0.4).measure_z(0).measure_z(1)
    with pytest.raises(NotImplementedError, match="rxx"):
        to_lqcloud(circuit)


def test_controlled_rotations_map_natively(fake_lqcloud):
    """crx/cry/crz 是 SDK 原生门，1:1 直译（UCCSD 线路依赖 cry）。"""
    circuit = tq.Circuit(2, ops=[
        ("crx", 0, 1, 0.1), ("cry", 0, 1, 0.2), ("crz", 1, 0, 0.3),
        ("measure_z", 0), ("measure_z", 1),
    ])
    native = to_lqcloud(circuit)
    assert _names(native)[:3] == ["crx", "cry", "crz"]


def test_rzz_decomposes_to_cx_rz_cx(fake_lqcloud):
    """SDK 无 rzz，分解为 CX·RZ·CX（SQD/LUCJ 的 CPhase 依赖 rzz）。"""
    circuit = tq.Circuit(2).rzz(0, 1, 0.5).measure_z(0).measure_z(1)
    native = to_lqcloud(circuit)
    assert _names(native)[:3] == ["cx", "rz", "cx"]


def test_single_qubit_unitary_maps_to_su2(fake_lqcloud):
    """单比特自定义酉→SDK su2（1:1 直译）。"""
    import numpy as np
    mat = np.array([[0, 1], [1, 0]], dtype=complex)  # X
    circuit = tq.Circuit(1).unitary(0, matrix=mat).measure_z(0)
    native = to_lqcloud(circuit)
    assert _names(native)[0] == "su2"


def _two_mode_fock(v):
    """复现 LUCJ/SQD 的 two_mode_fock_matrix(v)：把 2x2 轨道旋转提升为 4x4 Fock 酉。"""
    import numpy as np
    v = np.asarray(v, dtype=complex)
    m = np.eye(4, dtype=complex)
    m[1, 1] = v[1, 1]
    m[1, 2] = v[1, 0]
    m[2, 1] = v[0, 1]
    m[2, 2] = v[0, 0]
    m[3, 3] = v[0, 0] * v[1, 1] - v[0, 1] * v[1, 0]
    return m


def test_two_qubit_unitary_requires_core_decomposition(fake_lqcloud):
    """R1 起 lowering 不再自带 2-qubit 酉分解（消除与核心 MatchgateDecompose 的重复实现）。

    数守恒 matchgate 必须由核心先分解为原生门（经委托路径，见 test_lqcloud_delegation.py）；
    直接把裸 2-qubit unitary 交给 to_lqcloud 应清晰报错，指引“先由核心分解”。
    """
    import numpy as np
    theta, phi = 0.4, 0.7
    v = np.array(
        [[np.cos(theta), -np.sin(theta) * np.exp(1j * phi)],
         [np.sin(theta) * np.exp(-1j * phi), np.cos(theta)]],
        dtype=complex,
    )
    circuit = tq.Circuit(2).unitary(0, 1, matrix=_two_mode_fock(v)).measure_z(0).measure_z(1)
    with pytest.raises(NotImplementedError, match="核心"):
        to_lqcloud(circuit)


def test_non_matchgate_two_qubit_unitary_is_rejected(fake_lqcloud):
    """非数守恒的 2-qubit 酉（|00>↔|11> 耦合，有跨子空间泄露）不属 fSim 家族，须清晰报错。"""
    import numpy as np
    mat = np.eye(4, dtype=complex)
    mat[0, 0] = 0.0
    mat[3, 3] = 0.0
    mat[0, 3] = 1.0
    mat[3, 0] = 1.0
    circuit = tq.Circuit(2).unitary(0, 1, matrix=mat).measure_z(0).measure_z(1)
    with pytest.raises(NotImplementedError, match="matchgate"):
        to_lqcloud(circuit)


def test_barrier_after_measure_is_tolerated(fake_lqcloud):
    """HEA 场景：measure_z 在 ops、barrier 在 instructions，合并后 barrier 落到测量之后。

    barrier 是编译指令（无物理作用），应被容忍并丢弃，不得触发“测量必须在末尾”；
    但测量后出现真正的门（如 x）仍须拒绝（见 test_non_terminal_measurement_is_rejected）。
    """
    circuit = tq.Circuit(2, ops=[("ry", 0, 0.1), ("ry", 1, 0.2), ("measure_z", 0), ("measure_z", 1)])
    circuit = circuit.add_barrier(0, 1)  # barrier 进 instructions
    native = to_lqcloud(circuit)
    names = _names(native)
    assert names[-2:] == ["measure", "measure"]
    assert names.count("ry") == 2


@pytest.mark.skipif(importlib.util.find_spec("lqcloud") is None, reason="lqcloud extra not installed")
def test_official_sdk_object_can_be_built(monkeypatch):
    monkeypatch.delitem(sys.modules, "lqcloud", raising=False)
    from lqcloud.backend.serialization import serialize_circuit

    circuit = tq.Circuit(4).h(0).ry(1, 0.4).cx(0, 1).cz(2, 3)
    circuit = circuit.add_measure(0, 1, 2, 3)
    native = to_lqcloud(circuit)
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
