"""LQCloud 一等 compile_engine 的纯离线测试（IR -> SDK QuantumCircuit 对象直译）。

与 test_guodun_compiler.py 对称，但目标方言是 SDK **对象**而非文本：
  - 用 _FakeQuantumCircuit monkeypatch sys.modules['lqcloud']，无需安装官方 SDK；
  - 校验 compile(compile_engine='lqcloud') 产出对象 + metadata（conversion=对象直译）；
  - 校验 physical_qubits 是运行期参数（不烘焙进对象），随 Circuit.run() 流向驱动。
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest

from tyxonq.compiler.api import compile as compile_ir
from tyxonq.core.ir import Circuit


class _FakeGate:
    def __init__(self, name):
        self.name = name


class _FakeQubit:
    def __init__(self, index):
        self.index = index


class _FakeQuantumCircuit:
    """最小可用的 lqcloud.QuantumCircuit 替身：暴露 .data 与原生门方法。"""

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


def _compile(circuit):
    return compile_ir(circuit, compile_engine="lqcloud", output="lqcloud")


def test_bell_add_measure_compiles_to_sdk_object(fake_lqcloud):
    circuit = Circuit(2)
    circuit.h(0).cx(0, 1).add_measure(0, 1)
    result = _compile(circuit)

    qc = result["compiled_source"]
    assert isinstance(qc, _FakeQuantumCircuit)
    assert qc.num_clbits == 2
    # 直译：h -> cx -> barrier（测量前强制）-> measure x2
    assert _names(qc) == ["h", "cx", "barrier", "measure", "measure"]

    meta = result["metadata"]
    assert meta["output"] == "lqcloud"
    assert meta["target"] == "lqcloud.QuantumCircuit"
    # R1 起先委托 gate_compiler 核心再直译，conversion 标注随之变更。
    assert meta["conversion"] == "core_compile_then_ir_to_lqcloud_object"
    assert meta["num_qubits"] == 2
    assert meta["gate_stats"]["h"] == 1
    assert meta["gate_stats"]["cx"] == 1
    assert meta["gate_stats"]["measure"] == 2


def test_compile_caches_compiled_source_on_circuit(fake_lqcloud):
    circuit = Circuit(2).h(0).cx(0, 1).add_measure(0, 1)
    result = _compile(circuit)
    # api.compile 会把产物缓存回 circuit._compiled_source（与其它 engine 一致）
    assert circuit._compiled_source is result["compiled_source"]


def test_unsupported_output_is_rejected(fake_lqcloud):
    circuit = Circuit(1).x(0).measure_z(0)
    with pytest.raises(ValueError, match="output"):
        compile_ir(circuit, compile_engine="lqcloud", output="qasm3")


def test_physical_qubits_is_not_baked_into_object(fake_lqcloud):
    # physical_qubits 是运行期 initial_layout，误传入编译选项时应被忽略，
    # 对象里的比特索引仍是逻辑索引（对象方言无“物理比特”概念）。
    circuit = Circuit(2).h(0).cx(0, 1).add_measure(0, 1)
    result = compile_ir(
        circuit,
        compile_engine="lqcloud",
        output="lqcloud",
        options={"physical_qubits": [21, 28]},
    )
    qc = result["compiled_source"]
    measured = [item[1][0].index for item in qc.data if item[0].name == "measure"]
    assert measured == [0, 1]


def test_missing_measurement_is_auto_added_by_core(fake_lqcloud):
    # 委托核心后，无显式测量的线路由核心 AutoMeasure 自动补 Z 测量（带警告），
    # 不再直接抛错；而 to_lqcloud 直译路径仍要求显式测量（见 test_lqcloud_lowering）。
    circuit = Circuit(1).h(0)
    with pytest.warns(UserWarning, match="自动补"):
        result = _compile(circuit)
    qc = result["compiled_source"]
    measured = [item[1][0].index for item in qc.data if item[0].name == "measure"]
    assert measured == [0]


def test_unsupported_gate_is_rejected(fake_lqcloud):
    # SDK 与我方都未提供的门（rxx）必须本地 loudly raise，绝不提交。
    circuit = Circuit(2).rxx(0, 1, 0.4).measure_z(0).measure_z(1)
    with pytest.raises(NotImplementedError, match="rxx"):
        _compile(circuit)


def test_lqcloud_provider_run_passes_object_and_keeps_layout(monkeypatch, fake_lqcloud):
    # Circuit.run(provider="lqcloud") 应经 compile() 产出对象作 source；
    # physical_qubits 保留在 dev_opts 中流向驱动（运行期映射为 initial_layout），不烘焙进对象。
    captured = {}

    class _Task:
        def get_result(self, wait=False):
            return {"result": {}, "result_meta": {}, "uni_status": "completed", "error": ""}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return [_Task()]

    monkeypatch.setattr("tyxonq.devices.base.run", fake_run)
    circuit = Circuit(2).h(0).cx(0, 1).add_measure(0, 1)
    circuit.device(provider="lqcloud", device="lq_test").run(
        shots=128,
        physical_qubits=[3, 4],
    )

    assert captured["provider"] == "lqcloud"
    assert captured["device"] == "lq_test"
    assert captured["shots"] == 128
    # source 是 SDK 对象（非字符串文本）
    assert isinstance(captured["source"], _FakeQuantumCircuit)
    assert _names(captured["source"]) == ["h", "cx", "barrier", "measure", "measure"]
    # physical_qubits 作为运行期参数保留，随 dev_opts 流向驱动（驱动再映射为 initial_layout）
    assert captured["physical_qubits"] == [3, 4]


@pytest.mark.skipif(
    importlib.util.find_spec("lqcloud") is None, reason="lqcloud extra not installed"
)
def test_official_sdk_object_via_compiler(monkeypatch):
    # 装了真 SDK 时，确认编译产物确为官方 lqcloud.QuantumCircuit 对象。
    monkeypatch.delitem(sys.modules, "lqcloud", raising=False)
    circuit = Circuit(2).h(0).cx(0, 1).add_measure(0, 1)
    result = compile_ir(circuit, compile_engine="lqcloud", output="lqcloud")
    qc = result["compiled_source"]
    assert qc.__class__.__module__.startswith("lqcloud.")
    assert result["metadata"]["conversion"] == "core_compile_then_ir_to_lqcloud_object"
