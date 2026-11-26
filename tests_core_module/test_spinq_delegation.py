"""Circuit.run 的 SpinQ 原生编译、拓扑路由与提交接线测试。"""

from __future__ import annotations

import pytest

from tyxonq.core.ir import Circuit
from tyxonq.core.ir import circuit as circuit_mod
from tyxonq.devices import base as device_base
from tyxonq.devices.hardware import topology as device_topology

CM4 = [[a, b] for a in range(4) for b in range(4)
       if a != b and (a, b) not in {(0, 3), (3, 0)}]
OPS = [("h", 0), ("cx", 0, 3), ("measure_z", 0), ("measure_z", 3)]


def _circuit(**kwargs):
    return Circuit(num_qubits=4, ops=list(OPS), **kwargs)


@pytest.fixture
def harness(monkeypatch):
    calls = {"discovered": [], "compile": [], "run": []}

    def fake_get_device_rule(provider, device, token=None, **opts):
        calls["discovered"].append({"provider": provider, "device": device, "opts": opts})
        return {"provider": provider, "device": device, "coupling_map": [list(p) for p in CM4], "num_qubits": 4}

    def fake_compile(circuit, **kwargs):
        calls["compile"].append(kwargs)
        return {"compiled_source": "SPINQ_EXE", "metadata": {"target": "spinqit.native.executable"}}

    def fake_run(**kwargs):
        calls["run"].append(kwargs)
        return []

    monkeypatch.setattr(device_topology, "get_device_rule", fake_get_device_rule)
    monkeypatch.setattr(circuit_mod, "compile_api", fake_compile)
    monkeypatch.setattr(device_base, "run", fake_run)
    return calls


def test_topology_discovery_enables_core_routing(harness):
    _circuit().run(
        provider="spinq", device="gemini_vp", shots=8,
        username="user", keyfile="key", host="host",
    )
    discovery = harness["discovered"][0]
    assert discovery["provider"] == "spinq"
    assert discovery["opts"] == {"username": "user", "keyfile": "key", "host": "host"}
    kwargs = harness["compile"][0]
    assert kwargs["platform"] == "spinq" and kwargs["output"] == "spinq"
    assert kwargs["route"] is True
    assert kwargs["device_rule"]["coupling_map"] == CM4
    assert harness["run"][0]["source"] == "SPINQ_EXE"
    assert harness["run"][0]["compile_metadata"]["target"] == "spinqit.native.executable"


def test_no_topology_keeps_native_direct_translation(harness, monkeypatch):
    monkeypatch.setattr(device_topology, "get_device_rule", lambda *a, **k: {"coupling_map": []})
    _circuit().run(provider="spinq", device="gemini_vp", shots=8)
    assert harness["compile"][0]["route"] is False
    assert harness["compile"][0]["device_rule"] == {"coupling_map": []}


def test_user_device_rule_skips_discovery(harness):
    supplied = {"coupling_map": [[0, 1], [1, 0]], "num_qubits": 4}
    _circuit(compile_options={"device_rule": supplied}).run(
        provider="spinq", device="gemini_vp", shots=8
    )
    assert harness["discovered"] == []
    assert harness["compile"][0]["route"] is True


def test_discovery_failure_warns_and_still_submits(harness, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(device_topology, "get_device_rule", boom)
    with pytest.warns(RuntimeWarning, match="拓扑发现失败"):
        _circuit().run(provider="spinq", device="gemini_vp", shots=8)
    assert harness["compile"][0]["route"] is False
    assert harness["run"][0]["source"] == "SPINQ_EXE"


def test_physical_qubits_are_compile_time_only(harness):
    _circuit().run(
        provider="spinq", device="gemini_vp", shots=8, physical_qubits=[4, 5, 6, 7]
    )
    assert harness["compile"][0]["options"]["physical_qubits"] == [4, 5, 6, 7]
    assert harness["compile"][0]["device_rule"]["initial_layout"] == [4, 5, 6, 7]
    assert "physical_qubits" not in harness["run"][0]


def test_route_and_device_rule_are_not_duplicated_in_options(harness):
    _circuit(compile_options={"route": True, "device_rule": {"coupling_map": CM4}, "spinq_optimization_level": 2}).run(
        provider="spinq", device="gemini_vp", shots=8
    )
    options = harness["compile"][0]["options"]
    assert "route" not in options and "device_rule" not in options
    assert options["spinq_optimization_level"] == 2


def test_runtime_native_optimization_is_a_compile_option(harness):
    _circuit().run(
        provider="spinq", device="gemini_vp", shots=8, spinq_optimization_level=2
    )
    assert harness["compile"][0]["options"]["spinq_optimization_level"] == 2
    assert "spinq_optimization_level" not in harness["run"][0]


def test_sqc25_uses_runtime_layout_by_default(harness):
    _circuit().run(provider="spinq", device="SQC-25", shots=8)
    assert harness["compile"][0]["options"]["spinq_runtime_layout"] is True


def test_triangulum_keeps_compile_time_mapping(harness):
    _circuit().run(provider="spinq", device="Triangulum-pro-1", shots=8)
    assert harness["compile"][0]["options"]["spinq_runtime_layout"] is False


def test_source_ir_is_not_mutated(harness):
    circuit = _circuit()
    before = list(circuit.ops)
    circuit.run(provider="spinq", device="gemini_vp", shots=8)
    assert circuit.ops == before
