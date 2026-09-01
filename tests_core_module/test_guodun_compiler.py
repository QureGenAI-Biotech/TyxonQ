"""国盾门线路编译器的纯离线测试。"""

from __future__ import annotations

import importlib.util
import math

import pytest

from tyxonq.compiler.api import compile as compile_ir
from tyxonq.core.ir import Circuit


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cqlib") is None,
    reason="需要可选依赖 cqlib==1.3.11",
)


def _compile(circuit: Circuit, physical_qubits=None):
    options = {}
    if physical_qubits is not None:
        options["physical_qubits"] = physical_qubits
    return compile_ir(
        circuit,
        compile_engine="guodun",
        output="qcis",
        options=options,
    )


def test_bell_add_measure_compiles_to_parseable_qcis():
    from cqlib import Circuit as CqlibCircuit

    circuit = Circuit(2)
    circuit.h(0).cx(0, 1).add_measure(0, 1)
    result = _compile(circuit, [60, 55])

    qcis = result["compiled_source"]
    CqlibCircuit.load(qcis)
    assert qcis.splitlines()[:4] == [
        "H Q60",
        "H Q55",
        "CZ Q60 Q55",
        "H Q55",
    ]
    assert "CZ Q60 Q55" in qcis
    assert qcis.splitlines()[-2:] == ["M Q60", "M Q55"]
    assert "X2P" not in qcis
    assert result["metadata"]["logical_physical_mapping"] == {0: 60, 1: 55}


def test_parameter_rotation_gates_compile():
    circuit = Circuit(2)
    circuit.rx(0, math.pi / 3).ry(1, -math.pi / 5).rz(0, 0.25)
    circuit.add_measure(0, 1)

    result = _compile(circuit)
    assert result["metadata"]["gate_stats"]["M"] == 2


def test_iswap_and_two_qubit_rotations_lower_to_allowed_qcis():
    circuit = Circuit(2)
    circuit.iswap(0, 1).rxx(0, 1, 0.2).ryy(0, 1, -0.3).rzz(0, 1, 0.4)
    circuit.measure_z(0).measure_z(1)

    result = _compile(circuit)
    allowed = {"X", "H", "RZ", "CZ", "M"}
    assert set(result["metadata"]["gate_stats"]) <= allowed
    assert result["metadata"]["gate_stats"]["CZ"] > 0


def test_missing_explicit_measurement_fails():
    with pytest.raises(ValueError, match="显式调用"):
        _compile(Circuit(1).h(0))


@pytest.mark.parametrize(
    ("mapping", "error_type", "message"),
    [
        ([0], ValueError, "长度"),
        ([0, -1], ValueError, "负数"),
        ([1, 1], ValueError, "重复"),
        ([0, "1"], TypeError, "整数"),
    ],
)
def test_invalid_physical_mapping_fails(mapping, error_type, message):
    circuit = Circuit(2).measure_z(0).measure_z(1)
    with pytest.raises(error_type, match=message):
        _compile(circuit, mapping)


def test_physical_mapping_does_not_confuse_q1_and_q10():
    mapping = list(range(11))
    mapping[1] = 60
    mapping[10] = 55
    circuit = Circuit(11).x(1).x(10).measure_z(1).measure_z(10)
    qcis = _compile(circuit, mapping)["compiled_source"]
    assert qcis == "X Q60\nX Q55\nM Q60\nM Q55"


def test_guodun_provider_auto_compiles_before_device_run(monkeypatch):
    captured = {}

    class _Task:
        def get_result(self, wait=False):
            return {
                "result": {},
                "result_meta": {},
                "uni_status": "completed",
                "error": "",
            }

    def fake_run(**kwargs):
        captured.update(kwargs)
        return [_Task()]

    monkeypatch.setattr("tyxonq.devices.base.run", fake_run)
    circuit = Circuit(2)
    circuit.h(0).cx(0, 1).add_measure(0, 1)
    circuit.device(provider="guodun", device="gd_test").run(
        shots=10,
        physical_qubits=[60, 55],
    )

    assert captured["provider"] == "guodun"
    assert captured["device"] == "gd_test"
    assert captured["shots"] == 10
    assert "CZ Q60 Q55" in captured["source"]
    assert "physical_qubits" not in captured
