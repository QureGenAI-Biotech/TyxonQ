from typing import Any, Dict

from tyxonq.compiler import CompileRequest, Compiler
from tyxonq.core.ir import Circuit
from tyxonq.devices import Device, DeviceCapabilities


class DummyCompiler:
    def compile(self, request: CompileRequest):  # type: ignore[override]
        assert "circuit" in request and "target" in request and "options" in request
        return {"circuit": request["circuit"], "metadata": {"ok": True}}


class DummyDevice:
    name = "dummy"
    capabilities: DeviceCapabilities = {"supports_shots": True}

    def run(self, circuit: Circuit, shots: int | None = None, **kwargs: Any):
        return {"samples": None, "expectations": {}, "metadata": {"shots": shots}}

    def expval(self, circuit: Circuit, obs: Any, **kwargs: Any) -> float:
        return 0.0


def test_compiler_contract_minimal():
    compiler: Compiler = DummyCompiler()  # type: ignore[assignment]
    circ = Circuit(num_qubits=1, ops=[])
    req: Dict[str, Any] = {"circuit": circ, "target": {}, "options": {}}
    res = compiler.compile(req)  # type: ignore[arg-type]
    assert "circuit" in res and "metadata" in res


def test_device_contract_minimal():
    dev: Device = DummyDevice()  # type: ignore[assignment]
    circ = Circuit(num_qubits=1, ops=[])
    rr = dev.run(circ, shots=100)
    assert "metadata" in rr and rr["metadata"]["shots"] == 100
    assert isinstance(dev.expval(circ, obs=None), float)


