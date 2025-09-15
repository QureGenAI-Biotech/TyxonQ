from tyxonq.core.ir import Circuit
from tyxonq.compiler.api import compile as compile_ir
from tyxonq.compiler.stages.scheduling.shot_scheduler import schedule
from tyxonq.devices.session import device_job_plan


class _DummyDevice:
    name = "dummy"
    capabilities = {"supports_shots": True}

    def run(self, circuit: Circuit, shots: int | None = None, **kwargs):
        return {"expectations": {"Z0": float(shots or 0)}, "metadata": {}}

    def expval(self, circuit: Circuit, obs, **kwargs) -> float:
        return 0.0


def test_runtime_run_end_to_end():
    circ = Circuit(num_qubits=1, ops=[("h", 0), ("measure_z", 0)])
    dev = _DummyDevice()

    comp = compile_ir(circ, compile_engine="default", output="ir", options={"total_shots": 25})
    plan = comp["metadata"].get("execution_plan") or schedule(comp["circuit"], total_shots=25)
    plan["circuit"] = comp["circuit"]

    agg = device_job_plan(dev, plan)
    assert agg["metadata"]["total_shots"] == 25
    assert agg["expectations"]["Z0"] == 25.0


