from tyxonq.compiler.pipeline import build_pipeline
from tyxonq.compiler.stages.scheduling.shot_scheduler import schedule
from tyxonq.core.ir import Circuit
from tyxonq.core.measurements import Expectation
from tyxonq.compiler.api import compile as compile_ir
from tyxonq.devices.session import execute_plan


class DummyDevice:
    name = "dummy"
    capabilities = {"supports_shots": True}

    def run(self, circuit: Circuit, shots: int | None = None, **kwargs):
        return {"samples": None, "expectations": {"Z0": float(shots or 0)}, "metadata": {"shots": shots}}

    def expval(self, circuit: Circuit, obs, **kwargs) -> float:
        return 0.0


class _DummyDevice:
    name = "dummy"
    capabilities = {"supports_shots": True}

    def run(self, circuit: Circuit, shots: int | None = None, **kwargs):
        return {"samples": None, "expectations": {"Z0": float(shots or 0)}, "metadata": {}}

    def expval(self, circuit: Circuit, obs, **kwargs) -> float:
        return 0.0


def test_scheduler_and_session_execute_plan_flow():
    circ = Circuit(num_qubits=2, ops=[])
    ms = [Expectation(obs="Z", wires=(0,)), Expectation(obs="X", wires=(1,))]
    pl = build_pipeline(["rewrite/measurement"])
    circ = pl.run(circ, caps={}, measurements=ms)
    plan = schedule(circ, total_shots=100)
    dev = DummyDevice()
    agg = execute_plan(dev, plan)
    assert agg["metadata"]["total_shots"] in (99, 100, 101)
    # per-segment entries should contain grouping context
    assert all("basis" in s and "wires" in s for s in agg["metadata"]["per_segment"])


def test_compile_emits_execution_plan_and_session_consumes():
    circ = Circuit(num_qubits=1, ops=[("h", 0), ("measure_z", 0)])
    res = compile_ir(circ, compile_engine="tyxonq", output="ir", options={"total_shots": 30})
    plan = res["metadata"].get("execution_plan")
    assert plan is not None and isinstance(plan, dict)
    assert plan["circuit"] is res["circuit"]
    assert sum(seg.get("shots", 0) for seg in plan.get("segments", [])) == 30

    dev = _DummyDevice()
    agg = execute_plan(dev, plan)
    assert agg["expectations"]["Z0"] == 30.0


