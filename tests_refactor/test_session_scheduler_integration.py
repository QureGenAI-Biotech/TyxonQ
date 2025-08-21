from tyxonq.compiler.pipeline import build_pipeline
from tyxonq.compiler.stages.scheduling.shot_scheduler import schedule
from tyxonq.core.ir import Circuit
from tyxonq.core.measurements import Expectation
from tyxonq.devices.session import execute_plan


class DummyDevice:
    name = "dummy"
    capabilities = {"supports_shots": True}

    def run(self, circuit: Circuit, shots: int | None = None, **kwargs):
        return {"samples": None, "expectations": {"Z0": float(shots or 0)}, "metadata": {"shots": shots}}

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


