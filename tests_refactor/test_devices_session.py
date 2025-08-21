from tyxonq.core.ir import Circuit
from tyxonq.devices import Device
from tyxonq.devices.session import execute_plan


class DummyDevice:
    name = "dummy"
    capabilities = {"supports_shots": True}

    def run(self, circuit: Circuit, shots: int | None = None, **kwargs):
        # Return expectations proportional to shots for testing aggregation
        return {"samples": None, "expectations": {"Z0": float(shots or 0)}, "metadata": {"shots": shots}}

    def expval(self, circuit: Circuit, obs, **kwargs) -> float:
        return 0.0


def test_execute_plan_aggregates_shots_and_metadata():
    dev: Device = DummyDevice()  # type: ignore[assignment]
    circ = Circuit(num_qubits=1, ops=[])
    plan = {"circuit": circ, "segments": [{"shots": 10}, {"shots": 20}]}
    agg = execute_plan(dev, plan)
    assert agg["expectations"]["Z0"] == 30.0
    assert agg["metadata"]["per_segment"][0]["shots"] == 10
    assert agg["metadata"]["total_shots"] == 30


