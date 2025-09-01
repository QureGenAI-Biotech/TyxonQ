from tyxonq.core.ir import Circuit
from tyxonq.compiler.api import compile as compile_ir
from tyxonq.compiler.stages.scheduling.shot_scheduler import schedule
from tyxonq.devices.session import execute_plan


def test_examples_minimal_end_to_end_like_simple_demo():
    # Minimal 2-qubit Bell state + measure z on qubit 1
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("measure_z", 1)])
    res = compile_ir(circ, compile_engine="default", output="ir", target={"max_shots_per_job": 16}, options={"total_shots": 33})
    plan = res["metadata"].get("execution_plan") or schedule(res["circuit"], total_shots=33, caps={"max_shots_per_job": 16})
    # Dummy device behavior: Z1 expectation equals shots
    class _D:
        name = "dummy"; capabilities = {"supports_shots": True}
        def run(self, circuit, shots=None, **kw):
            return {"expectations": {"Z1": float(shots or 0)}, "metadata": {}}
        def expval(self, circuit, obs, **kw):
            return 0.0
    agg = execute_plan(_D(), plan)
    assert agg["metadata"]["total_shots"] == 33
    assert agg["expectations"]["Z1"] == 33.0


