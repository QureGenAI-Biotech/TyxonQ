from tyxonq.core.ir import Circuit
from tyxonq.devices.simulators.matrix_product_state import MatrixProductStateEngine
from tyxonq.compiler.stages.scheduling.shot_scheduler import schedule
from tyxonq.devices.session import device_job_plan


def test_compressed_state_engine_end_to_end_minimal():
    eng = MatrixProductStateEngine()
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("measure_z", 1)])
    plan = schedule(circ, total_shots=7)
    out = device_job_plan(eng, plan)
    assert out["metadata"]["total_shots"] == 7
    # Bell state's Z on qubit 1 expectation is 0
    assert abs(out["expectations"].get("Z1", 0.0)) <= 1e-12



