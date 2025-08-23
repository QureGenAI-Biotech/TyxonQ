from tyxonq.core.ir import Circuit
from tyxonq.devices.simulators.density_matrix import DensityMatrixEngine
from tyxonq.compiler.stages.scheduling.shot_scheduler import schedule
from tyxonq.devices.session import execute_plan


def test_density_matrix_engine_end_to_end_expectation():
    eng = DensityMatrixEngine()
    circ = Circuit(num_qubits=1, ops=[("h", 0), ("rz", 0, 0.0), ("measure_z", 0)])
    plan = schedule(circ, total_shots=10)
    out = execute_plan(eng, plan)
    # exact Z expectation after H is 0.0; allow small tol
    assert abs(out["expectations"].get("Z0", 0.0)) <= 1e-12
    assert out["metadata"]["total_shots"] == 10


