from tyxonq.core.ir import Circuit
from tyxonq.devices.simulators.wavefunction import WavefunctionEngine
from tyxonq.compiler.stages.scheduling.shot_scheduler import schedule
from tyxonq.devices.session import execute_plan


def test_wavefunction_simulator_minimal_end_to_end():
    sim = WavefunctionEngine()
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("measure_z", 1)])
    plan = schedule(circ, total_shots=12)
    out = execute_plan(sim, plan)
    assert out["metadata"]["total_shots"] == 12
    # Bell state |00>+|11| / sqrt(2) => Z on qubit 1 has expectation 0.0
    assert out["expectations"].get("Z1", 0.0) == 0.0


