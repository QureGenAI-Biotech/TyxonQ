from tyxonq.core.ir import Circuit
from tyxonq.devices.simulators.statevector import StatevectorEngine
from tyxonq.devices.simulators.density_matrix import DensityMatrixEngine
from tyxonq.devices.simulators.matrix_product_state import MatrixProductStateEngine
from tyxonq.compiler.stages.scheduling.shot_scheduler import schedule
from tyxonq.devices.session import execute_plan


def _smoke(engine_cls):
    dev = engine_cls()
    circ = Circuit(num_qubits=1, ops=[("measure_z", 0)])
    plan = schedule(circ, total_shots=5)
    out = execute_plan(dev, plan)
    assert out["metadata"]["total_shots"] == 5


def test_statevector_engine_smoke():
    _smoke(StatevectorEngine)


def test_density_matrix_engine_smoke():
    _smoke(DensityMatrixEngine)


def test_matrix_product_state_engine_smoke():
    _smoke(MatrixProductStateEngine)


