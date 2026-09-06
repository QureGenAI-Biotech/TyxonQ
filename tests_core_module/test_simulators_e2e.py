# 模拟器端到端测试合并：Statevector / DensityMatrix / MatrixProductState 三种引擎
# 经 shot 调度与 device_job_plan 的期望值输出、MPS 的 max_bond 截断选项，以及各引擎
# 的最小 smoke 校验。
from __future__ import annotations

from tyxonq.core.ir import Circuit
from tyxonq.devices.simulators.statevector import StatevectorEngine
from tyxonq.devices.simulators.density_matrix import DensityMatrixEngine
from tyxonq.devices.simulators.matrix_product_state import MatrixProductStateEngine
from tyxonq.compiler.stages.scheduling.shot_scheduler import schedule
from tyxonq.devices.session import device_job_plan


def _smoke(engine_cls):
    dev = engine_cls()
    circ = Circuit(num_qubits=1, ops=[("measure_z", 0)])
    plan = schedule(circ, total_shots=5)
    out = device_job_plan(dev, plan)
    assert out["metadata"]["total_shots"] == 5


def test_density_matrix_engine_end_to_end_expectation():
    eng = DensityMatrixEngine()
    circ = Circuit(num_qubits=1, ops=[("h", 0), ("rz", 0, 0.0), ("measure_z", 0)])
    plan = schedule(circ, total_shots=10)
    out = device_job_plan(eng, plan)
    # exact Z expectation after H is 0.0; allow small tol
    assert abs(out["expectations"].get("Z0", 0.0)) <= 1e-12
    assert out["metadata"]["total_shots"] == 10


def test_compressed_state_engine_end_to_end_minimal():
    eng = MatrixProductStateEngine()
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("measure_z", 1)])
    plan = schedule(circ, total_shots=7)
    out = device_job_plan(eng, plan)
    assert out["metadata"]["total_shots"] == 7
    # Bell state's Z on qubit 1 expectation is 0
    assert abs(out["expectations"].get("Z1", 0.0)) <= 1e-12


def test_compressed_state_engine_max_bond_option_smoke():
    eng = MatrixProductStateEngine(max_bond=1)
    circ = Circuit(num_qubits=3, ops=[("h", 0), ("cx", 0, 1), ("cx", 1, 2), ("measure_z", 2)])
    out = eng.run(circ, shots=0)
    assert "expectations" in out and "metadata" in out


def test_statevector_simulator_minimal_end_to_end():
    sim = StatevectorEngine()
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("measure_z", 1)])
    plan = schedule(circ, total_shots=12)
    out = device_job_plan(sim, plan)
    assert out["metadata"]["total_shots"] == 12
    # Bell state |00>+|11| / sqrt(2) => Z on qubit 1 has expectation 0.0
    assert out["expectations"].get("Z1", 0.0) == 0.0


def test_statevector_engine_smoke():
    _smoke(StatevectorEngine)


def test_density_matrix_engine_smoke():
    _smoke(DensityMatrixEngine)


def test_matrix_product_state_engine_smoke():
    _smoke(MatrixProductStateEngine)
