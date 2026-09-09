# 模拟器端到端测试：Statevector / DensityMatrix / MatrixProductState 三种引擎直接
# engine.run 驱动的期望值物理断言、MPS 的 max_bond 截断选项、采样档 counts，以及各引擎
# 的最小 smoke 校验。（原经 schedule/device_job_plan 的骨架链已随重构删除。）
from __future__ import annotations

from tyxonq.core.ir import Circuit
from tyxonq.devices.simulators.statevector import StatevectorEngine
from tyxonq.devices.simulators.density_matrix import DensityMatrixEngine
from tyxonq.devices.simulators.matrix_product_state import MatrixProductStateEngine


def _smoke(engine_cls):
    dev = engine_cls()
    circ = Circuit(num_qubits=1, ops=[("measure_z", 0)])
    # 解析档（shots=0）直接返回期望值；替代原 schedule/device_job_plan 骨架链
    out = dev.run(circ, shots=0)
    assert "expectations" in out and "metadata" in out


def test_density_matrix_engine_end_to_end_expectation():
    eng = DensityMatrixEngine()
    circ = Circuit(num_qubits=1, ops=[("h", 0), ("rz", 0, 0.0), ("measure_z", 0)])
    out = eng.run(circ, shots=0)
    # exact Z expectation after H is 0.0; allow small tol
    assert abs(out["expectations"].get("Z0", 0.0)) <= 1e-12


def test_compressed_state_engine_end_to_end_minimal():
    eng = MatrixProductStateEngine()
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("measure_z", 1)])
    out = eng.run(circ, shots=0)
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
    out = sim.run(circ, shots=0)
    # Bell state |00>+|11| / sqrt(2) => Z on qubit 1 has expectation 0.0
    assert out["expectations"].get("Z1", 0.0) == 0.0


def test_statevector_sampling_counts_sum_to_shots():
    # 采样档（shots>0）返回 counts（result），总数应等于 shots；保留原骨架链的 shot 覆盖
    sim = StatevectorEngine()
    circ = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1), ("measure_z", 0), ("measure_z", 1)])
    out = sim.run(circ, shots=64)
    assert sum(out["result"].values()) == 64


def test_statevector_engine_smoke():
    _smoke(StatevectorEngine)


def test_density_matrix_engine_smoke():
    _smoke(DensityMatrixEngine)


def test_matrix_product_state_engine_smoke():
    _smoke(MatrixProductStateEngine)
