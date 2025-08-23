import math

from tyxonq.core.ir import Circuit
from tyxonq.devices.simulators.statevector import StatevectorEngine
from tyxonq.devices.simulators.density_matrix import DensityMatrixEngine
from tyxonq.devices.simulators.matrix_product_state import MatrixProductStateEngine


def _expected_factor(p: float) -> float:
    return max(0.0, 1.0 - 4.0 * p / 3.0)


def test_statevector_noise_depolarizing_rx_pi():
    eng = StatevectorEngine()
    circ = Circuit(num_qubits=1, ops=[("rx", 0, math.pi), ("measure_z", 0)])
    base = eng.run(circ)
    assert abs(base["expectations"]["Z0"] + 1.0) < 1e-8
    p = 0.3
    noisy = eng.run(circ, use_noise=True, noise={"type": "depolarizing", "p": p})
    assert abs(noisy["expectations"]["Z0"] + _expected_factor(p)) < 1e-6


def test_density_matrix_noise_depolarizing_rx_pi():
    eng = DensityMatrixEngine()
    circ = Circuit(num_qubits=1, ops=[("rx", 0, math.pi), ("measure_z", 0)])
    base = eng.run(circ)
    assert abs(base["expectations"]["Z0"] + 1.0) < 1e-8
    p = 0.3
    noisy = eng.run(circ, use_noise=True, noise={"type": "depolarizing", "p": p})
    assert abs(noisy["expectations"]["Z0"] + _expected_factor(p)) < 1e-6


def test_compressed_state_noise_depolarizing_rx_pi():
    eng = MatrixProductStateEngine()
    circ = Circuit(num_qubits=1, ops=[("rx", 0, math.pi), ("measure_z", 0)])
    base = eng.run(circ)
    assert abs(base["expectations"]["Z0"] + 1.0) < 1e-8
    p = 0.3
    noisy = eng.run(circ, use_noise=True, noise={"type": "depolarizing", "p": p})
    assert abs(noisy["expectations"]["Z0"] + _expected_factor(p)) < 1e-6


