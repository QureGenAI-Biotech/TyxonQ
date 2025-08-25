import numpy as np

import pytest
from tyxonq.core.ir import Circuit
from tyxonq.devices.simulators.statevector import StatevectorEngine
from tyxonq.devices.simulators.density_matrix import DensityMatrixEngine


def test_statevector_mid_measure_and_reset():
    eng = StatevectorEngine()
    # Prepare |+0>, then project qubit 0 to |0>, then reset qubit 1 to |0|
    c = Circuit(num_qubits=2, ops=[("h", 0), ("project_z", 0, 0), ("reset", 1), ("measure_z", 0), ("measure_z", 1)])
    out = eng.run(c)
    # After projection and reset, state is |00>
    assert np.isclose(out["expectations"]["Z0"], 1.0)
    assert np.isclose(out["expectations"]["Z1"], 1.0)


def test_density_matrix_mid_measure_and_reset():
    eng = DensityMatrixEngine()
    # Validate Z before and after projection
    # Before projection, Z0 should be 0.0 for |+>
    c = Circuit(num_qubits=2, ops=[("h", 0), ("measure_z", 0)])
    out0 = eng.run(c)
    assert np.isclose(out0["expectations"]["Z0"], 0.0)

    c2 = Circuit(num_qubits=2, ops=[("h", 0), ("project_z", 0, 1), ("reset", 1), ("measure_z", 0), ("measure_z", 1)])
    out = eng.run(c2)
    # After projection to |1> on qubit 0 and reset qubit 1 to |0>
    assert np.isclose(out["expectations"].get("Z1", 0.0), 1.0)
    assert np.isclose(out["expectations"].get("Z0", 0.0), -1.0)


