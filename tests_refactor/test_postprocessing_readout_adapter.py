from tyxonq.postprocessing.readout import ReadoutMit
import numpy as np


def test_readout_mit_basic_inverse_and_square():
    # Calibration: single qubit with small asymmetry
    mit = ReadoutMit()
    mit.set_single_qubit_cals({0: np.array([[0.95, 0.1], [0.05, 0.9]])})

    # True counts: |0> occurs 80 times, |1> occurs 20 times
    true_counts = {"0": 80, "1": 20}

    # Simulate measurement noise: p_meas = A @ p_true
    p_true = np.array([0.8, 0.2])
    A = mit.single_qubit_cals[0]
    p_meas = A @ p_true
    meas_counts = {"0": int(round(p_meas[0] * 100)), "1": int(round(p_meas[1] * 100))}

    inv_counts = mit.apply_readout_mitigation(meas_counts, method="inverse")
    assert sum(inv_counts.values()) == 100
    # Expect roughly close to original split
    assert inv_counts.get("0", 0) > inv_counts.get("1", 0)

    sq_counts = mit.apply_readout_mitigation(meas_counts, method="square")
    assert sum(sq_counts.values()) == 100


