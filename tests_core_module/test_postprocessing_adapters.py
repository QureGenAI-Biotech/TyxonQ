# postprocessing 适配器测试合并：counts 层面的噪声注入（bitflip/depolarizing）与
# 读出误差缓解（ReadoutMit 的 inverse/square 两种反演方法）。
from __future__ import annotations

import numpy as np

from tyxonq.postprocessing.noise_analysis import apply_bitflip_counts, apply_depolarizing_counts
from tyxonq.postprocessing.readout import ReadoutMit


def test_apply_bitflip_counts_simple():
    counts = {"0": 80, "1": 20}
    out = apply_bitflip_counts(counts, p=0.1)
    # Expect probability mass move between 0 and 1
    assert abs(sum(out.values()) - sum(counts.values())) < 1e-9
    assert out["0"] > out["1"]


def test_apply_depolarizing_counts_simple():
    counts = {"00": 50, "01": 0, "10": 0, "11": 50}
    out = apply_depolarizing_counts(counts, p=0.2)
    assert abs(sum(out.values()) - sum(counts.values())) < 1e-9
    # With depolarizing, probability mass spreads more uniformly
    assert out["01"] > 0 and out["10"] > 0


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
    # Ensure p_meas is numpy array to avoid issues with Tensor.__round__
    p_meas = np.asarray(p_meas, dtype=float)
    meas_counts = {"0": int(np.round(p_meas[0] * 100)), "1": int(np.round(p_meas[1] * 100))}

    inv_counts = mit.apply_readout_mitigation(meas_counts, method="inverse")
    assert sum(inv_counts.values()) == 100
    # Expect roughly close to original split
    assert inv_counts.get("0", 0) > inv_counts.get("1", 0)

    sq_counts = mit.apply_readout_mitigation(meas_counts, method="square")
    assert sum(sq_counts.values()) == 100
