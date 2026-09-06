# 测量相关测试合并：measurement 重写 pass 对非重叠期望值的分组（basis/wires/估计 shots），
# 以及中途测量 + 投影 + reset 在 statevector / density_matrix 引擎上的后选择行为。
import numpy as np

import pytest

from tyxonq.compiler.compile_engine.native.compile_plan import build_plan
from tyxonq.core.ir import Circuit
from tyxonq.core.measurements import Expectation
from tyxonq.devices.simulators.statevector import StatevectorEngine
from tyxonq.devices.simulators.density_matrix import DensityMatrixEngine


def test_measurement_rewrite_groups_non_overlapping_expectations():
    circ = Circuit(num_qubits=2, ops=[])
    ms = [
        Expectation(obs="Z", wires=(0,)),
        Expectation(obs="X", wires=(1,)),
        Expectation(obs="ZZ", wires=(0, 1)),
    ]
    pl = build_plan(["rewrite/measurement"])
    out = pl.execute_plan(circ, device_rule={}, measurements=ms)
    assert out is circ
    groups = circ.metadata.get("measurement_groups")
    assert isinstance(groups, list)
    # Expect two groups with basis tags; independent single-qubit X/Z can coexist
    # One group for {Z(0), X(1)} and another for {ZZ(0,1)}
    assert len(groups) == 2
    assert sum(len(g["items"]) for g in groups) == 3
    assert {g["basis"] for g in groups} <= {"pauli"}
    # basis map should reflect per-wire pauli bases for the single-qubit group
    g0 = next(g for g in groups if len(g["items"]) == 2)
    assert g0.get("basis_map", {})
    assert set(g0["basis_map"].keys()) == {0, 1}
    # estimated_settings should be present for downstream shot planning
    assert all("estimated_settings" in g for g in groups)
    assert all("estimated_shots_per_group" in g for g in groups)


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
