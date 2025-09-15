from tyxonq.compiler.compile_engine.native.compile_plan import build_plan
from tyxonq.core.ir import Circuit
from tyxonq.core.measurements import Expectation


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


