from tyxonq.core.ir import Circuit
from tyxonq.compiler.pipeline import build_pipeline


def test_lightcone_multiple_measures_propagation():
    # Chain entanglement 0->1->2, only measure qubit 2; ops on qubit 3 are irrelevant
    c = Circuit(num_qubits=4, ops=[
        ("h", 0),
        ("cx", 0, 1),
        ("cx", 1, 2),
        ("h", 3),
        ("rz", 3, 0.123),
        ("measure_z", 2),
    ])
    p = build_pipeline(["simplify/lightcone"])
    c2 = p.run(c, caps={})
    kept = set(c2.ops)
    assert ("h", 0) in kept
    assert ("cx", 0, 1) in kept
    assert ("cx", 1, 2) in kept
    assert ("measure_z", 2) in kept
    assert ("h", 3) not in kept
    assert not any(op[0] == "rz" and op[1] == 3 for op in c2.ops)


def test_lightcone_respects_reset_and_project():
    # project and reset should be kept if measured; unrelated ops on other qubits pruned
    c = Circuit(num_qubits=3, ops=[
        ("h", 0),
        ("h", 2),
        ("project_z", 1, 1),
        ("reset", 0),
        ("measure_z", 0),
        ("measure_z", 1),
    ])
    p = build_pipeline(["simplify/lightcone"])
    c2 = p.run(c, caps={})
    ops = set(c2.ops)
    assert ("reset", 0) in ops
    assert ("project_z", 1, 1) in ops
    assert ("measure_z", 0) in ops and ("measure_z", 1) in ops
    assert ("h", 2) not in ops

