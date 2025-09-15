from tyxonq.core.ir import Circuit
from tyxonq.compiler.compile_engine.native.compile_plan import build_plan


def test_lightcone_pass_prunes_irrelevant_ops():
    # Ops on qubit 2 should be pruned since only qubit 1 is measured
    c = Circuit(num_qubits=3, ops=[
        ("h", 0),
        ("cx", 0, 1),
        ("h", 2),
        ("rz", 2, 0.3),
        ("measure_z", 1),
    ])
    p = build_plan(["simplify/lightcone"])  # lightcone enabled
    c2 = p.execute_plan(c, device_rule={})
    assert ("h", 2) not in c2.ops
    assert not any(op[0] == "rz" and op[1] == 2 for op in c2.ops)
    # ops that influence measured qubit are kept
    assert ("h", 0) in c2.ops and ("cx", 0, 1) in c2.ops and ("measure_z", 1) in c2.ops


def test_lightcone_pass_keeps_all_when_assume_measure_all():
    # Without explicit measure, default keeps circuit; with assume_measure_all, treat all qubits as measured
    c = Circuit(num_qubits=2, ops=[
        ("h", 0), ("cx", 0, 1)
    ])
    p = build_plan(["simplify/lightcone"])  # default: no measures → no change
    c2 = p.execute_plan(c, device_rule={})
    assert len(c2.ops) == len(c.ops)
    # assume all measured → still no pruning (all qubits in lightcone)
    c3 = p.execute_plan(c, device_rule={}, assume_measure_all=True)
    assert len(c3.ops) == len(c.ops)


