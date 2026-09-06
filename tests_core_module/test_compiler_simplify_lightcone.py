# lightcone 简化 pass 测试合并：仅测量部分量子位时裁剪无关算符、assume_measure_all
# 行为、多测量点的纠缠传播，以及 reset/project 指令的保留规则。
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
    p = build_plan(["simplify/lightcone"])
    c2 = p.execute_plan(c, device_rule={})
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
    p = build_plan(["simplify/lightcone"])
    c2 = p.execute_plan(c, device_rule={})
    ops = set(c2.ops)
    assert ("reset", 0) in ops
    assert ("project_z", 1, 1) in ops
    assert ("measure_z", 0) in ops and ("measure_z", 1) in ops
    assert ("h", 2) not in ops
