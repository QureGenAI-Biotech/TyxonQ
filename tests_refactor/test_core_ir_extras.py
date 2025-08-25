from tyxonq.core.ir import Circuit


def test_gate_count_and_summary_and_extended_and_json():
    c = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1)])
    assert c.gate_count() == 2
    assert c.gate_count(["h"]) == 1
    s = c.gate_summary()
    assert s["h"] == 1 and s["cx"] == 1

    c2 = c.extended([("rz", 1, 0.1)])
    assert c2.gate_count() == 3
    assert c.gate_count() == 2  # no mutation

    js = c2.to_json_str()
    c3 = Circuit.from_json_str(js)
    assert c3.num_qubits == 2
    assert [op[0] for op in c3.ops] == ["h", "cx", "rz"]


