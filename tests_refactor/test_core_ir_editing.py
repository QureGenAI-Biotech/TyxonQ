from tyxonq.core.ir import Circuit


def test_compose_and_remap_and_inverse_and_measure_mapping():
    a = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1)])
    b = Circuit(num_qubits=2, ops=[("rz", 1, 0.5), ("measure_z", 1)])

    c = a.compose(b)  # same size
    assert [op[0] for op in c.ops] == ["h", "cx", "rz", "measure_z"]

    # remap b: logical 0->1, 1->0 onto a (2 qubits)
    c2 = a.compose(b, indices=[1, 0])
    assert ("rz", 0, 0.5) in c2.ops

    # remap explicit
    c3 = c2.remap_qubits({0: 1, 1: 0})
    assert ("rz", 1, 0.5) in c3.ops

    inv = a.inverse()
    assert [op[0] for op in inv.ops] == ["cx", "h"]  # inverse order

    # positional->logical mapping from measures
    circ_m = Circuit(num_qubits=3, ops=[("h", 0), ("measure_z", 2), ("measure_z", 0)])
    plm = circ_m.positional_logical_mapping()
    assert plm == {0: 2, 1: 0}


