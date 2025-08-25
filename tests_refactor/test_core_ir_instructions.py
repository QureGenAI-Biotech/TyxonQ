from tyxonq.core.ir import Circuit


def test_instruction_add_and_mapping_and_json():
    c = Circuit(num_qubits=3, ops=[("h", 0)])
    c2 = c.add_measure(2).add_measure(0).add_barrier(0, 2).add_reset(1)
    plm = c2.positional_logical_mapping()
    assert plm == {0: 2, 1: 0}

    js = c2.to_json_str()
    c3 = Circuit.from_json_str(js)
    assert c3.instructions[:2] == [("measure", (2,)), ("measure", (0,))]


