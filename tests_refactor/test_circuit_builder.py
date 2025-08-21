from tyxonq.core.ir import CircuitBuilder


def test_circuit_builder_basic_usage():
    with CircuitBuilder(num_qubits=2) as cb:
        cb.h(0)
        cb.cx(0, 1)
        cb.measure_z(1)
    circ = cb.circuit()
    assert circ.num_qubits == 2
    # Expect recorded ops: h, cx, measure_z
    assert len(circ.ops) == 3
    names = [op[0] for op in circ.ops]
    assert names == ["h", "cx", "measure_z"]


