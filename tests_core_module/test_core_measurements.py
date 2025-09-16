from tyxonq.core.measurements import Expectation, Probability, Sample


def test_measurement_structs_minimal():
    e = Expectation(obs="Z", wires=(0,))
    p = Probability(wires=(0, 1))
    s = Sample(wires=(0,), shots=1000)
    assert e.obs == "Z" and e.wires == (0,)
    assert p.wires == (0, 1)
    assert s.shots == 1000


