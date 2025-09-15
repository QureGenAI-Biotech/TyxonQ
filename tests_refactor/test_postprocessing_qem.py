from tyxonq.postprocessing.error_mitigation import apply_zne, apply_dd, apply_rc


def test_apply_zne_minimal_average():
    class Dummy:
        pass

    circ = Dummy()

    def exec_numeric(c):
        assert c is circ
        return 0.5

    val = apply_zne(circ, exec_numeric, factory=None, num_to_average=5)
    assert abs(val - 0.5) < 1e-12


def test_apply_dd_average_counts_and_numeric():
    class Dummy:
        pass

    circ = Dummy()

    def exec_counts(c):
        assert c is circ
        return {"0": 2, "1": 4}

    out_counts = apply_dd(circ, exec_counts, rule=None, num_trials=2, iscount=True)
    # average of identical dicts equals itself
    assert out_counts == {"0": 2.0, "1": 4.0}

    def exec_numeric(c):
        assert c is circ
        return 1.0

    out_num = apply_dd(circ, exec_numeric, rule=None, num_trials=3, iscount=False)
    assert abs(out_num - 1.0) < 1e-12


def test_apply_rc_average_and_circuit_list():
    class Dummy:
        pass

    circ = Dummy()

    def exec_numeric(c):
        assert c is circ
        return 2.0

    val, clist = apply_rc(circ, exec_numeric, num_to_average=4, iscount=False)
    assert abs(val - 2.0) < 1e-12
    assert len(clist) == 4


