from tyxonq.compiler.targets.qiskit.dialect import (
    DEFAULT_BASIS_GATES,
    DEFAULT_OPT_LEVEL,
    comment_dict,
    comment_qasm,
    free_pi,
    normalize_transpile_options,
)


def test_normalize_transpile_options_defaults():
    opts = normalize_transpile_options(None)
    assert opts["basis_gates"] == DEFAULT_BASIS_GATES
    assert opts["optimization_level"] == DEFAULT_OPT_LEVEL


def test_free_pi_and_comments():
    s = "u(2*pi, pi/2, -pi/2) q[0];"
    out = free_pi(s)
    assert "pi" not in out
    commented = comment_qasm(out)
    assert commented.startswith("//circuit begins")
    mapping = comment_dict({0: 2, 1: 3})
    assert "logical_physical_mapping" in mapping


