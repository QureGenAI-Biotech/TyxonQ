from tyxonq.core.ir import Circuit, Hamiltonian


def test_circuit_minimal_fields():
    c = Circuit(num_qubits=2, ops=[("rx", 0, 0.3), ("cx", 0, 1)])
    assert c.num_qubits == 2
    assert len(c.ops) == 2


def test_hamiltonian_terms_container():
    ham = Hamiltonian(terms=[("Z", 0, 1.0), ("ZZ", (0, 1), 0.5)])
    assert hasattr(ham, "terms")
    assert len(ham.terms) == 2


