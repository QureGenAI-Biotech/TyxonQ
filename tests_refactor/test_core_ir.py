from tyxonq.core.ir import Circuit, Hamiltonian


def test_circuit_minimal_fields():
    c = Circuit(num_qubits=2, ops=[("rx", 0, 0.3), ("cx", 0, 1)])
    assert c.num_qubits == 2
    assert len(c.ops) == 2
    assert isinstance(c.metadata, dict)


def test_hamiltonian_terms_container():
    ham = Hamiltonian(terms=[("Z", 0, 1.0), ("ZZ", (0, 1), 0.5)])
    assert hasattr(ham, "terms")
    assert len(ham.terms) == 2


def test_circuit_validates_qubits_and_with_metadata():
    c = Circuit(num_qubits=2, ops=[("h", 0), ("cx", 0, 1)])
    c2 = c.with_metadata(tag="test")
    assert c2.metadata.get("tag") == "test"
    import pytest

    with pytest.raises(ValueError):
        Circuit(num_qubits=1, ops=[("h", 1)])


