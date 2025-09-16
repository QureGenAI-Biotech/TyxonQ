from __future__ import annotations

from tyxonq.postprocessing.classical_shadows import (
    random_pauli_basis,
    estimate_z_from_counts,
    random_pauli_bases,
    bitstrings_to_bits,
    estimate_expectation_pauli_product,
)


def test_random_pauli_basis_len_and_values():
    b = random_pauli_basis(4, seed=42)
    assert len(b) == 4
    for x in b:
        assert x in {"X", "Y", "Z"}


def test_estimate_z_from_counts_simple():
    counts = {"0": 75, "1": 25}
    z = estimate_z_from_counts(counts, 0)
    assert abs(z - 0.5) < 1e-12


def test_shadows_end_to_end_pauli_product_estimation():
    num_qubits = 2
    bases = [["Z", "Z"], ["Z", "Z"], ["Z", "Z"], ["Z", "Z"]]
    # outcomes for |Φ+> measured in Z: half 00, half 11
    outcomes = [[0, 0], [1, 1], [0, 0], [1, 1]]
    pauli = {0: "Z", 1: "Z"}
    est = estimate_expectation_pauli_product(num_qubits, pauli, bases, outcomes)
    # Expectation of Z⊗Z for |Φ+> is +1
    assert abs(est - 1.0) < 1e-12



