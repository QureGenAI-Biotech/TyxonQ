import pennylane as qml
import numpy as np
from itertools import combinations

def generate_ucc_operator_pool(singles, doubles, op_times, num_qubits):
    """Generate the UCC operator pool including single and double excitations."""
    operator_pool = []
    for single in singles:
        for time in op_times:
            operator_pool.append(qml.SingleExcitation(time, wires=single))
            operator_pool.append(qml.SingleExcitationMinus(time, wires=single))
    for double in doubles:
        for time in op_times:
            operator_pool.append(qml.DoubleExcitation(time, wires=double))
            operator_pool.append(qml.DoubleExcitationMinus(time, wires=double))
    return operator_pool

def generate_molecule_data(molecules="H2", use_ucc=True):
    # H2, LiH, etc.
    datasets = qml.data.load("qchem", molname=molecules, attributes=["molecule", "hf_state", "hamiltonian", "fci_energy"])
    op_times = np.sort(np.logspace(-2, 0, num=8)) / 160
    molecule_data = {}
    for dataset in datasets:
        molecule = dataset.molecule
        num_electrons, num_qubits = molecule.n_electrons, 2*molecule.n_orbitals
        singles, doubles = qml.qchem.excitations(num_electrons, num_qubits)

        if use_ucc:
            operator_pool = generate_ucc_operator_pool(singles, doubles, op_times, num_qubits)
        else:
            operator_pool = [qml.Identity(wires=w) for w in range(num_qubits)]

        molecule_data[dataset.molname] = {
            "op_pool": operator_pool,
            "num_qubits": num_qubits,
            "hf_state": dataset.hf_state,
            "hamiltonian": dataset.hamiltonian,
            "expected_ground_state_E": dataset.fci_energy,
        }

    return molecule_data

