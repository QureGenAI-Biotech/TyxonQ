from __future__ import annotations

import numpy as np
from openfermion import jordan_wigner
import tyxonq as tq

from tyxonq.applications.chem.constants import (
    ad_a_hc2,
    adad_aa_hc2,
    ad_a_hc,
    adad_aa_hc,
)
from tyxonq.libs.hamiltonian_encoding.pauli_io import ex_op_to_fop


def _apply_excitation_numeric(statevector, f_idx, mode):
    n_qubits = int(round(np.log2(statevector.shape[0])))
    qubit_idx = [n_qubits - 1 - idx for idx in f_idx]

    circuit = tq.Circuit(n_qubits, inputs=statevector)
    # fermion operator index, not sorted
    if len(qubit_idx) == 2:
        circuit.any(*qubit_idx, unitary=ad_a_hc)
    else:
        assert len(qubit_idx) == 4
        circuit.any(*qubit_idx, unitary=adad_aa_hc)

    if mode != "fermion":
        return circuit.state()

    # apply all Z operators (Jordan–Wigner phase)
    # pauli string index, already sorted
    fop = ex_op_to_fop(f_idx)
    qop = jordan_wigner(fop)
    z_indices = []
    for idx, term in next(iter(qop.terms.keys())):
        if term != "Z":
            assert idx in f_idx
            continue
        z_indices.append(n_qubits - 1 - idx)
    sign = 1 if sorted(qop.terms.items())[0][1].real > 0 else -1
    phase_vector = [sign]
    for i in range(n_qubits):
        if i in z_indices:
            phase_vector = np.kron(phase_vector, [1, -1])
        else:
            phase_vector = np.kron(phase_vector, [1, 1])
    return phase_vector * circuit.state()


def apply_excitation_statevector(statevector, n_qubits, n_elec, f_idx, mode):
    """Apply one excitation on a statevector (numeric path only).

    This is a direct numeric helper used by UCC numeric runtime. It relies on
    Circuit(inputs=state) to express small unitaries (ad_a_hc/adad_aa_hc) and
    then correct the Jordan–Wigner phase if needed.
    """
    return _apply_excitation_numeric(statevector, f_idx, mode).real


