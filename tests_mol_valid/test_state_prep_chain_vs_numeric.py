import numpy as np
import pytest

from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
from tyxonq.libs.circuits_library.qubit_state_preparation import (
    get_device_init_circuit,
    get_numeric_init_circuit,
)
from tyxonq.applications.chem.chem_libs.quantum_chem_library.ci_state_mapping import get_ci_strings
from tyxonq.applications.chem.chem_libs.quantum_chem_library.pyscf_civector import get_init_civector


@pytest.mark.parametrize("mode,n_qubits,n_elec_s,givens_swap", [
    ("fermion", 4, (2, 2), False),
    ("qubit", 4, (2, 2), False),
    ("hcb", 4, 2, False),
    ("hcb", 4, 2, True),
])
def test_device_vs_numeric_state_prep(mode, n_qubits, n_elec_s, givens_swap):
    # device path: gate-level circuit → statevector
    circ = get_device_init_circuit(n_qubits, n_elec_s, mode, givens_swap=givens_swap)
    eng = StatevectorEngine()
    psi_device = np.asarray(eng.state(circ), dtype=np.complex128)

    # numeric path: direct statevector build (no Circuit)
    psi_numeric = get_numeric_init_circuit(n_qubits, n_elec_s, mode, givens_swap=givens_swap)

    # pyscf/ci baseline: construct civector (HF-like) and map to statevector via numeric helper
    ci_strings = get_ci_strings(n_qubits, n_elec_s, mode == "hcb")
    civ = get_init_civector(len(ci_strings))
    psi_pyscf = get_numeric_init_circuit(
        n_qubits, n_elec_s, mode, civector=civ, givens_swap=givens_swap
    )

    # up to global phase equality — align device to numeric, then compare with pyscf baseline
    def align(ref: np.ndarray, vec: np.ndarray) -> np.ndarray:
        phase = np.vdot(ref, vec)
        return vec if phase == 0 else vec * (phase / abs(phase))

    psi_device_aligned = align(psi_numeric, psi_device)
    np.testing.assert_allclose(psi_device_aligned, psi_numeric, atol=1e-12)
    psi_pyscf_aligned = align(psi_numeric, psi_pyscf)
    np.testing.assert_allclose(psi_pyscf_aligned, psi_numeric, atol=1e-12)


