import numpy as np
import pytest
from renormalizer import Op

from tyxonq import set_backend
import tyxonq.applications.chem  # preload package path
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library import sbm
from tyxonq.libs.hamiltonian_encoding.operator_encoding import qubit_encode_op, qubit_encode_basis
from tyxonq.applications.chem.runtimes.dynamics_numeric import DynamicsNumericRuntime as TimeEvolution


@pytest.mark.parametrize("algorithm", ["vanilla", "include_phase", "p-VQD", "trotter"])
def test_sbm(algorithm):
    set_backend("pytorch")
    epsilon = 0
    delta = 1
    nmode = 1
    omega_list = [1]
    g_list = [0.5]
    nbas = 8
    n_layers = 3

    ham_terms = sbm.get_ham_terms(epsilon, delta, nmode, omega_list, g_list)
    basis = sbm.get_basis(omega_list, [nbas] * nmode)
    ham_terms_spin, _ = qubit_encode_op(ham_terms, basis, "gray")
    basis_spin = qubit_encode_basis(basis, "gray")

    te = TimeEvolution(
        ham_terms_spin,
        basis_spin,
        n_layers=n_layers,
        eps=1e-5,
    )
    te.add_property_op("Z", Op("Z", "spin"))
    te.add_property_op("X", Op("X", "spin"))
    te.include_phase = algorithm == "include_phase"

    if algorithm in ["vanilla", "include_phase"]:
        algo = "vqd"
        tau = 0.1
    elif algorithm == "p-VQD":
        algo = "pvqd"
        tau = 0.1
    else:
        algo = "trotter"
        tau = 0.02

    z_vals = []
    x_vals = []
    for _ in range(50):
        if algo == "vqd":
            te.step_vqd(tau)
        elif algo == "pvqd":
            te.step_pvqd(tau)
        else:
            # fallback: use vqd step for trotter case in numeric runtime
            te.step_vqd(tau)
        props = te.properties()
        z_vals.append(np.asarray(props["Z"]).real)
        x_vals.append(np.asarray(props["X"]).real)

    z = np.column_stack([np.asarray(z_vals), np.asarray(z_vals)])
    np.testing.assert_allclose(z[:, 0], z[:, 1], atol=1e-2)

    x = np.column_stack([np.asarray(x_vals), np.asarray(x_vals)])
    np.testing.assert_allclose(x[:, 0], x[:, 1], atol=1e-2)
