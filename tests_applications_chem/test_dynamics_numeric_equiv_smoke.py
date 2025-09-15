from __future__ import annotations

import numpy as np
import tyxonq as tq

from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library import sbm
from tyxonq.applications.chem.runtimes.dynamics_numeric import DynamicsNumericRuntime
from renormalizer import Op

def test_dynamics_numeric_sbm_small_smoke():
    # tiny spin-boson model
    nmode = 1
    omega_list = [0.5]
    nlevels = [2]
    g_list = [0.1]
    epsilon = 0.2
    delta = 0.3

    # avoid Op.product(np.product) path by building simple non-multiplied terms
    ham_terms = [
        Op("sigma_z", "spin", epsilon),
        Op(r"b^\dagger b", "v0", omega_list[0]),
    ]
    basis = sbm.get_basis(omega_list, nlevels)

    runtime = DynamicsNumericRuntime(
        ham_terms,
        basis,
        boson_encoding="gray",
        init_condition={"spin": 0},
        n_layers=1,
        eps=1e-6,
    )

    # shapes
    assert runtime.h.shape[0] == len(runtime.state_list[0])
    assert runtime.params.shape == (runtime.n_params,)

    # single Euler step should produce finite params/state
    theta0 = runtime.params.copy()
    new_params = runtime.step_vqd(0.05)
    assert np.all(np.isfinite(new_params))
    assert np.all(np.isfinite(runtime.state_list[-1]))

    # energy expectation should be real
    state = runtime.state_list[-1]
    e = state.conj().T @ (runtime.h @ state)
    assert abs(e.imag) < 1e-8
