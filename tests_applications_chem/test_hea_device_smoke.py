import math
import pytest


def test_headevice_energy_smoke():
    from tyxonq.applications.chem.algorithms import HEA

    # Simple 2-qubit Z Hamiltonian: H = Z0 + Z1
    H = [(1.0, [("Z", 0)]), (1.0, [("Z", 1)])]
    hea = HEA(n=2, layers=1, hamiltonian=H, runtime="device")

    e = hea.energy(shots=1024, provider="simulator", device="statevector")
    assert isinstance(e, float) and math.isfinite(e)

    e2, g = hea.energy_and_grad(hea.init_guess, shots=512, provider="simulator", device="statevector")
    assert isinstance(e2, float)
    assert g.shape == (hea.n_params,)

