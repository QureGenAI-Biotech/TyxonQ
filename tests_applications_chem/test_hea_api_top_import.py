import math


def test_hea_top_import_smoke():
    from tyxonq.applications.chem import HEA

    H = [(1.0, [("Z", 0)]), (1.0, [("Z", 1)])]
    hea = HEA(n_qubits=2, layers=1, hamiltonian=H, runtime="device")
    e = hea.energy(shots=512, provider="simulator", device="statevector")
    assert isinstance(e, float) and math.isfinite(e)

