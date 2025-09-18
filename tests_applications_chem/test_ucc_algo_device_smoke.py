import math


def test_ucc_algo_device_smoke():
    from openfermion import QubitOperator
    from tyxonq.applications.chem.algorithms import UCC

    H = QubitOperator("Z0") + QubitOperator("Z1")
    algo = UCC(n_qubits=2, n_elec_s=(1, 1), h_qubit_op=H, runtime="device")
    e = algo.energy(shots=512, provider="simulator", device="statevector")
    assert isinstance(e, float) and math.isfinite(e)
    e2, g = algo.energy_and_grad(shots=256, provider="simulator", device="statevector")
    assert isinstance(e2, float) and g.shape == (0,)

