import math
import numpy as np


def test_uccalgo_param_device_non_trotter_smoke():
    from openfermion import QubitOperator
    from tyxonq.applications.chem.algorithms import UCC

    # 2-qubit Hamiltonian: H = Z0 + Z1
    H = QubitOperator("Z0") + QubitOperator("Z1")

    # Simple single-excitation ansatz: ex_ops=[(1,0)], param_ids=[0]
    algo = UCC(
        n_qubits=2,
        n_elec_s=(1, 1),
        h_qubit_op=H,
        runtime="device",
        mode="fermion",
        ex_ops=[(1, 0)],
        param_ids=[0],
        decompose_multicontrol=False,
        trotter=False,
    )
    params = np.array([0.1], dtype=np.float64)

    e = algo.energy(params, shots=512, provider="simulator", device="statevector")
    assert isinstance(e, float) and math.isfinite(e)

    e2, g = algo.energy_and_grad(params, shots=256, provider="simulator", device="statevector")
    assert isinstance(e2, float) and math.isfinite(e2)
    assert g.shape == (1,)


def test_uccalgo_param_device_trotter_smoke():
    from openfermion import QubitOperator
    from tyxonq.applications.chem.algorithms import UCC

    H = QubitOperator("Z0") + QubitOperator("Z1")
    algo = UCC(
        n_qubits=2,
        n_elec_s=(1, 1),
        h_qubit_op=H,
        runtime="device",
        mode="fermion",
        ex_ops=[(1, 0)],
        param_ids=[0],
        decompose_multicontrol=False,
        trotter=True,
    )
    params = np.array([0.2], dtype=np.float64)

    e = algo.energy(params, shots=512, provider="simulator", device="statevector")
    assert isinstance(e, float) and math.isfinite(e)

    e2, g = algo.energy_and_grad(params, shots=256, provider="simulator", device="statevector")
    assert isinstance(e2, float) and math.isfinite(e2)
    assert g.shape == (1,)


