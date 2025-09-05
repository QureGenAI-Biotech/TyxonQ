import math


def test_ucc_device_runtime_smoke():
    from openfermion import QubitOperator
    from tyxonq.applications.chem.runtimes.ucc_device_runtime import UCCDeviceRuntime

    # 2-qubit H = Z0 + Z1
    H = QubitOperator("Z0") + QubitOperator("Z1")
    rt = UCCDeviceRuntime(n_qubits=2, n_elec_s=(1, 1), h_qubit_op=H, mode="fermion")
    e = rt.energy(shots=1024, provider="simulator", device="statevector")
    assert isinstance(e, float) and math.isfinite(e)

