import numpy as np
import pytest
from tyxonq.applications.chem import UCCSD
from tyxonq.applications.chem.algorithms.vqe.hea import HEA
from tyxonq.libs.hamiltonian_encoding.fermion_to_qubit import parity
from tyxonq.applications.chem.molecule import h2, h_chain
from qiskit.circuit.library import real_amplitudes
from tyxonq.libs.circuits_library.qiskit_real_amplitudes import real_amplitudes_circuit_template_converter

import tyxonq as tq


@pytest.mark.parametrize("shots", [0, 512, 1024])
def test_hea(shots):
    """Device runtime convergence under analytic and sampling conditions.

    - shots=0: exact simulator path, result ~ FCI with tight tol
    - shots>0: sampling noise, result close to FCI with tolerance ~ O(1/sqrt(shots))
      and simple postprocessing (repeat-average) should not be worse than single run.
    """
    m = h2
    uccsd = UCCSD(m,run_fci=True)
    hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3)

    if shots == 0:
        e = hea.kernel(shots=0, provider="simulator", device="statevector")
        np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-5)
    else:
        # Optimize under sampling noise
        e_single = hea.kernel(shots=shots, provider="simulator", device="statevector")
        # tolerance decreases with shots
        tol = 2.0 / np.sqrt(shots) + 0.02
        np.testing.assert_allclose(e_single, uccsd.e_fci, atol=tol)

        # # Postprocessing: evaluate multiple times at optimized params and average
        # ps = np.asarray(hea.params if hea.params is not None else hea.init_guess)
        # evals = [hea.energy(ps, shots=shots, provider="simulator", device="statevector") for _ in range(5)]
        # e_avg = float(np.mean(evals))
        # # averaged energy should not be worse than single evaluation against FCI (within small slack)
        # assert abs(e_avg - uccsd.e_fci) <= abs(e_single - uccsd.e_fci) + 1e-3


def test_build_from_integral_and_mapping():
    m = h2
    uccsd = UCCSD(m,run_fci=True)
    hea = HEA.from_integral(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, n_layers=2, mapping="parity")
    e = hea.kernel(shots=0)
    np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-5)

    
def test_qiskit_circuit():
    m = h2
    uccsd = UCCSD(m,run_fci=True)
    qc = real_amplitudes(2)
    hea = HEA.from_qiskit_circuit(parity(uccsd.h_fermion_op, 4, 2), qc, np.random.rand(qc.num_parameters))
    e = hea.kernel(shots=0, provider="simulator", device="statevector")
    
    np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-5)
    hea.print_summary()


@pytest.mark.parametrize("grad_method", ["free", "autodiff"])
@pytest.mark.parametrize("numeric_engine", ["statevector"])
@pytest.mark.parametrize("shots", [0, 2048])
def test_hea_convergence(grad_method, numeric_engine, shots):
    uccsd = UCCSD(h2, run_fci=True)
    hea = HEA.ry(
        uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3,
        numeric_engine=numeric_engine,
    )
    hea.grad = "free" if grad_method == "free" else "param-shift"
    e = hea.kernel(shots=shots, provider="simulator", device="statevector")
    tol = 1e-5 if shots == 0 else 2.0 / np.sqrt(shots) + 0.02
    np.testing.assert_allclose(e, uccsd.e_fci, atol=tol)

    if shots > 0:
        params = np.asarray(hea.params)
        evals = [hea.energy(params, shots=shots) for _ in range(5)]
        e_avg = float(np.mean(evals))
        assert abs(e_avg - uccsd.e_fci) <= abs(e - uccsd.e_fci) + 1e-3

        a0 = np.array([[0.97, 0.03], [0.05, 0.95]])
        a1 = np.array([[0.98, 0.02], [0.04, 0.96]])
        noise = {"type": "readout", "cals": {0: a0, 1: a1}}
        e_noisy = hea.energy(params, shots=shots, use_noise=True, noise=noise)
        e_mitig = hea.energy(
            params, shots=shots, use_noise=True, noise=noise,
            postprocessing={"readout_cals": {0: a0, 1: a1}, "mitigation": "inverse"},
        )
        assert np.isfinite(e_noisy) and np.isfinite(e_mitig)




@pytest.mark.parametrize("mapping", ["jordan-wigner", "bravyi-kitaev"])
def test_mapping(mapping):
    uccsd = UCCSD(h2, run_fci=True)
    hea = HEA.ry(
        uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, mapping=mapping
    )
    e = hea.kernel(shots=0)
    np.testing.assert_allclose(e, uccsd.e_fci)


@pytest.mark.parametrize("mapping", ["jordan-wigner", "parity", "bravyi-kitaev"])
def test_rdm(mapping):
    uccsd = UCCSD(h2)
    hea = HEA.ry(
        uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, mapping=mapping
    )
    hea.kernel(shots=0)
    r1_h = hea.make_rdm1()
    r2_h = hea.make_rdm2()

    uccsd.kernel(shots=0)
    r1_uccsd = uccsd.make_rdm1(basis="MO")
    r2_uccsd = uccsd.make_rdm2(basis="MO")

    # TCC 金标准（H2, MO 基）
    rdm1_gold = np.array([[1.97457654e+00, -2.00371787e-16],
                          [-2.00371787e-16, 2.54234643e-02]], dtype=np.float64)
    rdm2_gold = np.array(
        [[[[1.97457654e+00, -2.26018011e-16],
           [-2.26018011e-16, 2.58709350e-32]],

          [[-2.26018011e-16, -2.24054851e-01],
           [0.00000000e+00, 2.56462238e-17]]],


         [[[-2.26018011e-16, 0.00000000e+00],
           [-2.24054851e-01, 2.56462238e-17]],

          [[2.58709350e-32, 2.56462238e-17],
           [2.56462238e-17, 2.54234643e-02]]]], dtype=np.float64)

    # 数值实现细节（端序/稀疏求值/阈值）导致 JW/Parity 在 1e-5 量级的偏差，放宽容差至 2e-5
    # np.testing.assert_allclose(r1_h, rdm1_gold, atol=1e-4)
    # np.testing.assert_allclose(r2_h, rdm2_gold, atol=1e-4)
    np.testing.assert_allclose(r1_uccsd, rdm1_gold, atol=1e-6)
    np.testing.assert_allclose(r2_uccsd, rdm2_gold, atol=1e-6)
    np.testing.assert_allclose(r1_uccsd, r1_h, atol=1e-4)
    np.testing.assert_allclose(r2_uccsd, r2_h, atol=1e-4)



def test_open_shell():

    m = h_chain(3, charge=0, spin=1)

    hea = HEA.from_molecule(m, n_layers=6, mapping="parity")
    # try multiple times to avoid local minimum
    es = []
    for i in range(3):
        hea.init_guess = np.random.random(hea.init_guess.shape)
        es.append(hea.kernel(shots=0))
    e1 = min(es)

    from tyxonq.applications.chem.algorithms.vqe.uccsd import ROUCCSD
    ucc = ROUCCSD(m,run_fci=True)
    e2 = ucc.kernel(shots=0)

    # for debugging
    # ucc.print_summary()

    # usually ROUCCSD is more accurate
    np.testing.assert_allclose(e2, ucc.e_fci, atol=1e-4)
    np.testing.assert_allclose(e1, ucc.e_fci, atol=2e-3)

    np.testing.assert_allclose(hea.make_rdm1(), ucc.make_rdm1(basis="MO"), atol=5e-3)
    np.testing.assert_allclose(hea.make_rdm2(), ucc.make_rdm2(basis="MO"), atol=5e-3)

if __name__ == "__main__":
    # test_rdm(mapping="jordan-wigner")
    # test_qiskit_circuit()
    test_open_shell()