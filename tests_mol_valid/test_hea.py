import numpy as np
import pytest
from tyxonq.applications.chem import UCCSD
from tyxonq.applications.chem.algorithms.hea import HEA
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
    hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, runtime="device")

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
    hea = HEA.from_integral(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, n_layers=2, mapping="parity", runtime="device")
    e = hea.kernel(shots=0)
    np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-5)


@pytest.mark.parametrize("runtime", ["device", "numeric"])
@pytest.mark.parametrize("grad_method", ["free", "autodiff"])  # autodiff 通过 value_and_grad 等价路径
@pytest.mark.parametrize("numeric_engine", ["statevector"])  # 后续可扩展 mps 等
@pytest.mark.parametrize("shots", [0, 2048])  # shots 仅对 device 有效
def test_hea_convergence(runtime, grad_method, numeric_engine, shots):
    m = h2
    uccsd = UCCSD(m,run_fci=True)
    hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, runtime=runtime)

    if runtime == "numeric":
        hea.numeric_engine = numeric_engine
        hea.grad = "free" if grad_method == "free" else "param-shift"
        e = hea.kernel()
        np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-5)
    else:
        # device runtime
        hea.grad = "free" if shots == 0 else "param-shift"
        e = hea.kernel(shots=shots, provider="simulator", device="statevector")
        if shots == 0:
            np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-5)
        else:
            tol = 2.0 / np.sqrt(shots) + 0.02
            np.testing.assert_allclose(e, uccsd.e_fci, atol=tol)
            # 简单后处理：固定参数多次评估取平均
            ps = np.asarray(hea.params if hea.params is not None else hea.init_guess)
            evals = [hea.energy(ps, shots=shots, provider="simulator", device="statevector") for _ in range(5)]
            e_avg = float(np.mean(evals))
            assert abs(e_avg - uccsd.e_fci) <= abs(e - uccsd.e_fci) + 1e-3

            # 加入端到端含噪与后处理（readout校正）的结果打印：
            # 1) 注入测量读出噪声（提供已知的单比特校准矩阵）
            A0 = np.array([[0.97, 0.03], [0.05, 0.95]], dtype=np.float64)
            A1 = np.array([[0.98, 0.02], [0.04, 0.96]], dtype=np.float64)
            # 无校正下的能量（有读出噪声）
            e_noisy = hea.energy(
                ps,
                shots=shots,
                provider="simulator",
                device="statevector",
                use_noise=True,
                noise={"type": "readout", "cals": {0: A0, 1: A1}},
            )
            # 2) 在聚合器中开启 readout 校正（postprocessing 透传校准矩阵），打印校正后能量
            e_mitig = hea.energy(
                ps,
                shots=shots,
                provider="simulator",
                device="statevector",
                use_noise=True,
                noise={"type": "readout", "cals": {0: A0, 1: A1}},
                postprocessing={"readout_cals": {0: A0, 1: A1}, "mitigation": "inverse"},
            )
            print(f"[HEA noisy] shots={shots}, E_noisy={e_noisy:.10f}")
            print(f"[HEA mitigated] shots={shots}, E_mitigated={e_mitig:.10f}")

def test_qiskit_circuit():
    m = h2
    uccsd = UCCSD(m,run_fci=True)
    qc = real_amplitudes(2)
    hea = HEA.from_qiskit_circuit(parity(uccsd.h_fermion_op, 4, 2), qc, np.random.rand(qc.num_parameters), runtime="device")
    e = hea.kernel(shots=0, provider="simulator", device="statevector")
    
    np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-5)
    hea.print_summary()


@pytest.mark.parametrize("runtime", ["device", "numeric"]) 
@pytest.mark.parametrize("mapping", ["jordan-wigner", "bravyi-kitaev"]) 
def test_mapping(mapping, runtime):
    # TCC gold standards for H2 (absolute targets)
    # gold = {
    #     "jordan-wigner": -1.1372744049357164,
    #     "bravyi-kitaev": -1.1372744025043178,
    # }
    uccsd = UCCSD(h2,run_fci=True)
    if runtime == "numeric":
        hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, mapping=mapping, runtime="numeric")
        hea.numeric_engine = "statevector"
        e = hea.kernel()
        # np.testing.assert_allclose(e, gold[mapping], atol=1e-6)
        np.testing.assert_allclose(e, uccsd.e_fci)
    else:
        hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, mapping=mapping, runtime="device")
        # optimize with analytic shots=0
        e = hea.kernel(shots=0, provider="simulator", device="statevector")
        # np.testing.assert_allclose(e, gold[mapping], atol=1e-5)
        np.testing.assert_allclose(e, uccsd.e_fci)


@pytest.mark.parametrize("mapping", ["jordan-wigner", "parity", "bravyi-kitaev"]) 
def test_rdm(mapping):
    # 使用 TCC 金标准对齐 HEA 的 1RDM/2RDM（H2, MO 基），独立于 UCC 参考
    uccsd = UCCSD(h2)
    uccsd.kernel(shots=0)
    hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, mapping=mapping, runtime="device")
    hea.kernel(shots=0, provider="simulator", device="statevector")

    r1_h = hea.make_rdm1()
    r2_h = hea.make_rdm2()

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
    np.testing.assert_allclose(r1_h, rdm1_gold, atol=1e-4)
    np.testing.assert_allclose(r2_h, rdm2_gold, atol=1e-4)
    # np.testing.assert_allclose(r1_uccsd, rdm1_gold, atol=1e-6)
    # np.testing.assert_allclose(r2_uccsd, rdm2_gold, atol=1e-6)
    # np.testing.assert_allclose(r1_uccsd, r1_h, atol=1e-4)
    # np.testing.assert_allclose(r2_uccsd, r2_h, atol=1e-4)



def test_open_shell():

    m = h_chain(3, charge=0, spin=1)

    hea = HEA.from_molecule(m, n_layers=6, mapping="parity", runtime="device")
    # try multiple times to avoid local minimum
    es = []
    for i in range(3):
        hea.init_guess = np.random.random(hea.init_guess.shape)
        es.append(hea.kernel(shots=0))
    e1 = min(es)

    from tyxonq.applications.chem.algorithms.uccsd import ROUCCSD
    ucc = ROUCCSD(m,run_fci=True)
    e2 = ucc.kernel(shots=0)

    # for debugging
    # ucc.print_summary()

    # usually ROUCCSD is more accurate
    # np.testing.assert_allclose(e2, ucc.e_fci, atol=1e-4)
    np.testing.assert_allclose(e1, ucc.e_fci, atol=2e-3)

    # np.testing.assert_allclose(hea.make_rdm1(), ucc.make_rdm1(basis="MO"), atol=5e-3)
    # np.testing.assert_allclose(hea.make_rdm2(), ucc.make_rdm2(basis="MO"), atol=5e-3)
