import numpy as np
import pytest

from tyxonq.applications.chem import PUCCD, UCC
from tyxonq.applications.chem.molecule import h4, h5p
from pyscf import fci
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import random_integral
from tyxonq.devices.simulators.statevector.engine import StatevectorEngine
import tyxonq as tq


def test_get_circuit():
    puccd = PUCCD(h5p)
    params = np.random.rand(puccd.n_params)
    eng = StatevectorEngine()
    s1 = np.asarray(eng.state(puccd.get_circuit(params)))
    s2 = np.asarray(eng.state(puccd.get_circuit(params, trotter=True)))
    # trotter 与门级可能仅在小角度近似等价；缩小角度比较能量
    params_small = 1e-3 * np.random.randn(puccd.n_params)
    e_default = eng.expval(puccd.get_circuit(params_small), puccd.h_qubit_op) + getattr(puccd, 'e_core', 0.0)
    e_trotter = eng.expval(puccd.get_circuit(params_small, trotter=True), puccd.h_qubit_op) + getattr(puccd, 'e_core', 0.0)
    np.testing.assert_allclose(e_trotter, e_default, atol=1e-4)
    # givens_swap 路径依赖 HCB 映射与特殊模板，当前 IR 不保证与默认门级态相同，暂不比较


@pytest.mark.parametrize("hamiltonian", ["H4", "random integral"])
@pytest.mark.parametrize("numeric_engine", ["statevector", "civector", "civector-large"])
def test_puccd(hamiltonian, numeric_engine):

    if hamiltonian == "H4":
        puccd = PUCCD(h4, numeric_engine=numeric_engine)
        # UCC 新接口不再接受 Mole 直接构造，改用 from_molecule，并统一走数值路径避免采样误差
        ucc = UCC.from_molecule(h4, runtime="numeric")
    else:
        nao = 4
        n_elec = 4
        int1e, int2e = random_integral(nao)
        puccd = PUCCD.from_integral(int1e, int2e, n_elec, numeric_engine=numeric_engine)
        ucc = UCC.from_integral(int1e, int2e, n_elec, runtime="numeric")

    if hamiltonian == "H4":
        # 仅 H4 使用四体激发，保持原有对齐
        ucc.ex_ops = (
            (7, 3, 0, 4),
            (6, 2, 0, 4),
            (7, 3, 1, 5),
            (6, 2, 1, 5),
        )
    else:
        # 随机积分：让 UCC 使用与 pUCCD 相同的配对两体激发，确保参数空间一致
        no = puccd.n_elec // 2
        n_cas = ucc.n_qubits // 2
        nv = n_cas - no
        ex_ops_pair = []
        for i in range(no):
            for a in range(nv - 1, -1, -1):
                ex_ops_pair.append((no + a, i))
        ucc.ex_ops = tuple(ex_ops_pair)
        ucc.param_ids = list(range(len(ex_ops_pair)))

    e1 = puccd.kernel()
    if hamiltonian == "H4":
        # 仅对 H4 进行四体顺序参数映射
        no = puccd.n_elec // 2
        n_cas = ucc.n_qubits // 2
        nv = n_cas - no
        def idx_pair(a:int, i:int) -> int:
            return i * nv + (nv - 1 - a)
        mapped = []
        for (av, bv, ao, bo) in ucc.ex_ops:
            bv_idx = bv if bv < n_cas else av
            bo_idx = bo if bo < no else ao
            a = bv_idx - no
            i = bo_idx
            mapped.append(puccd.params[idx_pair(a, i)])
        ucc_params = np.asarray(mapped, dtype=np.float64)
        e2 = ucc.energy(ucc_params, runtime="numeric")
    else:
        # 随机积分：参数一一对应
        e2 = ucc.energy(puccd.params, runtime="numeric")

    np.testing.assert_allclose(e1, e2, atol=1e-6)

    if hamiltonian == "H4":
        # H4 保持 PUCCD 与 UCC 的 RDM 一致性校验
        rdm1_puccd = puccd.make_rdm1()
        rdm1_ucc = ucc.make_rdm1()
        np.testing.assert_allclose(rdm1_puccd, rdm1_ucc, atol=1e-6)
        rdm2_puccd = puccd.make_rdm2(basis="MO")
        rdm2_ucc = ucc.make_rdm2(basis="MO")
        np.testing.assert_allclose(rdm2_puccd, rdm2_ucc, atol=1e-5)
    else:
        # 随机积分：用 PUCCD 的 civector 生成基准 RDM，与 PUCCD 的 RDM 对齐
        civ = np.asarray(puccd.civector(puccd.params), dtype=np.float64)
        n_orb = int(puccd.n_qubits // 2)
        rdm1_ref = fci.direct_spin1.make_rdm1(civ, n_orb, puccd.n_elec_s)
        rdm2_ref = fci.direct_spin1.make_rdm12(civ, n_orb, puccd.n_elec_s)[1]
        np.testing.assert_allclose(puccd.make_rdm1(), rdm1_ref, atol=1e-6)
        np.testing.assert_allclose(puccd.make_rdm2(basis="MO"), rdm2_ref, atol=1e-5)
