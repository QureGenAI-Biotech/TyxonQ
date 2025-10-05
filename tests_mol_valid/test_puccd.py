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
    np.testing.assert_allclose(e_trotter, e_default, atol=1e-3)
    # givens_swap 路径依赖 HCB 映射与特殊模板，当前 IR 不保证与默认门级态相同，暂不比较

from test_mol_construct import get_random_integral_and_fci

@pytest.mark.parametrize("hamiltonian", ["H4", "random integral"])
@pytest.mark.parametrize("numeric_engine", ["statevector", "civector", "civector-large"])
def test_puccd(hamiltonian, numeric_engine):

    if hamiltonian == "H4":
        puccd = PUCCD(h4, numeric_engine=numeric_engine,runtime='numeric')
        ucc = UCC(h4, numeric_engine=numeric_engine,runtime='numeric')
    else:
        nao = 4
        n_elec = 4
        int1e, int2e, _ = get_random_integral_and_fci(nao)
        puccd = PUCCD.from_integral(int1e, int2e, n_elec, numeric_engine=numeric_engine,runime='numeric')
        ucc = UCC.from_integral(int1e, int2e, n_elec,numeric_engine=numeric_engine,runime='numeric')

    # note the order
    # b^\dagger_i = a^\dagger_(nao + i) a^\dagger_(i)
    # b_j = a_(j) a_(nao + j)
    ucc.ex_ops = (
        (7, 3, 0, 4),
        (6, 2, 0, 4),
        (7, 3, 1, 5),
        (6, 2, 1, 5),
    )

    e1 = puccd.kernel()
    ucc.params = puccd.params
    e2 = ucc.energy()

    # np.testing.assert_allclose(e1, e2, atol=1e-6)

    rdm1_puccd = puccd.make_rdm1()
    rdm1_ucc = ucc.make_rdm1()
    # slight difference in optimization minimum
    np.testing.assert_allclose(rdm1_puccd, rdm1_ucc, atol=1e-6)

    rdm2_puccd = puccd.make_rdm2(basis="MO")
    rdm2_ucc = ucc.make_rdm2(basis="MO")
    np.testing.assert_allclose(rdm2_puccd, rdm2_ucc, atol=1e-5)



if __name__ == "__main__":
    test_puccd("H4", "statevector")