import numpy as np
from pyscf import fci
from pyscf.scf import RHF
from pyscf.mcscf import CASCI
import pytest

from tyxonq.applications.chem import UCCSD, KUPCCGSD, ROUCCSD
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import get_integral_from_hf, random_integral
from tyxonq.applications.chem.molecule import _random, h4, h8, h_chain, c4h4,h2
from tyxonq.applications.chem.chem_libs.hamiltonians_chem_library.hamiltonian_builders import canonical_mo_coeff


def get_random_integral_and_fci(n):
    nao = n_elec = n
    int1e, int2e = random_integral(nao)
    e, _ = fci.direct_spin1.kernel(int1e, int2e, nao, n_elec)
    return int1e, int2e, e


@pytest.mark.parametrize("hamiltonian", ["H2", "H2 integral", "random integral"])
@pytest.mark.parametrize("ansatz_str", ["UCCSD", "kUpCCGSD"])
def test_ucc(hamiltonian, ansatz_str):
    m = h2
    nao = m.nao
    n_elec = m.nelectron
    if ansatz_str == "UCCSD":
        ansatz = UCCSD
        kwargs = {}
        atol = 1e-4
        if hamiltonian == "random integral":
            atol = 1e-3
    elif ansatz_str == "kUpCCGSD":
        ansatz = KUPCCGSD
        kwargs = {"n_tries": 1}
        # too few tries and the error could be large
        atol = 3e-3
    else:
        assert False
    if hamiltonian == "H2":
        # from mol
        ucc = ansatz(m, **kwargs)
    elif hamiltonian == "H2 integral":
        if ansatz_str == 'UCCSD':
            atol = 2e-2
        int1e = m.intor("int1e_kin") + m.intor("int1e_nuc")
        int2e = m.intor("int2e")
        ovlp = m.intor("int1e_ovlp")
        n_elec = m.nelectron
        e_nuc = m.energy_nuc()
        ucc = ansatz.from_integral(int1e, int2e, n_elec, e_nuc, ovlp, **kwargs)
    else:
        int1e, int2e, _ = get_random_integral_and_fci(nao)
        ucc = ansatz.from_integral(int1e, int2e, n_elec, **kwargs)
    e = ucc.kernel(shots=0)
    np.testing.assert_allclose(e, ucc.e_fci, atol=atol)


def test_rdm():
    m = _random(4, 4)
    hf = RHF(m)
    hf.chkfile = None
    hf.kernel()
    hf.mo_coeff = canonical_mo_coeff(hf.mo_coeff)
    my_fci = fci.FCI(hf)
    e1, fcivec = my_fci.kernel()
    # rdm in MO basis
    rdm1, rdm2 = fci.direct_spin1.make_rdm12(fcivec, 4, 4)
    uccsd = UCCSD(m)
    e2 = uccsd.kernel()
    np.testing.assert_allclose(e1, e2, atol=1e-3)
    rdm1_uccsd = uccsd.make_rdm1(basis="MO")
    rdm2_uccsd = uccsd.make_rdm2(basis="MO")
    np.testing.assert_allclose(rdm1_uccsd, rdm1, atol=5e-3)
    np.testing.assert_allclose(rdm2_uccsd, rdm2, atol=5e-3)

    int1e, int2e, e_core = get_integral_from_hf(hf)
    rdm_e = int1e.ravel() @ rdm1_uccsd.ravel() + 1 / 2 * int2e.ravel() @ rdm2_uccsd.ravel() + e_core
    np.testing.assert_allclose(e2, rdm_e, atol=1e-5)


def test_active_space():
    m = h_chain(12)
    m.verbose = 0
    ncas = 4
    nelecas = 2

    hf = m.HF()
    hf.kernel()
    hf.mo_coeff = canonical_mo_coeff(hf.mo_coeff)
    casci = CASCI(hf, ncas, nelecas)
    e1 = casci.kernel()[0]
    uccsd = UCCSD(m, active_space=(nelecas, ncas))
    e2 = uccsd.kernel()
    np.testing.assert_allclose(e2, e1, atol=1e-5)
    np.testing.assert_allclose(uccsd.make_rdm1(), casci.make_rdm1(), atol=1e-3)
    from pyscf.mcscf.addons import make_rdm12

    _, rdm2 = make_rdm12(casci)
    np.testing.assert_allclose(uccsd.make_rdm2(), rdm2, atol=1e-3)

    uccsd.print_summary(include_circuit=True)


def test_active_space_aslst():
    m = h_chain(12)
    m.verbose = 0
    ncas = 4
    nelecas = 4
    aslst = [0, 1, 8, 10]
    uccsd = UCCSD(m, active_space=(nelecas, ncas), aslst=aslst)
    casci = CASCI(uccsd.hf, ncas, nelecas)
    mo = casci.sort_mo(aslst, base=0)
    e1 = casci.kernel(mo)[0]
    e2 = uccsd.kernel()
    uccsd.print_summary()
    np.testing.assert_allclose(e2, e1, atol=1e-5)


def test_get_circuit():
    """验证不同分解选项生成的电路在理想模拟器下等价。

    说明：架构下无法直接从电路对象访问“量子设备纯态”，
    这里用本地 statevector 模拟器对 IR Circuit 求态，再做数值比对。
    """
    from tyxonq.devices.simulators.statevector.engine import StatevectorEngine

    uccsd = UCCSD(h4)
    params = np.random.rand(uccsd.n_params)
    eng = StatevectorEngine()

    s1 = np.asarray(eng.state(uccsd.get_circuit(params)))
    s2 = np.asarray(eng.state(uccsd.get_circuit(params, decompose_multicontrol=True)))
    np.testing.assert_allclose(s2, s1, atol=1e-10)
    # trotter 分解与门级实现一般不严格相同；在小角度极限下两者能量应近似一致
    params_small = 1e-3 * np.random.randn(uccsd.n_params)
    c_trotter = uccsd.get_circuit(params_small, trotter=True)
    e_default = eng.expval(uccsd.get_circuit(params_small), uccsd.h_qubit_op) + uccsd.e_core
    e_trotter = eng.expval(c_trotter, uccsd.h_qubit_op) + uccsd.e_core
    np.testing.assert_allclose(e_trotter, e_default, atol=1e-4)


@pytest.mark.parametrize("init_method", ["mp2", "ccsd", "zeros", "fe"])
def test_init_guess(init_method):
    pick_ex2 = sort_ex2 = True
    if init_method == "zeros":
        pick_ex2 = sort_ex2 = False
    ucc = UCCSD(h4, init_method, pick_ex2=pick_ex2, sort_ex2=sort_ex2)
    e = ucc.kernel()
    np.testing.assert_allclose(e, ucc.e_fci, atol=1e-4)


def test_mf_input():
    m = c4h4(1.46, 1.46, basis="ccpvdz", symmetry=False)
    hf = RHF(m)
    hf.kernel()
    dm, _, stable, _ = hf.stability(return_status=True)
    while not stable:
        print("Instability detected in RHF")
        hf.kernel(dm)
        dm, _, stable, _ = hf.stability(return_status=True)
        if stable:
            break
    ucc = UCCSD(hf, active_space=(4, 4))
    e = ucc.kernel()
    np.testing.assert_allclose(ucc.e_hf, -153.603405, atol=1e-4)
    np.testing.assert_allclose(e, ucc.e_fci, atol=2e-2)


@pytest.mark.parametrize("method", [UCCSD, ROUCCSD])
def test_pyscf_solver(method):
    if method == UCCSD:
        m = h8
        ncas = 4
        nelecas = 4
        hf = m.HF()
    else:
        m = h_chain(7, spin=1)
        ncas = 4
        nelecas = 3
        hf = m.ROHF()

    hf.kernel()
    e_ref = ROUCCSD(hf, active_space=(nelecas, ncas)).kernel()
    mc = CASCI(hf, ncas, nelecas)
    mc.fcisolver = method.as_pyscf_solver()
    mc.kernel()
    np.testing.assert_allclose(mc.e_tot, e_ref, atol=1e-4)
