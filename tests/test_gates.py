import sys
import os
import numpy as np

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tyxonq as tq


def test_rgate(highp):
    np.testing.assert_almost_equal(
        tq.gates.r_gate(1, 2, 3).tensor, tq.gates.rgate_theoretical(1, 2, 3).tensor
    )


def test_phase_gate():
    c = tq.Circuit(1)
    c.h(0)
    c.phase(0, theta=np.pi / 2)
    np.testing.assert_allclose(c.state()[1], 0.7071j, atol=1e-4)


def test_cu_gate():
    c = tq.Circuit(2)
    c.cu(0, 1, theta=np.pi / 2, phi=-np.pi / 4, lbd=np.pi / 4)
    m = c.matrix()
    print(m)
    np.testing.assert_allclose(m[2:, 2:], tq.gates._wroot_matrix, atol=1e-5)
    np.testing.assert_allclose(m[:2, :2], np.eye(2), atol=1e-5)


def test_get_u_parameter(highp):
    for _ in range(6):
        hermitian = np.random.uniform(size=[2, 2])
        hermitian += np.conj(np.transpose(hermitian))
        unitary = tq.backend.expm(hermitian * 1.0j)
        params = tq.gates.get_u_parameter(unitary)
        unitary2 = tq.gates.u_gate(theta=params[0], phi=params[1], lbd=params[2])
        ans = unitary2.tensor
        unitary = unitary / np.exp(1j * np.angle(unitary[0, 0]))
        np.testing.assert_allclose(unitary, ans, atol=1e-3)


def test_ided_gate():
    g = tq.gates.rx.ided()
    np.testing.assert_allclose(
        tq.backend.reshapem(g(theta=0.3).tensor),
        np.kron(np.eye(2), tq.gates.rx(theta=0.3).tensor),
        atol=1e-5,
    )
    g1 = tq.gates.rx.ided(before=False)
    np.testing.assert_allclose(
        tq.backend.reshapem(g1(theta=0.3).tensor),
        np.kron(tq.gates.rx(theta=0.3).tensor, np.eye(2)),
        atol=1e-5,
    )


def test_fsim_gate():
    theta = 0.2
    phi = 0.3
    c = tq.Circuit(2)
    c.iswap(0, 1, theta=-theta)
    c.cphase(0, 1, theta=-phi)
    m = c.matrix()
    ans = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.95105654 + 0.0j, 0.0 - 0.309017j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 - 0.309017j, 0.95105654 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.9553365 - 0.29552022j],
        ]
    )
    np.testing.assert_allclose(m, ans, atol=1e-5)
    print(m)


def test_exp_gate():
    c = tq.Circuit(2)
    c.exp(
        0,
        1,
        unitary=tq.gates.array_to_tensor(
            np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        ),
        theta=tq.gates.num_to_tensor(np.pi / 2),
    )
    np.testing.assert_allclose(c.wavefunction()[0], -1j)


def test_any_gate():
    c = tq.Circuit(2)
    c.any(0, unitary=np.eye(2))
    np.testing.assert_allclose(c.expectation((tq.gates.z(), [0])), 1.0)


def test_iswap_gate():
    t = tq.gates.iswap_gate().tensor
    ans = np.array([[1.0, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1.0]])
    np.testing.assert_allclose(t, ans.reshape([2, 2, 2, 2]), atol=1e-5)
    t = tq.gates.iswap_gate(theta=0).tensor
    np.testing.assert_allclose(t, np.eye(4).reshape([2, 2, 2, 2]), atol=1e-5)


def test_gate_list():
    assert tq.Circuit.sgates == tq.abstractcircuit.sgates


def test_controlled():
    xgate = tq.gates.x
    cxgate = xgate.controlled()
    ccxgate = cxgate.controlled()
    assert ccxgate.n == "ccx"
    assert ccxgate.ctrl == [1, 1]
    np.testing.assert_allclose(
        ccxgate().tensor, tq.backend.reshape2(tq.gates._toffoli_matrix)
    )
    ocxgate = cxgate.ocontrolled()
    c = tq.Circuit(3)
    c.x(0)
    c.any(1, 0, 2, unitary=ocxgate())
    np.testing.assert_allclose(c.expectation([tq.gates.z(), [2]]), -1, atol=1e-5)
    print(c.to_qir()[1])


def test_variable_controlled():
    crxgate = tq.gates.rx.controlled()
    c = tq.Circuit(2)
    c.x(0)
    tq.Circuit.crx_my = tq.Circuit.apply_general_variable_gate_delayed(crxgate)
    c.crx_my(0, 1, theta=0.3)
    np.testing.assert_allclose(
        c.expectation([tq.gates.z(), [1]]), 0.95533645, atol=1e-5
    )
    assert c.to_qir()[1]["name"] == "crx"


def test_adjoint_gate():
    np.testing.assert_allclose(
        tq.gates.sd().tensor, tq.backend.adjoint(tq.gates._s_matrix)
    )
    assert tq.gates.td.n == "td"


def test_rxx_gate():
    c1 = tq.Circuit(3)
    c1.rxx(0, 1, theta=1.0)
    c1.ryy(0, 2, theta=0.5)
    c1.rzz(0, 1, theta=-0.5)
    c2 = tq.Circuit(3)
    c2.exp1(0, 1, theta=1.0 / 2, unitary=tq.gates._xx_matrix)
    c2.exp1(0, 2, theta=0.5 / 2, unitary=tq.gates._yy_matrix)
    c2.exp1(0, 1, theta=-0.5 / 2, unitary=tq.gates._zz_matrix)
    np.testing.assert_allclose(c1.state(), c2.state(), atol=1e-5)
