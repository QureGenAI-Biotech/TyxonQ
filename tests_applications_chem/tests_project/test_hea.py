import numpy as np
import pytest
from qiskit.circuit.library import RealAmplitudes

from tyxonq.applications.chem import UCCSD, HEA, parity
from tyxonq.applications.chem.molecule import h2, h_chain

import tyxonq as tq


@pytest.mark.parametrize(
    "numeric_engine",
    [
        "tensornetwork",
        "tensornetwork-noise",
        "tensornetwork-shot",
        "tensornetwork-noise&shot",
    ],
)
@pytest.mark.parametrize("backend_str", ["pytorch", "numpy"])
@pytest.mark.parametrize("grad", ["param-shift", "autodiff", "free"])
def test_hea(numeric_engine, backend_str, grad):
    prev = getattr(tq, "backend", None)
    try:
        try:
            tq.set_backend(backend_str)
        except Exception:
            pytest.xfail(f"Backend {backend_str} not available")
        if backend_str in ["numpy", "cupy"] and grad == "autodiff":
            pytest.xfail("Incompatible backend and gradient method")
        if numeric_engine in ["tensornetwork-shot", "tensornetwork-noise&shot"] and grad == "autodiff":
            pytest.xfail("Incompatible numeric_engine and gradient method")
        m = h2
        uccsd = UCCSD(m)
        hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, numeric_engine=numeric_engine)
        hea.grad = grad
        e = hea.kernel()
        atol = 0.1
        if numeric_engine == "tensornetwork-noise&shot" and grad == "free":
            atol *= 2
        np.testing.assert_allclose(e, uccsd.e_fci, atol=atol)
    finally:
        if prev is not None:
            try:
                tq.set_backend(prev.name if hasattr(prev, "name") else prev)
            except Exception:
                pass


def test_qiskit_circuit():
    m = h2
    uccsd = UCCSD(m)
    circuit = RealAmplitudes(2)
    hea = HEA(parity(uccsd.h_fermion_op, 4, 2), circuit, np.random.rand(circuit.num_parameters))
    e = hea.kernel()
    np.testing.assert_allclose(e, uccsd.e_fci, atol=1e-5)
    hea.print_summary()


@pytest.mark.parametrize("mapping", ["jordan-wigner", "bravyi-kitaev"])
def test_mapping(mapping):
    uccsd = UCCSD(h2)
    hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, mapping=mapping)
    e = hea.kernel()
    np.testing.assert_allclose(e, uccsd.e_fci)


@pytest.mark.parametrize("mapping", ["jordan-wigner", "parity", "bravyi-kitaev"])
def test_rdm(mapping):
    # placeholder: original test depended on legacy reset_backend; keep structure
    assert mapping in {"jordan-wigner", "bravyi-kitaev"}
    uccsd = UCCSD(h2)
    uccsd.kernel()
    hea = HEA.ry(uccsd.int1e, uccsd.int2e, uccsd.n_elec, uccsd.e_core, 3, mapping=mapping)
    hea.kernel()
    np.testing.assert_allclose(hea.make_rdm1(), uccsd.make_rdm1(basis="MO"), atol=1e-4)
    np.testing.assert_allclose(hea.make_rdm2(), uccsd.make_rdm2(basis="MO"), atol=1e-4)


def test_open_shell():

    m = h_chain(3, charge=0, spin=1)

    hea = HEA.from_molecule(m, n_layers=6, mapping="parity")
    # try multiple times to avoid local minimum
    es = []
    for i in range(3):
        hea.init_guess = np.random.random(hea.init_guess.shape)
        es.append(hea.kernel())
    e1 = min(es)

    ucc = ROUCCSD(m)
    e2 = ucc.kernel()

    # for debugging
    # ucc.print_summary()

    # usually ROUCCSD is more accurate
    np.testing.assert_allclose(e2, ucc.e_fci, atol=1e-4)
    np.testing.assert_allclose(e1, ucc.e_fci, atol=2e-3)

    np.testing.assert_allclose(hea.make_rdm1(), ucc.make_rdm1(basis="MO"), atol=5e-3)
    np.testing.assert_allclose(hea.make_rdm2(), ucc.make_rdm2(basis="MO"), atol=5e-3)
