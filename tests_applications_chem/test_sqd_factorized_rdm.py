"""用独立 PySCF 收缩逐自旋块验证 selected-CI 密度。"""
import numpy as np
import pytest
from pyscf.fci import cistring, selected_ci
from tyxonq.applications.chem.algorithms.sqd.gpu.rdm_factorized import spin_rdms, spin_sum


@pytest.mark.parametrize("algorithm", ["auto", "pyscf"])
@pytest.mark.parametrize("n,ne,cap", [(4,(1,1),2), (5,(3,2),6), (6,(2,4),9), (4,(4,4),1), (4,(2,0),3)])
def test_spin_blocks_match_pyscf(n, ne, cap, algorithm):
    rng = np.random.default_rng(1994)
    labels = tuple(np.sort(rng.permutation(cistring.make_strings(range(n),k))[:cap]) for k in ne)
    c = rng.normal(size=tuple(map(len,labels)))
    vec = selected_ci._as_SCIvector(c/np.linalg.norm(c), labels)
    report = {}
    spin = spin_rdms(vec, n, ne, report=report, algorithm=algorithm)
    assert report["algorithm"] == (
        "spin_links_fp64" if algorithm == "auto" and min(ne) > 0 else "pyscf_spin_blocks"
    )
    reference = selected_ci.make_rdm1s(vec,n,ne), selected_ci.make_rdm2s(vec,n,ne)
    for blocks, other in zip(spin, reference):
        for x, y in zip(blocks, other):
            np.testing.assert_allclose(x,y,rtol=0,atol=2e-14)
    d1, d2 = spin_sum(*spin)
    np.testing.assert_allclose(d1, selected_ci.make_rdm1(vec,n,ne), atol=2e-14)
    np.testing.assert_allclose(d2, selected_ci.make_rdm2(vec,n,ne), atol=2e-14)


def test_pyscf_golden_never_calls_factorized_contractions(monkeypatch):
    labels = (np.array([1]), np.array([1]))
    vec = selected_ci._as_SCIvector(np.ones((1, 1)), labels)

    def forbidden(*args, **kwargs):
        raise AssertionError("pure PySCF golden must not use production contractions")

    monkeypatch.setattr(
        "tyxonq.applications.chem.algorithms.sqd.gpu.rdm_factorized._one_spin", forbidden
    )
    monkeypatch.setattr(
        "tyxonq.applications.chem.algorithms.sqd.gpu.rdm_factorized._mixed", forbidden
    )
    report = {}
    spin_rdms(vec, 1, (1, 1), report=report, algorithm="pyscf")
    assert report["algorithm"] == "pyscf_spin_blocks"
    with pytest.raises(ValueError, match="RDM algorithm"):
        spin_rdms(vec, 1, (1, 1), algorithm="links")
